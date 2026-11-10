#!/usr/bin/env python3
"""Автономный заезд по записанному маршруту (route.json от teach.py).

    cd ~/hack && python3 mission.py            # заезд с распознаванием и объездом сугробов
    cd ~/hack && python3 mission.py --dry      # только проезд маршрута (проверка навигации)
    cd ~/hack && python3 mission.py --direct   # ехать только по точкам осмотра (+ финиш),
                                               # без старых точек обхода сугробов
    флаг --nosnow — сугробы не объезжать вообще (большие колёса их переезжают):
                    без камеры, без проверок наклона и застревания; стенки/стойки — по лидару

Робот ставится на старт ТАК ЖЕ, как при записи маршрута.
На каждой точке осмотра: стоп -> несколько кадров -> голосование ->
лог в формате регламента (экран + logs/mission_*.txt) + картинка logs/armature_N.jpg
с рамками (доказательство для судей).
Ctrl+C — аварийный стоп.
"""
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

from nav import Nav, norm, _load_local
from aruco_detect import detect_aruco
from animals_detect import AnimalDetector
from inspect_vision import read_plate, read_gauge, draw_results, put_text

ROUTE = "route.json"
MODEL = "best_v2.onnx"
ARUCO_DICT = "DICT_6X6_50"   # найден в заезде; None = автоподбор (медленнее)
MIN_MARKER_FRAC = 0.04       # маркер меньше 4% ширины кадра — это чужая арматура вдали
FRAMES = 6                   # кадров на осмотр
SETTLE = 0.6                 # с — дать роботу успокоиться перед кадрами
ANIMAL_MIN_SHARE = 0.4       # животное «есть», если найдено хотя бы на 40% кадров
SCAN_ANGLES = [0]            # например [0, -30, 30] — доглядеть животное сбоку от арматуры
LOG_DIR = "logs"
ANIMAL_CONF = 0.75           # уверенность модели, ниже — не животное (0.5 ловило робота на фоне)
ANIMAL_MIN_BOTTOM = 0.65     # низ рамки животного должен быть в нижних 35% кадра — фигурка стоит
                             # на полу у арматуры; всё, что «висит» выше (роботы, люди на фоне), — мимо
SHOW_CONF = True             # писать уверенность (0.86) рядом с рамками на скринах
TIME_LIMIT = 270             # с — после этого бросаем оставшиеся осмотры и едем на старт
VIEW_DIST = 0.45             # м — примерно на таком расстоянии от арматуры записаны точки осмотра
# запасные точки обзора, если записанная занята сугробом: (поворот вокруг арматуры °, +дальше м)
ALT_VIEWS = [(0, 0.20), (30, 0.0), (-30, 0.0), (30, 0.20), (-30, 0.20), (50, 0.10), (-50, 0.10)]
VIEW_BUDGET = 60             # с — максимум на поиск точки обзора одной арматуры
NEAR_VIEW = 0.22             # м — для осмотра хватит встать так близко к точке обзора

LOCAL = _load_local(globals())           # ANIMAL_CONF и т.п. можно задать в local_settings.json


def animals_on_floor(found, img_h):
    """Отбросить «животных», которые не стоят на полу (робот/человек на заднем плане)."""
    return [f for f in found if f["box"][3] >= ANIMAL_MIN_BOTTOM * img_h]


class Mission:
    def __init__(self, dry=False):
        self.dry = dry
        nosnow = "--nosnow" in sys.argv     # большие колёса: сугробы переезжаем, не объезжаем
        self.nav = Nav(camera=not nosnow, snow_avoid=not nosnow)
        self._msg = None
        self._msg_t = 0.0
        self.nav.create_subscription(CompressedImage, "/front_camera/image_raw/compressed",
                                     self._on_cam, qos_profile_sensor_data)
        self.animals = None if dry else AnimalDetector(MODEL, conf_th=ANIMAL_CONF)
        self.aruco_dict = ARUCO_DICT
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = f"{LOG_DIR}/run_{stamp}"            # своя папка на каждый заезд
        os.makedirs(f"{self.run_dir}/raw", exist_ok=True)
        self.log_path = f"{self.run_dir}/mission.txt"
        self.log_f = open(self.log_path, "w", encoding="utf-8")
        self.evidence = []                                 # картинки осмотров для сводки

    # ---- камера / лог ------------------------------------------------------
    def _on_cam(self, msg):
        self._msg = msg
        self._msg_t = time.time()

    def frame(self, after, timeout=2.0):
        """Свежий кадр, пришедший позже момента after."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._msg is not None and self._msg_t > after:
                img = cv2.imdecode(np.frombuffer(self._msg.data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    return img
            time.sleep(0.02)
        return None

    def write(self, text):
        print(text, flush=True)
        self.log_f.write(text + "\n")
        self.log_f.flush()

    def log(self, key, value):
        self.write(f"Время: {datetime.now():%H:%M:%S}\n{key}: {value}")

    # ---- осмотр арматуры ---------------------------------------------------
    def analyze(self, img):
        """Всё распознавание по одному кадру."""
        H, W = img.shape[:2]
        res = {"img": img, "marker": None, "plate": None, "gauge": None, "animals": []}
        exclude = []
        name, ms = detect_aruco(img, self.aruco_dict)
        ms = [m for m in ms if m["side_px"] >= MIN_MARKER_FRAC * W]
        if ms:
            if self.aruco_dict is None:
                self.aruco_dict = name
                print("словарь ArUco:", name, flush=True)
            res["marker"] = max(ms, key=lambda m: m["side_px"])   # ближайший маркер
            for m in ms:
                c, pad = m["corners"], 0.25 * m["side_px"]
                exclude.append((c[:, 0].min() - pad, c[:, 1].min() - pad,
                                c[:, 0].max() + pad, c[:, 1].max() + pad))
        res["gauge"] = read_gauge(img)
        if res["gauge"]:
            exclude.append(res["gauge"]["box"])
        res["plate"] = read_plate(img, exclude=exclude)
        res["animals"] = animals_on_floor(self.animals.detect(img), H)
        res["n"] = sum(x is not None for x in (res["marker"], res["plate"], res["gauge"])) \
            + bool(res["animals"])
        return res

    def inspect(self, k):
        markers, plates, gauges, animal_hits = Counter(), Counter(), Counter(), Counter()
        n_frames = 0
        best = None                       # кадр, где распознано больше всего — для судей
        base_yaw = self.nav.pose()[2]

        for da in SCAN_ANGLES:
            if da:
                self.nav.rotate_to(base_yaw + math.radians(da))
            time.sleep(SETTLE)
            t = time.time()
            for _ in range(FRAMES if da == 0 else max(2, FRAMES // 2)):
                img = self.frame(t)
                if img is None:
                    break
                t = time.time()
                n_frames += 1
                cv2.imwrite(f"{self.run_dir}/raw/armature_{k}_{n_frames:02d}_"
                            f"{datetime.now():%H%M%S}.jpg", img)
                if da == 0:
                    r = self.analyze(img)
                    if r["marker"]:
                        markers[r["marker"]["id"]] += 1
                    if r["plate"]:
                        plates[r["plate"]["text"]] += r["plate"]["score"]
                    if r["gauge"]:
                        gauges[r["gauge"]["color"]] += r["gauge"]["margin"]
                    if best is None or r["n"] > best["n"]:
                        best = r
                    found = r["animals"]
                else:
                    found = animals_on_floor(self.animals.detect(img), img.shape[0])
                    if found and (best is None or not best["animals"]):
                        best = dict(best or {}, img=img, animals=found, n=1,
                                    marker=None, plate=None, gauge=None)
                for sp in {f["ru"] for f in found}:
                    animal_hits[sp] += 1
        if SCAN_ANGLES != [0]:
            self.nav.rotate_to(base_yaw)

        # голосование по кадрам
        marker = markers.most_common(1)[0][0] if markers else None
        plate = plates.most_common(1)[0][0] if plates else None
        gauge = gauges.most_common(1)[0][0] if gauges else None
        animal = None
        if animal_hits and n_frames:
            sp, cnt = animal_hits.most_common(1)[0]
            if cnt / n_frames >= ANIMAL_MIN_SHARE:
                animal = sp.capitalize()

        self.write(f"\n=== Арматура {k} ===")
        self.log("Табличка", plate or "не распознано")
        self.log("Маркер", marker if marker is not None else "не распознан")
        self.log("Манометр", gauge or "не распознано")
        self.log("Животное", f"Есть, {animal}" if animal else "Нет")

        # картинка-доказательство: рамки вокруг всего распознанного + итог внизу
        if best is not None:
            vis = best["img"].copy()
            if best.get("marker"):
                m = best["marker"]
                cv2.polylines(vis, [m["corners"].astype(int)], True, (0, 255, 0), 2)
                x, y = m["corners"][:, 0].min(), m["corners"][:, 1].min()
                put_text(vis, f"маркер {m['id']}", (int(x), max(20, int(y) - 8)), 18,
                         (0, 255, 0))
            draw_results(vis, best.get("plate"), best.get("gauge"), show_conf=SHOW_CONF)
            for f in best.get("animals", []):
                x1, y1, x2, y2 = f["box"]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 128, 0), 2)
                put_text(vis, f"{f['ru']} {f['conf']:.2f}" if SHOW_CONF else f["ru"],
                         (x1, max(20, y1 - 6)), 18,
                         (255, 128, 0))
            txt = (f"#{k}  Табличка: {plate or '—'}   Маркер: "
                   f"{marker if marker is not None else '—'}   Манометр: {gauge or '—'}   "
                   f"Животное: {'Есть, ' + animal if animal else 'Нет'}   "
                   f"{datetime.now():%H:%M:%S}")
            H = vis.shape[0]
            cv2.rectangle(vis, (0, H - 30), (vis.shape[1], H), (0, 0, 0), -1)
            put_text(vis, txt, (6, H - 8), 17, (255, 255, 255))
            cv2.imwrite(f"{self.run_dir}/armature_{k}.jpg", vis)
            self.evidence.append(vis)

    # ---- заезд -------------------------------------------------------------
    def run(self, direct=False):
        with open(ROUTE, encoding="utf-8") as f:
            route = json.load(f)
        if direct:      # только осмотры + последняя точка (старт): обход сугробов — по ходу
            route = [p for p in route[:-1] if p.get("inspect")] + route[-1:]
        self.nav.init_walls()               # запомнить стенки, пока робот ровно на старте
        t0 = time.time()
        self.write(f"Старт: {datetime.now():%H:%M:%S}, точек {len(route)}, "
                   f"осмотров {sum(p['inspect'] for p in route)}")
        k = 0
        for i, p in enumerate(route, 1):
            last = i == len(route)
            if time.time() - t0 > TIME_LIMIT and not last:
                if p.get("inspect"):
                    k += 1
                print(f"-> {i}/{len(route)} пропуск: время {time.time() - t0:.0f} с, "
                      f"еду на старт", flush=True)
                continue
            kind = "ОСМОТР" if p.get("inspect") else "проезд"
            print(f"-> {i}/{len(route)} {kind} x={p['x']:+.2f} y={p['y']:+.2f}  "
                  f"[{time.time() - t0:5.1f} с]", flush=True)
            if p.get("inspect"):
                self.reach_view(p)
                k += 1
                if self.dry:
                    print(f"  (dry) осмотр {k}: стою 1 с", flush=True)
                    time.sleep(1.0)
                else:
                    self.inspect(k)
                continue
            # до финиша даём больше попыток объезда, до проходной точки — меньше
            ok = self.nav.go_to(p["x"], p["y"], timeout=90 if last else 45)
            if not ok:
                x, y, _ = self.nav.pose()
                print(f"  точка {i}: не доехал (стою в x={x:+.2f} y={y:+.2f}), пропускаю",
                      flush=True)
        if route:
            self.nav.rotate_to_deg(route[-1]["yaw_deg"])
        if self.nav.slips:
            print(f"(пробуксовок поймано и вычтено из пути: {self.nav.slips})", flush=True)
        self.write(f"\nФиниш: {datetime.now():%H:%M:%S}, заняло {time.time() - t0:.0f} с")
        self.save_summary()
        self.write(f"Лог и скрины: {os.path.abspath(self.run_dir)}")

    def reach_view(self, p):
        """Доехать до точки осмотра. Если она занята (сугроб), годится любое место
        в 0.3–0.8 м перед арматурой (не дальше ±60° от записанного направления).
        Пробуем запасные точки вокруг арматуры и разворачиваемся камерой на неё."""
        yaw = math.radians(p["yaw_deg"])
        ax, ay = p["x"] + VIEW_DIST * math.cos(yaw), p["y"] + VIEW_DIST * math.sin(yaw)
        t_end = time.time() + VIEW_BUDGET

        def face():
            px, py, _ = self.nav.pose()
            if math.hypot(px - p["x"], py - p["y"]) < 0.05:
                self.nav.rotate_to_deg(p["yaw_deg"])          # ровно на записанной точке
            else:
                self.nav.rotate_to(math.atan2(ay - py, ax - px))  # камерой на арматуру

        def good_spot():
            px, py, _ = self.nav.pose()
            d = math.hypot(px - ax, py - ay)
            off = abs(norm(math.atan2(py - ay, px - ax) - (yaw + math.pi)))
            return 0.30 <= d <= 0.80 and off <= math.radians(60)

        if self.nav.go_to(p["x"], p["y"], timeout=25, max_detours=3, near_ok=NEAR_VIEW) \
                or good_spot():
            face()
            return True
        alts = []
        for rot, extra in ALT_VIEWS:
            back = yaw + math.pi + math.radians(rot)          # направление от арматуры к точке
            d = VIEW_DIST + extra
            alts.append((rot, extra, ax + d * math.cos(back), ay + d * math.sin(back)))
        px, py, _ = self.nav.pose()
        # сначала ближние и смотрящие на арматуру прямее (под углом табличка читается хуже)
        alts.sort(key=lambda v: math.hypot(v[2] - px, v[3] - py) + 0.012 * abs(v[0]))
        for rot, extra, vx, vy in alts:
            if time.time() > t_end:
                break
            if self.nav.near_obstacle(vx, vy):
                continue                                       # там уже упирались
            print(f"  точка осмотра занята — пробую запасную ({rot:+d}°, +{extra:.2f} м)",
                  flush=True)
            if self.nav.go_to(vx, vy, timeout=min(15, max(3, t_end - time.time())),
                              max_detours=1, near_ok=NEAR_VIEW) or good_spot():
                face()
                return True
        print("  ! до арматуры не подобраться — осматриваю отсюда", flush=True)
        face()
        return False

    def save_summary(self):
        """Все осмотры на одной картинке summary.jpg — показать судьям одним кадром."""
        if not self.evidence:
            return
        w, h = 640, 480
        tiles = [cv2.resize(v, (w, h)) for v in self.evidence]
        while len(tiles) % 2:
            tiles.append(np.zeros((h, w, 3), np.uint8))
        rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
        cv2.imwrite(f"{self.run_dir}/summary.jpg", np.vstack(rows))


if __name__ == "__main__":
    m = Mission(dry="--dry" in sys.argv)
    try:
        m.run(direct="--direct" in sys.argv)
    except KeyboardInterrupt:
        print("\nОстановлено вручную")
        m.save_summary()                 # скрины уже сделанных осмотров не пропадут
        print("Скрины:", os.path.abspath(m.run_dir))
    finally:
        m.log_f.close()
        m.nav.close()
