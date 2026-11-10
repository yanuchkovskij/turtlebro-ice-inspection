#!/usr/bin/env python3
"""Сугробы по камере: светлые пятна на полу перед роботом.

Лидар сугробы (5 см) не видит — он выше. Камера видит: сугроб заметно ярче
серого пола, широкий и низкий, лежит на полу и снизу имеет тёмную кромку;
у «льда» (белой плёнки) кромки нет, поэтому лёд не объезжаем.

Проверка и настройка (робот перед сугробом):
    python3 snow.py --ros            # печатает «впереди сугроб / свободно», пишет last_snow.jpg
    python3 snow.py frames/*.jpg     # то же по сохранённым кадрам -> *_snow.jpg

На last_snow.jpg: найденный сугроб закрашен фиолетовым, коридор движения — жёлтый,
линия порога — красная. Поставь робота так, чтобы край сугроба был в ~25 см
перед носом, и посмотри «ближний край y=…» — это значение впиши в TRIGGER_Y.

В своём коде:
    from snow import DriftDetector
    det = DriftDetector()
    r = det.analyze(bgr)   # r["blocked"], r["near_y"], r["left"], r["right"]
"""
import glob
import sys

import cv2
import numpy as np

# ---- настройки (подбираются по кадрам с камеры робота) ------------------------
MODE = "white"             # "white" — светлое пятно ярче пола (не зависит от освещения),
                           # "cream" — по кремовому оттенку, "both" — любое из двух
WHITE_DV = 40              # сугроб ярче пола хотя бы на столько (яркость V, 0..255)
WHITE_S_MAX = 30           # и почти без цвета (насыщенность)
FLOOR_ROWS = 0.10          # нижние 10% кадра — пол перед роботом, по нему меряем яркость пола
H_MIN, H_MAX = 14, 38      # режим cream: оттенок сугроба (кремовый, ~26 в OpenCV)
S_MIN, S_MAX = 12, 55      # режим cream: насыщенность
V_MIN = 140                # режим cream: яркость
HORIZON = 0.35             # выше этой доли высоты кадра не смотрим (стенки, люди)
TRIGGER_Y = 0.85           # ближний край сугроба ниже этой линии (доля высоты) -> объезд
MIN_FRAC = 0.03            # доля коридора, занятая сугробом, чтобы считать «перекрыто»
MIN_WIDTH = 0.20           # сугроб широкий; узкие пятна (фланцы стоек, блики) — не сугроб
MIN_ASPECT = 1.8           # сугроб низкий и широкий (ширина/высота); трубы арматуры — высокие
REQUIRE_RIM = True         # у сугроба внизу тёмная коричневая кромка; у льда её нет
TOP_MIN = 0.30             # сугроб (5 см) целиком ниже уровня камеры; выше — это стойки, люди
SMALL_W = 240              # кадр уменьшаем до этой ширины — быстро


# ---- местные настройки робота: файл local_settings.json рядом со скриптами ----
# Обновления .py-файлов его не трогают, поэтому калибровка не слетает.
def _load_local(names):
    import json, os
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_settings.json")
    try:
        data = json.load(open(p, encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as e:
        print("! local_settings.json не прочитан:", e, flush=True)
        return {}
    used = {k: v for k, v in data.items() if k in names}
    names.update(used)
    return used


LOCAL = _load_local(globals())

class DriftDetector:
    def __init__(self):
        self._shape = None

    def _corridor(self, h, w):
        """Трапеция «куда сейчас поедет робот»: узкая у горизонта, широкая у низа кадра."""
        if self._shape != (h, w):
            m = np.zeros((h, w), bool)
            hz = int(HORIZON * h)
            for y in range(hz, h):
                t = (y - hz) / max(1, h - 1 - hz)
                half = w * (0.07 + 0.25 * t)
                m[y, max(0, int(w / 2 - half)):min(w, int(w / 2 + half))] = True
            self._corr, self._shape = m, (h, w)
        return self._corr

    def mask(self, img):
        k = SMALL_W / img.shape[1]
        small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hue, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        cream = (hue >= H_MIN) & (hue <= H_MAX) & (s >= S_MIN) & (s <= S_MAX) & (v >= V_MIN)
        # яркость пола — по нижней полосе кадра (если там сугроб — берём «пол» потемнее)
        bottom = v[int((1 - FLOOR_ROWS) * v.shape[0]):]
        floor_v = min(float(np.percentile(bottom, 25)), 200.0)   # даже если сугроб наполз на низ кадра
        white = (v.astype(np.int16) >= floor_v + WHITE_DV) & (s <= WHITE_S_MAX)
        m = {"white": white, "cream": cream}.get(MODE, white | cream)
        self._floor_v = floor_v
        m[: int(HORIZON * m.shape[0])] = False
        m = m.astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        # мелочь (блики, фигурки) отбрасываем
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        keep = np.zeros_like(m, bool)
        H_, W_ = m.shape
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if area < 0.01 * m.size or w < MIN_WIDTH * W_:
                continue
            if w / max(1, h) < MIN_ASPECT or y < TOP_MIN * H_:
                continue                       # высокое — это арматура/стенка/человек
            if not self._on_floor(lab == i, v, s, floor_v, hue):
                continue                       # висит в воздухе (труба) — под ним не пол
            keep |= lab == i
        return keep, small

    @staticmethod
    def _on_floor(comp, v, s, floor_v, h=None, max_gap=12):
        """Сугроб лежит на полу и заслоняет его: от его нижнего края до начала сплошного
        пола (если идти от низа кадра вверх) — ничего или только тёмная кромка сугроба.
        Под трубой арматуры в этом зазоре синяя стенка — значит, не сугроб."""
        H_ = comp.shape[0]
        floorish = (np.abs(v.astype(np.int16) - floor_v) < 28) & (s < 40)
        rev = floorish[::-1]
        run = np.where(rev.all(axis=0), H_, np.argmax(~rev, axis=0))   # длина пола снизу
        floor_top = H_ - run                                           # строка, где пол кончился
        cols = np.nonzero(comp.any(axis=0))[0]
        if len(cols) == 0:
            return False
        bottom = H_ - 1 - np.argmax(comp[::-1, cols], axis=0)         # нижний край пятна
        ok = rim = 0
        for c, b0 in zip(cols, bottom):
            gap = floor_top[c] - b0 - 1
            seg = slice(b0 + 1, b0 + 1 + max(0, min(gap, max_gap)) + 2)   # зазор + пару строк
            if (v[seg, c].astype(np.int16) < floor_v - 35).any():
                rim += 1                                               # тёмная кромка есть
            if gap <= 1:
                ok += 1
            elif gap <= max_gap:
                seg_h, seg_s = h[b0 + 1:floor_top[c], c], s[b0 + 1:floor_top[c], c]
                wall = (seg_h >= 95) & (seg_h <= 130) & (seg_s > 50)  # синяя стенка
                if not wall.any():
                    ok += 1                                            # кромка сугроба, не стенка
        if REQUIRE_RIM and rim / len(cols) < 0.3:
            return False                                               # белое без кромки — лёд
        return ok / len(cols) >= 0.5

    def analyze(self, img):
        m, small = self.mask(img)
        h, w = m.shape
        corr = self._corridor(h, w)
        inside = m & corr
        frac = inside.sum() / max(1, corr.sum())
        ys = np.nonzero(inside.any(axis=1))[0]
        near_y = float(ys.max() + 1) / h if len(ys) else None
        floor = np.zeros_like(m)
        floor[int(HORIZON * h):] = True
        left = (m & floor)[:, : w // 2].mean() * 2
        right = (m & floor)[:, w // 2:].mean() * 2
        blocked = bool(near_y is not None and near_y >= TRIGGER_Y and frac >= MIN_FRAC)
        return {"blocked": blocked, "near_y": near_y, "frac": float(frac),
                "left": float(left), "right": float(right), "mask": m, "small": small}

    def draw(self, img, r):
        vis = r["small"].copy()
        h, w = vis.shape[:2]
        vis[r["mask"]] = (0.4 * vis[r["mask"]] + 0.6 * np.array([200, 0, 200])).astype(np.uint8)
        c = self._corridor(h, w).astype(np.uint8)
        cs, _ = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, (0, 255, 255), 1)
        ty = int(TRIGGER_Y * h)
        cv2.line(vis, (0, ty), (w, ty), (0, 0, 255), 1)
        vis = cv2.resize(vis, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        txt = ("BLOCKED" if r["blocked"] else "free") + \
              (f" near_y={r['near_y']:.2f}" if r["near_y"] else "")
        cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 255) if r["blocked"] else (0, 200, 0), 2)
        return vis


def fmt(r):
    s = "СУГРОБ ВПЕРЕДИ" if r["blocked"] else "свободно"
    if r["near_y"] is not None:
        s += f" | ближний край y={r['near_y']:.2f} (порог {TRIGGER_Y})"
    return s + f" | в коридоре {r['frac']*100:.0f}% | слева {r['left']*100:.0f}% справа {r['right']*100:.0f}%"


def _main_files(patterns):
    det = DriftDetector()
    paths = []
    for p in patterns:
        paths += [x for x in glob.glob(p) if "_snow." not in x] or [p]
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"{p}: не открылся")
            continue
        r = det.analyze(img)
        print(f"{p}: {fmt(r)}")
        cv2.imwrite(p.rsplit(".", 1)[0] + "_snow.jpg", det.draw(img, r))


def _main_ros():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage

    rclpy.init()
    node = Node("snow_probe")
    det = DriftDetector()

    def cb(msg):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        r = det.analyze(img)
        print(fmt(r), flush=True)
        cv2.imwrite("last_snow.jpg", det.draw(img, r))
        cv2.imwrite("last_raw.jpg", img)

    node.create_subscription(CompressedImage, "/front_camera/image_raw/compressed",
                             cb, qos_profile_sensor_data)
    print("Слушаю камеру, Ctrl+C — выход", flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--ros":
        _main_ros()
    else:
        _main_files(sys.argv[1:])
