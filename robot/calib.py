#!/usr/bin/env python3
"""Калибровка колёс: сколько робот реально проезжает и поворачивается.
Результат сам записывается в local_settings.json (остальные настройки не трогает).

    cd ~/hack
    python3 calib.py dist          # проедет 0.8 м «по колёсам», спросит реальное расстояние
    python3 calib.py dist 1.0      # то же на 1 м
    python3 calib.py rot           # 4 поворота по 90° = полный оборот, спросит, сколько реально
    python3 calib.py show          # что сейчас записано

Мерить без рулетки: лист A4 — 29,7 x 21 см. Три листа подряд длинной стороной — 89 см.
"""
import json
import math
import os
import sys

import nav as N

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_settings.json")


def load():
    try:
        return json.load(open(PATH, encoding="utf-8"))
    except FileNotFoundError:
        return {}


def save(key, value):
    data = load()
    data[key] = round(value, 3)
    json.dump(data, open(PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nЗаписал {key} = {value:.3f} в local_settings.json")
    print("Сейчас там:", json.dumps(data, ensure_ascii=False))


def ask_number(prompt):
    while True:
        try:
            return float(input(prompt).replace(",", "."))
        except ValueError:
            print("Введи число, например 112 или -15")


def calib_dist(meters):
    N.ODOM_SCALE = 1.0                       # меряем «сырые» колёса
    nav = N.Nav(camera=False, snow_avoid=False)
    try:
        print(f"Отметь, где сейчас передний край робота. Впереди нужно ~{meters + 0.5:.1f} м свободно.")
        input("Enter — поехали...")
        x0, y0, _ = nav.pose()
        ok = nav.forward(meters, speed=0.15)
        N.time.sleep(0.5)
        x1, y1, _ = nav.pose()
        odo_cm = math.hypot(x1 - x0, y1 - y0) * 100
        if not ok:
            print("! Остановился раньше (препятствие?). Освободи путь и повтори.")
            return
        real_cm = ask_number(f"Робот считает, что проехал {odo_cm:.0f} см. "
                             "Сколько реально, см? ")
    finally:
        nav.close()
    k = real_cm / odo_cm
    print(f"Коэффициент пути: {k:.3f}"
          + ("  (колёса больше родных)" if k > 1.03 else "  (колёса меньше родных)" if k < 0.97
             else "  (почти как родные)"))
    save("ODOM_SCALE", k)
    print("Проверка: python3 nav.py fwd 0.8 — должно выйти ~80 см.")


def calib_rot():
    N.YAW_SCALE = 1.0
    nav = N.Nav(camera=False, snow_avoid=False)
    try:
        print("Поставь робота ровно вдоль шва плитки или края листа — чтобы видеть исходное "
              "направление. Вокруг ~30 см свободно.")
        input("Enter — крутимся (4 раза по 90°)...")
        yaw0 = nav.pose()[2]
        for k in range(1, 5):
            nav.rotate_to(yaw0 + math.radians(90 * k))
        delta = ask_number("Робот считает, что повернулся ровно на 360°.\n"
                           "Если смотрит точно как в начале — введи 0.\n"
                           "Перекрутил — плюс, недокрутил — минус (градусы, на глаз), например -20: ")
    finally:
        nav.close()
    k = (360 + delta) / 360
    if abs(k - 1) < 0.03:
        print("Повороты точные (курс, видимо, с гироскопа) — поправка не нужна.")
        k = 1.0
    save("YAW_SCALE", k)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "dist":
        calib_dist(float(sys.argv[2]) if len(sys.argv) > 2 else 0.8)
    elif cmd == "rot":
        calib_rot()
    elif cmd == "show":
        print(json.dumps(load(), ensure_ascii=False, indent=1))
    else:
        print(__doc__)
