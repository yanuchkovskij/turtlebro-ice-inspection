#!/usr/bin/env python3
"""Запись маршрута «обучением»: ведёшь робота с клавиатуры и отмечаешь точки.

ВАЖНО: перед записью и перед каждым заездом ставь робота в стартовую ячейку
ОДИНАКОВО (место и направление) — отметь контур колёс скотчем или запомни по краю ячейки.

    cd ~/hack && python3 teach.py          # результат: route.json

Управление (нажал — едет, пока не нажмёшь другую клавишу или пробел):
    w — вперёд        s — назад
    a — поворот влево d — поворот вправо
    пробел — стоп     + / - — быстрее / медленнее
    p — записать точку проезда (поворот маршрута, обход сугроба)
    i — записать точку ОСМОТРА: робот стоит перед арматурой лицом к ней,
        ~40–50 см, табличка, маркер и манометр в кадре
    u — удалить последнюю точку
    q — сохранить route.json и выйти
Последней точкой запиши «p» в стартовой ячейке — туда робот вернётся в конце заезда.
Если 3 с не нажимать клавиш, робот сам остановится (на случай обрыва SSH).
"""
import json
import math
import select
import sys
import termios
import time
import tty

from nav import Nav

SAFETY_STOP = 3.0   # с без клавиш — стоп


def main():
    nav = Nav()
    lin, ang = 0.10, 0.6
    pts = []
    v = w = 0.0
    last_key = time.time()
    old = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    print(__doc__, flush=True)
    try:
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                k = sys.stdin.read(1).lower()
                last_key = time.time()
                if k in ("w", "ц"):
                    v, w = lin, 0.0
                elif k in ("s", "ы"):
                    v, w = -lin, 0.0
                elif k in ("a", "ф"):
                    v, w = 0.0, ang
                elif k in ("d", "в"):
                    v, w = 0.0, -ang
                elif k == " ":
                    v = w = 0.0
                elif k in ("+", "="):
                    lin, ang = min(lin + 0.02, 0.2), min(ang + 0.1, 1.2)
                    print(f"скорость {lin:.2f} м/с, поворот {ang:.1f} рад/с", flush=True)
                elif k == "-":
                    lin, ang = max(lin - 0.02, 0.04), max(ang - 0.1, 0.2)
                    print(f"скорость {lin:.2f} м/с, поворот {ang:.1f} рад/с", flush=True)
                elif k in ("p", "з", "i", "ш"):
                    v = w = 0.0
                    nav.stop()
                    time.sleep(0.3)
                    x, y, yaw = nav.pose()
                    inspect = k in ("i", "ш")
                    pts.append({"x": round(x, 3), "y": round(y, 3),
                                "yaw_deg": round(math.degrees(yaw), 1), "inspect": inspect})
                    kind = "ОСМОТР" if inspect else "проезд"
                    print(f"#{len(pts)} {kind}: x={x:+.2f} y={y:+.2f} "
                          f"курс={math.degrees(yaw):+.0f}°", flush=True)
                elif k in ("u", "г"):
                    if pts:
                        print("удалена точка", pts.pop(), flush=True)
                elif k in ("q", "й"):
                    break
            if time.time() - last_key > SAFETY_STOP and (v or w):
                v = w = 0.0
                print("автостоп: нет клавиш 3 с", flush=True)
            nav._send(v, w)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        nav.stop()
        # сохраняем даже при Ctrl+C / ошибке, чтобы не потерять маршрут
        if pts:
            with open("route.json", "w", encoding="utf-8") as f:
                json.dump(pts, f, ensure_ascii=False, indent=1)
            n_insp = sum(p["inspect"] for p in pts)
            print(f"\nСохранено {len(pts)} точек (осмотров: {n_insp}) в route.json")
        nav.close()


if __name__ == "__main__":
    main()
