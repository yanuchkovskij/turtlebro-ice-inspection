#!/usr/bin/env python3
"""Замер максимальной скорости TurtleBro по одометрии.

!!! КОЛЁСА ПОДНЯТЬ: поставь робота на коробку, чтобы колёса крутились в воздухе !!!

    cd ~/hack && python3 speedtest.py
    cd ~/hack && python3 speedtest.py rot   # только повороты — робот НА ПОЛУ, вокруг свободно

Шлёт всё большие команды и печатает реальную скорость колёс (по /odom).
Где реальная скорость перестала расти — это потолок. Ctrl+C — стоп.
"""
import math
import sys
import time

from nav import Nav, norm

LIN_CMDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]   # м/с
ANG_CMDS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]       # рад/с
HOLD = 2.5                                      # с на каждую команду


def measure(nav, v, w):
    """Держит команду HOLD секунд, меряет скорость за последнюю секунду."""
    t0 = time.time()
    while time.time() - t0 < HOLD - 1.0:
        nav._send(v, w)
        time.sleep(0.05)
    x0, y0, a0 = nav.pose()
    t1 = time.time()
    dist = turn = 0.0
    px, py, pa = x0, y0, a0
    while time.time() - t1 < 1.0:
        nav._send(v, w)
        time.sleep(0.05)
        x, y, a = nav.pose()
        dist += math.hypot(x - px, y - py)
        turn += norm(a - pa)
        px, py, pa = x, y, a
    dt = time.time() - t1
    return dist / dt, turn / dt


def main():
    nav = Nav()
    try:
        only_rot = "rot" in sys.argv[1:]
        if only_rot:
            print("Робот на полу, вокруг свободно? Старт через 3 с...", flush=True)
        else:
            print("Колёса подняты? Старт через 3 с...", flush=True)
        time.sleep(3)
        if not only_rot:
            print("\nЕзда вперёд:")
            for v in LIN_CMDS:
                real, _ = measure(nav, v, 0.0)
                print(f"  команда {v:.2f} м/с -> реально {real:.3f} м/с", flush=True)
            nav.stop()
            time.sleep(1)
        print("\nПоворот на месте:")
        for w in ANG_CMDS:
            _, real = measure(nav, 0.0, w)
            print(f"  команда {w:.1f} рад/с -> реально {real:.2f} рад/с", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        nav.close()


if __name__ == "__main__":
    main()
