#!/usr/bin/env python3
"""Запуск миссии кнопкой на плате (D22–D25) или ладонью перед лидаром — без SSH и интернета.

Как пользоваться:
    1. Поставь робота на старт. Горит светодиод D26 — робот готов.
    2. Нажми любую кнопку D22–D25 (если прошит скетч buttons.ino) —
       ИЛИ поднеси ладонь к носу робота (5–15 см) на ~2 с и убери.
    3. Через 1 с робот поедет (горит D27). Финиш — D28, ошибка — D29.
После миссии робот снова ждёт — так можно запустить вторую попытку.

Запуск вручную (для проверки):     python3 launcher.py --nosnow
Как сервис при включении робота:  см. launcher.sh / hack-mission.service
Флаги передаются в mission.py: --nosnow, --direct, --dry.
Журнал: ~/hack/launcher.log, результаты заездов: ~/hack/logs/run_*/
"""
import os
import sys
import time
import traceback
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # логи пишутся в ~/hack/logs

import nav as N

HAND_DIST = 0.28      # м от центра робота (лидара): ладонь ближе этого = «нажата»
HOLD = 1.5            # с — держать ладонь
RELEASE_DIST = 0.45   # м — рука убрана, когда впереди свободно дальше этого
START_DELAY = 1.0     # с — пауза после того, как убрали руку


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def open_buttons():
    try:
        from arduino_buttons import Buttons
        b = Buttons()
        log(f"кнопки Arduino: {b.port}")
        return b
    except Exception as e:
        log(f"кнопки Arduino недоступны ({e}) — запуск только ладонью")
        return None


def wait_for_start(btn):
    """Ждём кнопку на плате или ладонь перед лидаром. True — стартуем."""
    nav = N.Nav(camera=False, snow_avoid=False)
    try:
        if "--nowiggle" not in sys.argv:          # подтверждение «готов»: покачивание
            y0 = nav.pose()[2]
            nav.rotate_to(y0 + 0.15)
            nav.rotate_to(y0 - 0.15)
            nav.rotate_to(y0)
        if btn:
            btn.leds_off()
            btn.led(0)
            btn.poll()                            # выкинуть старые нажатия
        log("Готов. Кнопка D22–D25 или ладонь перед носом на 2 с -> старт.")
        held_since = None
        while True:
            if btn and btn.poll():
                log("нажата кнопка — старт через 1 с")
                time.sleep(START_DELAY)
                btn.poll()                        # отпускание кнопки тоже событие — выкинуть
                return True
            d = nav.corridor()
            if d < HAND_DIST:
                if held_since is None:
                    held_since = time.time()
                    log(f"вижу ладонь ({d:.2f} м), держи...")
                elif time.time() - held_since >= HOLD:
                    log("принято! убери руку")
                    break
            else:
                held_since = None
            time.sleep(0.1)
        t0 = time.time()
        while nav.corridor() < RELEASE_DIST:          # ждём, пока руку уберут
            if time.time() - t0 > 20:
                log("рука не убрана 20 с — отмена, жду снова")
                return False
            time.sleep(0.1)
        time.sleep(START_DELAY)
        return True
    finally:
        nav.close()


def run_mission(btn):
    import mission as M
    if btn:
        btn.leds_off()
        btn.led(1)
    ok = True
    m = M.Mission(dry="--dry" in sys.argv)
    try:
        m.run(direct="--direct" in sys.argv)
    except Exception:
        ok = False
        log("ОШИБКА в миссии:\n" + traceback.format_exc())
        m.save_summary()
    finally:
        m.log_f.close()
        m.nav.close()
    if btn:
        btn.leds_off()
        btn.led(2 if ok else 3)
    log(f"Миссия окончена. Результаты: {os.path.abspath(m.run_dir)}")


def main():
    log(f"launcher запущен, флаги: {' '.join(sys.argv[1:]) or '—'}")
    btn = open_buttons()
    while True:
        try:
            if wait_for_start(btn):
                log("СТАРТ миссии")
                run_mission(btn)
                time.sleep(3)                   # показать светодиод результата
        except KeyboardInterrupt:
            log("остановлен вручную")
            break
        except Exception:
            # ROS ещё не поднялся после включения и т.п. — подождать и пробовать снова
            log("ошибка, повтор через 5 с:\n" + traceback.format_exc(limit=2))
            time.sleep(5)


if __name__ == "__main__":
    main()
