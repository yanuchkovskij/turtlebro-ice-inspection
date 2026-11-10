#!/usr/bin/env python3
"""Кнопки и светодиоды Arduino-части платы TurtleBro (скетч buttons/buttons.ino).

Проверка:   python3 arduino_buttons.py      # жми кнопки — печатает нажатия, мигает огнями
Порт ищется сам (переходник CP2102 с номером 0010); можно задать в local_settings.json:
            "BUTTON_PORT": "/dev/ttyUSB0"
"""
import glob
import os
import termios
import time
import tty

BUTTON_PORT = None          # None — найти автоматически
try:                        # местная настройка порта, если задана
    import json as _json
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_settings.json")
    BUTTON_PORT = _json.load(open(_p, encoding="utf-8")).get("BUTTON_PORT", BUTTON_PORT)
except Exception:
    pass


def find_port():
    if BUTTON_PORT:
        return BUTTON_PORT
    for pat in ("/dev/serial/by-id/*CP210*0010*", "/dev/serial/by-id/*0010*"):
        found = sorted(glob.glob(pat))
        if found:
            return found[0]
    return None


class Buttons:
    """Неблокирующее чтение событий кнопок. Без pyserial — через termios."""

    def __init__(self, port=None):
        self.port = port or find_port()
        if not self.port:
            raise RuntimeError("порт Arduino (CP2102 ...0010) не найден")
        self.fd = os.open(self.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        tty.setraw(self.fd)
        a = termios.tcgetattr(self.fd)
        a[4] = a[5] = termios.B115200
        termios.tcsetattr(self.fd, termios.TCSANOW, a)
        self.buf = b""
        time.sleep(2.0)            # открытие порта перезагружает Arduino
        self.poll()                # выкинуть приветствие

    def poll(self):
        """-> список (пин, значение) с прошлого вызова."""
        try:
            self.buf += os.read(self.fd, 1024)
        except BlockingIOError:
            pass
        events = []
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            parts = line.decode(errors="ignore").strip().split()
            if len(parts) == 3 and parts[0] == "BTN":
                try:
                    events.append((int(parts[1]), int(parts[2])))
                except ValueError:
                    pass
        return events

    def led(self, i, on=True):
        """i = 0..3 (D26..D29)."""
        try:
            os.write(self.fd, (str(i) if on else "abcd"[i]).encode())
        except OSError:
            pass

    def leds_off(self):
        try:
            os.write(self.fd, b"x")
        except OSError:
            pass

    def close(self):
        try:
            os.close(self.fd)
        except OSError:
            pass


if __name__ == "__main__":
    b = Buttons()
    print("Порт:", b.port, "— жми кнопки, Ctrl+C — выход", flush=True)
    k = 0
    try:
        while True:
            for pin, val in b.poll():
                print(f"кнопка D{pin}: {val}", flush=True)
            b.leds_off()
            b.led(k % 4)
            k += 1
            time.sleep(0.25)
    except KeyboardInterrupt:
        b.leds_off()
        b.close()
