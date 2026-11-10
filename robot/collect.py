#!/usr/bin/env python3
"""Сбор кадров с камеры робота для датасета.

    python3 collect.py              # кадр каждые 0.5 с в ~/hack/frames
    python3 collect.py 1.0 bear     # кадр раз в секунду, имена bear_0000.jpg ...

Ctrl+C — стоп. Потом скачай папку frames через левую панель MobaXterm
и загрузи кадры в Roboflow (в тот же проект).
"""
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

step = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
prefix = sys.argv[2] if len(sys.argv) > 2 else "frame"
out = os.path.expanduser("~/hack/frames")
os.makedirs(out, exist_ok=True)

n = len([f for f in os.listdir(out) if f.startswith(prefix)])  # дописываем, не затирая
last = 0.0

rclpy.init()
node = Node("collector")


def cb(msg):
    global n, last
    now = time.time()
    if now - last < step:
        return
    img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return
    path = f"{out}/{prefix}_{n:04d}.jpg"
    cv2.imwrite(path, img)
    print("сохранён", path, flush=True)
    n += 1
    last = now


node.create_subscription(CompressedImage, "/front_camera/image_raw/compressed",
                         cb, qos_profile_sensor_data)
print(f"Пишу кадр каждые {step} с в {out}, Ctrl+C — стоп", flush=True)
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
print("Всего кадров с префиксом", prefix, ":", n)
