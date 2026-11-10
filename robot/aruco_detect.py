#!/usr/bin/env python3
"""ArUco: распознавание номера маркера на арматуре.

Ноутбук (проверка на фото):
    pip install opencv-python
    python aruco_detect.py папка_с_фото\\*.jpg
    -> печатает словарь и id по каждому фото, рядом сохраняет *_aruco.jpg с рамками

Робот (живая камера):
    python3 aruco_detect.py --ros                 # автоподбор словаря на каждом кадре (медленно)
    python3 aruco_detect.py --ros DICT_4X4_50     # когда словарь уже известен (быстро)

В своём коде:
    from aruco_detect import detect_aruco
    dict_name, markers = detect_aruco(bgr_image, "DICT_4X4_50")
    for m in markers: m["id"], m["center"], m["side_px"], m["corners"]
"""
import sys
import glob
from collections import Counter

import cv2
import numpy as np

# Порядок важен: при равенстве побеждает словарь, стоящий раньше (меньше ложных срабатываний)
CANDIDATE_DICTS = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL", "DICT_APRILTAG_36h11",
]

_detectors = {}


def _make_detector(dict_name):
    """Детектор, совместимый и со старым (<4.7), и с новым API OpenCV."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    if hasattr(cv2.aruco, "ArucoDetector"):  # OpenCV >= 4.7
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        det = cv2.aruco.ArucoDetector(dictionary, params)
        return det.detectMarkers
    params = cv2.aruco.DetectorParameters_create()  # OpenCV 4.6 и старше
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return lambda gray: cv2.aruco.detectMarkers(gray, dictionary, parameters=params)


def _detector(dict_name):
    if dict_name not in _detectors:
        _detectors[dict_name] = _make_detector(dict_name)
    return _detectors[dict_name]


def detect_aruco(img, dict_name=None):
    """Ищет ArUco-маркеры.

    img       — BGR- или серое изображение
    dict_name — например "DICT_4X4_50"; None = перебрать все словари
    Возвращает (имя_словаря | None, [ {id, center, side_px, corners}, ... ])
    Сначала ищет в исходном размере, если пусто — в увеличенном вдвое
    (маркер 70 мм издалека занимает мало пикселей).
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    names = [dict_name] if dict_name else CANDIDATE_DICTS

    for scale in (1.0, 2.0):
        g = gray if scale == 1.0 else cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        best_name, best = None, []
        for name in names:
            corners, ids, _ = _detector(name)(g)
            if ids is None:
                continue
            found = []
            for c, i in zip(corners, ids.flatten()):
                pts = c.reshape(4, 2) / scale
                cx, cy = pts.mean(axis=0)
                side = float(np.mean([np.linalg.norm(pts[k] - pts[(k + 1) % 4])
                                      for k in range(4)]))
                found.append({"id": int(i), "center": (float(cx), float(cy)),
                              "side_px": side, "corners": pts})
            if len(found) > len(best):
                best_name, best = name, found
        if best:
            return best_name, best
    return None, []


def draw(img, markers):
    vis = img.copy()
    for m in markers:
        cv2.polylines(vis, [m["corners"].astype(int)], True, (0, 255, 0), 2)
        x, y = (int(v) for v in m["center"])
        cv2.putText(vis, str(m["id"]), (x - 10, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return vis


def _main_files(patterns):
    paths = []
    for p in patterns:  # в Windows звёздочку раскрываем сами
        paths += [x for x in glob.glob(p) if "_aruco." not in x] or [p]
    stat = Counter()
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"{p}: не открылся")
            continue
        name, found = detect_aruco(img)
        print(f"{p}: {name or '-'} {[m['id'] for m in found]}")
        if found:
            stat[name] += 1
            cv2.imwrite(p.rsplit(".", 1)[0] + "_aruco.jpg", draw(img, found))
    print("\nСловари (сколько фото):", dict(stat) or "ничего не найдено")


def _main_ros(dict_name):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage

    rclpy.init()
    node = Node("aruco_probe")

    def cb(msg):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        name, found = detect_aruco(img, dict_name)
        if found:
            print(name, [(m["id"], f"{m['side_px']:.0f}px") for m in found], flush=True)

    node.create_subscription(CompressedImage, "/front_camera/image_raw/compressed",
                             cb, qos_profile_sensor_data)
    print("Слушаю камеру, Ctrl+C — выход", flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    print("OpenCV", cv2.__version__)
    if not hasattr(cv2, "aruco"):
        sys.exit("В этом OpenCV нет модуля aruco: pip install opencv-python>=4.7 "
                 "(на роботе: sudo apt install python3-opencv)")
    args = sys.argv[1:]
    if args and args[0] == "--ros":
        _main_ros(args[1] if len(args) > 1 else None)
    elif args:
        _main_files(args)
    else:
        print(__doc__)
