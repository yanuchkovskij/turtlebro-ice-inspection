#!/usr/bin/env python3
"""Животные: YOLOv8 (ONNX) на Raspberry Pi.

Установка на роботе (один раз, нужен интернет):
    pip install onnxruntime --break-system-packages

Проверка на фото:
    python3 animals_detect.py best.onnx фото1.jpg фото2.jpg
    -> печатает находки, рядом сохраняет *_animals.jpg с рамками

Живая камера:
    python3 animals_detect.py best.onnx --ros

В своём коде:
    from animals_detect import AnimalDetector
    det = AnimalDetector("best.onnx")
    found = det.detect(bgr_image)   # [{name, ru, conf, box:(x1,y1,x2,y2)}, ...]
"""
import ast
import glob
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

# Русские названия для лога; поправь под формулировки регламента
RU = {"bear": "медведь", "elk": "лось"}

CONF_TH = 0.75  # ниже — дубли и ложные срабатывания (робот/люди на фоне)
NMS_TH = 0.5


class AnimalDetector:
    def __init__(self, onnx_path, conf_th=CONF_TH, nms_th=NMS_TH):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.inp_name = inp.name
        self.h, self.w = int(inp.shape[2]), int(inp.shape[3])  # 320x320
        self.conf_th, self.nms_th = conf_th, nms_th
        # Имена классов ultralytics кладёт в метаданные ONNX
        try:
            meta = self.sess.get_modelmeta().custom_metadata_map
            self.names = ast.literal_eval(meta["names"])  # {0: 'bear', 1: 'elk'}
        except Exception:
            self.names = {0: "bear", 1: "elk"}

    def detect(self, img):
        H, W = img.shape[:2]
        # Растягиваем до 320x320 без полей — так же, как Roboflow при подготовке датасета
        x = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])

        out = self.sess.run(None, {self.inp_name: x})[0][0].T  # (N, 4 + классы)
        scores = out[:, 4:]
        cls = scores.argmax(1)
        conf = scores.max(1)
        keep = conf >= self.conf_th
        if not keep.any():
            return []
        out, cls, conf = out[keep], cls[keep], conf[keep]

        sx, sy = W / self.w, H / self.h
        cx, cy, bw, bh = out[:, 0] * sx, out[:, 1] * sy, out[:, 2] * sx, out[:, 3] * sy
        boxes = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1)

        # NMS без учёта класса: убирает и дубли, и «два класса на одном животном»
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), self.conf_th, self.nms_th)
        idx = np.array(idx).flatten()

        found = []
        for i in idx:
            x0, y0, bw_, bh_ = boxes[i]
            name = self.names.get(int(cls[i]), str(int(cls[i])))
            found.append({
                "name": name,
                "ru": RU.get(name, name),
                "conf": float(conf[i]),
                "box": (int(max(0, x0)), int(max(0, y0)),
                        int(min(W - 1, x0 + bw_)), int(min(H - 1, y0 + bh_))),
            })
        found.sort(key=lambda f: f["box"][0])  # слева направо
        return found


def draw(img, found):
    vis = img.copy()
    for f in found:
        x1, y1, x2, y2 = f["box"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{f['name']} {f['conf']:.2f}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return vis


def _main_files(det, patterns):
    paths = []
    for p in patterns:
        paths += [x for x in glob.glob(p) if "_animals." not in x] or [p]
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"{p}: не открылся")
            continue
        t = time.time()
        found = det.detect(img)
        ms = (time.time() - t) * 1000
        print(f"{p} ({ms:.0f} мс): " +
              (", ".join(f"{f['ru']} {f['conf']:.2f}" for f in found) or "животных нет"))
        cv2.imwrite(p.rsplit(".", 1)[0] + "_animals.jpg", draw(img, found))


def _main_ros(det):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage

    rclpy.init()
    node = Node("animals_probe")

    def cb(msg):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        t = time.time()
        found = det.detect(img)
        ms = (time.time() - t) * 1000
        print(f"{ms:4.0f} мс | " +
              (", ".join(f"{f['ru']} {f['conf']:.2f}" for f in found) or "—"), flush=True)
        # Последний кадр с рамками — скачай через левую панель MobaXterm, чтобы посмотреть
        cv2.imwrite("last_animals.jpg", draw(img, found))

    node.create_subscription(CompressedImage, "/front_camera/image_raw/compressed",
                             cb, qos_profile_sensor_data)
    print("Слушаю камеру, Ctrl+C — выход", flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    args = sys.argv[1:]
    conf = CONF_TH
    if "--conf" in args:  # например: --conf 0.25 для отладки
        k = args.index("--conf")
        conf = float(args[k + 1])
        del args[k:k + 2]
    sys.argv = [sys.argv[0]] + args
    detector = AnimalDetector(sys.argv[1], conf_th=conf)
    print("Порог уверенности:", conf)
    print("Классы:", detector.names, "| вход", detector.w, "x", detector.h)
    if sys.argv[2] == "--ros":
        _main_ros(detector)
    else:
        _main_files(detector, sys.argv[2:])
