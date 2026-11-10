"""Обучение детектора животных (YOLOv8n -> ONNX) на датасете из Roboflow.

    pip install ultralytics roboflow onnx onnxslim
    set ROBOFLOW_API_KEY=ваш_ключ          (Linux/macOS: export ROBOFLOW_API_KEY=...)
    python train.py

Что делает: скачивает датасет -> чинит пути в data.yaml -> обучает (с best.pt, если он
лежит рядом, иначе с yolov8n.pt) -> экспортирует ONNX 320x320 -> best_v<версия>.onnx.
Готовый .onnx кладётся на робота в ~/hack рядом с mission.py.
"""
import os
import shutil

import yaml
from roboflow import Roboflow
from ultralytics import YOLO

API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")      # ключ не храним в коде
WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "artem-1nhv1")
PROJECT = os.environ.get("ROBOFLOW_PROJECT", "yoloanimals-mkdd0")
VERSION = int(os.environ.get("ROBOFLOW_VERSION", "2"))


def main():
    # 1. Датасет
    if not API_KEY:
        raise SystemExit("Задай ключ: переменная окружения ROBOFLOW_API_KEY")
    rf = Roboflow(api_key=API_KEY)
    ds = (rf.workspace(WORKSPACE).project(PROJECT)
          .version(VERSION).download("yolov8", location=f"yoloanimals-v{VERSION}", overwrite=True))
    loc = os.path.abspath(ds.location)

    # 2. Абсолютные пути в data.yaml (относительные от Roboflow часто ломаются)
    p = os.path.join(loc, "data.yaml")
    d = yaml.safe_load(open(p, encoding="utf-8"))
    d["train"] = os.path.join(loc, "train", "images")
    d["val"] = os.path.join(loc, "valid", "images")
    d["test"] = os.path.join(loc, "test", "images")
    yaml.safe_dump(d, open(p, "w", encoding="utf-8"), allow_unicode=True)
    print("Классы:", d["names"])
    if len(d["names"]) != 2:
        print("ВНИМАНИЕ: классов не 2. Если там есть лишний 'yoloanimals' — "
              "удали его в Roboflow и создай версию заново.")
    for split in ("train", "valid", "test"):
        folder = os.path.join(loc, split, "images")
        n = len(os.listdir(folder)) if os.path.isdir(folder) else 0
        print(f"  {split}: {n} картинок")

    # 3. Дообучение со старых весов (если их нет — с нуля от yolov8n)
    start = "best.pt" if os.path.exists("best.pt") else "yolov8n.pt"
    print("Стартовые веса:", start)
    model = YOLO(start)
    model.train(data=p, imgsz=320, epochs=60, batch=16, patience=20,
                workers=2, project="runs_v2", name="animals", exist_ok=True)
    save_dir = str(model.trainer.save_dir)
    best_pt = os.path.join(save_dir, "weights", "best.pt")
    print("Лучшие веса:", best_pt)

    # 4. Экспорт в ONNX с теми же параметрами, что и в первый раз
    onnx_path = YOLO(best_pt).export(format="onnx", imgsz=320, opset=12,
                                     simplify=True, dynamic=False)
    shutil.copy(onnx_path, f"best_v{VERSION}.onnx")
    shutil.copy(best_pt, f"best_v{VERSION}.pt")
    print(f"\nГОТОВО: best_v{VERSION}.onnx -> скопируй на робота в ~/hack")
    print(f"Графики: {save_dir}\\results.png и confusion_matrix.png")


if __name__ == "__main__":   # обязательно на Windows, иначе обучение зависает
    main()
