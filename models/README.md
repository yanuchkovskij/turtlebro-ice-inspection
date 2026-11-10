# Модели

Сюда кладётся модель детектора животных — **`best_v2.onnx`** (YOLOv8n, вход 320×320, классы
`0: bear`, `1: elk`, ~12 МБ). Файл с робота можно просто положить сюда и закоммитить —
12 МБ для git нормально (или приложить к GitHub Release).

Получить модель:

```bash
pip install -r ../training/requirements.txt
export ROBOFLOW_API_KEY=<ваш ключ Roboflow>
export ROBOFLOW_VERSION=2
python ../training/train.py        # → best_v2.onnx
```

На роботе модель лежит рядом с кодом: `~/hack/best_v2.onnx` (имя задаётся `MODEL` в `mission.py`).
