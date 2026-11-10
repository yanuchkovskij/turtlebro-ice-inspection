# Генерирует шаблоны букв (кириллица + цифры) из жирных шрифтов в виде base64-блока
# для robot/inspect_vision.py (_TPL_CHARS / _TPL_N / _TPL_B64).
#     python make_plate_templates.py tpl_block.py
# Нужны Pillow, OpenCV и шрифты Liberation Sans / DejaVu Sans (есть в Ubuntu).
import base64, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

CHARS = "АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЭЮЯ0123456789"
FONTS = ["/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"]
S = 32


def norm_glyph(binary):
    """binary: uint8 0/255, буква белая. -> 32x32 с сохранением пропорций + аспект"""
    ys, xs = np.nonzero(binary)
    crop = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    k = (S - 2) / max(h, w)
    nh, nw = max(1, round(h * k)), max(1, round(w * k))
    r = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((S, S), np.uint8)
    y0, x0 = (S - nh) // 2, (S - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return (out > 127).astype(np.uint8), w / h


recs = []
for fp in FONTS:
    font = ImageFont.truetype(fp, 120)
    for ch in CHARS:
        img = Image.new("L", (200, 200), 0)
        ImageDraw.Draw(img).text((30, 20), ch, fill=255, font=font)
        g, asp = norm_glyph(np.array(img))
        recs.append((ch, asp, g))

chars = "".join(r[0] for r in recs)
asp = np.array([r[1] for r in recs], np.float32)
bits = np.packbits(np.stack([r[2] for r in recs]).reshape(len(recs), -1), axis=1)
blob = base64.b64encode(np.concatenate([asp.view(np.uint8), bits.ravel()]).tobytes()).decode()
with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write(f'_TPL_CHARS = "{chars}"\n_TPL_N = {len(recs)}\n_TPL_B64 = (\n')
    for i in range(0, len(blob), 100):
        f.write(f'    "{blob[i:i+100]}"\n')
    f.write(")\n")
print(len(recs), "templates,", len(blob), "b64 chars")
