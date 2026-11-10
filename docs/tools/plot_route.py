#!/usr/bin/env python3
"""Рисует docs/img/route_plot.png — схему финального заезда (точки из лога mission.txt)."""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "img", "route_plot.png")
# x, y цели точки (из лога), курс камеры при осмотре, результат
PTS = [(0.83, 0.52, -178.4, "1: В · 2 · красный · лось"),
       (0.84, 1.20, 95.1, "2: Б · 1 · зелёный · лось*"),
       (2.19, 1.16, 91.7, "3: В · 1 · посередине · медведь"),
       (2.26, 1.36, -91.7, "4: А · 3 · красный · нет")]
LABEL_OFFS = [(0.10, -0.10, "left"), (-0.05, 0.14, "right"), (0.0, -0.22, "center"), (0.0, 0.17, "center")]

ink, muted, acc, grid = "#2b2b2b", "#8a8a8a", "#2f6fdb", "#e6e6e6"
fig, ax = plt.subplots(figsize=(8, 5.8), dpi=150)
path = [(0, 0)] + [(p[0], p[1]) for p in PTS] + [(0, 0.06)]
xs, ys = zip(*path)
ax.plot(xs, ys, color=acc, lw=2, zorder=2)
for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    ax.annotate("", xy=(mx + (x1 - x0) * 0.05, my + (y1 - y0) * 0.05), xytext=(mx, my),
                arrowprops=dict(arrowstyle="-|>", color=acc, lw=1.5), zorder=3)
ax.add_patch(plt.Rectangle((-0.2, -0.2), 0.4, 0.4, fc="#ecebf7", ec="#5b4fb5", lw=1.5, zorder=1))
ax.text(0, -0.28, "старт / финиш", ha="center", va="top", color=ink, fontsize=9)
for (x, y, yaw, lab), (dx, dy, ha) in zip(PTS, LABEL_OFFS):
    t = math.radians(yaw)
    ax.annotate("", xy=(x + 0.22 * math.cos(t), y + 0.22 * math.sin(t)), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=ink, lw=1.2), zorder=4)
    ax.plot(x, y, "o", ms=9, mfc="white", mec=ink, mew=2, zorder=5)
    ax.text(x + dx, y + dy, lab, ha=ha, va="center", fontsize=9, color=ink,
            bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.9), zorder=6)
ax.set_aspect("equal")
ax.set_xlim(-0.6, 3.0)
ax.set_ylim(-0.5, 1.9)
ax.grid(color=grid, lw=0.8)
ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_color(grid)
ax.tick_params(colors=muted, labelsize=8)
ax.set_xlabel("x, м (вперёд от старта)", color=muted, fontsize=9)
ax.set_ylabel("y, м (влево)", color=muted, fontsize=9)
ax.set_title("Финальный заезд: 4 арматуры, 56 с", loc="left", color=ink, fontsize=12, fontweight="bold")
fig.text(0.02, 0.015, "Чёрная стрелка — куда смотрит камера при осмотре.  "
         "* ложное срабатывание, исправлено после заезда.", fontsize=8, color=muted)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT, facecolor="white")
print("->", os.path.abspath(OUT))
