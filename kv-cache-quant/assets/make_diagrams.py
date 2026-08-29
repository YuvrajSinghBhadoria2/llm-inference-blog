import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ACCENT = "#1a73e8"
RED = "#d93025"
GREEN = "#188038"
GREY = "#5f6368"
LIGHT = "#e8f0fe"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})

# ---------- Diagram 1: the KV cache tax ----------
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

ax.text(50, 96, "The KV cache is a tax you pay on every token", ha="center",
        fontsize=15, fontweight="bold", color="#202124")

# tokens
ax.text(12, 80, "Prompt +\ngenerated\ntokens", ha="center", fontsize=10, color=GREY)
for i, y in enumerate(range(74, 40, -6)):
    b = FancyBboxPatch((6, y), 12, 4, boxstyle="round,pad=0.3",
                       linewidth=1, edgecolor=GREY, facecolor=LIGHT)
    ax.add_patch(b)

# arrow tokens -> kv cache
ax.add_patch(FancyArrowPatch((20, 58), (34, 58), arrowstyle="-|>", mutation_scale=14, color=ACCENT))

# KV cache box (growing)
kv = FancyBboxPatch((36, 38), 26, 40, boxstyle="round,pad=0.4",
                    linewidth=1.5, edgecolor=ACCENT, facecolor="#fff")
ax.add_patch(kv)
ax.text(49, 76, "KV Cache", ha="center", fontweight="bold", color=ACCENT)
ax.text(49, 70, "(K & V per layer,\nper token)", ha="center", fontsize=8.5, color=GREY)
# stacked precision rows
rows = [("fp16  → 17.2 GB @128K", RED),
        ("4-bit → 4.3 GB", GREEN),
        ("2-bit → 2.1 GB", GREEN)]
for i, (t, c) in enumerate(rows):
    yy = 60 - i*11
    ax.add_patch(FancyBboxPatch((39, yy), 20, 8, boxstyle="round,pad=0.2",
                 linewidth=1, edgecolor=c, facecolor="#fff"))
    ax.text(49, yy+4, t, ha="center", fontsize=8.5, color=c, fontweight="bold")

# arrow -> gpu
ax.add_patch(FancyArrowPatch((62, 58), (74, 58), arrowstyle="-|>", mutation_scale=14, color=ACCENT))

# GPU memory
gpu = FancyBboxPatch((76, 34), 20, 48, boxstyle="round,pad=0.4",
                     linewidth=1.5, edgecolor="#202124", facecolor="#fff")
ax.add_patch(gpu)
ax.text(86, 78, "GPU 80GB", ha="center", fontweight="bold", color="#202124")
ax.add_patch(FancyBboxPatch((79, 64), 14, 8, boxstyle="round,pad=0.2",
             linewidth=1, edgecolor=GREY, facecolor=LIGHT))
ax.text(86, 68, "Weights 16GB", ha="center", fontsize=8, color=GREY)
# overflow bar (fp16)
ax.add_patch(FancyBboxPatch((79, 44), 14, 16, boxstyle="round,pad=0.2",
             linewidth=1, edgecolor=RED, facecolor="#fce8e6"))
ax.text(86, 52, "KV fp16\nOVERFLOW", ha="center", fontsize=7.5, color=RED, fontweight="bold")
# fit bar (4-bit)
ax.add_patch(FancyBboxPatch((79, 38), 14, 4, boxstyle="round,pad=0.2",
             linewidth=1, edgecolor=GREEN, facecolor="#e6f4ea"))
ax.text(86, 40, "KV 4-bit fits", ha="center", fontsize=7.5, color=GREEN, fontweight="bold")

ax.text(49, 12, "Quantize the cache → the same 80GB GPU now serves 4× longer context / 4× more batch",
        ha="center", fontsize=10.5, color=ACCENT, fontweight="bold")
fig.tight_layout()
fig.savefig("assets/kv_flow.png", dpi=140)
print("saved kv_flow.png")

# ---------- Diagram 2: asymmetric quantization ----------
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
ax.text(50, 95, "Keys and values break differently — quantize them differently",
        ha="center", fontsize=14, fontweight="bold", color="#202124")

def draw_grid(x0, y0, title, highlight, label, color):
    ax.text(x0+13, y0+30, title, ha="center", fontweight="bold", color="#202124")
    # simple matrix of cells
    cols, rows = 13, 8
    cw, ch = 1.7, 2.4
    for r in range(rows):
        for c in range(cols):
            xx = x0 + c*cw
            yy = y0 - (r+1)*ch
            if (c, r) in highlight:
                fc = color; ec = color
            else:
                fc = "#f1f3f4"; ec = "#dadce0"
            ax.add_patch(plt.Rectangle((xx, yy), cw-0.15, ch-0.15, facecolor=fc, edgecolor=ec, lw=0.6))
    ax.text(x0+13, y0-rows*ch-4, label, ha="center", fontsize=9.5, color=color, fontweight="bold")

# Key cache: highlight a few fixed COLUMNS (channels) -> per-channel outliers
key_hl = {(3, r) for r in range(8)} | {(9, r) for r in range(8)}
draw_grid(4, 70, "Key cache", key_hl, "fixed outlier CHANNELS → quantize PER-CHANNEL", RED)

# Value cache: highlight a few ROWS (tokens) -> per-token
val_hl = {(c, 2) for c in range(13)} | {(c, 5) for c in range(13)}
draw_grid(56, 70, "Value cache", val_hl, "per-TOKEN outliers → quantize PER-TOKEN", GREEN)

ax.text(50, 6, "Naive uniform (per-token) key quantization → ~5× larger attention-score error",
        ha="center", fontsize=9.5, color=RED)
fig.tight_layout()
fig.savefig("assets/kv_asym.png", dpi=140)
print("saved kv_asym.png")
