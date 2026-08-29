import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
A = "#1a73e8"; R = "#d93025"; G = "#188038"; GR = "#5f6368"; L = "#e8f0fe"

def arrow(ax, x1, y1, x2, y2, c="#202124"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=13, color=c, lw=1.6))

# ---------- kv_flow.png ----------
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
ax.text(50, 96, "The KV cache is a memory tax paid on every token",
        ha="center", fontsize=14, fontweight="bold", color="#202124")

# tokens
ax.add_patch(FancyBboxPatch((4, 42), 13, 18, boxstyle="round,pad=0.3",
             lw=1.2, edgecolor=GR, facecolor=L))
ax.text(10.5, 60, "Prompt +\ngenerated\ntokens", ha="center", fontsize=9.5, color=GR)
for i, y in enumerate(range(56, 46, -3)):
    ax.add_patch(FancyBboxPatch((6, y), 4, 2.2, boxstyle="round,pad=0.1",
                 lw=0.8, edgecolor=GR, facecolor="#fff"))
arrow(ax, 17, 51, 28, 51, A)

# kv cache
ax.add_patch(FancyBboxPatch((28, 30), 30, 42, boxstyle="round,pad=0.4",
             lw=1.6, edgecolor=A, facecolor="#fff"))
ax.text(43, 70, "KV Cache", ha="center", fontsize=12, fontweight="bold", color=A)
ax.text(43, 65, "(K & V per layer, per token)", ha="center", fontsize=8, color=GR)
rows = [("fp16  17.2 GB @128K", R, 56), ("4-bit  4.3 GB", G, 46), ("2-bit 2.1 GB", G, 38)]
for t, c, y in rows:
    ax.add_patch(FancyBboxPatch((31, y), 24, 7, boxstyle="round,pad=0.2",
                 lw=1.1, edgecolor=c, facecolor="#fff"))
    ax.text(43, y+3.5, t, ha="center", fontsize=8.5, color=c, fontweight="bold")
arrow(ax, 58, 51, 69, 51, A)

# gpu
ax.add_patch(FancyBboxPatch((69, 26), 25, 50, boxstyle="round,pad=0.4",
             lw=1.6, edgecolor="#202124", facecolor="#fff"))
ax.text(81.5, 73, "GPU 80 GB", ha="center", fontsize=12, fontweight="bold", color="#202124")
ax.add_patch(FancyBboxPatch((72, 62), 18, 7, boxstyle="round,pad=0.2",
             lw=1, edgecolor=GR, facecolor=L))
ax.text(81, 65.5, "Weights 16 GB", ha="center", fontsize=8, color=GR)
ax.add_patch(FancyBboxPatch((72, 45), 18, 13, boxstyle="round,pad=0.2",
             lw=1.1, edgecolor=R, facecolor="#fce8e6"))
ax.text(81, 52, "KV fp16\nOVERFLOW", ha="center", fontsize=7.5, color=R, fontweight="bold")
ax.add_patch(FancyBboxPatch((72, 36), 18, 6, boxstyle="round,pad=0.2",
             lw=1.1, edgecolor=G, facecolor="#e6f4ea"))
ax.text(81, 39, "KV 4-bit fits", ha="center", fontsize=7.5, color=G, fontweight="bold")

ax.text(50, 16, "Quantize the cache  →  same 80 GB GPU serves 4x longer context / 4x more batch",
        ha="center", fontsize=10.5, color=A, fontweight="bold")
fig.tight_layout(); fig.savefig("assets/kv_flow.png", dpi=140); print("kv_flow.png")

# ---------- kv_asym.png ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
def grid(ax, title, hl, label, color):
    ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(6.5, 10.6, title, ha="center", fontsize=11, fontweight="bold", color="#202124")
    cols, rows = 13, 8
    cw, ch = 0.95, 1.05
    for r in range(rows):
        for c in range(cols):
            xx = c*cw + 0.3; yy = 9 - (r+1)*ch
            fc = color if (c, r) in hl else "#f1f3f4"
            ec = color if (c, r) in hl else "#dadce0"
            ax.add_patch(plt.Rectangle((xx, yy), cw-0.12, ch-0.12, facecolor=fc, edgecolor=ec, lw=0.7))
    ax.text(6.5, -0.8, label, ha="center", fontsize=9.5, color=color, fontweight="bold")
key_hl = {(3, r) for r in range(8)} | {(9, r) for r in range(8)}
val_hl = {(c, 2) for c in range(13)} | {(c, 5) for c in range(13)}
grid(axes[0], "Key cache", key_hl, "fixed outlier CHANNELS → quantize PER-CHANNEL", R)
grid(axes[1], "Value cache", val_hl, "per-TOKEN outliers → quantize PER-TOKEN", G)
fig.suptitle("Keys and values break differently — quantize them differently",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout(); fig.savefig("assets/kv_asym.png", dpi=140); print("kv_asym.png")

# ---------- kv_serve.png ----------
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
ax.text(50, 92, "KV-cache quantization in a real serving stack", ha="center",
        fontsize=13, fontweight="bold", color="#202124")
steps = [("Prompt +\ntokens", 3, GR), ("Prefill\n(self-attn)", 22, A), ("Paged KV\ncache (fp16)", 44, A),
         ("Quantize\nK pc / V pt\n(2–4 bit)", 66, R), ("Compressed\nKV store", 85, G)]
bx = []
for t, x, c in steps:
    ax.add_patch(FancyBboxPatch((x, 45), 16, 20, boxstyle="round,pad=0.3",
                 lw=1.4, edgecolor=c, facecolor="#fff"))
    ax.text(x+8, 55, t, ha="center", va="center", fontsize=8.5, color=c, fontweight="bold")
    bx.append((x, x+16))
for i in range(len(steps)-1):
    arrow(ax, bx[i][1], 55, bx[i+1][0], 55, "#202124")
ax.add_patch(FancyBboxPatch((3, 18), 16, 12, boxstyle="round,pad=0.3", lw=1.2, edgecolor=GR, facecolor=L))
ax.text(11, 24, "GPU\nVRAM", ha="center", fontsize=8.5, color=GR)
ax.text(50, 8, "decode dequant-matmul fuses Q·K′ on the fly  ·  matches vLLM fp8 / KIVI / KVQuant kernels",
        ha="center", fontsize=8.5, color=GR)
fig.tight_layout(); fig.savefig("assets/kv_serve.png", dpi=140); print("kv_serve.png")
