import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
A="#1a73e8"; R="#d93025"; G="#188038"; GR="#5f6368"; L="#e8f0fe"

def arrow(ax,x1,y1,x2,y2,c="#202124"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=12,color=c,lw=1.4))
def box(ax,x,y,w,h,t,sub,c,fc="#fff"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3",lw=1.4,edgecolor=c,facecolor=fc))
    ax.text(x+w/2,y+h-0.5,t,ha="center",va="top",fontsize=9.5,color=c,fontweight="bold")
    if sub: ax.text(x+w/2,y+h-3.4,sub,ha="center",va="top",fontsize=7.2,color=GR)

# ---------- exp_flow.png ----------
fig,ax=plt.subplots(figsize=(10,5.4))
ax.set_xlim(0,100);ax.set_ylim(0,100);ax.axis("off")
ax.text(50,96,"How we measured the trap (real experiment)",ha="center",fontsize=13,fontweight="bold",color="#202124")
box(ax,3,62,26,22,"1 · Load model","Qwen2.5-0.5B\n24 layers · GQA",A,L)
box(ax,37,62,26,22,"2 · Forward pass","77-token real prompt",A,L)
box(ax,71,62,26,22,"3 · Capture K,V","actual cache\ntensors / layer",A,L)
box(ax,71,18,26,22,"4 · Quantize 4 ways","uniform pt · pc\nKIVI · wrong",R,"#fce8e6")
box(ax,37,18,26,22,"5 · Recompute attn","A = softmax(QKᵀ/√d)",GR,L)
box(ax,3,18,26,22,"6 · Measure error","‖A − Â‖ / ‖A‖",G,"#e6f4ea")
arrow(ax,29,73,37,73); arrow(ax,63,73,71,73)
arrow(ax,84,62,84,40); arrow(ax,71,29,63,29)
arrow(ax,37,29,37,40); arrow(ax,37,40,29,40)
arrow(ax,63,29,37,29)  # back to 5 from 4? draw 4->5
# fix: 4 -> 5
arrow(ax,71,29,63,29)
arrow(ax,29,29,3,29)
ax.text(50,7,"every scheme reuses the SAME real K,V → apples-to-apples attention error",ha="center",fontsize=8.5,color=GR)
fig.tight_layout();fig.savefig("assets/exp_flow.png",dpi=140);print("exp_flow.png")

# ---------- kv_serve.png (richer) ----------
fig,ax=plt.subplots(figsize=(11,5))
ax.set_xlim(0,100);ax.set_ylim(0,100);ax.axis("off")
ax.text(50,95,"KV-cache quantization in a serving stack",ha="center",fontsize=13,fontweight="bold",color="#202124")
comp=[("1 · Prefill Worker","runs self-attention,\nproduces K & V",A,L,3),
      ("2 · Paged KV Cache","per-layer, per-token\nK,V held in VRAM",A,L,28),
      ("3 · KV Quantizer","K per-channel,\nV per-token → 2/4-bit",R,"#fce8e6",53),
      ("4 · Decode Worker","fused dequant-matmul,\ngenerates tokens",G,"#e6f4ea",78)]
for t,s,c,fc,x in comp:
    box(ax,x,46,21,26,t,s,c,fc)
arrow(ax,24,59,28,59);arrow(ax,49,59,53,59);arrow(ax,74,59,78,59)
ax.text(50,30,"compressed KV store (4–8× smaller) absorbs bigger batches / longer context",ha="center",fontsize=9,color=GR)
ax.text(50,22,"decode pulls compressed KV; fused dequant hides the cost → vLLM fp8 / HuggingFace-KIVI / KVQuant kernels",ha="center",fontsize=8,color=GR)
fig.tight_layout();fig.savefig("assets/kv_serve.png",dpi=140);print("kv_serve.png")

# ---------- bench.png ----------
fig,ax=plt.subplots(1,2,figsize=(10,4.2))
pre=["fp16","8-bit","4-bit","2-bit"]; mem=[17.18,8.59,4.29,2.15]
bars=ax[0].bar(pre,mem,color=[R, "#f9ab00", G, G])
for b,v in zip(bars,mem): ax[0].text(b.get_x()+b.get_width()/2,v+0.2,f"{v}",ha="center",fontsize=8)
ax[0].set_title("KV cache @ 128K ctx (GB, 1 req)",fontsize=10)
ax[0].set_ylabel("GB"); ax[0].grid(axis="y",alpha=0.3)
names=["KIVI","GEAR","KVQuant"]; thr=[3.0,4.0,1.7]
bars=ax[1].bar(names,thr,color=A)
for b,v in zip(bars,thr): ax[1].text(b.get_x()+b.get_width()/2,v+0.05,f"{v}×",ha="center",fontsize=8)
ax[1].set_title("Throughput vs fp16 (per paper)",fontsize=10)
ax[1].set_ylabel("relative"); ax[1].grid(axis="y",alpha=0.3)
fig.suptitle("Memory saved vs speedup at a glance",fontsize=12,fontweight="bold",y=1.02)
fig.tight_layout();fig.savefig("assets/bench.png",dpi=140);print("bench.png")
