import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open("assets/real_results.json"))
res = d["results"]
schemes = list(res["2"].keys())
labels = ["uniform\n(K,V pt)", "uniform\n(K,V pc)", "KIVI\n(K pc, V pt)", "wrong\n(K pt, V pc)"]
x = np.arange(len(schemes))
w = 0.38
fig, ax = plt.subplots(figsize=(9, 5))
b2 = [res["2"][s][0] for s in schemes]
b4 = [res["4"][s][0] for s in schemes]
ax.bar(x - w/2, b2, w, label="2-bit", color="#d93025")
ax.bar(x + w/2, b4, w, label="4-bit", color="#1a73e8")
for i, v in enumerate(b2): ax.text(i - w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
for i, v in enumerate(b4): ax.text(i + w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Relative attention-score error")
ax.set_title("Measured KV-cache quantization error on Qwen2.5-0.5B (77-tok prompt, 24 layers)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig("assets/real_attn_err.png", dpi=140)
print("saved real_attn_err.png")

# ---- Serving architecture diagram ----
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")
def box(x,y,w,h,t,c,fc="#fff"):
    ax.add_patch(plt.Rectangle((x,y),w,h,ec=c,fc=fc,lw=1.4))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=9,color=c,fontweight="bold")
def arrow(x1,y1,x2,y2,c="#202124"):
    ax.add_patch(plt.Arrow(x1,y1,x2-x1,y2-y1,color=c,width=1.2,lw=0))
ax.text(50,95,"KV-cache quantization in a serving stack",ha="center",fontsize=13,fontweight="bold")
box(2,60,18,18,"Prompt\n+ tokens","#5f6368")
arrow(20,69,30,69)
box(30,60,20,18,"Prefill\n(self-attn)","#1a73e8")
arrow(50,69,60,69)
box(60,60,22,18,"Paged KV\ncache (fp16)","#1a73e8")
arrow(82,69,90,69)
box(90,58,8,18,"GPU\nVRAM","#5f6368")
# quantize branch
arrow(71,60,71,46,"#d93025")
box(58,28,26,16,"Quantize: K per-channel\nV per-token (2/4-bit)","#d93025","#fce8e6")
arrow(71,28,71,18,"#d93025")
box(56,6,30,12,"Compressed KV store\n(4–8× smaller)","#188038","#e6f4ea")
ax.text(50,2.5,"decode dequant-matmul fuses Q·K' on the fly",ha="center",fontsize=8,color="#5f6368")
fig.tight_layout(); fig.savefig("assets/kv_serve.png", dpi=140)
print("saved kv_serve.png")
