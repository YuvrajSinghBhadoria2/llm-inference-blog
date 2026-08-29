import torch, json, sys
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"
torch.set_num_threads(max(4, torch.get_num_threads()))

print("loading", MODEL, flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
                                             torch_dtype=torch.float32).to(DEVICE)
model.eval()
L = len(model.model.layers)
H = model.config.num_attention_heads
HD = model.config.head_dim if hasattr(model.config, "head_dim") else (model.config.hidden_size // H)
KVH = model.config.num_key_value_heads
print(f"layers={L} q_heads={H} kv_heads={KVH} head_dim={HD}", flush=True)

# Real prompt -> real activations
prompt = ("Long-context inference is memory bound because the key-value cache grows with "
          "every token. Quantizing that cache is the obvious fix, but naive rounding collapses "
          "attention. The trick is to respect the outlier structure of the activations. "
          + "attention key value cache quantization inference optimization serving throughput. " * 3)
ids = tok(prompt, return_tensors="pt").input_ids[:, :256].to(DEVICE)
T = ids.shape[1]
print("seq len", T, flush=True)

captured = {}
def hook(mod, inp, out, name):
    captured[name] = out.detach()
hs = []
for l in range(L):
    a = model.model.layers[l].self_attn
    hs.append((
        a.q_proj.register_forward_hook(lambda m, i, o, n="q", lyr=l: hook(m, i, o, f"q{lyr}")),
        a.k_proj.register_forward_hook(lambda m, i, o, n="k", lyr=l: hook(m, i, o, f"k{lyr}")),
        a.v_proj.register_forward_hook(lambda m, i, o, n="v", lyr=l: hook(m, i, o, f"v{lyr}")),
    ))
with torch.no_grad():
    model(ids)
for grp in hs:
    for h in grp: h.remove()

def fake_quant(x, bits, dim):
    qmax = 2 ** bits - 1
    scale = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-9) / qmax
    return torch.round(x / scale).clamp(-qmax - 1, qmax) * scale

def attn_err(q, k, v, bits, kdim, vdim):
    kq = fake_quant(k, bits, kdim)
    vq = fake_quant(v, bits, vdim)
    d = q.shape[-1] ** 0.5
    A = torch.softmax(q @ k.transpose(-1, -2) / d, -1)
    Ap = torch.softmax(q @ kq.transpose(-1, -2) / d, -1)
    O = A @ v
    Op = Ap @ vq
    eA = (A - Ap).norm() / A.norm()
    eO = (O - Op).norm() / O.norm()
    return eA.item(), eO.item()

schemes = {
    "uniform (K,V per-token)":  (3, 3),
    "uniform (K,V per-channel)":(2, 2),
    "K per-channel, V per-token (KIVI)": (2, 3),
    "K per-token, V per-channel (wrong)": (3, 2),
}
results = {b: {} for b in (2, 4)}
for b in (2, 4):
    for name, (kd, vd) in schemes.items():
        es = []
        for l in range(L):
            q = captured[f"q{l}"].reshape(1, T, H, HD).transpose(1, 2)
            k = captured[f"k{l}"].reshape(1, T, KVH, HD).transpose(1, 2)
            v = captured[f"v{l}"].reshape(1, T, KVH, HD).transpose(1, 2)
            # GQA: expand kv heads to match query heads (dim 1 after transpose)
            k = k.repeat_interleave(H // KVH, dim=1)
            v = v.repeat_interleave(H // KVH, dim=1)
            eA, eO = attn_err(q, k, v, b, kd, vd)
            es.append((eA, eO))
        results[b][name] = (sum(e[0] for e in es)/L, sum(e[1] for e in es)/L)
        print(f"bits={b} {name:38s} attn_err={results[b][name][0]:.4f} out_err={results[b][name][1]:.4f}", flush=True)

json.dump({"T": T, "L": L, "results": results}, open("assets/real_results.json", "w"), indent=2)
print("saved assets/real_results.json", flush=True)
