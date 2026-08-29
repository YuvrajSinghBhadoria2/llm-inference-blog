import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Llama-3-8B KV-cache config (GQA)
n_layers = 32
n_kv_heads = 8
head_dim = 128
bytes_fp16 = 2

# Bytes per token (K and V) at fp16
bytes_per_token_fp16 = 2 * n_layers * n_kv_heads * head_dim * bytes_fp16
print(f"KV cache bytes/token (fp16): {bytes_per_token_fp16} = {bytes_per_token_fp16/1024:.1f} KiB")

# bits -> bytes per element (K and V share the same precision)
def kv_bytes(seq_len, batch, bits):
    return 2 * n_layers * n_kv_heads * head_dim * (bits/8) * seq_len * batch

contexts = [4096, 8192, 32768, 131072]
batches = [1, 64]
bit_options = [16, 8, 4, 2]

print("\n=== Single request (batch=1) KV cache size by context ===")
print(f"{'ctx':>8} | {'fp16':>8} | {'8-bit':>8} | {'4-bit':>8} | {'2-bit':>8}")
for c in contexts:
    row = []
    for b in [16,8,4,2]:
        gb = kv_bytes(c, 1, b) / 1e9
        row.append(f"{gb:8.2f}")
    print(f"{c:>8} | " + " | ".join(row))

print("\n=== Batched (batch=64) at 32K context ===")
for b in bit_options:
    gb = kv_bytes(32768, 64, b) / 1e9
    print(f"  {b:>2}-bit: {gb:8.2f} GB")

# A100-80GB: how much context fits at batch=1 leaving 40GB for weights
weights_gb = 16  # Llama-3-8B fp16
free_gb = 80 - weights_gb
print("\n=== Max context (batch=1) on A100-80GB (16GB weights reserved) ===")
for b in [16,4,2]:
    max_ctx = free_gb*1e9 / kv_bytes(1,1,b)
    print(f"  {b:>2}-bit: ~{max_ctx:,.0f} tokens")

# Chart: KV cache size vs context (batch=1) for fp16/4bit/2bit
fig, ax = plt.subplots(figsize=(8,5))
for b, color, label in [(16, "#1f77b4", "fp16"), (4, "#ff7f0e", "4-bit"), (2, "#2ca02c", "2-bit")]:
    sizes = [kv_bytes(c,1,b)/1e9 for c in contexts]
    ax.plot(contexts, sizes, marker="o", color=color, label=label)
ax.set_xscale("log", base=2)
ax.set_xlabel("Context length (tokens, log2)")
ax.set_ylabel("KV cache size (GB, batch=1)")
ax.set_title("Llama-3-8B KV-cache memory vs context length")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("assets/kv_memory.png", dpi=130)
print("\nsaved assets/kv_memory.png")
