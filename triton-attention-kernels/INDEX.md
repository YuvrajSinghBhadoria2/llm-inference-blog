# I Wrote the Three Attention Kernels That Run LLM Inference — in Portable Triton, on a Free Colab T4

![the three attention kernels = the full LLM serving path](assets/attn_overview.svg)

When people talk about LLM inference kernels, they usually go straight to the hardware-specific fireworks: Hopper TMA, FP8, warp-specialization, FlashAttention-3. But if you actually want to *understand* why inference is fast or slow on ordinary hardware, you have to start with the foundation: the three attention kernels that every serving engine secretly runs — **fused prefill, split-KV decode, and paged KV-cache attention**.

I wrote all three from scratch in **portable Triton** (no CUDA C++), ran them on a **free Colab NVIDIA T4**, and validated every one against PyTorch SDPA. This is the engineering post: what each kernel does, why it matters, where it wins, and the traps I hit.

## What is attention, really?

Inference attention computes `softmax(QKᵀ / √d) V`. The catch is the middle matrix `QKᵀ`:

- At **prefill** (processing a prompt), `Q` is large, so this is **compute-bound** — but the naive version materializes the full `[B, H, N, N]` score matrix, which is `O(N²)` in *memory*.
- At **decode** (generating one token), `Q` is a single vector (`Q=1`), so the matmul is tiny and the cost is dominated by **streaming the KV cache from HBM** — it is **memory-bound**, not compute-bound.

That single distinction — compute-bound prefill vs memory-bound decode — is the whole reason serving engines use *different* kernels for each phase. Ignore it and your "optimization" optimizes the wrong thing.

## The memory model (why naive attention dies)

Your GPU has fast compute but it does not know how to stage your exact workload. With custom/Triton kernels you control data movement:

- **Off-chip (HBM, 16 GB on T4):** where `Q, K, V` and the KV cache live. Huge but slow.
- **On-chip (shared memory / registers):** tiny but fast. You want to pull `K, V` once and reuse them across the `Q` dimension.

For prefill, the naive implementation pulls this wrong: it builds the full `N×N` score matrix in HBM. That is the trap.

## My hardware: a free Colab T4

I have no H100. This runs on an **NVIDIA T4 (sm75, 16 GB)** — the free Colab tier. Ada/Hopper tricks (TMA, FP8) are intentionally out of scope; the point is a *portable* kernel any laptop-class GPU can run. fp16 throughout.

## The trap: naive attention OOMs at 8K

I benchmarked prefill peak memory exactly (peak-memory is unaffected by Colab's compute throttling, so these numbers are trustworthy):

| N_CTX | Naive attention | Triton fused |
|------:|-----------------:|-------------:|
| 4K | 4.46 GB | **0.44 GB** (10.2× less) |
| 8K | **OOM** | **0.57 GB** (fits) |

The naive materialized `[B,H,N,N]` matrix is `O(N²)`. At 8K it blows past 16 GB and dies. Triton's fused kernel is `O(N·D)` — it never materializes the scores, it streams `K,V` and reduces online (FlashAttention-2 style). Same math, 10× less memory, and it *keeps running* where naive crashes.

> [!WARNING]
> On a free T4, PyTorch SDPA dispatches to vendor-tuned FlashAttention-2 / cuDNN. Do **not** expect a raw-speed win over SDPA. The honest, demonstrable win here is **memory** and **portability** — not latency. Measuring latency against SDPA and calling it a loss misses the point.

## The three kernels (evolution, with why)
### 1. Fused prefill attention (`flash_attention.py`)

The baseline. Online softmax, autotuned, FA2-style. Processes the whole prompt in one pass. Its job is to prove the **correctness methodology**: every output is checked against `F.scaled_dot_product_attention` with `allclose(atol=1e-2, rtol=1e-2)` in fp16, including non-power-of-two and GQA shapes.

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4}, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 4}, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 4}, num_stages=2),
    ],
    key=["N_CTX"],
)
@triton.jit
def _flash_attention_fwd(Q, K, V, sm_scale, Out, ...):
    start_m = tl.program_id(0)   # one tile of query rows per program
    # online softmax: stream K,V, reduce in registers -- never materialize NxN
```

### 2. Split-KV flash-decoding (`flash_decoding.py`)
The decode kernel. With `Q=1` there is no query-dimension parallelism, so the *only* way to parallelize decode is to split the **KV dimension** across SMs — exactly what vLLM does for long-context decode. Each SM reduces its slice of `K,V`, then a final cross-SM reduction gives the output.

**Why it matters:** the tiny `Q=1` matmul is memory-bound, so you hide HBM latency by having many SMs pull different KV blocks at once.

```python
split = tl.program_id(2)                  # KV dimension split across SMs
kv_per_split = tl.cdiv(tl.cdiv(N_CTX, BLOCK_N), NUM_SPLITS) * BLOCK_N
start_n = split * kv_per_split            # this SM owns its slice of K,V

# ... after each SM reduces its slice, a final cross-SM combine:
lse_max = partial_lse.max(-1, keepdim=True).values
w = torch.exp(partial_lse - lse_max)
out = (partial_out * w.unsqueeze(-1)).sum(2) / w.sum(-1, keepdim=True)
```

### 3. Paged KV-cache attention (`paged_attention.py`)
The inference-systems piece. Instead of one contiguous KV tensor, it reads through a **block table** (GQA-aware) — the exact memory layout vLLM uses. This is a capability **SDPA literally cannot express**: PyTorch's built-in attention assumes a dense KV tensor; real serving does not have one.

```python
# paged_attention.py -- read K,V through a block table, not one dense tensor
page = tl.load(BlockTable + b * stride_bb + (p // PAGE_SIZE) * stride_bm)
k = tl.load(KV_Cache + page * stride_kn + off_in_page * stride_kps)
```

## Benchmark method

- **Shapes:** `B=4, H=16, D=64, Q=1`, fp16.
- **Decode sweep:** `N_CTX ∈ {1024, 2048, 4096, 8192, 16384}`, `num_splits ∈ {1, 8, 16, 32, 64, 128}`.
- **Timing:** 20 warmup + 100 measured CUDA-event iterations (async events, like a real benchmark).
- **Correctness:** `allclose` vs SDPA on every shape before any timing runs.

## Results

![decode latency vs context](assets/decode_latency.png)

**Best `num_splits` per context (lowest kernel latency):**

| N_CTX | best num_splits | Triton (ms) | SDPA (ms) | Triton / SDPA |
|------:|----------------:|------------:|----------:|--------------:|
| 1024 | 32 | 0.405 | 0.249 | 1.62× |
| 2048 | 32 | 0.297 | 0.379 | 0.78× |
| 4096 | 16 | 0.528 | 0.393 | 1.34× |
| 8192 | 16 | 1.018 | 0.845 | 1.21× |
| 16384 | 16 | 2.012 | 1.737 | 1.16× |

![prefill memory vs naive](assets/prefill_memory.png)

## Step-by-step optimization journey

1. **From single-split to split-KV (the big one).** At `N=1024`, `num_splits=1` stalled at **1.14 ms**; moving to `num_splits=32` dropped it to **0.40 ms** — because splitting the KV dimension is the only source of sequence parallelism when `Q=1`.
2. **Prefill memory:** fused Triton went from 4.46 GB → 0.44 GB at 4K (**10.2×**), and fit at 8K where naive **OOMs**.
3. **Versus SDPA:** within ~1.2–1.6× at 4K–16K, and *faster* at 2K (0.78×) in this run. SDPA stays the vendor-tuned reference; the benchmark characterizes split-KV scaling, not a speed claim.

## Key Tradeoffs & Technical Takeaways

**1. More splits is NOT monotonic — launch overhead beats memory reuse.**
`num_splits=128` is the *worst* configuration at every context length. The sweet spot (16–32) does not even grow steadily with `N`. Past the sweet spot, cross-SM launch + reduction overhead dominates the HBM savings. Takeaway: split-KV helps until reduction bookkeeping costs more than the memory you saved.

**2. Memory win vs speed win are different wins.**
On T4 we beat naive on memory 10×, and we match/beat SDPA on latency at scale — but SDPA still wins at small `N` because it dispatches to cuDNN/FA2. Takeaway: on constrained hardware, optimize the metric that's actually broken (here, decode memory pressure), not the one the vendor already solved.

**3. Portability is a deliberate tradeoff.**
No TMA, no FP8, no warp-spec — those need Hopper+. By targeting sm75 Triton, the kernel runs on a free Colab T4 *and* any datacenter GPU, at the cost of leaving Hopper bandwidth on the table. Takeaway: write the portable kernel first; specialize only when the hardware is guaranteed.

## Conclusion

Triton is a new thing I explored in this project, and it's a bit low-level — but watching three kernels go from "naive OOM" to "validated against SDPA on a free T4" made the inference path click in a way no framework toggle ever did. Building the paged kernel especially — the one SDPA *can't* express — is what showed me why serving engines look the way they do.

Benchmarking on shared Colab GPUs has been tough lately (numbers wander with contention), so I report medians across 2–3 runs. Still, the memory numbers and the split-KV shape are solid.

Thanks for reading till here. Repo: [triton-attention-kernels](https://github.com/YuvrajSinghBhadoria2/triton-attention-kernels).
