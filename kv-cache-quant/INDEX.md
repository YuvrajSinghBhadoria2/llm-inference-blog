# KV-Cache Quantization: How 2–4 Bits Cut LLM Memory by 4–8× Without Killing Quality

Every token an LLM generates, it pays rent. Not to the weights — those are loaded once — but to the
**KV cache**: the stored Key and Value vectors from every previous token that attention needs to avoid
recomputing. As context windows stretched from 4K to 128K and beyond, that rent became the dominant
line item in inference cost. On long prompts it is no longer the weights that decide whether a request
fits on the GPU — it is the cache.

The obvious lever is quantization: store the cache in 4-bit or 2-bit instead of fp16. But the KV cache
is not a passive buffer you can round carelessly. Get the scheme wrong and attention silently collapses.
Get it right and you can shrink the cache 4–8× with perplexity unchanged. This post walks through why
naive quantization fails, what the leading methods actually do, how far the wins go, and where the
technique still breaks — with numbers pulled from primary papers and a small computation you can
reproduce yourself.

## The problem, in bytes

The KV cache holds, for every layer, the K and V projections of every token processed so far. Its size
is:

```
KV_bytes = 2 · n_layers · n_kv_heads · head_dim · bytes_per_elem · seq_len · batch
```

For **Llama-3-8B** (32 layers, 8 KV heads via GQA, head_dim 128), that is **128 KiB per token at fp16**.
Linear in both sequence length and batch. Here is what that means in practice:

| Context (tokens) | fp16   | 8-bit | 4-bit | 2-bit |
|------------------|--------|-------|-------|-------|
| 4,096            | 0.54 GB| 0.27 GB| 0.13 GB| 0.07 GB|
| 8,192            | 1.07 GB| 0.54 GB| 0.27 GB| 0.13 GB|
| 32,768           | 4.29 GB| 2.15 GB| 1.07 GB| 0.54 GB|
| 131,072          | 17.18 GB| 8.59 GB| 4.29 GB| 2.15 GB|

![KV cache memory vs context length for Llama-3-8B](assets/kv_memory.png)

Now multiply by batch. At **64 concurrent requests and 32K context**, a single fp16 cache is **~275 GB** —
far past any single GPU. Even 4-bit leaves **~69 GB**. This is why throughput on long-context servers is
memory-bound, not compute-bound: the GPU's compute units sit idle while the KV cache is shuttled from
main memory into SRAM for every generated token (KIVI, ICML 2024).

> [!WARNING]
> Memory-bound ≠ free lunch. Cutting the cache's footprint raises *how many requests you can batch*,
> which raises throughput — but only if your serving stack actually supports KV quantization end to end
> (paged allocation, dequant at the matmul, correct calibration). Quantizing the tensor but not the
> memory manager gets you neither the memory win nor the speedup.

## Why naive quantization fails

The first instinct — uniform, per-token quantization of every K/V vector — is quietly catastrophic.
The reason is in the *distribution* of the cache, not the arithmetic:

- **Keys have persistent per-channel outliers.** A few fixed channels carry very large magnitudes across
  all tokens (consistent with the activation-outlier findings in LLM.int8() and AWQ). Quantizing keys
  *per-token* smears those outliers into every other channel.
- **Values have no such pattern**, but they are mixed by attention into the output, so the error couples
  across tokens. Quantizing values *per-channel* lets one token's error leak into another's.

KIVI measured the consequence directly: per-token key quantization produces nearly **5× larger attention
score error** than per-channel key quantization, and per-channel value quantization is **~15× worse** than
per-token. The fix is asymmetric and non-obvious — **quantize keys per-channel, values per-token** — and
that single insight is what makes 2-bit KV cache viable.

A minimal sketch of the right scheme:

```python
# Asymmetric KV quantization (KIVI-style)
def quantize_key_per_channel(K, bits=2):
    # K: [n_kv_heads, seq, head_dim] -> scale per channel (head_dim dim)
    scale = K.abs().amax(-1, keepdim=True) / (2**bits - 1)
    return round(K / scale).clamp(...), scale        # dequant: Q * scale

def quantize_value_per_token(V, bits=2):
    # V: [n_kv_heads, seq, head_dim] -> scale per token (seq dim)
    scale = V.abs().amax(-2, keepdim=True) / (2**bits - 1)
    return round(V / scale).clamp(...), scale
```

## What the leading methods actually do

Three papers define the modern frontier. They agree on the outlier structure and diverge on how hard they
push the bit-width.

| Method | Scheme | Bits | Quality | Memory / throughput win |
|--------|--------|------|---------|--------------------------|
| **KIVI** (ICML'24) | per-channel K, per-token V; recent tokens kept fp16 ("residual window") | 2 | ~2% drop on Llama-2/Mistral (GSM8K); Falcon needs 4-bit | 2.6× peak mem (incl. weights), 4× larger batch, **2.35–3.47× throughput** |
| **KVQuant** (NeurIPS'24) | pre-RoPE per-channel K, non-uniform (NUQ), dense-and-sparse (1% outliers) | 3 | **<0.1 perplexity drop** (WikiText-2, C4) | 4.8× compression; serves LLaMA-7B at **1M ctx on 1× A100-80GB**, 10M on 8× GPU; ~1.7× matvec speedup |
| **GEAR** (ICML'24) | quantization + low-rank error approx + sparse outliers | 2 | near-lossless; up to **24.4% accuracy gain** over SOTA baselines at 2-bit | 2.39× peak mem, **2.1–5.07× throughput** |

A few things worth pulling out:

- **KIVI is the practical baseline.** It is tuning-free, plug-and-play, and ships as a drop-in HuggingFace
  wrapper. The residual (full-precision) window for the most recent tokens is what rescues hard reasoning
  tasks — without it, fake 2-bit quantization on GSM8K craters.
- **KVQuant is the "go long" play.** By quantizing *before* RoPE (which otherwise mixes outlier channels)
  and using sensitivity-weighted non-uniform codebooks plus a 1% sparse outlier store, it reaches 3-bit
  with sub-0.1 perplexity loss and turns context length from a memory wall into a tunable dial.
- **GEAR is the "go low" play.** Rather than fight residuals, it models them: a low-rank matrix recovers
  the coherent part of the quantization error and a sparse matrix catches the rest. It augments any base
  quantizer (KIVI, KCVT) and is the only one of the three that stays near-lossless at 2-bit on complex
  generation.

> [!WARNING]
> These numbers are method-specific, not universal. KIVI reports that **Falcon-7B (multi-query attention,
> a single KV head) needs 4-bit**, not 2-bit, because MQA is already so compressed that there is no
> redundancy left to trade. If your model uses MQA, do not assume 2-bit works.

## Where it still breaks

Quantizing the KV cache is mature but not magic. The honest failure modes:

1. **Small models + long context are the danger zone.** Perplexity degradation grows as you shrink the
   model and lengthen the prompt; the headroom that hides quantization error in a 70B model does not exist
   in a 7B one.
2. **Reasoning tasks are unforgiving.** GSM8K-style math is where fake quantization fails hardest; the
   full-precision recent-token window (KIVI) or low-rank residual (GEAR) is not optional.
3. **Throughput only follows if the kernel is there.** KVQuant and GEAR both ship custom CUDA kernels;
   without fused dequant-matmul, the quantization/dequantization overhead eats the memory win. Production
   stacks (vLLM, TensorRT-LLM) now support **fp8 KV cache** as the default safe setting — 2× smaller,
   near-lossless, and hardware-accelerated on Hopper/Ada. 4-bit and below still require the research
   kernels above.
4. **Calibration matters at the low end.** KVQuant calibrates key scales offline; value scales are computed
   online. Skip calibration and 3-bit quality degrades sharply.

## A practical decision guide

- **Just want it to work, now:** use **fp8 KV cache** in your serving framework. ~2× memory, hardware
  accelerated, essentially lossless, zero custom code.
- **Need 4× memory and run consumer/Ada GPUs:** **KIVI at 4-bit** is the safe, tuned-free choice.
- **Serving very long context (hundreds of K to millions of tokens):** **KVQuant** — it is purpose-built
  for the long-context regime and quantizes *before* RoPE to keep quality.
- **Pushing to 2-bit on complex generation:** **GEAR** (or GEAR-L for the efficient variant), layered on
  top of a base quantizer.

## Reproducing the memory numbers

The table and chart above come from a 30-line script (`experiment.py`) using only the public Llama-3-8B
spec. The core formula:

```python
def kv_bytes(seq_len, batch, bits):
    # 2 (K and V) * layers * kv_heads * head_dim * (bits/8) * seq * batch
    return 2 * 32 * 8 * 128 * (bits / 8) * seq_len * batch

print(kv_bytes(131072, 1, 16) / 1e9)   # 17.18 GB  (fp16, single request, 128K)
print(kv_bytes(131072, 1, 4)  / 1e9)   # 4.29 GB   (4-bit)
```

Run it and you get the figures in this post. They are not benchmark results — they are the deterministic
memory accounting that tells you *whether the request fits at all*, which is the question KV-cache
quantization exists to answer.

## Takeaway

KV-cache quantization is the rare optimization that is both mathematically simple and empirically
subtle. The win is real and large — **4–8× memory reduction, 2–5× throughput** — but it is earned by
respecting the cache's outlier structure, not by rounding harder. For most teams the right move today is
fp8 in the serving stack; for the long-context frontier, KVQuant and GEAR show that 2–3-bit caches are no
longer research curiosities but deployable.

## References

- Liu et al., **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache** (ICML 2024).
  [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) · [github.com/jy-yuan/KIVI](https://github.com/jy-yuan/KIVI)
- Hooper et al., **KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization**
  (NeurIPS 2024). [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) · [github.com/SqueezeAILab/KVQuant](https://github.com/SqueezeAILab/KVQuant)
- Kang et al., **GEAR: An Efficient Error Reduction Framework for KV Cache Compression in LLM Inference**
  (ICML 2024). [arXiv:2403.05527](https://arxiv.org/abs/2403.05527) · [github.com/opengear-project/GEAR](https://github.com/opengear-project/GEAR)
- Kwon et al., **vLLM / PagedAttention** (SOSP 2023) — the system layer KV quantization slots into.
  [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
