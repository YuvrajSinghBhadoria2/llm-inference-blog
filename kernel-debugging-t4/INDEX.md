# I Found Three Real Inference-Kernel Bugs With a Free T4 (and No Local GPU)

When you read a kernel fix in a pull request, you see the final diff. You rarely see the part that matters most: *how do you know the bug is real, and how do you know the fix works?*

I don't have a GPU at my desk. So over a few weeks I used a **free Kaggle NVIDIA T4 (sm75)** to reproduce three separate inference-kernel bugs on real hardware, then verify the fixes against PyTorch references. This is the method and the measurements — not the marketing version.

## Why a T4 is enough

The stack on Kaggle was **torch 2.11.0+cu130** with **real Triton 3.6.0**, 2× Tesla T4, capability `(7,5)`.

There is a persistent claim that "Triton dropped Turing support after 3.3." I checked it the only way that counts: I compiled and ran actual Triton kernels on sm75. They run. `TRITON_INTERPRET` was never used — every number below is from a real T4 executing real compiled kernels.

The T4 has **no bf16 and no fp8**, which sounds disqualifying for LLM kernels. It isn't, because numerical bugs don't care about your dtype:

- The quantized-attention bug reproduces in **fp16 + int8/int4** KV.
- The softmax-NaN bug reproduces in **fp32**.
- The softplus-tail bug is an **fp32** rounding problem.

If anything, the T4's limits make the evidence *cleaner*. You cannot hide behind a vendor-tuned Hopper path; the kernel you wrote is the kernel that runs.

## The method

Every investigation followed the same four steps:

1. **Clone the repo twice** — once at the parent commit (before), once at the fix commit (after).
2. **Import the kernel standalone.** I extracted a minimal import of the pure-Triton kernel with a small stub package, so no compiled extensions or model weights are needed. The reproducer is self-contained and rerunnable.
3. **Run the exact same script on both commits**, same GPU, same inputs.
4. **Compare against a PyTorch reference** with a deliberately strict tolerance.

That last point is where most "it works" claims quietly die. Project test helpers often have a loose absolute-error shortcut — FLA's `assert_close` has a 1e-6 absolute shortcut that would have completely hidden one of these bugs, whose erroneous value differs by only **4.1e-8**. I used `torch.testing.assert_close(atol=1e-12)` instead. If your tolerance is looser than your bug, you have no test.

## Bug 1 — Quantized PagedAttention read the wrong request's row

**Repo:** InternLM/lmdeploy · PR #5175 · fixes issue #5154

`flash_attn_with_kvcache` accepts `num_tokens = batch * seq_len` query rows, and allocates its split-K scratch with `num_tokens` rows. The unquantized kernel splits its program id into `(kv_head, subtile)` and derives the token from the offset:

```
cur_head  = cur_kv_head * kv_group_num + offs_h % kv_group_num
cur_token = cur_batch * seq_len + offs_h // kv_group_num
```

The **quantized** kernel used `program_id(0)` directly as `cur_kv_head` and indexed both query loads and *all* scratch stores with `cur_batch` (the request id). With `seq_len > 1` that means:

- every query token of a request reads request 0's query row,
- every query token writes request 0's scratch rows,
- the remaining `num_tokens - batch` scratch rows are never written, so the reducer reads stale values.

The reporter's `B=2, S=4, INT8` case returned `2.0` for query row 1 where `0.0` was expected.

### The fix

Port the unquantized tiling to the quantized kernel: add `seq_len` as a constexpr, derive `cur_kv_head`, `cur_head` and `cur_token` from the tile id exactly as the unquantized path does, and use `cur_token` for query loads and scratch stores. Per-request accesses (`cache_seqlens`, `page_table`) still use `cur_batch`. For `seq_len == 1` the mapping is identical to before.

### T4 evidence

| Commit | nbits=8 result |
|---|---|
| Base `d958506` | **FAIL** — 740/1024 mismatched (72.3%), max abs diff **1.875**, allowed 0.001 |
| Fix `9793305` | **PASS** — max abs error **0** |

Reproduce: https://www.kaggle.com/code/yuvrajsinghbhadoria/lmdeploy-5175-quant-kv-multi-token-t4

**The honest part:** reviewing the int4 leg on real hardware exposed a bug in *my own test harness*, not the kernel. My nibble unpacking used `stack(..., dim=-1).flatten(-1)` — a no-op on the last dimension that interleaves nibbles. Fixed to `cat([low, high], dim=-1)` to match the kernel's layout. A wrong test can fail for the wrong reason; hardware made that visible.

## Bug 2 — Online softmax returns NaN when the first tile is fully masked

**Repo:** linkedin/Liger-Kernel · PR #1664 · fixes issue #1660

Three kernels — cross entropy, fused CE+TVD, and GRPO log-softmax — use online softmax with a running max initialized to `-inf`. If the first streamed tile of logits is entirely `-inf` (a fully masked leading block), the max stays `-inf`, and the rescale factor becomes:

```
exp(old_max - new_max) = exp(-inf - -inf) = NaN
```

That NaN propagates into the log-sum-exp, so all three kernels return NaN **even though later tiles contain finite logits**.

### The fix

Use `0.0` as the rescale reference while the running max is `-inf`. At that point the running sum is still `0`, so this is numerically a no-op; the next finite tile rescales normally.

### T4 evidence

All inputs fp32; first tile of logits set to `-inf`.

| Kernel | Before `33b66dc` | After `9bd13cd` |
|---|---|---|
| cross_entropy | NaN (6 nonfinite) | finite, max abs vs torch `9.54e-7` |
| fused_ce_tvd / ce | NaN (4 nonfinite) | finite, `9.54e-7` |
| fused_ce_tvd / tvd | NaN (4 nonfinite) | finite, `1.79e-7` |
| grpo_logsoftmax | NaN (6 nonfinite) | finite, `9.54e-7` |

The regression test file: `3 passed in 1.84s`. Reproduce: https://www.kaggle.com/code/yuvrajsinghbhadoria/verify-liger-1664-t4

## Bug 3 — Softplus tail cancellation silently drops residual gate mass

**Repo:** fla-org/flash-linear-attention · PR #1787 · fixes issue #1776

The fused GDN/KDA gate computes `softplus(x) = log2(1 + exp2(x))`. For raw gates below about **-17**, `1 + exp(x)` rounds to exactly `1.0` in fp32, and the log path returns `0`. The true value is a small but nonzero residual (about `8.6e-8` at `x = -17`), and `lg2.approx` could not recover it below about `-16.5`.

Why does a `1e-7` gate value matter? In an 8192-token state recurrence, dropping that residual **per key** compounds: the accumulated state came out as `1.0` instead of the expected `0.9945884`. It's the same tail-cancellation class of bug as Bug 2 — a numerically tiny error that becomes a large one after a long reduction.

### The fix

Both `softplus_nv` (PTX) and `softplus_triton` now use `exp(x)` directly for `x < -8`, where `softplus(x)` and `exp(x)` agree within the existing approximation error. The `log(1 + exp(x))` path is unchanged elsewhere.

### T4 evidence

- `test_softplus_stability.py` — **3 passed** (x in [-20, -10], both implementations, 8192-token GDN accumulation)
- 4 gate/GDN/KDA test groups — **102 passed**
- Benchmark `gdn_gate_fwd`, 16384×64 fp32: **0.1447 ms → 0.1456 ms (+0.6%, run-to-run noise)**

That last row matters. A numerical fix that cost 15% would need a different conversation. This one is free.

## What this adds up to

Three bugs, three repos, three different failure modes — wrong memory index, NaN from an undefined `inf - inf`, and fp32 tail cancellation. All three were invisible to casual inspection and caught by running the real kernel against a strict reference on hardware I didn't own.

The transferable lesson is not "use a T4." It is:

> **A bug you can't reproduce is a bug you can't fix confidently.** You do not need an H100 to find numerical and indexing errors in inference kernels. You need a compiled kernel, a strict reference, and the discipline to run the before and the after on the same machine.

And a second, less comfortable lesson: **two of the three "bugs" I first thought I had found were in my own test harness or measurement path** — the nibble unpack, and the loose project tolerance that would have hidden the softplus regression. Hardware doesn't just validate your fix. It validates your *method*.

## References

- lmdeploy PR #5175 — https://github.com/InternLM/lmdeploy/pull/5175
- Liger-Kernel PR #1664 — https://github.com/linkedin/Liger-Kernel/pull/1664
- FLA PR #1787 — https://github.com/fla-org/flash-linear-attention/pull/1787
- T4 stack evidence — torch 2.11.0+cu130, Triton 3.6.0, sm75
