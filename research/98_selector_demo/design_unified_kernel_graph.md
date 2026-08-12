# Unified design: paged FA3 in a full-chain graph, plus a quantized decode kernel

Date: 2026-08-12
Status: design proposal. The measurements it rests on are X6/X7 (recorded in
`design_quant_skip_penalty.md`); nothing here is implemented or measured yet.

## 1. Why paged FA3 cannot be captured today

`unified_attention_with_output` is the piecewise split op, which is why the
draft chain pays ~116 host round-trips per step. The reason it must be a split
is visible in the trace:

| kernel | piecewise | wholechain |
| --- | --- | --- |
| `prepare_varlen_num_blocks_kernel` | **113 calls/step** | 1 call/step |
| `_vllm_fa3_C::fwd` | 112 calls/step | 0 |

`prepare_varlen_num_blocks` computes FA3's varlen **work distribution** — how
tiles are assigned to sequences — from the runtime sequence lengths. FA3 decode
schedules per launch. A CUDA graph freezes whatever schedule existed at capture,
so replaying it with different sequence lengths would compute against the wrong
tile assignment. That is the replay-safety problem, and it is why the runtime
falls back to the `flash_fwd_splitkv` scratchpad path under FULLCG.

Note what is *not* the blocker: the block table itself. A graph can read a
tensor whose **contents** change as long as its **address** is stable, and the
block table is a persistent in-place-updated buffer. Paging is compatible with
capture. Only the host-computed schedule is not.

## 2. The window lever is what makes the unified design possible

The draft's KV read is capped at `min(context, window + sinks)` — 272 keys at
`window=256`. Once context exceeds that bound, **the draft's attention length is
constant for the rest of the sequence.** The quantity whose variation blocks
capture stops varying.

So the unified design is:

* Pad the draft's attention length to the window bound **always**, and mask.
  Then the varlen schedule is a function of `(batch, K, window)` only — all
  known at capture time — rather than of runtime sequence lengths.
* Wasted work is bounded by the window, and it is small: the splitkv kernel over
  272 keys costs 5.88 µs/call against a 19.40 µs GEMM in the same layer.
* Capture the whole K-step chain with **paged FA3** inside it, keyed on
  `(batch, K)` instead of today's `(batch, table_width, K)` — the table width
  stops mattering once the read is window-bounded.

This is the interesting part for the paper: **the window lever and the
graph-capture lever are not independent.** Windowing is what converts attention
from a shape-varying op into a shape-static one, and that is what makes the
chain capturable. It also explains cleanly why FULLCG requires `KV_WINDOW > 0` —
that is not an implementation quirk, it is the precondition.

`window: off` therefore cannot take this path at all, and must either stay
piecewise or re-capture per context bucket. That belongs in the cost model as a
term, not as a footnote.

## 3. Why the quantized draft needs its own GEMM kernel

Once the host stall is removed, the remaining cost is dominated by the quantized
draft's GEMM. At R1 (batch 1), of 21.27 ms/step of device time:

| kernel | ms/step | share | calls | µs/call |
| --- | --- | --- | --- | --- |
| **cutlass `GemmUniversal` (W4A16 draft)** | **8.69** | **41%** | 448 | 19.40 |
| `nvjet_tst_192x8` (bf16 target) | 2.48 | 12% | 36 | 68.99 |
| `nvjet_tst_384x8` (logits) | 2.06 | 10% | 5 | 412.79 |
| `nvjet_tst_64x8_splitK` | 1.79 | 8% | 72 | 24.86 |
| draft attention (splitkv + combine) | 1.09 | 5% | 224 | 4.85 |

The draft GEMM is the single largest line, and it is inefficient:

```text
active W4A16 weights     4.744 GB  (28 of 36 layers)
GEMM calls               448/step = 112/forward x 4 forwards
bytes per call           42.4 MB   (every forward re-reads the full weight set)
weight traffic           18.98 GB/step

H100 HBM3 @ 3.35 TB/s -> ideal 12.65 us/call
measured                          19.40 us/call   =  65% of bandwidth-bound
headroom                                             3.03 ms/step
```

At decode the draft runs **M = batch tokens** (M=1 at R1, M=8 at R5). That is
not a GEMM, it is a GEMV: the arithmetic intensity is ~1, and the only thing
that matters is streaming 4-bit weights at full bandwidth. `GemmUniversal` is a
general tiled GEMM; at M=1 most of each tile is padding, and it reaches 65% of
the achievable rate.

The bf16 draft does not have this problem, which is the asymmetry worth naming:
cuBLASLt's `nvjet` kernels already carry thin-M shapes (the `64x8` tiling in the
table is exactly that). **bf16 has a decode-shaped kernel and W4A16 does not.**

### The graph makes specialization easier, not harder

Under a captured chain, `M` is fixed at capture time. A shape-specialized W4A16
decode kernel — Marlin-style split-K, or a dedicated GEMV with the dequant fused
into the load — can be selected once per graph rather than dispatched per call.
The full-graph design and the specialized kernel reinforce each other: the graph
removes the host cost that hid the GEMM inefficiency, and fixed shapes make the
faster kernel safe to pick.

Expected combined effect at R1, if both land: 25.38 ms/step (today, whole-chain)
− ~3.0 ms (GEMM at bandwidth) ≈ 22.4 ms, against 35.46 ms for the piecewise
baseline this phase measured.

## 4. Proposed order of work

1. **Measure the crossover first.** X7 shows the kernels FULLCG *adds* grow ~6x
   faster with workload than the ones it removes (+74% vs +13% from R1 to R5).
   Before building a unified path, find where paged-FA3-piecewise wins back, by
   sweeping batch and context. If the crossover sits inside the operating range,
   FULLCG is a lever the selector must choose and the cost model needs a term
   for it — which changes what "unified" should mean.
2. **Window-bounded static attention shape**, so the FA3 schedule depends only
   on `(batch, K, window)`. This is the enabling change for item 3.
3. **Capture the chain with paged FA3**, keyed on `(batch, K)`.
4. **W4A16 decode kernel** for M ≤ 8, selected at capture time.
5. Keep piecewise as the fallback for `window: off`, non-greedy sampling, and
   any `(batch, K)` the capture does not cover — and **log** when it is taken,
   so a silent fallback cannot be mistaken for a fix that did not work. That
   failure mode already happened once in this phase.

## 5. What is assumed and not yet verified

* That FA3's schedule becomes constant under a padded, window-bounded length.
  This is inferred from `prepare_varlen_num_blocks` being called once per
  attention under piecewise and once per step under wholechain; it has not been
  confirmed by reading FA3's scheduler.
* That the W4A16 GEMM is bandwidth-bound rather than limited by the dequant
  path. The 65% figure assumes the former; a roofline check at M=1 against a
  pure weight-streaming baseline would settle it.
* Whole-chain capture requires `all_greedy` and keys on batch. Coverage across
  the selector's dynamic-K schedule is unmeasured. If capture misses common
  shapes, the unified design's benefit is smaller than these numbers suggest.
