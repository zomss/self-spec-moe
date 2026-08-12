# Unified design: paged FA3 in a full-chain graph, plus a quantized decode kernel

Date: 2026-08-12
Status: **revised by X8.** The original proposal here was to put paged FA3
inside the full-chain graph. X8 measured the crossover and shows that is the
wrong unification for this operating range: the draft's paged-FA3 path costs
1.88x-3.35x MORE than the scratchpad+splitkv path it would replace, at every
batch and both contexts. Section 2 is superseded by section 6; sections 1 and 3
stand.

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

## 6. X8 — the crossover, and why the unification target changes

16 cells: batch x context, fully crossed, both modes, wall with no profiler plus
kernel mix from the same boot. Whole-chain capture verified per boot.

### There is no crossover inside the K=4 operating range

| batch | R1 (~80 tok) | R5 (~14k tok) |
| --- | --- | --- |
| 1 | 1.408x | 1.373x |
| 4 | 1.379x | 1.277x |
| 8 | — | 1.241x |
| 16 | 1.240x | 1.196x |
| 32 | **1.070x** | — |

Whole-chain wins all 16. But the margin collapses with batch, and the dynamic-K
schedule is `[[1, 32, 4]]` — above batch 32 the chain length itself changes, so
the arms stop being comparable. **The margin runs out exactly where the K=4
range does.**

### Batch drives it; context does not

| R1 | pw bubble | wc bubble | device gain | bubble gain |
| --- | --- | --- | --- | --- |
| batch 1 | 9.51 | 3.88 | −4.60 | −5.64 |
| batch 4 | 9.97 | 4.05 | −4.00 | −5.92 |
| batch 16 | 7.22 | 4.36 | −3.99 | −2.86 |
| batch 32 | **3.04** | **5.25** | −4.70 | **+2.21** |

The device gain is **constant at ~4 ms** across every batch and both contexts.
All of the batch dependence lives in the bubble: piecewise's host bubble
collapses 9.5 → 3.0 ms as batch grows, because more GPU work per step hides the
same ~116 round-trips. At batch 32 it **inverts** — whole-chain carries the
larger bubble and stays ahead only on device time.

At matched batch, context is nearly irrelevant (R1 b16 bubble 7.22 vs R5 b16
8.27). That is the predicted signature: the draft chain's GPU work scales with
batch but not with context, because the window caps its attention at 272 keys
either way. The earlier "+74% with workload" reading of R1 → R5 was
**confounded** — R1 is batch 1 and R5 is batch 8, so that comparison moved both
axes at once. It was batch.

### The unification target is the scratchpad path, not paged FA3

| cell | paged FA3 path | splitkv + scratchpad | ratio |
| --- | --- | --- | --- |
| R1/b1 | 3936 µs | 1174 | 3.35x |
| R1/b4 | 3998 | 1706 | 2.34x |
| R1/b16 | 4983 | 2599 | 1.92x |
| R1/b32 | 6348 | 2269 | 2.80x |
| R5/b16 | 4960 | 2631 | 1.88x |

Paged FA3 is more expensive for the draft **everywhere in range**, by roughly
2-3x, and the gap does not close with batch or context. Section 2's plan — make
FA3's schedule shape-static so it can be captured — would buy capturability at
the price of a 2-3x more expensive attention path. It solves a problem the
scratchpad path does not have.

**Revised recommendation:**

1. Unify on **scratchpad + splitkv inside the full-chain graph** for every
   windowed configuration. That is what FULLCG already does; the work is to make
   it the default and to widen capture coverage, not to build a new path.
2. Keep **paged FA3 + piecewise** only where the scratchpad cannot apply —
   `window: off`, non-greedy sampling, and any `(batch, K)` capture misses.
3. **Do not** invest in making paged FA3 capturable. The measurement that would
   justify it — paged FA3 being the faster kernel — is false at these shapes.
4. Revisit only if the operating range moves above batch 32, where the margin is
   1.07x and the bubble advantage has already inverted. That is a K-schedule
   question first: the K=4 range ends at 32.

Section 3 is unaffected. The W4A16 decode kernel remains the largest single
opportunity (41% of device time, 65% of bandwidth-bound, ~3.0 ms/step headroom),
and it is now the *only* large one on the draft side.
