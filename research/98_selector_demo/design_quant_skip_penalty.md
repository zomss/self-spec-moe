# Why does quantization × layer-skip cost more? — investigation design

Date: 2026-08-12
Status: **mechanism identified, not yet root-caused.** X1–X3 complete. The cost
is GPU idle while the host runs framework code between piecewise cudagraph
replays; the inter-replay host cost roughly triples at skip8. What has not been
identified is *which* host operation grows. X4 (CPU profile) is the next step.

## 1. The defect, stated precisely

**28 quantized draft layers cost more than 32 quantized draft layers.**

Measured in one session, same window, same prompts, `draft_forward` region
(CUDA-synced at both ends):

| arm | active layers | draft_forward | ms/layer |
| --- | --- | --- | --- |
| w4a16, skip4 `{2,4,7,16}` | 32 | 4.437 ms | 0.1386 |
| w4a16, skip8 `{2,4,7,11,16,20,25,30}` | 28 | **5.323 ms** | **0.1901** |
| bf16, skip4 `{2,4,7,16}` | 32 | 5.898 ms | 0.1843 |
| bf16, skip8 `{2,4,7,11,16,20,25,30}` | 28 | 5.205 ms | 0.1859 |

bf16 is linear in layer count. Quantized is not: removing four more identical
layers *adds* 0.886 ms, and the per-layer cost lands on the bf16 line — W4A16's
~27% per-layer advantage is exactly cancelled.

Qwen3-8B is dense, 36 homogeneous layers, uniform W4A16 (`group_size=128`,
`pack-quantized`, only `lm_head` in `ignore`). There is no layer heterogeneity
available to explain it.

**What it costs.** At R5 the composed configuration should deliver 18.93 ms of
draft chain against a 51.48 ms unlevered baseline — **2.72×**. It measures
27.76 ms — **1.85×**. Recovering this is worth **8.8 ms/step**, and it is the
difference between "levers compose" and "levers compose except when you use two
of the three best ones together."

## 2. What is already established

Numbered so the elimination table can cite them.

- **F1** `draft_forward` is 89% of `draft_chain`. Host orchestration is flat at
  2.77–3.10 ms in every cell, anomalous or not (fine-region pass, 18 sub-regions).
  *The anomaly is inside the model forward, not the runtime around it.*
- **F2** Flat in context: 27.76 ms at 431 tokens and 27.76 ms at 14,357 tokens.
  Flat in window: w128 and w1024 agree to 0.06 ms. *Nothing KV-shaped.*
- **F3** Identical `MacheteLinearKernel` / `CompressedTensorsWNA16` selection,
  identical `splitting_ops`, `graph_partition: False`, identical cudagraph
  capture progress, KV memory within 1% (48.52 vs 48.97 GiB).
- **F4** `wc_replay` never fires in any arm. This does **not** mean the chain
  skips cudagraph replay: X2's host counters show 117 `cudaGraphLaunch`/step at
  skip8 and 133 at skip4, matching 29 and 33 piecewise pieces x 4 forwards. The
  chain *is* replayed piecewise; `wc_replay` simply covers a different path.
- **F5** The Humming JIT prewarm logs `n_layers == 0` in every quantized boot,
  so the Humming path is not involved.
- **F6** Per-boot `stdev_armed_step_s` ≈ 0.17 ms; arms differ by 0.9 ms. The
  effect is ~5σ, and reproduced across two independent runs (Round 1 and the
  probe) and two window values.

## 3. What the arithmetic already eliminates

Fit the obvious two-term form to both families, where `n` = executed layers,
`s` = skipped layers, `n + s = 36`:

```text
T = n*c + s*b        c = cost of executing a layer, b = cost of skipping one
```

| family | c (per executed layer) | b (per skipped layer) | b as a fraction of c |
| --- | --- | --- | --- |
| bf16 | 0.1831 ms | **0.0098 ms** | 0.05× |
| w4a16 | 0.0986 ms | **0.3201 ms** | **3.2×** |

In bf16 a skipped layer costs about 5% of a computed one — a small, physical
bookkeeping cost, exactly what a passthrough module should cost. In the
quantized family the same fit demands that skipping a layer cost **3.2× more
than computing one**, which is not a physical cost of skipping.

This eliminates the entire **host/graph-structure family** in one step:
`_SkipDecoderLayer` passthroughs, Dynamo graph breaks from its `*args/**kwargs`
signature, and per-piece dispatch overhead are all properties of the *module
graph*, not of the *weight dtype*. Any such mechanism must charge the same `b`
to both families. It charges 33× more to one.

Whatever this is, **it lives in the quantized execution path and is triggered by
the skip configuration** — not in the skip machinery itself.

The same arithmetic makes a falsifiable prediction: if quantized cost really
were linear in skip count with `b = 0.32`, then skip0 must measure
36 × 0.0986 = **3.55 ms**. Round 1's estimate for quantized skip0 is ~4.75 ms.
X1 measures it directly. If skip0 ≈ 4.75, the relationship is **not** linear and
the penalty is a **threshold between 4 and 8 skips**, not a per-skip charge.

## 4. Hypothesis space

Surviving hypotheses only. Each states the mechanism, what it predicts, and what
would falsify it.

### C — compilation and graph structure

- **C1 · Larger compiled pieces change inductor codegen.** With splitting on
  `unified_attention_with_output`, skipping a layer deletes its attention
  registration and merges two pieces. skip8 yields 29 pieces where skip4 yields
  33, so each piece is larger. A larger piece can cross an inductor fusion or
  register-pressure budget and fall back to worse codegen for the quantized
  matmul, whose epilogue (scale application) is the fusible part.
  *Predicts:* different generated kernel names or counts between arms; effect
  depends on where the skips sit (which pieces merge), not just how many.
  *Falsified by:* identical kernel mix in X2, or a position-independent result
  in X1.

### D — kernel selection and execution

- **D1 · The W4A16 GEMM is not taken at skip8.** The strongest single clue is
  that quantized per-layer cost does not merely degrade — it lands *on* the bf16
  line (0.1901 vs 0.1859). The simplest mechanism that produces exactly that is
  that the layers are executing a dequantize-then-bf16-GEMM path rather than the
  4-bit path, making them bf16-speed plus a little.
  *Predicts:* X2 shows bf16/cutlass GEMM kernels in the slow arm where the fast
  arm shows `machete_*`.
  *Falsified by:* identical kernel names and counts, differing only in duration.
- **D2 · Same kernel, different schedule.** Machete picks a tile/schedule by
  heuristic. If some global state (workspace size, available SMs, an autotune
  cache keyed by call count) differs, the same kernel could run a worse schedule.
  *Predicts:* same kernel names, longer per-call duration, in X2.

### E — memory placement

- **E1 · Freeing the skipped layers fragments the weight arena.**
  `_apply_skip_layers` replaces modules *after* `get_model`, so the skipped
  layers' weights are allocated and then dropped, leaving 8/36 × 6.1 GB ≈ 1.36 GB
  of holes interleaved among the surviving weights. The KV cache is then sized to
  fill what is free and occupies those holes. A 4-bit GEMM at M≈1 is almost
  purely weight-bandwidth-bound, so it is the configuration *most* sensitive to
  weight placement — and bf16, being less bandwidth-bound per useful byte, would
  show it less. This is the one hypothesis that naturally explains why the
  penalty is quant-specific.
  *Predicts:* the penalty tracks the number and adjacency of holes, so different
  8-layer sets differ; and it should move under a different allocator policy.
  *Falsified by:* X4 (expandable segments) and X5 (never allocate the skipped
  layers) leaving the penalty unchanged.
- **E2 · KV sizing feedback.** Fewer layers → more free memory → a larger KV
  cache (353,328 vs 356,576 tokens) → different page tables and workspace.
  *Predicts:* pinning block count equalizes the arms.
  *Falsified by:* X3.

### A — measurement (must be closed out, not assumed)

- **A1 · Different forwards per chain.** If acceptance differs, the number of
  `draft_forward` calls per chain differs and a per-call mean is comparing
  different things. Partly checked (n/n_first ≈ 3.31 in both), but it must be
  reported per arm, not assumed. **CLOSED by X1**: acceptance varies 2.4x across
  the 8-skip arms (149 to 358 armed steps for the same token budget) while
  forwards-per-chain holds at 3.12–3.36 and forward time does not move with it.

### H — host-side cost (opened by X3, the only family still standing)

- **H1 · The per-piece host cost grows.** X3 localises the penalty to ~90 extra
  50–200 µs host gaps per step against ~116 piecewise replays, with 95% of the
  idle outside any CUDA call. Something the host does between replays costs
  2–4x more at skip8 than at skip4.
  *Predicts:* a CPU profile shows a specific frame growing; the growth is
  quant-specific (bf16 skip8 does not show it).
  *Open sub-hypotheses:* larger per-piece graphs raising `cudaGraphLaunch`
  argument-marshalling cost; per-piece Python dispatch over a module list whose
  passthroughs are not traced away; attention-metadata rebuild between pieces.
  *Falsified by:* a flat CPU profile, which would push the cause back to the
  driver.

## 5. Discriminating experiments

Ordered by information per GPU-minute.

- **X1 · Skip-set sweep — COMPLETE. Verdict: COUNT, not position.**
  Seven arms at window 256, R1, `draft_forward`:

  | arm | layers | forward | ms/layer | armed steps |
  | --- | --- | --- | --- | --- |
  | quant skip0 | 36 | 4.748 | 0.1319 | 139 |
  | quant skip4 `{2,4,7,16}` | 32 | 4.316 | 0.1349 | 138 |
  | bf16 skip4 | 32 | 5.898 | 0.1843 | 131 |
  | quant skip8 head `{1..8}` | 28 | 5.737 | 0.2049 | 263 |
  | quant skip8 tail `{28..35}` | 28 | 5.768 | 0.2060 | 358 |
  | quant skip8 spread `{4,8,…,32}` | 28 | 5.824 | 0.2081 | 150 |
  | quant skip8 declared | 28 | 6.051 | 0.2161 | 149 |

  All four 8-layer sets land together: spread 0.31 ms (5%) against a count
  effect of +1.4 ms (33%). Contiguous-head, contiguous-tail and scattered are
  indistinguishable at the scale of the defect.

  Three consequences:

  1. **C1 and hole-adjacency are eliminated.** Both are positional. Merging
     pieces at the head, at the tail, or scattered through the model would not
     produce the same penalty. Only hole *volume* (8/36 of the weights, wherever
     they were) or a count threshold survives.
  2. **Linearity is falsified.** The two-point fit demanded skip0 = 3.55 ms; it
     measures **4.748 ms**. The 1st–4th skipped layers save ~0.11 ms each; the
     5th–8th *cost* ~0.25 ms each. This is a threshold between 4 and 8, not a
     per-skip charge.
  3. **A1 is closed.** Acceptance varies enormously across these arms (149 to
     358 armed steps for the same token budget, since skipping layers 1–8 or
     28–35 wrecks draft quality) while forwards-per-chain holds at 3.12–3.36 and
     the forward time moves not at all with it. The anomaly is not an artifact
     of counting different work.

  And past the threshold quantization does not merely stop paying — it costs:
  6.051 ms for 28 quantized layers against **5.205 ms for the same 28 layers in
  bf16**.
- **X2 · Per-kernel profile — COMPLETE. Verdict: nothing is slower. The GPU is
  idle.** 25 steps, window 256, R1.

  | | machete calls | per call | cutlass GEMM/call | device/step |
  | --- | --- | --- | --- | --- |
  | quant skip4 | 22,400 | 12.225 µs | 19.741 µs | 27.61 ms |
  | quant skip8 | 19,600 | 12.362 µs | 20.036 µs | **25.79 ms** |

  The W4A16 path **is** taken at skip8, runs the **same** kernels at the **same**
  per-call speed, and does **less** total device work — exactly as fewer layers
  should. **D1 and D2 are eliminated.** E1 is eliminated with them: worse memory
  placement would show as slower kernels, and the kernels are not slower.

  Host CUDA API calls per step go *down* too, and match the bf16 arm exactly:

  | per step | skip0 | skip4 | skip8 | bf16 skip8 |
  | --- | --- | --- | --- | --- |
  | `cudaGraphLaunch` | 149 | 133 | 117 | 117 |
  | `cudaLaunchKernel` | 449 | 417 | 385 | 385 |
  | `cudaLaunchKernelExC` | 288 | 256 | 224 | 224 |

  Graphs are being replayed, and skip8 launches less than skip4.

  **Measurement correction — twice, and the second retracts a number I gave.**
  Busy/idle must not mix a region-profiled wall with an unprofiled device total:
  the region profiler syncs ~10x per step and drains the very overlap being
  measured. Nor can wall be read from inside a `torch.profiler` context — CPU
  activity tracing inflated it to 171 ms/step for an arm whose true step is
  ~29 ms.

  I then reported ~95% vs ~76% occupancy by pairing X2's `device_us_total` with
  v4's profiler-free wall. **That was also invalid** and is retracted:
  `self_device_time_total` sums per-kernel time across all streams, so it
  exceeds wall wherever compute and memcpy streams overlap — for bf16 skip8 it
  gives 32.52 ms of "device" against a 31.61 ms step, which is impossible as an
  occupancy. Device totals are sound for comparing **kernel mix and per-call
  duration** (their purpose here) and unsound as a numerator over wall.

  Occupancy requires one timeline measured by one instrument: that is X3.

  **This reframes the question.** Nothing got slower and nothing got busier. The
  host does less work, the device does less work, and the wall time still rises.
  Less host work with an idle GPU means the host is **blocking**, not lagging —
  so the remaining question is *what it blocks on*, which is X3.
- **X3 · Timeline gap attribution — COMPLETE. The whole penalty is GPU idle.**
  One chrome timeline per arm, 6 steps, same instrument, gaps >= 20 us
  attributed to the host call spanning them.

  | per step | span | busy | idle | occupancy |
  | --- | --- | --- | --- | --- |
  | quant skip4 | 43.47 ms | 22.76 | 20.70 | 52.4% |
  | quant skip8 | 47.08 ms | 21.04 | 26.04 | 44.7% |

  The decomposition is exact and internally consistent:

  ```text
  delta span = +3.62 ms  =  busy -1.72  +  idle +5.34
  ```

  The arm does **less** GPU work and still costs more, and the arithmetic says
  the cost is idle and nothing else. It matches the profiler-free wall delta
  (+4.94 ms/step) in sign and scale.

  **Where the idle is: in the host, outside CUDA.** 95.3% of gap-idle at skip8
  (123.7 of 129.8 ms) falls under *no host CUDA call at all* — the host is not
  blocked in `cudaMemcpy`, `cudaStreamSynchronize` or a graph launch, it is
  executing framework/Python code between launches. `cudaMemcpyAsync`,
  `cudaStreamSynchronize` and `cudaEventQuery` together account for ~1 ms.

  **The signature is a shift in gap size, not gap count:**

  | gap bucket | skip4 | skip8 |
  | --- | --- | --- |
  | 20–50 µs | 1301 | 975 |
  | **50–200 µs** | **347** | **884** |
  | 200 µs–1 ms | 45 | 45 |
  | >1 ms | 1 | 0 |

  skip8 has *fewer* small gaps (fewer layers, as expected) and 2.5x more medium
  ones. Roughly 90 extra 50–200 µs host gaps per step against ~116 piecewise
  graph replays per step — so the per-replay host cost roughly triples.

  Graph launches confirm the structure: 29 pieces x 4 forwards = 116 ≈ the 117
  `cudaGraphLaunch`/step measured, and 33 x 4 = 132 ≈ 133 at skip4. The draft
  chain **is** running piecewise cudagraph replay; the host runs Python between
  every piece, and that inter-piece cost is what grows.

- **X4 · Pin KV blocks.** Re-run the fast and slow arms with an identical
  `num_gpu_blocks_override`. *Closes E2.* Cheap; 2 boots.
- **X5 · Allocator policy.** Re-run the slow arm under
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. *Probes E1* without any
  code change. 1 boot.
- **X6 · Load-time skip.** Construct the draft so skipped layers are never
  allocated, instead of allocated-then-replaced, and re-run. *Decides E1
  directly* — it removes the holes rather than working around them. Requires a
  change to `_apply_skip_layers`; only worth doing if X4 moves the number.
- **X7 · bf16 at the alternative sets.** Whichever sets X1 shows to be fast and
  slow, run them in bf16 too. *Confirms the position effect is quant-specific*
  rather than a property of those particular layers.

## 6. Decision rule

The investigation closes when a single mechanism predicts all of: the 0.886 ms
sign flip, the landing of quantized per-layer cost on the bf16 line, the
flatness in context and window, and whatever pattern X1 returns across the four
8-layer sets. A mechanism that explains the magnitude but not the sign flip, or
the sign flip but not the bf16 coincidence, is not the mechanism.

Nothing here is scored and none of it touches the frozen Round-1 record. The
Round-1 finding stands as written: levers compose additively within ±7% on 13 of
15 configurations, and this one interaction is the exception.
