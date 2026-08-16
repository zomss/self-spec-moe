# Where each lever's time actually goes: an Nsight per-operation account

Record of 2026-08-16, h103 GPU 7 / NUMA 1, regime R1 (batch 1, short context).
Probe: `scripts/probe_w98_nsys_levers.py` (X31); analysis:
`scripts/analyze_w98_nsys_accounting.py`; records under
`data/probe_nsys_levers/`. DIAGNOSTIC, NOT SCORED.

Every cost number in this phase until now was an aggregate -- `d_hat`, the
fitted draft-chain mean. Aggregates say a lever is slower than it should be
but never where the time went, which is why the window's ~1% short-context
overhead took a fitted model, a wall-clock A/B and a no-window control to
pin down. This measures it directly.

## Two instruments

* **NVTX ranges** on every profiler region (`VLLM_SELF_SPEC_PROFILE_NVTX=1`).
  Unlike the wall-clock timers these need no `cuda.synchronize()`, so the
  fine-grained sub-regions that had to be gated -- costing +2.4 ms/step of
  lost overlap (X25) -- become measurable.
* **`nsys stats` kernel summaries at `--cuda-graph-trace=node`**, which split
  the GPU time inside those ranges into attention / GEMM / elementwise.

Two methodology bugs were found and fixed before any of this was believed;
both are recorded in section "What nearly went wrong" because each produced a
plausible false finding.

## Calibration: is nsys distorting this?

| | wall clock (X30) | nsys |
| --- | --- | --- |
| `draft_chain`, base, R1 | 27.281 ms | **27.750 ms** (+1.7%) |

Per-node tracing inflates by under 2% here, so absolutes are usable and not
just ratios. Two internal checks also pass: `base` total kernel time is
**5.13x** the OFF arm's, matching 4 draft forwards + 1 verify against 1; and
~97% of the NVTX span is kernel execution, so the GPU is saturated rather than
idle.

## The economics, per operation

| range | ms |
| --- | --- |
| `verify` (target forward, K+1 = 5 tokens) | 5.984 |
| `draft_forward` (1 token) | **6.237** |
| `draft_chain` (K=4, four forwards + orchestration) | 27.750 |

**A draft forward costs more than a verify forward.** At batch 1 both are
weight-bound, so verifying five tokens costs no more than one. The K=4 chain
therefore buys five tokens' worth of verify for four forwards' worth of draft,
which is why an unmodified draft measures 0.891x against no speculation. That
is not a subtle effect; it is the arithmetic, and every lever exists to cut
that 6.237 ms.

## Why quantization dominates: what the draft is made of

Draft-only GPU kernel time (armed minus OFF, so the unchanged verify is
subtracted out):

| class | ms/step | share of draft |
| --- | --- | --- |
| **GEMM** | 22.092 | **83.6%** |
| attention | 2.879 | 10.9% |
| elementwise | 0.956 | 3.6% |
| other | 0.504 | 1.9% |

This one table sets the lever hierarchy before any implementation question
arises. **Quantization attacks 83.6% of the draft; the window attacks 10.9%.**
A perfect window at short context is capped at an 11% saving, and a perfect
skip-4 at 11%. No implementation quality changes those ceilings.

## Where each lever's promise goes

* **bound** -- what the lever could give if its kernels hit the physical limit
  (layers removed, KV not read, weight bytes not loaded);
* **kernels** -- what the kernels actually delivered;
* **model** -- what the draft forward actually cost.

So `kernel gap = kernels - bound` is kernel efficiency, and
`fixed gap = model - kernels` is per-forward cost that does not scale with the
work removed.

| combo | bound | kernels | model | kernel gap | fixed gap |
| --- | --- | --- | --- | --- | --- |
| **quant** | 0.3731 | 0.5945 | 0.6383 | **+0.2214** | +0.0438 |
| skip4 | 0.8889 | 0.8987 | 0.8905 | **+0.0098** | -0.0082 |
| skip8 | 0.7778 | 0.7953 | 0.7809 | **+0.0175** | -0.0144 |
| w1024 | 1.0000 | 0.9998 | 0.9999 | -0.0002 | +0.0001 |
| w128 | 0.9217 | 1.0069 | 1.0521 | +0.0852 | +0.0452 |
| composed (`w4a16/w512/skip4`) | 0.3317 | 0.5332 | 0.5887 | +0.2015 | +0.0556 |

### cost composes multiplicatively, to 0.2%

The composed cell is the check that the parts add up, and they do:

| | predicted product of singles | measured | error |
| --- | --- | --- | --- |
| **kernel time** | 0.5945 x 0.8987 x 0.9998 = **0.5342** | **0.5332** | **-0.18%** |
| model time | 0.6383 x 0.8905 x 0.9999 = 0.5683 | 0.5887 | +3.58% |

**GPU kernel time under composition is the product of the singles to within
0.2%.** That is the cost half of the phase's epistemic split, confirmed at
kernel granularity rather than inferred from a fit -- and it stands in exact
contrast to acceptance, where D2(a) measured composition to be *constructive*
(interaction ratio above 1 for every lever set, 1.670 at R4) and D2(b) found
the product's ranking inverting at k=16.

The model-time residual, +3.6%, is the fixed per-forward cost that no lever
scales: multiply three levers together and the orchestration each one failed
to shrink is still there once.

### skip is clean; its ceiling is arithmetic

skip4 and skip8 land within **1.0% and 1.8%** of their bounds, and every class
scales together -- at skip4, attention 0.897, GEMM 0.899, elementwise 0.906,
other 0.896 against the 0.889 that removing 4 of 36 layers implies. There is
nothing to fix in the layer-skip path. This confirms by a fully independent
route what `results_lever_mechanics.md` measured in wall clock (0.894-0.902
against an 0.889 ideal).

### quantization carries the largest recoverable overhead, and it is in the kernel

**+22.1 points**, by far the biggest gap in the table. Marlin delivers
**0.548x on GEMM against a 0.25x weight-bytes bound** -- 1.8x, not 4x. Had the
kernel reached its bound the draft forward would cost 0.373x instead of
0.638x. At batch 1 the GEMM is tiny-M and latency-bound rather than
bandwidth-bound, which is the likely cause; the 4x bound assumes a purely
bandwidth-bound decode and is optimistic by construction. Either way this is
the single largest cost-elimination target in the stack, and it is an order of
magnitude larger than the window issue that was just fixed.

Attention under quant reads 1.033x -- unchanged, as it must be. That is the
control that makes the attribution trustworthy.

### the window: free where it cannot help, harmful where it half-can

`w1024` at R1 takes the fast path shipped in `results_window_overhead.md`
(1024 + 16 sinks exceeds the context), so its rewrite is **0.000 ms** and the
lever is free: kernel gap -0.0002, fixed gap +0.0001.

`w128` does **not** take it -- 144 < ~512, so there genuinely is something to
trim. It pays **0.480 ms/chain** of rewrite and buys an attention reduction of
**0.6%**, ending 5.2% slower than no window at all. Two facts compound: at
batch 1 attention is launch-latency-bound rather than KV-bound, so shrinking
the KV read barely moves it, and attention is only 10.9% of the draft anyway.

This also bounds the fast path: it removes the rewrite exactly when the
rewrite is *provably* useless. For w128 the rewrite is doing real work and it
is the payoff that is absent, which is a lever-selection problem rather than
an implementation one -- and D3's selector already never picks w128 at short
context.

## The floor no lever touches

Orchestration (`step0_*`, `step_*`, `chain_setup`, `cpu_*`) is ~3.2 ms/chain
regardless of lever, and its SHARE therefore rises as levers work:

| combo | chain ms | orchestration ms | share |
| --- | --- | --- | --- |
| base | 27.750 | 3.193 | 11.5% |
| skip8 | 22.292 | 3.234 | 14.5% |
| quant | 19.150 | 3.252 | 17.0% |
| **composed** | 17.983 | 3.306 | **18.4%** |

The better the lever set, the more this dominates -- nearly a fifth of the
composed chain. At ~0.8 ms per draft forward it is the second-largest target
after the Marlin gap, and unlike the levers it is pure overhead: it buys no
acceptance. It is also the whole of the +3.6% composition residual above.

## What nearly went wrong

Two methodology bugs, both of which produced a plausible false finding:

1. **nsys graph granularity.** The default `--cuda-graph-trace=graph` reports
   each CUDA-graph replay as one entry, collapsing a 36-layer forward into a
   single row. The first sweep showed 200 GEMM instances for 200 forwards --
   arithmetically impossible for a 36-layer model, which is what gave it away
   -- and **no quantized kernel at all** in the w4a16 arm. Read naively that
   says the quantization lever does not quantize. It also inflated the idle
   fraction enough to suggest the batch-1 chain was ~80% GPU-idle, when it is
   ~97% busy. Fixed with `--cuda-graph-trace=node`; the collapsed traces are
   kept under `data/probe_nsys_levers_graphcollapsed/`.
2. **A double-counted expectation.** The skip bound applied the layer ratio to
   both the per-class scales and a separate model scale, squaring it: skip4's
   bound read 0.796 instead of 0.889, making a clean implementation look 10%
   inefficient.

A third, smaller: `cublasLt::splitKreduce_kernel` is a GEMM epilogue and was
landing in `elementwise` on a "reduce" match. Marlin does not use split-K, so
the mis-bucketing made the quantized draft look as though its elementwise work
had vanished (0.10x). Corrected, and the analyzer now recomputes classes from
raw rows so the fix is retroactive.

## Limitations

* **One run per combination.** Differences under ~2% are not reliable; the
  effects called out here (+22.1, +8.5, and skip's +1.0/+1.8 against their
  bounds) are well clear of that, but the small ones are not.
* **R1 only so far.** The composition shares are batch-1 shares; at 14k the
  attention share is much larger and the window's bound is real rather than
  nominal, which is exactly where it pays -39.8%.
* **`unattributed` is noisy** (-0.14 to -0.84 ms/chain, sometimes negative):
  NVTX child projections can slightly exceed the parent span, so treat that
  column as +/-0.8 ms rather than a quantity.
* **The 0.25x GEMM bound is optimistic by construction** -- it assumes a
  purely bandwidth-bound decode. The kernel gap it produces is an upper bound
  on what better kernels could recover, not a defect count.
