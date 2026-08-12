# W4A16 decode kernel: what already exists, and is new code needed?

Date: 2026-08-12
Status: answered. **It is a kernel-priority change, not new code — for about
half the gap.** A specialised kernel is still worth ~3.4 ms/step after that.

## The question

X7/X8 left the W4A16 GEMM as the largest remaining draft-side cost. Before
writing a decode kernel, two cheaper possibilities had to be excluded:

1. Machete's `schedule` argument is exposed (`ops.machete_mm(..., schedule=)`)
   and vLLM's wrapper never passes one, so the C++ default heuristic picks.
2. Marlin is the kernel designed for small batch, supports our config
   (group_size 128, `uint4b8`), and sits 4th in the CUDA priority list behind
   Machete on a general default ordering.

## What vLLM already ships

`MPLinearKernel` registry, CUDA priority order:
`CutlassW4A8 → Machete → AllSpark → Marlin → Humming → Conch → Exllama →
TritonW4A16`. Machete is selected for us. AllSpark is excluded by dtype — it
supports only `uint8b128` (8-bit). Separately there are W4A8
(`CutlassW4A8`, `compressed_tensors_w4a8_{int,fp8}`), W8A8 (a different
`scaled_mm` path), W8A16, and W4A4 (mxfp4/nvfp4) schemes — different quantization
schemes, not alternative kernels for ours.

## Measurement, and why the first attempt failed

`benchmark_machete.py` at Qwen3-8B's real layer shapes could not resolve it. Its
`bench_fns` times a Python loop (`for fn in fns: fn()`), so every call carries
dispatch. Fitting time against weight bytes across the four shapes exposes the
floor:

| kernel | per-call overhead |
| --- | --- |
| machete | 12.58 µs |
| machete_best (swept) | 14.43 µs |
| marlin | 24.34 µs |
| torch.matmul | 4.41 µs |

Those intercepts exceed the entire bandwidth-bound ideal for three of the four
shapes (o_proj 2.58 µs, qkv 3.87, down 7.75), and Marlin's fitted slope implies
a bandwidth 5x above HBM — impossible, and the signature of a measurement that
is all overhead.

X9b therefore captures 32 calls into a CUDA graph and replays it, timed with
CUDA events, so no per-call host work is included. That is also the regime the
draft actually runs in, since X6 established the chain is captured.

## Result: Marlin wins at every shape and every M in range

Device µs per call, and % of that shape's own bandwidth bound:

| shape (bytes) | ideal | machete | marlin |
| --- | --- | --- | --- |
| qkv_proj (12.98 MB) | 3.87 | 14.57 (27%) | **9.99 (39%)** |
| o_proj (8.65 MB) | 2.58 | 14.09 (18%) | **7.92 (33%)** |
| gate_up_proj (51.90 MB) | 15.49 | 37.66 (41%) | **25.79 (60%)** |
| down_proj (25.95 MB) | 7.75 | 20.84 (37%) | **16.22 (48%)** |

Aggregated over 28 layers x 4 chain forwards:

| M | machete | marlin | ideal | saving | machete eff | marlin eff |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 9.76 ms | 6.71 | 3.33 | **3.05** | 34% | 50% |
| 4 | 9.72 | 6.89 | 3.33 | 2.83 | 34% | 48% |
| 8 | 9.79 | 6.88 | 3.33 | 2.91 | 34% | 48% |
| 16 | 9.80 | 7.42 | 3.33 | 2.38 | 34% | 45% |
| 32 | 9.47 | 8.58 | 3.33 | 0.90 | 35% | 39% |

Anchor: the graph benchmark puts machete at 9.76 ms/step against the in-engine
CUPTI measurement of 8.69 ms/step (+12%), so it is not being flattered by L2
residency.

**Hypothesis 1 is dead.** The schedule sweep did not help: `machete_best` was
within noise of the default, and at qkv/o_proj slightly worse. The default
heuristic is not the problem.

**Hypothesis 2 holds.** Marlin captures **47% of the available headroom for zero
new code** — a kernel-priority change for the draft. A specialised decode kernel
would be worth a further **3.38 ms/step** on top, which is still the largest
single remaining item.

## Correction to an earlier figure

This phase previously reported the W4A16 GEMM at "65% of bandwidth-bound, ~3.0
ms/step headroom". That derived weight bytes by dividing the whole 6.1 GB
checkpoint by layer count, but the checkpoint includes the **bf16 `lm_head` and
embeddings**, which are neither quantized nor touched per layer. Counting the
actual quantized projections gives 99.5 MB/layer → **2.79 GB active, not 4.744
GB**; 11.14 GB of traffic per step; ideal 3.33 ms against 8.69 measured =
**38%, with 5.36 ms/step of headroom**. The opportunity is ~1.8x larger than
first stated.

## At M<=32 the activation precision is irrelevant

A 4096-wide bf16 activation row is 8 KB against tens of MB of weights — 0.02% of
traffic. **W4A8 and W4A16 have identical decode cost.** A-precision pays only at
prefill or large-batch verify, where the work is compute-bound. So W4A8 is not a
route to this 3-5 ms: it would change the checkpoint and the draft's acceptance
rate (a new lever value needing full re-measurement) while buying nothing where
the draft is bottlenecked. For decode cost the quantization lever is purely
**weight bits**.

## Recommendation

1. Switch the draft's W4A16 kernel to Marlin (priority change, gated to the
   draft so target/serving defaults are untouched). Worth ~3.0 ms/step at
   M<=8, ~2.4 at M=16, ~0.9 at M=32.
2. Re-measure in-engine to confirm the microbenchmark transfers.
3. Only then consider a specialised decode kernel, against a measured Marlin
   baseline, for the remaining ~3.4 ms/step.
