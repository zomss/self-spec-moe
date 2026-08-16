# The stack is not free when it declines to speculate: +3.7%

Record of 2026-08-16, h104 GPU 7 / NUMA 1. Probe:
`scripts/probe_w98_offpath_vs_stock.py` (X32); record:
`data/probe_offpath_stock/summary.json`.

The question a reviewer asks first: *you added a speculative-decoding runtime
to vLLM; what does it cost me when it decides not to speculate?* If the answer
is anything but zero, every OFF cell in this phase is measured against a
baseline our own stack degraded -- including the fail-closed rule, whose entire
value is choosing that baseline.

## Method

Two arms, interleaved and repeated:

* **`stock`** -- vLLM with no `speculative_config` and every `VLLM_SELF_SPEC_*`
  variable stripped from the environment. Plain autoregressive decode.
* **`koff_off`** -- our engine, `speculative_config` loaded, K/OFF policy
  pinned to K=0. The draft is `target-matching`, so weights are shared and the
  comparison is compute, not memory.

**Both arms run with the profiler and trace DISABLED.** `region()` syncs at
both ends and costs +2.4 ms/step ungated (X25); leaving it on would measure
the instrument rather than the stack, and stock vLLM has no trace to read.

That forces wall clock, which includes prefill. So each arm is timed at two
token budgets (64 and 576) and the per-token decode cost is taken as the
slope, `per_token = (t_hi - t_lo) / (N_hi - N_lo)`, which cancels prefill
exactly. The implied intercepts came out physically sensible in every cell
(R1 2.64 ms for one ~111-token prompt; R6 17.45 ms for 32 of them), which is
the check that the fit is not fitting noise.

## Result

| regime | batch | stock tok/s | koff-OFF tok/s | ratio | **our cost** |
| --- | --- | --- | --- | --- | --- |
| R1 | 1 | 141.60 | 136.58 | 0.9645 | **+3.68%** |
| R8 | 16 | 2058.30 | 1991.21 | 0.9674 | **+3.37%** |
| R6 | 32 | 3856.05 | 3704.92 | 0.9608 | **+4.08%** |
| | | | | | **mean +3.71%** |

**Our runtime costs 3.4-4.1% even with K=0.** In absolute terms that is
0.23-0.32 ms per decode step: the engine still walks the speculative path --
policy evaluation, `spec_decode_metadata` construction, the spec-shaped
model-runner branch -- after deciding not to speculate.

The effect is well clear of noise. Identical stock repeats differ by
0.37-1.02%; the two `koff_off` repeats differ by 0.11-0.15%, tighter than
stock's, and the sign is the same in all three regimes at three different
batch sizes.

## What this changes

**Every speedup in this phase is a ratio to OUR OFF, not to stock vLLM.** The
armed path carries the same overhead, so the correction is a single
multiplicative factor and changes no ranking, no lever comparison and no
verdict:

> D3's **1.436x** over static-OFF is approximately **1.383x** over stock vLLM
> autoregressive decode (x0.9629).

The paper should either report against stock AR or state explicitly which
baseline it uses. Reporting "1.44x over no speculation" while the no-
speculation arm is our own degraded path would be the kind of comparison this
phase has retracted others for.

It also partly explains a discrepancy noticed while running this: the stock
rates here run ~12% above the D3 grid's OFF rates (141.6 vs 125.3 at R1). Of
that, ~3.7 points is this effect; the rest is the profiler's syncs (D3 ran
with it on) and the h103/h104 box difference.

## Is it fixable?

The overhead is not intrinsic to speculation -- it is paid on steps where no
speculation happens. A genuine K=0 fast path that bypasses the spec branch
entirely (no `spec_decode_metadata`, no rejection-sampler call, the stock
model-runner path) should recover most of it. The nsys account supports the
scale: `cpu_exec_prepare_inputs` measures 0.411 ms/step in our OFF arm, and
the overhead here is 0.23-0.32 ms/step.

That is a worthwhile engineering item **precisely because it is not a lever
trade-off**: it costs acceptance nothing, it applies to every OFF step in
every regime, and it is exactly the regime the fail-closed rule steers into.
At R4 the selector now declines to arm -- and lands on a path 3.7% slower than
the stock AR it is trying to fall back to.

## Limitations

* Two repeats per arm, three regimes, one box, one model.
* Wall clock, not decode currency -- unavoidable, since stock vLLM emits no
  trace. The two-budget slope removes prefill but not scheduler-level effects
  that scale with token count.
* `target-matching` only. A quantized draft is a separate resident, so its OFF
  path also pays memory it is not using; that is not measured here.
