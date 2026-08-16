# Phase 99 — KV pressure: when the lever set stops being affordable

**Source phase:** 98 (`research/98_selector_demo/`), specifically
`results_switching_law.md` and `results_d3_failclosed.md`.

**Status (2026-08-16):** design + preflight. Nothing preregistered yet.

---

## Why this phase exists

Phase 98 measured per-regime switching at **+6.2%** over the best static
once the selector could decline to arm, and derived an exact law for why it
is not more. Under time-weighted aggregation,

```
gain = sum_R  t_R * ( rate_sel(R) / rate_static(R) )
```

with `t_R` the selector's **time** share. Switching therefore requires a
regime that is simultaneously **time-dominant** and **optimum-distinctive**.
Phase 98's grid has each property separately and never together: R1 carries
59% of the time budget and contributes +0.14%, because at batch 1 every
quantized configuration performs within 0.3% of every other.

That ceiling is a property of the six regimes measured, not of the design —
the fail-closed selector already captures 93% of the switching value the
grid contains. To raise it, the regimes have to differ in a way this grid
cannot express.

**The axis that supplies both conditions is feasibility, not preference.**
The `w4a16-quantized` draft is a separate resident, so at a large enough KV
working set it is not deployable at all. No single static configuration can
then serve a mix spanning both sides of that line, and switching stops being
a marginal ranking improvement.

Phase 98 sized this at **1.305x** by applying a modelled feasibility mask to
measured rates. This phase measures it.

## What the preflight already changed

The obvious design — hold batch 8 and sweep output length — **measures
nothing**, and the preflight caught it before anything was registered.

`scripts/preflight_w99_kv_budget.py` boots the engine on the campaign lane
and reads vLLM's own KV accounting rather than a derived estimate
(`data/preflight_kv_budget.json`):

| | KV tokens available |
| --- | --- |
| target only, no separate draft | **400,400** (25,025 blocks x 16) |
| with the w4a16 draft resident | **356,304** (22,269 blocks x 16) |
| **the draft's price** | **44,096 KV tokens ≈ 6.5 GB** |

Closed-form arithmetic predicted 396.2k and 354.8k, so the model of the
budget is sound to ~1%. What it implies:

| batch | max seq len, no draft | max seq len, with draft | reachable within the 40960 context limit? |
| --- | --- | --- | --- |
| 8 | 50,050 | 44,538 | **NO** — the crossover is beyond the model's context limit |
| 16 | 25,025 | 22,269 | yes |
| 32 | 12,512 | 11,134 | yes |
| 64 | 6,256 | 5,567 | yes |

**At batch 8 the crossover is unreachable.** A batch-8 sweep, however long
its outputs, never puts the draft under pressure, and would have produced a
null result that looked like a refutation of the hypothesis rather than a
mis-designed campaign. The phase runs at **batch >= 16**.

### The straddle cell runs — measured, not assumed

```
batch 16 x (14336 input + 8192 output) = 360,448 KV tokens
    vs the with-draft capacity          357,120  -> 100.9%, over the line
    vs the no-draft   capacity          400,400  ->  90.0%, fits
```

Run with the draft resident: **16/16 sequences completed, 131,072 tokens in
188.8 s (694 tok/s wall), `equal_work_ok` true** — every sequence emitted
exactly 8192 tokens under `ignore_eos`. So the cell degrades rather than
thrashing, and the campaign's central risk is retired.

(That run drove plain `speculative_config` at K=4, not the phase's K/OFF
policy, so 694 tok/s is a feasibility check and not a scored rate.)

### The effect is lost concurrency, not a crash

vLLM does not refuse to boot when the working set exceeds KV capacity — it
admits fewer sequences concurrently and queues the rest. So the draft's
44,096 tokens do not manifest as an OOM but as **~11% of the KV budget, and
therefore of the achievable batch, spent on draft weights instead of
requests**. That is a more realistic and more damaging failure than a crash,
and it is what the campaign measures.

### What the measurement then corrected: the window is narrow

The feasible-without / infeasible-with band is only **12.1% wide**
(357,120 -> 400,400). The first draft of this matrix put its two "draft
infeasible" cells at 360,448 — **0.9% past the line**, a crossing so thin it
would have measured scheduler noise rather than lever economics. Cells have
to sit near the TOP of the window to apply real pressure while still fitting
without the draft.

Two independent boots reported 356,304 and 357,120 KV tokens (0.23% apart),
so the campaign takes the **lower** figure as the line and re-reads capacity
per boot rather than assuming it.

## Objective

Establish, by measurement rather than by mask, that per-regime selection
beats the best uniformly-feasible static when the mix spans the feasibility
line — and determine where that line falls.

## Assumptions

1. Qwen3-8B dense, bf16 target, `w4a16-quantized` draft as a separate
   resident (~6.5 GB measured).
2. One H100 80 GiB, `gpu_memory_utilization = 0.90`, h104 GPU 7 / NUMA 1.
3. `max_model_len` must rise from Phase 98's 20480 to **24576** to admit the
   straddle cell. This is a run-contract change and is recorded as one.
4. Decode currency only, equal work (`ignore_eos`, fixed budget), as in
   Phase 98. Phase 96 retracted a claim to mixing decode with wall clock;
   Phase 98's asymmetric-drain defect is why `ignore_eos` is not optional.

## Design

### Axis A — the feasibility line (primary)

Sweep the KV working set across the measured crossover by varying batch and
total sequence length, at fixed content:

A **dose-response ladder** across the line, stated as a fraction of the
with-draft capacity (357,120) so the pressure is explicit:

| cell | batch | input | output | total KV | % of draft cap | % of free cap | side |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | 16 | 14336 | 2048 | 262,144 | 73.4% | 65.5% | both feasible |
| P2 | 16 | 14336 | 4096 | 294,912 | 82.6% | 73.7% | both feasible |
| P3 | 16 | 14336 | 6144 | 327,680 | 91.8% | 81.8% | both, near |
| P4 | 16 | 14336 | 8192 | 360,448 | **100.9%** | 90.0% | just over — the marginal control |
| **P5** | **16** | **14336** | **10240** | **393,216** | **110.1%** | **98.2%** | **draft infeasible** |
| **P6** | **32** | **8192** | **4096** | **393,216** | **110.1%** | **98.2%** | **draft infeasible** |

P4 is retained deliberately as the **marginal control**: if a 0.9% crossing
produces a 0.9%-scale effect and a 10% crossing produces a large one, the
mechanism is KV pressure. If P4 already shows a large effect, something other
than capacity is driving it and the campaign has found a confound.

P5 and P6 carry **identical total KV at different (batch, length) shapes** —
P5 at the `max_model_len` ceiling, P6 with sequence length less than half of
it. Agreement between them attributes the effect to pressure; disagreement
attributes it to batch, which is a different claim.

Each cell runs the Phase 98 lattice restricted to what fits: all 30 composed
configurations plus OFF where the draft is affordable, and the 15
`target-matching` configurations plus OFF where it is not.

### Axis B — within-request drift (secondary)

Phase 98's u-axis instrument carries bucket edges at **[256, 1024, 3072]**
and has never been fed: every campaign ran a 640-token budget, leaving the
top two buckets permanently empty. Within the 640 tokens that were measured,
acceptance moves with position (R5 +16.3%, R6 +11.4% from bucket 0 to 1) but
the configuration **ranking does not** — Spearman +0.82 to +0.97, and the
argmax is identical in both buckets in all six regimes.

So no within-request switching opportunity is visible in the measured span,
and the outputs here are long enough to populate the instrument's full
range. Axis B scores acceptance and rate **per position window** rather than
per request, and asks whether the argmax moves as a request generates.

This is the axis that separates *compiled per-regime policy* from *runtime
switching*: a request that starts with room for the draft can lose it as its
KV grows, which no per-regime selector and no static configuration can
express.

## Decision criteria

Registered before the first scored boot, with a commitment barrier and
digest per the Phase 98 protocol:

* **W99-1 (primary).** On a mix spanning P2 and P5, per-regime selection
  beats the best **uniformly-feasible** static by **>= 1.20x** in decode
  currency. Phase 98's modelled estimate was 1.305x.
* **W99-2.** The measured feasibility crossover falls within **+/-10%** of
  356,304 KV tokens, i.e. the budget model transfers from the preflight to
  the campaign.
* **W99-4 (dose response).** The selector's advantage over the best
  uniformly-feasible static is **monotone non-decreasing** across
  P1 -> P2 -> P3 -> P4 -> P5, and P5 and P6 agree within **5%**. This is the
  control that separates KV pressure from batch and from scheduler noise.
* **W99-3 (secondary, exploratory).** The per-regime argmax changes at least
  once **within** a request at 14336 input / 8192 output, scored per
  position window.

W99-3 is registered as exploratory: Phase 98's evidence points the other way
within its measured span, and a null result is informative rather than a
failure.

## Commands

```bash
# Preflight (done): KV budget and the straddle cell
.venv/bin/python research/99_kv_pressure/scripts/preflight_w99_kv_budget.py \
    --largest-cell
```

Campaign runners are not written yet; they follow Phase 98's create-only,
resumable, gate-per-boot structure and reuse its lattice and instrument
arms.

## Expected next artifact

`design_w99_campaign.md` with the preregistered matrix and digest, then
`results_w99_feasibility.md`.

## Risks

* ~~The straddle cell may thrash rather than degrade.~~ **RETIRED by the
  preflight**: 16/16 sequences completed at exactly 8192 tokens each, 694
  tok/s wall, no preemption pathology.
* **The window is narrow (12.1%), so cell placement is delicate.** P5/P6 sit
  at 98.2% of the no-draft capacity — if the campaign's actual boots report
  slightly less capacity than the preflight, those cells stop fitting on the
  feasible side too and measure nothing. Capacity is therefore re-read per
  boot and a cell that does not fit is recorded as such rather than scored.
* **Raising `max_model_len` to 24576 changes the run contract**, so Phase
  98's cost fits do not transfer directly and the phase carries its own
  anchors. P5 sits exactly at that ceiling with no headroom; P6 is its
  same-pressure alternative at less than half the sequence length.
* **One box, one model** — unchanged from Phase 98 and not addressed here.
