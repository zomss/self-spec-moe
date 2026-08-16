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

### The straddle cell

```
batch 16 x (14336 input + 8192 output) = 360,448 KV tokens
    exceeds the with-draft capacity     (356,304)
    fits    the no-draft   capacity     (400,400)
```

This is the design point: a cell that is feasible for `target-matching`
levers and infeasible for the quantized draft, on the same box at the same
memory setting.

### The effect is lost concurrency, not a crash

vLLM does not refuse to boot when the working set exceeds KV capacity — it
admits fewer sequences concurrently and queues the rest. So the draft's
44,096 tokens do not manifest as an OOM but as **~11% of the KV budget, and
therefore of the achievable batch, spent on draft weights instead of
requests**. That is a more realistic and more damaging failure than a crash,
and it is what the campaign measures.

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

| cell | batch | input | output | total KV tokens | side of the line |
| --- | --- | --- | --- | --- | --- |
| P1 | 16 | 14336 | 2048 | 261,888 | both feasible |
| P2 | 16 | 14336 | 4096 | 294,912 | both feasible |
| P3 | 16 | 14336 | 6144 | 327,680 | both feasible, near |
| **P4** | **16** | **14336** | **8192** | **360,448** | **draft infeasible** |
| P5 | 32 | 8192 | 2048 | 327,680 | both feasible, near |
| **P6** | **32** | **8192** | **3072** | **360,448** | **draft infeasible** |

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

* **W99-1 (primary).** On a mix spanning P2 and P4, per-regime selection
  beats the best **uniformly-feasible** static by **>= 1.20x** in decode
  currency. Phase 98's modelled estimate was 1.305x.
* **W99-2.** The measured feasibility crossover falls within **+/-10%** of
  356,304 KV tokens, i.e. the budget model transfers from the preflight to
  the campaign.
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

* **The straddle cell may thrash rather than degrade.** If vLLM preempts
  aggressively at 360,448 tokens the measured rate may reflect scheduler
  behaviour more than lever economics. The preflight's largest-cell run is
  the check; if it thrashes, the cell moves down to a smaller excess.
* **Raising `max_model_len` to 24576 changes the run contract**, so Phase
  98's cost fits do not transfer directly and the phase carries its own
  anchors.
* **One box, one model** — unchanged from Phase 98 and not addressed here.
