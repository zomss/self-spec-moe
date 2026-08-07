# W14 — uncertainty-aware selector search (awaiting GO)

Source: W12's search specification, W13's failed profiler validation, and the
2026-08-08 design review. Nothing in this work order has started.

## Objective

Validate on dense Qwen3-8B the narrow claim needed by the two-round selector:

> At a matched execution state, normalized speculative-step cost is reusable
> across content, while acceptance must be measured on the workload.

Round 1 is not a ranker. It estimates cost with uncertainty and eliminates
only configurations that cannot pay even under favorable assumptions. Round
2 measures acceptance for every survivor and measures cost only where Round 1
could not certify transfer. Deployment chooses from one global,
resource-feasible resident pool using action-specific live evidence.

The design is allowed to succeed only on part of the surface. A cell with an
inaccurate cost model passes through to Round 2; it does not invalidate cells
where cost transfer is certified.

## Definitions and invariants

Let

- `x = (n_active, total_scheduled_kv, graph_bucket)` be the execution state;
- `g` be the workload/content regime;
- `u` be generated-suffix position or interval;
- `c = (quant, realization, window, skip-set, kv-quant)` be a composition;
- `a = (c, K)` be a complete configuration, with `OFF` always available;
- `T(x)` be the unperturbed AR decode-step time;
- `P_a(x)` be the complete unperturbed armed-step time, including all `K`
  draft forwards, verification, and serving overhead; and
- `tau_a(g, u)` be expected emitted tokens per armed step, obtained from the
  position-resolved acceptance counters.

Define the required-acceptance label

```text
q_a(x) = tau*_a(x) = P_a(x) / T(x)
S_dec(a, g, u, x)   = tau_a(g, u) / q_a(x)
```

For a homogeneous fixed-width run,

```text
T = 1 / rate_AR
P = tau / rate_spec
q = P / T = tau / S_dec
```

where the rates come from request decode time, not end-to-end wall clock.
This identity is valid only for equal-work measurements over a matched state
or state interval. An aggregate over different context trajectories is not a
matched-cell transfer test.

Round 1 emits a prediction interval `[q_lo, q_hi]`. Set the existing arming
rent `epsilon_arm = 0.015`. The most favorable possible speedup is

```text
S_max = (K + 1) / q_lo
```

Round 1 may eliminate `a` only when `S_max < 1 + epsilon_arm`, or when `a` is
resource-infeasible. A noisy estimate therefore retains candidates rather
than creating a false winner. Exact or top-1 cost ranking is never a binding
criterion.

After Round 2 supplies an acceptance interval `[tau_lo, tau_hi]`, the speedup
interval is `[tau_lo/q_hi, tau_hi/q_lo]`. The selector's epsilon-optimal
tie-set uses `epsilon_sel = 0.015`, matching the arming rent.

## The search in four stages

1. **Route measurement by leverage.** Analytic weight/KV shares determine
   where a lever is worth profiling. They prioritize measurement; they do not
   eliminate a configuration.
2. **Round 1 — measure reusable cost.** Measure `T` and `P`, fit them
   separately over execution state, derive `q = P/T`, and conservatively
   remove only impossible configurations.
3. **Round 2 — measure acceptance and choose a portfolio.** On real workload
   content, measure `tau(K, g, u)` for every survivor and measure `q` only
   for uncertified cells. Form speedup intervals, emit epsilon-optimal
   tie-sets, and choose one global resident pool under HBM and graph-capture
   budgets.
4. **Deployment — sequential selection.** Use the cell as a prior and
   generated-suffix position plus action-specific acceptance as evidence.
   Demote with a confidence test and re-promote through low-duty probes. The
   runtime does not need to identify the regime explicitly.

## Scope corrections carried from W0–W13

- **Decode currency is mandatory.** End-to-end `q = tau/S` contains prefill
  dilution and produced 8.4% mean / 17.7% maximum apparent transfer error in
  W13. All decision tests below use request decode time.
- **Measure latency; do not use a roofline as the predictor.** W8 found a
  2–15x roofline-to-realization gap. Byte shares remain useful only as a
  measurement-routing prior.
- **Fit raw latency, not `q`, as affine.** At fixed batch/capture regime,
  `T(Q)` and `P_a(Q)` may be approximately affine in total KV `Q`; their ratio
  is generally rational, not affine. A windowed draft component may be flat,
  but the complete armed step still contains context-dependent verification.
- **`K` is part of the action.** Acceptance at all `K <= KMAX` can be derived
  from one position-resolved stream, but cost must be measured or modeled for
  each selected `K` because verification and dispatch change with width.
- **No-window is a different realization today.** `{window=none} x {FULLCG}`
  is unavailable, so `w-off` is treated as a complete configuration, not as a
  clean intervention on window size.
- **Quant is boot-class in the current stack.** Extra resident weights price
  memory but do not create a runtime dispatch mechanism. W14 chooses one
  quant/realization boot class unless a separate multi-draft implementation is
  later justified.
- **Skip switching remains conditional.** W6's three winning skip identities
  motivate a portfolio test; they do not by themselves authorize G3.
- **The sync-bracketed profiler is diagnostic only.** Decision inputs come
  from unperturbed equal-work serving measurements. Region decomposition may
  explain a result but cannot decide it.
- **W14 is a dense validation.** Sparse expert routing can make execution
  cost depend on an additional coverage/routing state. No architecture-wide
  transfer claim follows until MLA/MoE receives its own matched-state test.

## Work items

### A — phase-local decode instrumentation

**Cost:** CPU, about 30 minutes; no GPU.

Create a phase-96 runner or adapter rather than mutating a prior phase's
runner. Snapshot `vllm:request_decode_time_seconds` sum/count around each
measurement interval, following `run_w6_calib.py`, and emit:

- raw histogram deltas and `dec_rate_req`;
- output-token, request, accepted-token, and draft-token counts;
- `K`, complete configuration id, realization, and graph bucket; and
- the observed `n_active`, total-scheduled-KV, and generated-suffix ranges.

The scorer must retain raw values and derive `T`, `P`, `q`, and their
uncertainty. Before any new run, score the banked W6 `dec_rate_req` data as a
non-binding plumbing check. W6 mixed natural and fixed-length protocols, so
it cannot decide transfer.

**Gate A:** unit tests over synthetic cumulative snapshots recover known
interval deltas exactly. On a live fixed-width sanity cell, the histogram
count delta equals the number of completed requests and
`sum(output_tokens - 1)` equals the decode-token numerator. Repeated snapshots
must be monotone and token accounting must close before B starts. This is an
instrumentation gate, not an independent validation of the algebraic
`q = tau/S_dec` identity.

### B — matched-cell cost-transfer pilot (the first GPU gate)

**Question:** Does `q_a(x)` transfer between content regimes when `x` and the
amount of work are actually matched?

**Matrix:**

| axis | values |
|---|---|
| model | Qwen3-8B dense |
| regimes | R5, R5cot |
| generated interval | first fixed 512 tokens for both regimes |
| batch | 1, 8 |
| configurations | `s-none/w512`, `s-none/w2048`, `s-none/w-off` |
| depth | K=4 pilot |

The prompt lengths are nearly matched and both regimes traverse the same
512-token suffix interval. `w-off` is labeled as a distinct realization.
The six transfer units are `3 configurations x 2 batches`; each compares R5
with R5cot. Each regime is then used as calibration for the other, producing
12 directional configuration predictions and four selector evaluations.
The scored total-KV ranges must differ by at most 1%; otherwise that unit is
invalid rather than a transfer failure.

**Protocol:** fixed length, notune, two disjoint content seeds, `ITERS=4`,
**three independent boots per speculative configuration and three fresh AR
anchor boots**. Randomize boot order. Prefer GPU1 and apply the W3
episode-rejection and cross-boot certification rules.

*Why three and not two.* The resampling unit is the boot (rounds are not
independent replicates), so two boots leave one degree of freedom for a 95%
interval. Worse, a source-B episode shifts an ENTIRE boot by ~11% (W1): at
n=2 it is undetectable — you cannot tell which boot is the outlier, and a
uniformly-shifted boot reads as a clean ~11% disagreement, i.e. as a transfer
FAILURE rather than a rejection. Three boots make the outlier identifiable
and let episode rejection act at boot level. The adaptive rule is retained
for boundary cases only: add a fourth speculative/anchor boot for any unit
whose interval lies within one percentage point of a binding threshold, or
whose surviving boot means still disagree by more than 2%.

Nominal cost: 3 configurations x 3 boots + 3 AR anchor boots = **12 boots**
(up to 16 with adaptive additions), about three to four GPU-hours. One boot
sweeps both regimes and both batches.

**Pre-registered predictions:**

- **P-W14a — matched transfer:** for at least 5/6 transfer units, the 95%
  confidence interval for `(q_R5cot / q_R5) - 1` lies inside +/-5%. No unit's
  point estimate may exceed +/-10%.
- **P-W14b — decision preservation:** transferred `q` intervals combined
  with held-out measured `tau` contain actual `S_dec` for at least 11/12
  directional configuration predictions. In all four selector evaluations,
  the predicted epsilon-optimal tie-set contains the measured best; maximum
  simple regret is at most the 1.5% arming rent. Exact top-1 agreement is not
  required.

**Fallback if nothing certifies.** If no unit passes, D is skipped and E is
reshaped rather than stalled: its KV-dominated stratum measures cost and
acceptance together per cell (no transferred surface to draw on), and its
weight-dominated stratum still runs, because the quant boot classes are
selected offline regardless of whether runtime-class cost transfers. E's
P-W14d (no false eliminations) remains scorable in this form; P-W14e is
reported against a Round-1 that prunes nothing.

**Decision rule:** certification is local to `(configuration, batch)`.
Passing units enter the transferable cost surface. Failing units retain a
wide interval and pass through unpruned to Round 2. If no unit transfers,
runtime-class cost is measured with Round 2 and the offline stage is limited
to boot-class residency selection. A global mean may be reported but cannot
override this rule.

### C — analytic leverage router

**Cost:** CPU, about one hour; no GPU.

Compute the model's byte shares at each state:

```text
quant potential       -> weight-byte share
window / KV quant     -> KV-byte share
skip-count potential  -> approximate k/L share of layer work
```

Use 15% as the profiling-priority threshold. This threshold controls only the
measurement budget:

- at or above 15%: measure a full cost curve;
- below 15%: retain the configuration, use a deliberately non-pruning wide
  interval or defer its cost measurement to Round 2, and perform at most an
  anchor spot-check; and
- never eliminate from byte share alone.

Validate the routing rule without tuning the threshold. The correct W13
target is the **draft `D/T` spread** across windows: about 2–7% at low-KV
cells and 97–110% at b8/14.5k.

**Caveat, to be stated wherever C is cited:** that `D/T` spread is available
only from the SYNC-BRACKETED profiler, i.e. the biased instrument. This is
acceptable because C routes measurement budget and can never eliminate a
configuration — but C's agreement is NOT evidence that the cost model is
accurate. The leverage principle predicts the DRAFT-COMPONENT spread;
`q_true` is a whole-step quantity. They are different tests and must not be
conflated in the writeup. W13's complete `q` spread was smaller, about
30–49% at the high-KV cells. Derive unperturbed banked `q_true = tau/S` where
available; do not use the biased profiled `q` as ground truth. W6's inert
R1 window and active R5/R5cot windows are qualitative checks only.

### D — transferable latency surface

**Cost:** approximately eight boots / two GPU-hours if K can be selected down
inside a KMAX=4 boot; add configuration boots if it cannot.

For every B-certified `(configuration, batch)` unit and `K in {2, 4}`, fit at
fixed batch and capture regime:

```text
T_b(Q)        = alpha_T + beta_T * Q
P_c,K,b(Q)    = alpha_c,K + beta_c,K * Q
q_c,K,b(Q)    = P_c,K,b(Q) / T_b(Q)
```

Use three pre-declared training contexts and two untouched held-out contexts
per curve. B's matched point may be one training point. Fit separate curves
across graph/capture or compute-regime boundaries. If the training residual
exceeds 5%, widen the interval and use piecewise interpolation or a table;
never force the affine form.

**P-W14c — held-out cost coverage:** at least 90% of held-out `q_true` values
fall inside their 95% prediction intervals, mean absolute point error is at
most 5%, and the optimistic-bound elimination rule makes zero false
eliminations. Failure narrows the certified surface; it does not create a
global architecture failure.

### E — corrected dense end-to-end validation

**Cost:** approximately eight new quant/AR boots / two GPU-hours. D supplies
the window configurations and their held-out cells.

Test full configurations where their expected leverage is high:

1. **Weight-dominated, b1/short-context:** W4A16-INT4, W8A16-INT8, and
   FP8-dynamic. Each quantization plus its kernel realization is one complete
   boot-class configuration. Fit on three short-context states and reserve
   two states for scoring. At the training anchor, run the same matched
   content-transfer comparison as B using R1 and R6 over their first fixed
   256 tokens (nearly identical short prompts); an uncertified quant
   configuration receives a regime-local cost interval rather than an
   offline-transfer claim.
2. **KV-dominated, b8/long-context:** `w512`, `w2048`, and `w-off`, using D's
   untouched long-context states. Treat `w-off` as a separate realization.
3. **Depth:** score `K in {2, 4}` in both strata. Position counters may share
   acceptance data, but each depth needs a cost estimate.

For validation only, measure actual `S_dec` and `tau` for every configuration,
including anything Round 1 would eliminate. The simulated selector sees only
training cost data and Round-2 acceptance; the complete measurements form the
oracle used after predictions are frozen.

**P-W14d — safe Round 1:** every configuration eliminated by the optimistic
cost bound is truly below `1 + epsilon_arm`; otherwise the soundness claim is
refuted.

**P-W14e — final selector:** on every held-out cell, the predicted
epsilon-optimal tie-set contains the measured best or has at most 1.5% simple
regret. Gate decisions outside the experimental tie band must match the
measured ARM/OFF verdict. Report survivor fraction and measurement savings;
do not convert unresolved ties into point winners.

### F — global resident portfolio and runtime feasibility

**Cost:** CPU analysis plus at most one GPU-hour of switching benchmarks.

Round 2 emits regime-local evidence, but deployment receives one global pool.
Choose boot class `h` and runtime pool `P` by the robust portfolio objective

```text
maximize    sum_g w_g * max_{a in P union {OFF}} LCB(S_a,g)
subject to  HBM(h, P) <= M_HBM
            graph_memory(h, P) <= M_CG
            a is executable by the current runtime
```

Declare workload weights `w_g` before optimization and report sensitivity to
them. Account for target and draft weights, graph captures, KV-cache capacity,
and any preemption increase caused by reduced KV capacity. A configuration
measured alone is not assumed to preserve its cost after other weights become
resident.

Current executable scope and gaps:

- K/OFF: runtime-class now;
- window: boot-class today; masked down-selection or per-window capture is a
  G2 implementation selected only if F values more than one window;
- skip identity: boot-class today; G3 is selected only if F values more than
  one skip set; and
- quant and kernel realization: one boot-class choice in W14.

Specify the deployment ladder, confidence test, hysteresis, and probe duty.
Acceptance is action-specific: evidence from the active composition is not a
counterfactual measurement of another composition.

**Pre-engineering Gate F:** the proposed pool fits both estimated budgets and
its robust gain clears the relevant W3 value threshold. Existing runtime
actions must also have measured p95 switch latency below one target decode
step with amortized switch/probe overhead below 0.5%. An unavailable but
valuable window/skip action proceeds to G; it does not proceed to deployment.

### G — conditional composition switching (G2/G3)

**Cost:** a fixed-length confirmation pass, then only the selected engineering
work. G2a is small; multi-capture G2b/G3 is approximately one to two
engineering days and must be re-estimated from F's exact pool.

If G3 is proposed, first remeasure every proposed skip set under fixed length
and the corrected currency. Then implement only the mechanisms required by
F's pool:

- masked window down-selection when acceptance-only switching is sufficient;
- per-window captured graphs when cost-true window switching is valuable; and
- gated skip passthrough plus per-set capture when multiple skip identities
  remain valuable.

Proceed only if all of the following hold:

1. the pool requires more than one value of a lever that is boot-class today;
2. their robust portfolio gain clears the relevant W3 selection threshold;
3. the pool fits HBM and capture budgets; and
4. if G3 is selected, a fixed-length remeasurement confirms the W6 skip-set
   ordering under the corrected currency and episode protocol.

After implementation, rerun the full post-engineering Gate F: every action
must be executable, p95 switch latency must remain below one target decode
step, and amortized switch/probe overhead must remain below 0.5%.
Multi-quant resident dispatch is outside W14; it requires its own value gate
and engineering plan.

## Order, cost, and gates

| step | estimated cost | binding result |
|---|---:|---|
| C | 1 h CPU | measurement routing only |
| A | 30 min CPU | metric accounting closes |
| B | 12–16 boots, about 3–4 h GPU | local transfer mask |
| D | about 8 boots, 2 h GPU | held-out cost coverage |
| E | about 8 new boots, 2 h GPU | safe pruning and final regret |
| F | CPU + at most 1 h GPU | feasible global runtime pool |
| G | confirmation + 1–2 days engineering | only mechanisms valued by F |

To the first decision: about three to four GPU-hours. Full dense validation
through F: about 6–8 GPU-hours, with B/D anchors reused where valid. Boot count
is the binding estimate; elapsed time must be updated from the actual runner
matrix before execution.

## Measurement and scoring rules

- Keep all new scripts, data, logs, and results under phase 96 (`data/w14/`,
  `logs/w14/`, and phase-local scripts).
- Freeze the matrix, scorer, exclusions, and held-out cells before the first
  scored run. Retain refuted predictions.
- Fixed output length is mandatory for cost comparisons. Natural EOS may be
  reported only as a deployment result after equal-work validation.
- Use two disjoint content seeds and at least two independent boots for every
  binding comparison. Bootstrap at the boot level; rounds are not independent
  replicates. Add the pre-registered third boot for boundary or cross-boot
  disagreement cases.
- Use notune and the W3 episode-rejection/cross-boot rules. Pair every surface
  with fresh AR anchors and randomize configuration order.
- Report intervals, tie-sets, false eliminations, simple regret, survivor
  fraction, and resource cost. Exact top-1 accuracy is descriptive only.

## Known environment dependencies

- Checkpoints are under `/data/smcho/ckpts` after the 2026-08-07 home reset.
- `VLLM_USE_FLASHINFER_SAMPLER=0` and
  `LD_LIBRARY_PATH=/usr/local/cuda-12/lib64` are required; retain the other
  W13 environment fixes in the phase-local runner.
- Humming W4A8 JIT prewarm hung twice in W13. B avoids skip variants; E uses
  the non-Humming quant checkpoints already present.
- Prefer GPU1 while GPU0 has a co-tenant. Never terminate a non-`smcho`
  process.

## Paper outcomes

- **Broad B/D/E pass:** dense-model cost is an amortizable offline surface,
  acceptance is workload-local, and the uncertainty-aware two-round selector
  is validated on the measured dense domain.
- **Partial transfer:** publish a hybrid selector with a certified Round-1
  reliability mask; uncertified cells pass through to measurement. This is a
  supported outcome, not a failed architecture.
- **No runtime-class transfer:** offline work chooses the boot/resident class;
  runtime-class cost and acceptance are measured together online. The
  +1.75%/+7.05% offline-only error floor still motivates the online stage.
- **Transfer without pruning:** claim cost amortization, not candidate
  elimination. Round 1 is useful only to the measured extent.
- **Any false elimination in E:** the sound Round-1 claim fails; retain all
  affected candidates and report the counterexample.

## Expected next artifact

`results_w14_transfer.md` with the six matched transfer units, 12 directional
predictions, four tie-set decisions, raw boot-level intervals, and the local
transfer mask.
