# W14 — the refined search: plan (awaiting GO)

Consolidates the corrections from W13 and the 2026-08-08 design review
into a concrete, costed work order. Nothing here is started.

## The design, in the user's four steps

1. **Round 1** — from (batch, context) compute the spec-forward latency
   for each composition; eliminate hopeless ones.
2. **Round 2** — on real workload, measure ONLY acceptance for the
   Round-1 survivors; emit the candidate pool per regime.
3. **Deployment** — switch composition at runtime on (context, batch,
   KV share) × acceptance × live signal.

Corrections folded in:

- **Residency budget, not boot-vs-runtime.** Quant and skip ARE
  runtime-switchable given resident weights / per-set graph capture
  (W0 prices quant as "a second DRAM copy"; skip is item G3, deferred
  engineering, not a barrier). What Round 1/2 choose is the resident
  POOL; the runtime switches inside it.
- **G3 is now justified by measurement**: W6's Round-2 winners use
  THREE distinct skip sets (none / {2,8} / count-8), so the
  "only if >1 skip set is shortlisted" condition is met.
- **Latency must be measured, then interpolated — not calculated.**
  The HBM roofline is off 2–15× (P-W8d), so analysis alone decides
  nothing. τ* is smooth in ctx (`a + b·ctx`; flat for a windowed
  draft), so 3–4 points per lever fit the form and it then EVALUATES
  analytically at any cell.
- **τ\* must be in DECODE currency.** Measured end-to-end it carries
  prefill dilution (R5: prefill ≈54% of wall clock; R5cot ≈16%), which
  is why cross-regime transfer failed at 8.4% mean / 17.7% max.
- **The profiler is demoted to a diagnostic.** τ* = (armed step)/(AR
  step) needs no decomposition; the D/V/C split stays only for
  explaining mechanism.

## Work items

### A — decode-currency instrumentation  [CPU, ~30 min, no GPU]
Add the per-request decode-time histogram delta to `run_grid.py`
(`vllm:request_decode_time_seconds`, sum/count snapshots per cell, as
`run_w6_calib.py` already does). Emit `dec_rate_req` per cell so
`τ* = τ / S_dec`. Prerequisite for B.

### B — τ\* TRANSFER TEST  ← **the decision point**  [6 boots, ~1.5 h]
Does τ* measured at one regime predict S at another, same cell?
Cells: R5 and R5cot at b1 and b8 (same batch, ~same context, 6×
different generation length — the pair that failed in e2e currency).
Compositions: the three W13 windows (no Humming dependency).
Protocol: fixed length, notune, ITERS=4, GPU1, decode currency.

**Pre-registered:**
- **P-W14a**: mean |transfer error| ≤ **5%** (e2e baseline: 8.4%,
  max 17.7%).
- **P-W14b**: ranking at the held-out regime preserved, ≥5/6 top-1.
- **Decision rule, binding:** if P-W14a and P-W14b hold, Round 1 stays
  a separate stage and C–E proceed. If they fail, **Round 1 folds into
  Round 2** for runtime-class levers and the offline stage keeps only
  the residency-pool choice — the search architecture changes, and the
  paper claim changes with it.

### C — leverage filter  [CPU, ~1 h, no GPU, free to validate]
Compute the weight/KV split per cell analytically from model config
and batch × ctx; admit only levers whose leverage clears a threshold
(quant → weight share, window/kvq → KV share, skip → k/L at every
cell). Validate against banked data at zero cost: it must predict
W13's measured τ* spread (2–7% at low-KV cells vs 97–110% at b8/14.5k)
and W6's winners (R1 won with an inert window; R5/R5cot with an active
one). Runs before D and E so neither profiles a lever that cannot
matter.

### D — τ\* interpolation model  [2 boots, ~30 min]
Fit `τ*(b, ctx)` per lever from B's points plus 2 held-out cells;
predict the held-out cells. **P-W14c**: held-out prediction within
±10%. Gated on B passing.

### E — the corrected dense validation (what W13 should have been)
[8 boots, ~2 h]
Test each lever where it HAS leverage:
- b1 / short ctx → **quantization** sweep (≈97% leverage):
  W4A16-INT4, W8A16-INT8, FP8-dynamic — all present in
  `/data/smcho/ckpts`, none needs Humming (avoids the JIT prewarm
  wedge that killed W13's skip arms).
- b8 / 14k → **window** sweep (≈79% leverage): w512 / w2048 / woff.
**P-W14d**: predicted ranking matches measured, ≥5/6 cells — the test
W13 could not perform because it varied a 3% lever and held the 97%
lever fixed.

### F — G3 gated passthrough  [engineering, ~1 day; CONDITIONAL]
Runtime-switchable skip sets (gated passthrough + per-set capture),
justified by W6's three distinct winning skip sets. Start only after
E validates the selection machinery.

## Order, cost, and gates

| step | cost | gate |
|---|---|---|
| C | free (CPU) | none — do first, prunes everything |
| A | 30 min CPU | none |
| **B** | 6 boots ≈1.5 h | **decides the architecture** |
| D | 2 boots ≈30 min | B passes |
| E | 8 boots ≈2 h | C + D |
| F | ~1 day eng | E passes |

To the decision point (C + A + B): **~1.5 h GPU, ~1.5 h CPU.**
Full validation (C–E): **~4 h GPU.**

## Known environment dependencies

- `/home/smcho` was reset 2026-08-07: checkpoints now at
  `/data/smcho/ckpts`; `VLLM_USE_FLASHINFER_SAMPLER=0` and
  `LD_LIBRARY_PATH=/usr/local/cuda-12/lib64` are required (all four
  fixes are in `run_w13_taustar.sh`).
- Humming JIT prewarm hangs on W4A8 draft boots (twice in W13, 90-min
  timeout each). Items B and E are designed to avoid that checkpoint.
- GPU0 had a co-tenant (`jhchoi`); prefer GPU1 and never kill
  non-`smcho` processes.

## What each outcome means for the paper

- **B passes** → Round 1 is a validated, separately-justified stage and
  the two-round claim stands as written.
- **B fails** → the honest claim becomes "offline decides the resident
  pool; everything else is measured online", which is a weaker but
  still novel contribution — and the error floor (+1.75% / +7.05%)
  still motivates the online half either way.
