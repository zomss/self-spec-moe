# Phase 78 results

## V0 + V1: the term decomposition holds; window is predictable from bytes alone

`scripts/fit_cost_model.py`, fit to 76's absolute TPOTs (`data/fit.json`).

| gate | dense | moe | verdict |
|---|---|---|---|
| V0 in-grid median \|err\| (≤5%) | **3.5% PASS** | 6.8% marginal FAIL | see noise note |
| V0 crossover orderings | **8/8** | **8/9** (miss = a known S4-noisy cell) | PASS |
| V1 held-out window (≤10%) | **2.2% PASS** | **6.9% PASS** | **PASS** |

**V1 is the load-bearing result**: with every window measurement EXCLUDED from
the fit, the model predicts window's full R curve from KV-byte accounting
alone — 2.2% on dense. The decomposition is physics, not curve-fitting.

## The fit found a real missing term (and the fix is physical)

First fit under-predicted exactly the large-allocated-KV MoE cells by 33-49%
(window/kvq at b32 long-ctx). Diagnosis: a **KV-management term that scales
with ALLOCATED context, not bytes read** — block tables / attention metadata
that draft-side read levers (window, kv-quant) do NOT remove. Adding
`h·n·(ctx)` (fitted h = 56 ms/Mtok on MoE serve; h→0 on dense TP1) moved MoE
from 8.0% → 6.8% and killed the structured residual. Lesson for the map:
window's R has a floor set by KV management, not KV bytes — visible in E1's
measured window TPOT still growing with ctx at fixed window.

## Fitted parameters (additive form; `data/fit.json` for all)

- dense: F = 2.76 + 0.021·b ms, BW_eff = 4.8 TB/s (UNPHYSICAL > 3.35 peak —
  same effective-parameter inflation P75-E1 saw: weight-read partially
  overlaps compute, so Δbytes/Δt exceeds true BW; fine for prediction, not
  for physics claims), κ_machete = 0.67 ms vs κ_marlin ≈ 0 (E1's kernel
  ordering, recovered by the fit), κ_fp8 ≈ 0.
- moe: F = 2.79 + 0.053·n, BW_eff = 1.0 TB/s (grouped-GEMM latency-bound,
  never BW-saturating), h = 56 ms/Mtok, comm = 1.64 + 0.13·n ms,
  κ_fp8marlin = 0.70, κ_fib = 1.19, κ_a2atile ≈ 0.
- Roofline form fits identically to additive on this grid (compute corner
  never binds at these cells) — carried but not preferred.

## Honest caveats

1. MoE V0 at 6.8% vs the 5% gate: the three worst residuals are two known
   S4-noisy cells (m_fp8marlin b8/32k, m_skip50 b4/32k — DP4 placement
   bimodality, ±10-15%) plus m_localroute b32/16k. The DP4 noise floor
   plausibly makes 5% unreachable on this data; V1's pass is the stronger
   evidence. Next iteration: noise-weighted fit + a comm term with ctx
   dependence (skipa2a deltas grow with ctx — comm and KV-management may be
   partially confounded in comm-on arms).
2. BW_eff unphysical (see above) — the model is EFFECTIVE, validated by
   prediction, not a bandwidth measurement.
3. MoE expert weight-read uses uniform-routing distinct-expert expectation;
   P21 skew makes it an overestimate at small n.

## Next: V2 (win128 R + b16/8k interpolation, one ~30-min E1 run), then V3
(predict the deferred MLA tier from config, run `e1_sweep.sh mla` as ground
truth), then `selector.py` → strategy map v3.

## V2 forward predictions: PASS (registered before measurement)

`data/v2_predictions.json` (registered) vs `data/v2_verify.json` (measured):

| family | cells | median |err| | gate |
|---|---|---|---|
| win128 R (cost of a never-costed arm) | 17 | **2.9%** | PASS @10% |
| b16×8k interpolation (never-swept cell) | 4 | **8.0%** | PASS @10% |

Dense is clean throughout (max 7.3%). The three MoE outliers are the known
suspects: b4/32k (+42%) and b32/32k (+25%) are S4 placement-noise cells;
b32/2k (−16%) is the short-ctx parity band where the model over-credits the
KV cut (measured ≈ parity) — same structure as the V0 residuals. The
dense b32/32k cell is correctly MISSING (bf16 denominator over-capacity).

With V1 (held-out arm) + V2 (forward, off-grid), the model predicts R for
unswept levers AND unswept cells. Next: V3 — predict the deferred MLA cost
tier from config constants, then one confirmation sweep (`e1_sweep.sh mla`).
