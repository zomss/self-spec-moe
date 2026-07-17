# The selection mechanism, in full detail — audit document (DRAFT)

Purpose: verify (A) the candidate space is complete, (B) the search is
efficient. Every stage cites its code artifact and measured cost.

## Stage 0 — the candidate space

**Levers (the taxonomy, research/84/README):** every (draft-cost-term x
training-free-reduction-op) cell is classified:

| status | members |
|---|---|
| IN POOL | int4 quant (RTN + GPTQ), fp8 quant (W-only, W8A8, Marlin/native realizations), window+sinks (128/512), whole-layer skip (contiguous + profiled sets), sub-layer skip (mlp/attn), MoE expert restriction (contiguous shard, frequency-profiled), MoE top-C routing, n-gram/model-free |
| GATED DEAD (measured, with domination argument) | 2:4 & unstructured pruning (.42/.60 vs int4 .92 at more byte cut), SVD low-rank (.044 at 50% bytes) |
| MEASURED-EXCLUDED (reason stated) | draft-only KV-quant (pool unbuilt; V-only rule stated), vocab restriction (RETRACTED: circular gate; honest C4-keep coverage .80-.90 -> loses at deep gamma) |
| SCOPED OUT (statement) | dynamic token selection (cited as upper bound on window), tree/multi-draft (changes tau formula), KV-head merge (svdr-class argument) |

**Settings grids**: window {128,512} (win2048 pruned by measured beta-
equivalence), skip budgets 1-7 per model (greedy-searched, not fixed
fractions), expert fracs {25,50}, top-C {2,4}, quant {int4-RTN, int4-GPTQ,
fp8 x3 realizations}. gamma 1-8, capped per lever semantics.

**Combinations**: all subsets <=4 levers, <=1 per class (win/quant/skip/
lr) = full class grid (26-53 configs/cell/arch). Winner- AND OFF-hardening:
the exhaustive argmax was verified over the extended sets at every cell
(79/winner_hardening: 26/26 winners confirmed; OFF cells max 1.13x under
optimistic pricing).

**Realizations** (the axis single-lever work ignores): per config, the
execution variant is priced separately -- e.g. expert restriction spans
{EP-shard local, fp8 full replica, bf16 partial replica} = measured
0.55x/0.93x/1.03x at the SAME beta. Chain implementations {piecewise,
scratchpad-FA3 fixed, eager-MLA} carry fitted (phi, psi).

**Known holes (explicit)**: hardware axis (one 4xH100 box), one model per
MoE/MLA class (dense has three: Q2.5-7B/Q3-8B/Q3-32B), per-layer mixed-
precision assembly (deferred), dynamic token selection unmeasured.

## Stage 1 — the two surfaces (measurement)

- **beta** (research/77 harness): teacher-forced rejection-sampling
  acceptance vs target refs; 12 prompts x 96 pos (1152, paired);
  anchor-gated (composed tau reproduces e2e accept to <=2%, dense+MoE);
  second-distribution check (math bank); disjoint-artifact rule (learned
  from the vres retraction: profiled artifacts built on data disjoint
  from evaluation). Cost: refs ~7-25min/model+ctx; ~1.5-4min/arm.
- **R** (research/76): serve-mode TPOT, one server/arm, warm-pass,
  infra-parity preflight (backend/kernel/CG asserted), over-capacity
  detection, dummy-weight denominators for skip. Cross-checked vs offline
  slope at 2%.

## Stage 2 — pricing

speedup = tau_beta(gamma) / (gamma*(R + phi) + 1 + psi), where
- R: measured cell > measured combo arm > term-model
  T = F + bytes/BW_eff + h*b*ctx_attended + comm + kappa(M); levers are
  term edits; kappa(M) = 0.28*max(0,1-M/0.2) for fp8-dequant kernels
  (fitted on paired realization deltas); one-anchor architecture
  transfer (R 9.3% on 53 held-out cells).
- beta: measured combo > product law (median dev .009-.013; exceptions
  MEASURED: window-rescues-kvq +0.13-0.26; MLA skip x ctx destructive;
  depth breaks the product within-lever -> deep sets must be measured).
- (phi, psi): per (arch, chain-impl), fitted from e2e throughput and
  cross-validated against independent kineto traces (phi_broken 0.144 vs
  traced 0.129-0.172; phi_fixed 0.000; the 77% delivery number
  reproduced as arithmetic).
- Feasibility: bytes_resident <= HBM (validated 6/6 recorded incidents).

## Stage 3 — selection

Exhaustive over the priced space; winner = argmax of LCB05 under
per-source sigma (measR 4%, modelR 10-30% by family, prod-beta 2%,
phi/psi tier sigma); Monte-Carlo P(win); conservatism tie-break (prefer
measured-R within model error); OFF = 1.0 exactly. Properties by
construction: under-measured realizations cannot win; every cell prints
winner + LCB + P(win) + provenance; the map emits its own measurement
queue.

## Stage 4 — allocation & audit (the efficiency claim)

- **Nominate-by-profile, confirm-by-measurement**: contested cells
  (top-2 gap < model error) get 1-2 measurements. Backtests:
  - Cell-level (MLA-as-new): 91 GPU-min recovers the map to max regret
    2.6% vs 305-min full profile (L0 priors alone: max regret 22% --
    measurement is load-bearing).
  - Set-level (17-set pool): profile-then-solve (KnapSpec recipe = m=0)
    mis-ranks at depth (Spearman .94->.60) and mis-prices 30%;
    prior-guided finds pool-best in 1-2 measurements.
- **Audit round record**: e2e contradicted the offline layer 3x and was
  right 3x (ngram realization psi=2.4; flr ctx-invariance; vres circular
  gate -> retraction). No confident cell moved. Rules absorbed:
  disjoint-artifact; realization pricing mandatory.
- **Head-to-head cost accounting (measured, Qwen3-8B column)**: refs+7
  singles ~35 min; 32-layer profile ~50 min; greedy frontier ~40 min;
  9 e2e arms ~45 min => **~2.8 GPU-h for a complete new-model column**
  including the e2e audit. 32B: ~6 GPU-h + 1 ckpt build (2.5h).

## Pre-registration record (Phase 86)

P1 QK-norm predictive (kvq .985/.988 on Qwen3) -- CONFIRMED.
P2 skip family-dependence -- CONFIRMED (and scale-NON-monotonicity found).
P3 8B match -- EXCEEDED (1.42 vs 1.28).
P4 32B composed beat -- CONFIRMED task-matched (1.63 vs 1.43; prose tie 1.41).
