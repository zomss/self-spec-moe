# W6 item 3 — the formal section (paper-bound statements, measured constants)

Every constant cited here is a committed artifact of phases 93–96.

## Setting

Per cell (batch b, context ℓ) and lever a = (composition, K):
`S_a = (1 + f_a K)/(K R_a + 1)` (exact identity; C2). L levers, regime
mixture w over cells.

## Result 1 (identifiability dichotomy — the spine)

(i) R is offline-identifiable: calibration R reproduces under deployment
kernels at ratio 1.00× (n=32 cells, range [0.91, 1.13]; W4b), transfers
to serving within 2% at matched decode length (W4c), and admits a
3-parameter roofline fit (v2r). (ii) f is not offline-identifiable:
across regimes ρ = f_live/f_cal ∈ [0.41, 1.23]; on llama ρ is not even
regime-scalar (35% composition interaction, W5); within the skip lever,
the cheap tier's per-layer order flips by content with seed-consistent
effect sizes (transfer test). **Empirical corollary (the floor)**: an
offline-only argmax over {OFF, w512, w2048} pays mean +1.75% (dense) /
+7.05% (llama), worst-cell +37.1%, against a measurement-informed
selection over the same actions (error_floor.json).

## Result 2 (soundness of Round 1)

S_a is strictly increasing in f_a, so `sup_f S_a = (1+K)/(K R_a + 1)`.
Eliminating a iff `sup_f S_a < 1 + ε` under R_a's favorable band never
eliminates an optimum (one-sided guarantee conditional only on the
R-band, which Result 1(i) licenses). Applied per realization; skip
enters via its COUNT (R physics) with identity deferred (Result 4).

## Result 3 (Round-2 sample complexity and the serving ladder's regret)

f̂_a is a Bernoulli-rate estimate over drafted tokens (measured seed
agreement 0.5–1.5% at n≈16-prompt bursts); the identity maps its CI to
S with Lipschitz constant K/(KR+1) ≤ 1.2. Hence ε-optimal per-regime
selection w.p. 1−δ in O(log(|pool|/δ)/ε²) drafted tokens; the tie-set
is the ε-optimal set at the budget. At serving: switching is free
(per-flip cost ≈ 0, W4a) and mis-arming costs c_dwell ≈ 1.06
step-equivalents per armed step, so the ladder's regret decomposes as
`regret ≤ d·c_dwell + T_detect·Δ_drift` with probe duty d a solvable
knob (drift constants: phases 91/92). Demotion is an SPRT on f̂ with
unbiased telemetry (no optimistic pooling — F5's +0.1 bias is a
measured counterexample to plug-in EMA estimators).

## Result 4 (budgeted portfolio; the KnapSpec generalization)

Offline stage: choose boot-class config and a resident-graph pool
P (|P| ≤ M, the capture budget) maximizing `E_w[max_{a∈P∪{OFF}} S_a]`
— a two-stage stochastic knapsack whose recourse (the inner max) is
realized online by the ladder. KnapSpec is the special case |P|=1 with
offline profits; the dichotomy forces the profit vector to Round-2
measurement. Within skip: weights (per-layer cost, uniform) are
Round-1 physics; values (per-layer acceptance cost) are measured — and
the transfer test shows the values' decisive entries (cheap tier) are
exactly the non-transferable ones.

## Instantiation (dense, measured)

Round 1 pruned to the q-hum pool; Round 2 (1 GPU-h) ranked it: R5→
s-none/w512 (1.532), R5cot→s-2,8/w512 (1.884), R1→s-2,8/w2048 (1.184);
gate PASS (+5.9% single-regime, W3 thresholds). Llama: fail-closed to
static + OFF gate — the floor's llama column shows what ignoring the
dichotomy would have cost instead (+7.05% mean).
