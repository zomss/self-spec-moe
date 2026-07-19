# Phase 89 — runtime lever adaptation via DRAM-cached weights (step 3)

Source: user program 2026-07-19 ("As efficient rollout paper loads
quantized model to device, we can try this way by caching required
model at DRAM and load this at runtime. This is final one and
essential for RL post training regime"), after steps 0-2 closed
(88: 9/9 8B regimes beat AR; 32B diversity column; KnapSpec cell
~1.51x vs 1.43).

## Objective

The compiled policy today switches K/OFF per step; the WEIGHT lever
(ckpt x kernel) is boot-fixed. Build the missing axis: draft weight
checkpoints pinned in host DRAM, hot-loaded to GPU at runtime on a
policy signal, EfficientRollout-style -- then demo on an RL-rollout
trace where the right lever CHANGES during training (accept drift,
batch drain) and across serving regime shifts (cross-column value:
fleet prediction +39.1%).

## Design sketch

1. E0: price the mechanics (no policy). Pin the W4A16 + W4A8 draft
   ckpts in pinned host memory; measure DRAM->GPU load for the 32B
   drafts (~18GB int4: PCIe gen4 ~25GB/s -> ~0.7s; 8B ~4.5GB ->
   ~0.2s; E0-toggle measured 0.11s for the 8B swap path) + weight
   REBIND cost into the live engine (param copy into existing
   tensors, no re-init) + first-step penalty (CUDA graph re-capture?
   -- wholechain graphs key on shapes not weights: captures should
   REPLAY with swapped weights if tensors are updated in place.
   VERIFY: in-place copy_ preserves graph validity).
2. E1: engine surgery -- a swap endpoint on the draft proposer:
   load_draft_weights(tag) copies the DRAM-cached state_dict into the
   draft model's parameters in place (quant configs must match shapes:
   W4A16<->W4A8 differ in scales layout -> either (a) same-format
   swaps only [K-lever + skip-set + window flips free; W4<->W4A8
   needs re-pack], or (b) pre-packed per-kernel tensors cached per
   ckpt -- cache BOTH packed forms in DRAM, swap wholesale).
3. E2: policy wiring -- extend the compiled table with per-lever
   options (not just per-K): cells price {(lever, K)} and the
   scheduler argmax emits a swap request when the winning lever
   changes (hysteresis: swap only if predicted gain amortizes the
   0.1-1s swap in <N seconds at current throughput).
4. E3: the RL demo -- rollout trace with (a) accept drift emulated by
   checkpoint sequence (or temperature schedule as proxy), (b) batch
   drain within steps; policy with DRAM swap vs best-static vs AR.
   Gate: policy beats every static on the trace aggregate; the swap
   fires at the drift point measured, not scripted.

## Constraints / notes

- K2/K3 options must enter the compiled tables (88 finding: shallow-K
  wins the low-accept cells; RL T=1.0 cell = Hum K2 at 8B).
- Humming odd-width wedge (M=3/5 tile hang at TP2, library bug) --
  8B/TP1 unaffected: run the demo at 8B first.
- Prior art anchor: EfficientRollout loads the quantized model per
  rollout phase; our delta = measured per-cell policy + zero-downtime
  in-place swap + the map deciding WHEN.

## Artifacts

results_swap.md; scripts/e0_swap_cost.py, e1_swap_hook (fork),
e2 policy table ext, e3 demo driver.
