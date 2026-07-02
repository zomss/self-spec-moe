# OV1(b) results — consuming the ahead chain (b1: correct; b2: designed)

## b1 — all-hit-gated consumption: CORRECT, not yet faster

`VLLM_SELF_SPEC_CONSUME_AHEAD=1`: when every row on a rank fully hit (all K
accepted AND the actual bonus equals the anchor token the in-flight ahead run
consumed), the ahead drafts are served and the lockstep propose is SKIPPED; any
miss re-bootstraps via the normal propose. Chain state is a pending-token
machine: each run consumes [anchor, d1..dK] (K+1 forwards; K+2 on bootstrap) and
ends on a trailing guess token — the next run's anchor, whose KV is exactly
right whenever the guess matched the sampled bonus.

Measured (DP4 GPUs 0-3, b64, K=2, a2a=0, 1 iter):

| config | tok/s | accept |
|---|---:|---:|
| lockstep (off-baseline) | 2478 | 2.91 |
| OV1(a) validation (ahead + propose both run) | 1995 | 2.91 |
| **OV1(b1) consume (all-hit gated)** | **1755** | **2.92** |

- **Correctness: PASS.** Accept fully restored — consumed drafts are byte-what
  propose would produce (same chain, verified anchors); misses cost nothing
  (fallback). No crashes (bounded overlap window), no DP deadlocks.
- **Speed: not yet.** The all-hit gate needs a per-cycle `.item()` host sync
  that serializes the pipeline (~+10 ms/cycle), costing more than the ~22%
  propose-skip rate (per-rank all-hit = 0.91^16) saves.

## En-route findings (each its own fix, committed)

1. **DP deadlock**: consume-vs-propose divergence across DP ranks deadlocked on
   propose's `coordinate_batch_across_dp` collective → the comm-free draft path
   now skips DP coordination entirely in consume mode (per-rank dispatch;
   locally-uniform `num_tokens_across_dp`). Also removes the Phase-45 DP-sync
   barrier from the draft path.
2. **Pipeline off-by-one family**: the first consume implementation anchored
   hit rows on the stashed bonus (already covered by the chain's own guess) —
   accept 1.09. The pending-token formulation fixes the algebra; the fully
   general per-row version needs a two-deep expectation register (see b2).

## b2 — per-row consumption (designed, not implemented)

Goal: remove the host sync and consume per-row (~91% of rows every cycle).
Design notes from the b1 derivation:

- Uniform K+2 forwards per run; per-row draft offsets: valid rows take
  outputs[0:K] (pending = outputs[K], last forward idempotently rewrites the
  pending KV); re-anchored rows take outputs[1:K+1] (outputs[0] is their guess
  of the intervening commit; pending = outputs[K+1]).
- Validity test is TWO pipeline stages deep (the stash at run t reflects
  verify t-1), so the expectation register must hold the anchor consumed one
  run back: valid = rej0 & (b == anchor_used_prev), plus per-row bookkeeping
  for rows whose assumption chain is one cycle shorter (fresh re-anchors).
- Misses recover in ~1-2 cycles (fully-rejected garbage + guess-matched
  correction), losslessly; partial acceptance of garbage falls back cleanly
  because the rej0 term fails.
- All decisions stay on-GPU (torch.where selects; no .item()).

Expected effect (from the OV1(a) numbers): remove ~30 ms lockstep chain from
~91% of cycles at the cost of the bounded ahead exposure (~13 ms at a2a=500)
-> ~+15-20% over lockstep at f~0.6, growing with f, before K-retune upside.
