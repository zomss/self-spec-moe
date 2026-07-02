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

## b2 — per-row consumption (WIP: implemented, debugging)

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

### b2 debug state (WIP checkpoint)

Implemented: GPU-side per-row gating (no host sync), uniform K+2 forwards,
per-row offset extraction, expectation registers (expect_rej/expect_tok:
normal rows expect rej==0 & bonus==anchor-consumed; re-anchored rows expect
rej==K & correction==out[0]). Measured accept 1.08; the W7_AHEAD_DEBUG=1
breakdown localizes it:

- cycle 1: valid=0.72-0.80, tok_match=0.72-0.80 -- the expectation test works.
- cycle 51: valid=0.00 with rej_match=1.00 (stale drafts fully rejected AS
  PREDICTED) but tok_match=0.00 -- the recovery guess out[0] NEVER equals the
  actual correction (systematic, not statistical): rows enter recovery and
  never leave; initially-valid rows fall in within a few cycles.

Since b1 validated the same extraction on the all-hit path (accept 2.92), the
suspect space is the b1->b2 delta: (a) the K+2th forward / in-run pending
consumption, (b) the expectation register pairing across mixed-row cycles,
(c) per-row divergent p1 interacting with the shared metadata build. Next
session: add a one-cycle A/B knob that gates consumption to all-hit (b1
semantics) while keeping the K+2/expect machinery -- separates (a) from (b/c)
in a single run; then instrument out[0] vs actual correction values directly.

### b2 session 2: the depth asymmetry (key structural finding)

Seven iterations converged on the load-bearing insight: **the free-running
loop has two vantage points with DIFFERENT pipeline depths.**

- `consume` runs AFTER the sampler: its stash is the verdict of the PREVIOUS
  run's drafts (depth 1) -- which is why b1's consume-side all-hit test
  (b == anchor-consumed-by-the-current-run) validates cleanly (accept 2.92).
- `run_ahead_chain` launches BEFORE the sampler: the stash it reads is one
  cycle older, i.e. the verdict of drafts from TWO runs back (depth 2) --
  inherent to the overlap structure, NOT async scheduling (draft_model spec
  disables async; the engine is sync).

Every run-side scheme tried (pending-continuation with expect registers /
uniform K+2 with per-row offsets / collapsed anchor=b@committed) fixed one
vantage point's alignment while breaking the other's. The collapsed anchor=b
design is depth-1-correct and therefore serves drafts one position early
under the real depth-2 (rej0 -> 0.00, the year-token +1 signature). The
original expect-register design was depth-2-correct structurally (its
validity transitions verify in the trace) but recovery drafts still failed at
the VALUE level -- the one unexplained defect.

**Next session needs a deterministic repro, not more blind DP4 iterations**:
DP=1, batch 1-2, dump per-cycle (draft consumed tokens+positions, served
drafts, verify input+verdict+commits) into one aligned table. With the depth
asymmetry now understood, that table should identify the recovery defect in
one pass. The consume-side (depth-1) test is trustworthy; a hybrid design --
run-side does pure continuation only, consume-side (post-sampler, depth-1
info) decides re-anchor and can LAUNCH the corrective re-run in the same
cycle -- avoids run-side depth-2 reasoning entirely and is likely the
simplest correct b2.
