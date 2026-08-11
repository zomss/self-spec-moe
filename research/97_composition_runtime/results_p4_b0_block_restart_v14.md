# P4 B0 block-1 restart authorization V14

Date: 2026-08-11

Decision: **APPROVE one restart of capture block 1 only.** Blocks 2 and 3 are
reused unmodified. The combined 432-capture score is emitted whether or not it
certifies. P4a engineering, action admission, and any performance claim remain
false.

## What the V13 screen produced

The screen ran to completion: **432/432 captures, 3/3 blocks, every boot
`rc=0`**, in ~7.9 h across the two lanes. The frozen scorer then refused:

```text
cross-boot certification exceeded 2% for target-matching-k4/R8/seed1: 0.024031
```

Three of 36 cells exceed the 2% bar after the frozen 95% episode filter
(median disagreement 0.987%, max 3.378%). No score was emitted. The complete,
unscored captures are preserved in `data/p4/run_b0_value_screen_v12` and the
diagnosis is recorded in that run's `certification_refusal.json`.

## The lane decision was not the cause

Post-filter median per-block offsets:

| block | GPU | condition | median offset |
| --- | --- | --- | --- |
| 1 | 0 | co-tenant | +0.099% |
| 2 | 1 | co-tenant | +0.208% |
| 3 | 0 | **solo** | −0.133% |

All three sit inside the 0.45–0.49% Phase 96 precedent. Two facts refute a
lane or contention explanation directly:

1. the solo block is not faster — it is marginally the slowest, the opposite
   of what an uncontended tail would produce; and
2. in the failing `k4/R8/seed1` cell, blocks 1 and 3 sit on the **same** GPU
   and still disagree by 2.4%, so the disagreement is temporal, not spatial.

## What did cause it

Episode noise concentrated in block 1. Rounds below the 95% floor:
**block 1 = 6/144, block 2 = 1/144, block 3 = 0/144.** The pattern is plainly
bimodal, e.g. `k4/R6/seed0` block 1: `125.7, 125.9, 94.9, 125.8`. This is the
co-tenant fabric/DRAM bimodality Phase 96's W1 traced and could not eliminate;
another user's ~70 GiB job occupies GPUs 2/3.

One mechanism deserves naming because it is methodological rather than
environmental: **the 95% rejection rule is one-sided.** It removes slow rounds
only. In `k4/R8/seed1` block 1's rounds were `89.3, 100.1, 83.0, 100.1`; the
filter stripped the two slow rounds and kept only the fast pair at 100.1,
placing block 1 *above* blocks 2 and 3 (~98) by 2.4%. The filter converted an
erratic block into an apparently fast one. Block 1 is erratic, not slow.

## Why this restart is not selection on the outcome

Rerunning until a screen certifies would be selection on the result under
test. Three commitments, all registered in the authorization **before** the
rerun and enforced by its validator, keep this a repair rather than a search:

1. **The block is selected on a rule independent of the outcome.** Block 1 is
   chosen for its episode-rejection count (6 vs 1 vs 0), computed from the
   frozen 95% rule. That count does not depend on which cells failed
   certification.
2. **The result is bound before it exists.** The combined score is emitted and
   kept whether or not it certifies; `rerun_until_pass_forbidden` is true and
   at most one restart of this block is permitted. A further restart requires
   a new declared rule.
3. **Nothing else moves.** The 2% certification bar, the 95% episode rule, the
   prompts, seeds, cells, and the reused blocks are all unchanged, and the
   source run is never overwritten.

Ten fail-closed tests assert exactly these: selecting a quieter block,
relaxing either threshold, permitting rerun-until-pass, discarding the result
on failure, allowing more than one restart, fabricating the rejection counts,
overwriting the source run, and remeasuring the reused blocks are each
refused.

## Execution

Block 1's three boots (`off`, `k4`, `w512`) rerun on lane-a / GPU 0, the same
lane they occupied, into the fresh create-only
`data/p4/run_b0_block1_restart_v1`. Scoring then combines the 144 fresh
captures with the 288 preserved captures from blocks 2 and 3, all through the
same frozen adapter, and writes `combination.json` alongside either
`score.json` or the recorded refusal. Expected ~3.8 h.

## Artifacts

- `data/p4/p4_b0_block_restart_authorization_v14.json`
- `data/p4/p4_b0_block_restart_authorization_v14_validation.json`
- `data/p4/run_b0_value_screen_v12/certification_refusal.json`
- `schemas/p4_b0_block_restart_authorization_v14.schema.json`
- `scripts/validate_p4_b0_block_restart_authorization_v14.py`
- `tests/test_p4_b0_block_restart_v14.py`
