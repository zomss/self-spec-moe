# P4 B0 value-screen run authorization V12

Date: 2026-08-10

Decision: **APPROVE one block-parallel, two-lane, nine-boot / 432-capture
matched B0 value screen.** This authorizes execution and the frozen score
only. P4a engineering, action admission, and any performance claim remain
false.

## What V12 replaces

V11 executed once on physical GPU 4. It completed the first boot's 48
captures, then GPU 4 was reserved by another user and the run was interrupted
by SIGINT with no runtime fault observed. The immutable record in
`data/p4/run_b0_value_screen_v10/failure.json` classifies the stop as
`external_resource_reassignment`, records 48 complete captures plus one empty
placeholder, and forbids retry, partial resume, fallback GPU, and scoring.
V11 is consumed; its output is preserved and none of its captures are reused.

## Why the contention probe is no longer on the critical path

Two consumed probes tried and failed to produce a GPU-0/GPU-1 contention
bound, both in the launcher rather than the runtime:

- V1 stopped after both serial baselines because its registered `47000`--
  `47399` rendezvous range overlaps the host ephemeral range; and
- V2 completed `serial-gpu0` (4/4 rounds, 96.75 tok/s, retention satisfied),
  then refused `serial-gpu1` with `source hash drifted for capture_adapter`.
  The adapter was rewritten inside the run window by a concurrent staging
  pass; its content is byte-identical to the registered hash today, so the
  refusal was correct but spurious.

No contention bound exists and V12 claims none. Instead V12 rests on Phase
96's registered block-per-GPU precedent, referenced read-only and never
rescored (`research/96_selector_foundations/data/w14/w14d_prereg.json`):
one COMPLETE block per GPU, boots sequential within a block, with a measured
lane spread of 0.45% (b1) and 0.49% (b8) under an identical AR boot and the
same frozen prompt.

That precedent transfers to this screen for four reasons that are properties
of the frozen matrix and scorer, not new assumptions:

1. acceptance metrics are pure counts (`tau_raw = 1 + A/H`) and carry no
   timing, so they are lane-independent by construction;
2. committed tokens per action are equal by design, so the scored ratio
   `s_k4 = rate_k4 / rate_off` reduces to a ratio of summed decode times,
   in which a lane-wide scale factor cancels exactly under action-by-block
   separability;
3. the Latin square gives every action exactly one boot in every block, so
   every action carries an identical lane mixture; and
4. the residual is a block-by-action interaction, and the frozen uncertainty
   is already a 4,000-draw paired complete-boot-block bootstrap, which
   resamples exactly that term.

A tripwire is retained rather than assumed away: `score_p4_b0.py` fails
closed when per-(action, regime, seed) cross-boot disagreement exceeds 2%.
The measured lane spread is a quarter of that bar. If a lane effect ever
exceeded it, the screen would refuse to score rather than absorb it.

## Registered lane assignment

| lane | GPU | UUID suffix | blocks | boots | CPUs |
| --- | --- | --- | --- | --- | --- |
| lane-a | 0 | `4938442e` | 1, 3 | 6 | 0-95 |
| lane-b | 1 | `ba39f4f0` | 2 | 3 | 96-191 |

Blocks are never split: block 1 (`off`, `k4`, `w512`), block 2, and block 3
each keep all three actions on one GPU, back to back. Lanes hold disjoint CPU
sets and disjoint compile-cache roots.

With three blocks over two lanes, lane-a runs block 3 after lane-b has
finished, so block 3 is measured without a co-tenant while blocks 1 and 2
overlap. That asymmetry is deliberate and recorded: it is a block-level
condition, which is exactly the term the paired complete-boot-block bootstrap
resamples. Every boot records its physical GPU index and UUID so the effect
stays detectable post hoc instead of hiding inside the score.

Expected wall clock is about 7.7 hours against roughly 11.5 hours serial,
derived from the V11 boot that completed 48 captures in 75.4 minutes. Boot
cost is not the constraint: engine init measured 48.5 s with the compile
cache warm. The capture time is concentrated in two regimes, R1 (46.5%) and
R5cot (33.6%).

## Repairs carried by V12

The V2 probe's failure mode is closed structurally rather than by hoping the
worktree stays quiet:

- **Source snapshot.** All 34 registered sources are copied into
  `<output>/source_snapshot/` under their repository-relative paths and
  verified against the authorization before any child starts. Children
  resolve registered paths through the snapshot, so later worktree churn
  cannot drift a hash mid-run.
- **Executing-code guard.** The snapshot fixes the hashes a child verifies,
  so the code actually executing is additionally compared byte-for-byte
  against its snapshot copy. A mismatch fails that boot only.
- **Worktree quiescence preflight.** The launcher refuses to start while any
  registered source is uncommitted or modified, and records the git HEAD.
  This turns the V2 condition into a refusal before any GPU work.
- **Block restart unit.** Each block writes its own `block_result.json`. A
  failed or interrupted block is preserved alone, its siblings stay valid,
  and a rerun requires a fresh block-scoped authorization. Scoring still
  requires all three blocks and all 432 captures.

## Boundary

A pass produces `p4_b0_value_screen_result` and nothing more. The scorer's
acceptance-dominance short circuit is unchanged: if it fires, P4 stops and
K4/OFF remain the executable pool. P4a authority, exact live aliases,
target-local controls, transition overhead, and post-engineering resource
remeasurement all remain required before any action admission. P5 skip and B1
remain blocked; private-KV profiling stays cancelled.

## Artifacts

- `data/p4/p4_b0_run_authorization_v12.json`
- `data/p4/p4_b0_run_authorization_v12_validation.json`
- `schemas/p4_b0_run_authorization_v12.schema.json`
- `scripts/run_p4_b0_value_screen_v12.py`
- `scripts/validate_p4_b0_run_authorization_v12.py`
- `tests/test_p4_b0_run_authorization_v12.py`
- `tests/test_p4_b0_value_screen_v12.py`

Registered create-only output: `data/p4/run_b0_value_screen_v11`.

## Status

CPU validation passes and the package is run-ready. V12 is **unexecuted**.
