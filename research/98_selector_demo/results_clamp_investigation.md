# The clamp: characterised, gated, not explained

Diagnostic record for X19–X23 and five G98-C campaign attempts. Nothing here
amends a frozen artifact.

## 1. What the clamp is

A boot that is otherwise identical measures the draft chain 8–60% slow. It is
not noise: it is a **discrete state** with a reproducible magnitude, and it is
wrong in a direction that mimics the very effect Round 2 exists to measure.

Comparing a clamped and a clean boot of the same cell, from their traces:

* **The work is bit-identical.** KV blocks in use (415/29/818 medians), their
  maxima, and `H_target_steps` (16/1/32) match exactly, regime for regime.
  Same batches, same requests, same allocation. Only the time differs.
* **It aligns to regime boundaries** and is flat within each regime — R1's
  thirds read 44.51 / 44.55 / 44.49 ms.
* **It is ordered by how much GPU work can hide it.** Added time per step
  across the measurement order R4, R5, R5cot, R8, R1, R6:

  | regime | batch / context | added |
  | --- | --- | --- |
  | R4 | 8 / 8.8k | +0.24 ms |
  | R5, R5cot | 8 / 14k | +0.17 ms |
  | R6 | 32 / short | +1.79 ms |
  | R8 | 16 / short | +3.91 ms |
  | R1 | 1 / short | +5.02 ms |

That is the signature of a **fixed ~4–5 ms per-step host-side delay,
progressively hidden as GPU work grows** — fully exposed at batch 1, absorbed
once the context is long. It is not a cost floor: R6 sits between R8 and R1 in
cost yet is spared.

Magnitude is reproducible across independent boots: R1 whole-step 44.51, 44.93,
44.93 ms against a clean 39.91 — three sessions agreeing to the hundredth of a
millisecond at 1.126x.

## 2. Seven hypotheses, seven refutations

| # | hypothesis | test | result |
| --- | --- | --- | --- |
| 1 | mid-measurement CPU starvation | X19, X19-rerun | runqueue wait ≤0.07% |
| 2 | lane CPU pinning | X20, X20-rerun | pinned/unpinned ratio 1.001, 1.003 |
| 3 | chassis power from neighbours | X21 | 3 clean boots at 524+542 W |
| 4 | thermal capping | X21 | 42–69 °C, clocks at max |
| 5 | GPU-0-local fault | X22 | lane-b (GPU 1, CPUs 96-111) clamps identically |
| 6 | neighbour GPU activity | run 4 | clamped with neighbours at 0% / 125 W |
| 7 | campaign code path | X23 | both paths clamp, interleaved |
| 8 | our own power cap | live telemetry | cap fires during the regimes that DON'T clamp |

**The cause is not identified.**

## 3. The methodological error, stated plainly

Four of those hypotheses were advanced on comparisons made ACROSS TIME — probe
A now, probe B ten minutes later — and every one dissolved on retest. The clamp
switches on a timescale of minutes, so an A-then-B design manufactures
differences from nothing. A 5-versus-6 tally of campaign-clamped against
probe-clean looked like 1-in-126 odds; interleaving the two paths (X23) showed
the path was irrelevant.

**Only interleaved comparisons survived.** X20 and X23 are the two probes that
alternated arms within a session, and they are the two that produced trustworthy
negatives.

A second, related error: three instruments — runqueue wait, per-regime clocks,
cheap-regime spread — were each read as evidence about the CAUSE while having
only ever observed CLEAN boots. An instrument that has never seen the event
cannot exonerate anything. The runqueue instrument has now failed five times to
be running when the clamp fired.

## 4. What the telemetry did show

Sampling continuously across three confirmed clamped boots (the first such
data):

| window | sm_med | sm_min | pw_max | throttle |
| --- | --- | --- | --- | --- |
| clamped ×3 | 1980 | 1290–1455 | 696–709 W | SW_POWER_CAP |
| no boot | 1980 | 1980 | 501 W | — |

GPU 0 reaches its 700 W TDP and the software power cap engages, dipping the
clock to ~1300 MHz. This is **our own workload**, with neighbours idle at 124 W
— which inverts how the power question had been framed all night.

It is nonetheless **not the mechanism**. The capping occurs during the
high-utilisation long-context regimes, which are exactly the regimes that do not
clamp, and X21 observed the same flag on wholly clean boots. A clock reduction
would also slow GPU-bound work most, whereas the observed pattern is the
opposite.

## 5. What to do about it

The gate detects the condition reliably without explaining it: **0 false
positives over 25 clean boots, 4/4 on known-clamped boots**, using two
independent within-boot signatures (batch monotonicity and relative spread).
It has now refused five contaminated campaigns.

That refusal is the whole value. A clamped boot inflates cheap configurations
toward a common floor — indistinguishable in shape from "composition costs more
than predicted", the residual Round 2 was built to explain. An ungated campaign
would have confirmed its own hypothesis.

Operationally the campaign needs a window in which the box is not clamping. It
is resumable, so a retry loop ratchets forward through the lattice across
windows.

## 6. Open items

1. **Resume is broken by a guard of its own.** `_require(not trace.exists())`
   correctly stops a boot reading another boot's steps, but also rejects
   legitimate resumption, because an aborted run leaves `attempt1.jsonl`
   behind. Currently worked around by deleting stale traces between attempts;
   the runner should key traces by run or clear them itself.
2. **Telemetry is discarded on cell failure.** It is written only when a cell is
   accepted, so the three most informative boots of the night lost theirs. It
   belongs in the `discard` callback.
3. **The sampler lacks temperature**, which matters if thermal state drives the
   power ceiling.
4. **A continuous instrument spanning many boots** is the only design that can
   catch an intermittent effect that is never present when a probe is aimed at
   it. Started; it produced section 4. It should record per-regime windows, not
   just per-boot.

Both (1) and (2) change hashes and need a v5 re-issue.
