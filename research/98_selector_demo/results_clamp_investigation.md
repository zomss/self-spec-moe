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

## 6. Nsight Systems: the step budget (X24)

Two profiled boots, both confirmed clamped by the gate (R1 34.16 and 34.28 ms
against a clean 30.1). No clean baseline was obtainable -- the box clamped every
boot for over an hour -- so this characterises the PATH, not the clamp.

Per batch-1 engine step, 48.0 ms wall under nsys, 60 steps profiled:

| component | ms/step | tail >50us |
| --- | --- | --- |
| GPU kernels (busy) | 5.14 | — |
| `cudaDeviceSynchronize` (waiting) | 7.91 | all 720 calls |
| kernel launches (3 APIs) | 6.01 | negligible |
| `cudaMemcpyAsync` | 0.59 | 33.6% |
| `mmap` + `munmap` | 1.05 | — |
| host work outside CUDA | ~33 | — |

**GPU utilisation is 10.7%.** At batch 1 the draft chain is overwhelmingly
host-bound, which confirms X8-X12 by direct measurement rather than inference
and explains the clamp's regime ordering: a host delay is fully exposed where
there is almost no GPU work to hide it, and invisible at long context.

`cudaLaunchKernel` shows a 0.5% tail, so this is NOT driver contention on
launches. The time is in Python and framework code between CUDA calls.

**One concrete inefficiency.** There is exactly one `mmap` and one `munmap` per
flash-attention call: 8640 of each against 8640 attention kernels, 144 pairs per
step = 36 layers x 4 chain steps, ~1.05 ms/step. Beyond the syscall time, each
pair takes the process's `mmap_lock` and forces TLB-shootdown IPIs across every
core the process's threads occupy, a cost paid by other threads and invisible in
the syscall's own duration. Worth fixing on its own merits; whether it varies
between clean and clamped boots is UNKNOWN, because no clean trace exists.

## 7. The profiler's own syncs cost 2.4 ms/step (X25)

X24 found 12 `cudaDeviceSynchronize` per engine step, all from
`SelfSpecProfiler.region`, which syncs at both ends of every timed region:
`draft_chain` (2), `draft_forward_first` (2), `draft_forward` x3 (6),
`verify` (2). `_fine_only()` gates per-step sub-timers behind
VLLM_SELF_SPEC_PROFILE_FINE but matches only `step_`, `step0_` and
`chain_setup`, so `draft_forward` -- which fires once per CHAIN STEP -- syncs by
default, putting 8 of the 12 syncs INSIDE the `draft_chain` region being
measured.

Interleaved A/B, four boots, all gating clean, with `_fine_only` monkeypatched
in the child to also match `draft_forward` (12 syncs -> 4):

| regime | chain ON | chain OFF | delta | step ON | step OFF | delta |
| --- | --- | --- | --- | --- | --- | --- |
| R1 (b1) | 30.21 | 27.80 | +2.41 | 40.13 | 37.66 | **+2.47** |
| R8 (b16) | 31.59 | 29.20 | +2.38 | 42.70 | 40.23 | **+2.46** |
| R6 (b32) | 33.86 | 31.48 | +2.38 | 46.23 | 43.77 | **+2.47** |

The decisive column is `mean_armed_step`, taken from the koff trace and measured
identically in both arms, independent of the profiler. Its delta matches the
`draft_chain` delta (2.47 against 2.40), so this is **not misattribution but
real time destroyed**: the inner syncs serialise host against device, costing
~0.3 ms of lost overlap each. The cost is constant across batch, as an
overlap loss should be.

**Consequence for `F`.** A constant per-step cost is absorbed by whatever
constant term a model carries, and in the Round-2 model that term is `F`,
registered as an irreducible floor at 3.66 ms at R1. Up to ~2.4 ms of it may be
the instrument rather than lm_head and embeddings. This is a hypothesis about
what the FITTED `F` will absorb -- `F` was originally derived from checkpoint
bytes, not from the fit -- and it is directly checkable once Round 2 fits `F`
from data. It should be checked, not assumed.

**Removing the syncs is safe for CUDA graphs** and was demonstrated: all four
`syncs_off` boots ran the identical piecewise graphs. The syncs never occur
inside capture -- the whole-chain path already disables the profiler while
capturing, since a sync is illegal there.

**Not changed for the campaign.** Round 1's v6 numbers carry the same +2.4 ms,
and the preregistration requires Round-2 cost to be comparable to Round 1.
Removing it mid-campaign would break that comparison AND silently shift `F`.
Round 2 runs as registered; the correction is a preregistration decision to be
taken with both measurements in hand.

## 8. Open items

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
