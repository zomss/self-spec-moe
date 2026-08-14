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

## 2. The refutation table

| # | hypothesis | test | result |
| --- | --- | --- | --- |
| 1 | mid-measurement CPU starvation | X19, X19-rerun | runqueue wait ≤0.07% |
| 2 | lane CPU pinning | X20, X20-rerun | pinned/unpinned ratio 1.001, 1.003 |
| 3 | chassis power from neighbours | X21 | 3 clean boots at 524+542 W |
| 4 | thermal capping | X21 | 42–69 °C, clocks at max |
| 5 | GPU-0-local fault | X22 | lane-b (GPU 1, CPUs 96-111) clamps identically |
| 6 | neighbour GPU activity | run 4 | clamped with neighbours at 0% / 125 W |
| 7 | campaign code path | X23, X23-rerun | both paths clamp, interleaved, twice |
| 8 | our own power cap | live telemetry | cap fires during the regimes that DON'T clamp |
| 9 | campaign GPU telemetry | v5 re-issue | telemetry disabled, campaign still clamps |
| 10 | co-tenant compute / power | X26 gemm | clean under 100% SM, 700 W on GPUs 2+3 |
| 11 | co-tenant memory bandwidth | X26 membw | clean under 100% mem utilisation |
| 12 | co-tenant residency (24.5 GiB) | X26 hold | clean; 76 GB untested |
| 13 | any co-tenant at all | X23-rerun | clamps on a fully idle box (section 8) |
| 14 | the Kimina server (user-controlled co-project) | shutdown test | attempts 13-14 clamped with the server fully down |
| 15 | kernel memory management | /proc snapshot | numa_balancing=0, THP madvise-only + idle, zero compact stalls, no pressure |
| 16 | hypervisor vCPU preemption | steal tracking | steal clusters in boot windows but at ~0.2% of wall vs the ~11% required |

**The cause is not identified — but it is now localised to the hypervisor
layer (section 9), with one mechanism class left standing.**

Entry 14 deserves its context: a host-side server was the first candidate that
fit ALL the evidence, because every "idle box" reading in this investigation
was nvidia-smi-based -- GPU utilisation and residency -- and a server's
host-side existence (polling, NVML queries, request threads) is invisible to
that check while the clamp itself is a host-side delay. The user could control
this server, which made it the first co-tenant hypothesis that was actually
manipulable. Its GPU jobs exited at ~12:50 and the clamp persisted (attempts
10-12); the server itself went down at ~13:15 and the clamp persisted
(attempts 13-14, 2026-08-14). Refuted like everything else -- but the
host-side-invisible-to-nvidia-smi argument survives it, and applies equally
to the other host-side actors observed on the box (an nvitop NVML poller and
terminal/agent sessions under two other users).

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

## 8. X23-rerun: the clamp on a fully idle box

The strongest result of the investigation arrived last, by accident. After X26
and X27 ran clean for ~90 minutes on an idle box (all four GPUs 0%
utilisation, 0 MiB resident, no co-tenant processes), the campaign was
relaunched at 04:51 and its first three boots were all rejected by the
measurement gate. That inverted the conditions that made the original X23
uninformative -- both arms clamped then, on a bad box -- so X23 was re-run
immediately, interleaved, on the SAME cell (`target-matching`/`woff`/`skip0`):

| boot | path | R1 ms | vs v6 quiet | spread | verdict |
| --- | --- | --- | --- | --- | --- |
| r0_campaign | campaign | 34.19 | +12.62% | 0.0126 | clamped |
| r0_probe | probe | 33.19 | +9.32% | 0.0299 | clamped |
| r1_campaign | campaign | 34.30 | +12.98% | 0.0098 | clamped |
| r1_probe | probe | 33.64 | +10.80% | 0.0207 | clamped |

Verdict: `PATH_IRRELEVANT_BOTH_CLAMP` (`data/probe_path_ab2/`). Two
consequences:

1. **The campaign path is exonerated a second time**, now under conditions
   with discriminating power: the probe arm had been clean twenty minutes
   earlier and clamps here, alternating with the campaign arm, at the same
   magnitude (the familiar ~1.13x at R1).
2. **Co-tenancy is not necessary for the clamp.** The box flipped from clean
   to clamped between ~04:35 and 04:51 with NOTHING else running on it --
   no neighbour process, no resident memory, no host load of ours beyond the
   probes themselves. Together with X26 (manufactured co-tenant load does not
   produce the clamp), the co-tenant framing is dead in both directions:
   co-tenant activity is neither sufficient nor necessary. The correlation
   that drove the first half of the night was time-of-night, not tenancy.

What remains is a box-intrinsic, time-varying host condition that switches on
a timescale of tens of minutes, adds a fixed ~4-5 ms of host time per step,
and survives every process-level control we can apply from userspace. In ~7
hours the box offered two clean windows (~00:10-00:20 and ~03:20-04:50); the
campaign needs 70-90 continuous minutes and was gated nine times.

**Investigation stopped here by decision.** The gate detects the state
reliably (13 refutations, 0 false accepts), so the campaign can be re-armed
whenever the box is next clean. The remaining diagnostic step is not ours to
take: it needs whoever administers the box to account for what changes
host-side on that cadence (the signature and timeline above are the evidence
to hand them).

## 9. The box is a virtual machine

Found while checking CPU frequency governors during the 2026-08-14 clamped
stretch: the cpufreq/cpuidle sysfs nodes do not exist, because **this machine
is a QEMU/KVM guest** (`hypervisor` flag in cpuinfo, `systemd-detect-virt:
kvm`, QEMU DMI strings, 192 vCPUs) running on **kvm-clock**.

This dissolves the paradox the first 15 refutations built. Under kvm-clock,
time the physical host spends elsewhere still advances the guest's clock, so
host-side interference presents as guest wall time passing with nothing
visible consuming CPU -- the clamp's exact presentation. The cause was never
on this box; it is on the physical host underneath, which no in-guest
instrument can see. Every property fits: box-wide (all vCPUs share the
host), a fixed per-operation cost, flips on tens-of-minutes timescales (host
operations -- co-located guests, snapshots, backups, migration pre-copy --
run on exactly that cadence), and complete indifference to everything running
inside the VM.

Two hypervisor mechanism classes, one already refuted:

1. **vCPU preemption** registers as steal time, and steal accounting is
   active (cpu0 carries ~11 minutes cumulative). Per-lane tracking over 76
   clamped minutes shows steal clustering exactly in our boot windows -- the
   host does preempt us when we are busy -- but at ~0.2% of wall time
   against the ~11% of one core the clamp requires. Sixty-fold short:
   REFUTED (entry 16).
2. **Memory-state amplification** -- dirty-page logging (snapshot, backup,
   migration), kernel same-page merging, EPT churn -- inflates every guest
   memory-mapping operation WITHOUT registering as steal. The engine step
   performs 144 mmap/munmap cycles (one per flash-attention call); ~31 us of
   added cost per cycle accounts for the entire clamp. The launch canary maps
   nothing and holds flat while gate boots clamp, consistent with this class.
   PRIME SUSPECT, directly instrumented: the sentinel's `mmap` channel
   measures the map-fault-unmap cycle every 30 s (clamped-state reading:
   ~4003 us per 8 MiB cycle). The clean-state twin at the next flip is the
   verdict.

The escalation therefore has a precise address: the VM host operator, with
the ask to check what runs on the physical host at the flip timestamps and
whether this guest's vCPUs/memory can be dedicated. Nothing inside the guest
ever moved the clamp because nothing inside the guest is causing it.

## 10. Open items

1. ~~Resume is broken by a guard of its own.~~ **FIXED in v6**: attempt
   numbering continues past whatever traces AND logs an aborted run left
   behind (both checked, because an operator-era cleanup could delete a trace
   while its log survives), so the audit trail is retained instead of
   hand-deleted and the `_require(not trace.exists())` guard becomes the
   invariant it was meant to be.
2. ~~Telemetry is discarded on cell failure.~~ **FIXED in v6**: telemetry is
   re-enabled (the v5 campaign clamped with it disabled -- table entry 9 --
   so the sampler is refuted as a cause) and every attempt's summary is
   persisted to a sidecar the moment the child exits, before any acceptance
   decision. Rejected boots -- the ones whose GPU state matters -- now keep
   their evidence for the box escalation.
3. **The sampler lacks temperature**, which matters if thermal state drives the
   power ceiling.
4. ~~A continuous instrument spanning many boots.~~ **BUILT (X28,
   `probe_w98_flip_sentinel.py`)**: a 30-second-cadence canary mimicking the
   draft chain's host profile (800 tiny launches + synchronize per iteration,
   GPU 1, lane-b pinned) with a per-burst box snapshot (per-user process
   census, GPU state, load, pressure, vmstat). Two questions it answers that
   nothing else could: whether the clamp is vLLM-specific (does a bare launch
   loop feel it?), and what changes on the box at the flip moment. Deployed
   2026-08-14 ~13:50 with the box reliably clamped, alongside a 30-minute-
   cadence campaign loop whose first boot per attempt doubles as ground truth
   on the real workload. On the next clean flip: verify with the gate, capture
   the clean nsys twin of X24, and let the campaign complete.
