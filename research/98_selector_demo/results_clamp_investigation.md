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

## 10. Relocation to h104 (2026-08-14)

Section 9 ended the in-guest investigation: the cause lives on the physical
host under the VM and recurs across the user's other (virtualised) servers.
The campaign therefore moved to **h104** (`h104.prometheus.msr`): bare metal
-- no `hypervisor` cpuinfo flag, `systemd-detect-virt: none` -- 2x Xeon
Platinum 8480C, 8x H100. The entire hypervisor mechanism class of section 9
is structurally absent here.

**Placement constraint.** NUMA node 0 of h104 carries other tenants' heavy
CPU jobs (and their vLLM processes on GPUs 0-1). Everything of ours is pinned
to NUMA node 1, whose CPUs are 56-111 (+HT 168-223) and whose PCIe root owns
GPUs 4-7:

| role | GPU | CPUs | notes |
| --- | --- | --- | --- |
| campaign lane-a | 4 | 96-111 | only 16-wide physical node-1 range clear of the telemetry set |
| GPU telemetry sampler | — | 64-95 | hash-bound constant in `w98_host_load.py`; lands on node 1 here unchanged |
| loop + runner parent | — | 56-63 | `numactl --physcpubind=56-63 --membind=1` in `run_g98c_loop.sh` |
| X28 sentinel | 5 | 72-87 | GPUs 6-7 left free |

`--membind=1` is inherited by every child, so first-touch allocation cannot
spill to the contended node-0 pool (node 0 had 11 GB free at relocation
time; node 1 had 608 GB).

**What changed, and what did not.** The authorization was re-issued as
**v7** (`w98_g98c_authorization_v7.json`), rebinding hashes after exactly two
relocations inside `run_w98_g98b_round1.py`: lane-a is redefined for h104
(the phase-97 lane table describes the old box and stays read-only), and
`QUANT_CKPT` repoints to h104 paths. The target-matching snapshot is
**byte-identical** to the registered one (same HF snapshot
`b968826d9c46dd...`); the W4A16 draft was rebuilt on h104 with the identical
data-free RTN recipe (`research/74 make_w4a16_int4_ckpt.py`, int4 symmetric
group-128 weight-only, lm_head ignored — `recipe.yaml` in the checkpoint
records it), which needs no calibration data and so cannot drift with the
box. The prereg documents, lattice, model, both gates, and sampling are
untouched; `validate_authorization` enforces that by construction.

**Closed gates became box-bound records.** G98-A's package embeds a live
`is_dir()` of the old box's checkpoint (two tests now skip off that box), and
G98-B's v6 package hash-binds the pre-relocation runner bytes, so its
round-trip test now asserts the guard REFUSES — v6 must never authorize new
Round-1 boots on a box it did not describe. The bytes it did authorize are
fixed in git history.

**Comparability caveat, stated in advance of any Round-2 number.** Round 1
was measured on the old box; Round 2 will be measured on h104. Prompt bytes,
measurement path, and the +2.4 ms profiler-sync overhead (section 7) carry
over unchanged, and everything Round 2 SCORES is internal to Round 2 —
anchors, sigma_repro, fits, and held-out predictions all come from h104
boots. But any comparison of ABSOLUTE per-step times across rounds now
contains a box term, and the section-7 check of what the fitted `F` absorbs
must remember `F` may also absorb host differences between the boxes.

**First bare-metal sentinel readings — CORRECTED.** The X28 twin deployed on
GPU 5 during the campaign's first attempts read canary ~5.17 ms and mmap
~4610 us; an idle re-measurement (X29, below) reads **3.21 ms and 2615 us**.
The first readings were inflated by our own campaign and warmup boots on the
same NUMA node — which is itself a finding: the mmap channel moves +76%
under same-node co-activity on h104, so old-box sentinel readings must be
interpreted against what else the guest was doing (on the VM the channel did
NOT move during engine boots; its lanes plausibly sit on a different guest
node). h104 idle references, steal identically zero: canary 3.21 ms, mmap
2615 us (1.28 us/fault vs the VM's clamped 1.92), clocksource `tsc`. The
verdict on the memory-state-amplification hypothesis still requires the VM's
OWN clean-state reading at its next flip.

**The first campaign attempt produced three FALSE 'loud' verdicts.**
Attempts 1-3 of the first anchor cell were rejected by the startup gate
(compile 14.70 / 13.02 / 15.10 s against the 12.5 s ceiling) on an idle box.
The per-attempt logs show why: the torch.compile cache directory hash
changed on every boot of the same cell. vllm hashes every registered env var
into its compile-cache key, and `VLLM_SELF_SPEC_KOFF_TRACE` carries the
attempt-numbered trace path — so each boot got a fresh key, compiled cold
(~14.7 s on these cores), and the gate read a cold cache as a loud host.
Those three rejections are cold-cache artifacts, not host-load events, and
must not be counted in any future gate tally. Two fixes, neither touching a
bound artifact's semantics: the trace path joined `ignored_factors` in
`vllm/envs.py` (an output path cannot affect compiled code), and
`warm_g98c_compile_cache.py` boots each unique configuration once before the
loop is armed — fit and held-out cells treated identically, engine
construction only, no measurement and no campaign outputs — so campaign
boots load the cache and the startup gate operates in the warm regime its
thresholds were calibrated against (warm 'Directly load' boots were part of
the old box's quiet population; the v3 re-issue exists because of them).

## 11. The clamp does not touch the probes (X28 record analysis, X29 design)

The committed sentinel record (13:45-15:52 +0900, 246 bursts) spans a
stretch the status checkpoint describes as reliably clamped, with four
campaign boot windows at the loop's ~35-minute cadence visible in the GPU
channel (GPU 0 at 74-100%, 73.7 GB resident, guest page-fault rate 25-38k/s
against a 1.5-2k/s idle floor). Across ALL of it:

* **canary: flat.** 5.13-5.38 ms, no step — idle-clamped and boot-clamped
  minutes are indistinguishable.
* **mmap: flat.** 3855-4012 us (4% total span over its 40 minutes),
  including DURING the engine's own boots — 30k faults/s of engine mmap
  storm on the neighbouring GPU does not move the channel by more than its
  noise. Contrast h104, where our own same-node boots moved it +76%.
* **steal: negligible.** 1-13 ticks per 2.5 min across 192 vCPUs.
* **co-tenants: negligible.** Other users' census CPU totals 13-22%
  throughout.

So while the engine measurably clamps (+4-5 ms/step), nothing the probes
exercise — kernel launch + spin-sync dispatch, single-threaded page faults,
CPU availability — degrades at all. Whatever the mechanism is, it lives in
a channel the engine step has and these probes lack. That kills one more
class and constrains the rest:

* **Blocking-sync wake latency is refuted by shape.** The engine issues 12
  `cudaDeviceSynchronize` per step in every regime; a per-sync wake penalty
  would add a constant ~12x per step everywhere, and post-completion wake
  delay cannot be hidden by concurrent GPU work. Observed: R4 +0.24 ms
  against R1 +5.02 ms. The added cost is proportional to EXPOSED HOST TIME,
  not to sync count.
* Three candidates fit everything and are exactly the engine-vs-probe gap:
  1. **Physical-host LLC/DRAM latency contention** (other guests on the
     socket). Slows the engine's large pointer-chasing host working set
     (~33 ms/step of Python and framework at batch 1; +15% of it is the
     whole clamp), spares the canary's cache-resident loop and the mmap
     probe's streaming page-zeroing, produces no steal, no clock effect,
     flips on neighbour-workload timescales, and is hidden wherever host
     work overlaps GPU work — the observed regime ordering.
  2. **Timekeeping fast-path degradation** (kvm-clock vDSO falling back to
     syscall when the host clears the TSC-stable bit). Two clock reads per
     5 ms probe iteration is invisible; thousands per engine step is
     milliseconds. Same hiding pattern, host-event flip cadence.
  3. **Synchronous TLB-shootdown amplification on a multi-core mm.** The
     engine's 144 munmaps/step must IPI every vCPU its threads occupy —
     vCPU kicks whose cost depends on host state. The X28 mmap probe sends
     none (single-threaded mm), and X24's 1.05 ms/step munmap figure sits
     in the one nsys profile that had no clean twin.

**X29 (`probe_w98_clamp_channels.py`) gives each candidate its own
channel**: serial pointer-chase at 1 MiB / 32 MiB / 512 MiB working sets
(contention steps the big chains, spares the small one); ns per monotonic
clock read with getpid as syscall baseline (vDSO loss is a ~20x step);
timed munmap of a mapping resident in 8 pinned toucher threads' TLBs
(shootdown cost); spin-vs-block GPU wait (wake latency, the control); plus
X28's canary and mmap unchanged for continuity, and the box snapshot.

h104 idle references (GPU 5, CPUs 72-87, node 1): chase 9-12 / 76-83 /
119-126 ns per hop, clock 52-98 ns/read, getpid 120-233 ns, munmap_mt
143-210 us, wake overhead ~0.7 us, canary 3.21 ms, mmap 2615 us,
clocksource `tsc`. Running continuously as `channels_h104.jsonl`.

**X29b: the clock-read census promotes candidate 2 to prime suspect.** An
LD_PRELOAD shim counting `clock_gettime`/`munmap` around a counted window of
batch-1 engine steps on h104 (`probe_w98_clock_reads.py`,
`x29b_h104.json`): the engine makes **~150-168k clock reads per step** —
one every ~230 ns of wall time — and the count is intrinsic, not
instrumentation (profiler off: 150k). Decomposition: libcuda's sync
spin-wait reads at ~6.7/us, and spin reads cannot amplify (the spin exits on
GPU completion regardless of read cost), but even granting 8-15 ms/step of
sync wait, the SERIAL path carries ~68-114k reads per step. The arithmetic:
**+40-66 ns per clock read reproduces the entire +4.5 ms clamp.** That is a
mild pvclock slow path — extra seqlock retries while the host updates
pvclock data, nowhere near a full syscall fallback (which would add
150-300 ms/step and is excluded by the observed magnitude). Every property
fits: serial-path cost (hidden exactly where host work overlaps GPU work —
the regime ordering), fixed per step, box-wide, zero steal, invisible to a
canary that reads the clock twice per 5 ms iteration, and flipping on
host-event cadence. Candidate 1 (LLC contention) remains live; candidate
3's substrate is VM-specific — **h104 makes ZERO munmaps per step**, so the
144/step storm X24 saw is the old box's allocator behaviour, not the
engine's intrinsic cost.

Independent of the clamp: ~160k reads x ~25 ns is ~4 ms of every batch-1
step — 10% — spent reading the clock on a healthy box. Like section 6's
mmap note, worth fixing on its own merits once the campaign closes.

**Deployment ask for the old box** (this is where the answer is): pull this
branch and run, pinned clear of the campaign lane —

    CUDA_VISIBLE_DEVICES=1 taskset -c 96-111 \
      .venv/bin/python research/98_selector_demo/scripts/probe_w98_clamp_channels.py \
      --out research/98_selector_demo/data/probe_flip_sentinel/channels_vm.jsonl

The first burst in a CLAMPED window plus the first burst after a clean flip
settle it: the channel that steps names the mechanism; if none steps while
the gate still clamps, the mechanism is narrower than all three and the
next probe is designed against THAT null. Expected VM signatures —
clocksource reads `kvm-clock`; chase values well above h104's given
virtualised EPT walks; the discriminator is the within-box step at the
flip, never the cross-box level. Under X29b's arithmetic the sharpest
prediction is the clock channel: a step of a few tens of ns per read
between clean and clamped windows — trivially resolvable, the channel
averages 200k reads per burst — confirms the mechanism AND quantifies it
(observed engine clamp per step should equal the per-read step times the
serial read count).

## 12. Open items

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
