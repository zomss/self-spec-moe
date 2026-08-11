# Plan — remaining checks on measurement spread

Date: 2026-08-11

Status: **plan only. No GPU command is authorized by this document.** Each
item below states its trigger, cost, and what it would settle.

## What is already established

- The V13 certification failure was **CPU dispatch contention**, not a GPU or
  lane effect. Post-filter median per-block offsets were +0.099%/+0.208%/
  −0.133%, the solo block was marginally the slowest, and the two blocks that
  disagreed most sat on the same GPU.
- The mechanism is specific: the PIECEWISE draft chain is host-launch-bound
  (~98 launches and ~16 ms host per step against ~5.3 ms GPU). All four
  blowouts landed in `k4`/`w512`; `off`, which replays one captured graph, had
  zero.
- Core reservation works. All four failing cells improved 9× to 251×, and
  rounds below the 95% floor fell from six to one. See
  `results_p4_b0_pinning_probe.md`.

## What is not established

1. **The noise is not eliminated.** `w512/R5/s0` returned 9.8%, a cell that
   passed in V13. One over-5% cell instead of four is a large improvement, but
   certification requires all 36 cells inside 2%.
2. **Attribution is still inferential.** No run records host load, so an
   episode is diagnosed by reasoning after the fact rather than measured.
3. **Coverage is partial.** The probe exercised block 1's two draft boots
   only — 24 of 108 cells. `off` boots and blocks 2 and 3 are unprobed.
4. **The residual sources are not ours.** `clangd` (~50%) and
   `zed-remote-server` belong to another user and remain unpinned. `taskset`
   on our own processes reserves nothing against them.

## Checks, in the order they earn their cost

### S1 — record host load per round (no GPU time, highest value)

Sample `/proc/stat` and the lane's per-CPU counters at each round boundary and
write them into the existing per-boot observation record: context switches,
involuntary switches, run-queue depth, and steal time for the reserved cores.

This converts contention from an inference into a covariate. A future episode
becomes attributable on the spot instead of requiring the reconstruction this
session went through. It also makes S2 and S4 interpretable.

Trigger: before the next scored run. Cost: CPU-only, no new GPU work, no
frozen-contract change — the per-boot observation record is already outside
the schema-locked capture.

### S2 — quiet-box preflight gate

Before any scored run, measure a short fixed calibration burst on the reserved
lane and compare against a registered reference. Refuse to start when the box
is loud, the same way the worktree-quiescence preflight refuses a dirty tree.

This prevents spending 7.7 h to discover the box was busy, which is exactly
what V13 did. Trigger: with S1, since it needs the same counters. Cost: about
one minute per run.

### S3 — threshold robustness, offline

Using the preserved V13 rounds and the probe's 96 captures, compute how many
rounds per cell the frozen 95%/2% rules need to tolerate one episode. The
question is concrete: at four rounds, one bad round leaves the filter keeping
the fast mode and shifting the cell upward — the one-sided-filter artifact
that produced the 2.4% failure.

This answers "more rounds per cell?" with arithmetic rather than another GPU
campaign. Note it would be a **preregistration change** if adopted, not a
config tweak. Trigger: any time. Cost: offline only.

### S4 — contention dose-response

At two fixed cells, vary the Lean server load (0, 1, 4, 8 repls) with
everything else held constant, and measure spread against load.

This turns the mechanism from inferred to characterized, and would produce a
defensible statement of how quiet the box must be. It is the experiment that
would let a future run *predict* rather than discover its noise. Trigger: only
if S1+S2 prove insufficient, or if the mechanism needs to be publishable.
Cost: about 8 short boots.

### S5 — cpuset isolation

`taskset` pins our processes but reserves nothing. A cgroup cpuset (or
`isolcpus`) would make the reserved cores genuinely exclusive, including
against `clangd`.

This is the only measure that addresses the residual sources. Trigger: if S1
attributes remaining episodes to the unpinned co-tenants. Cost: needs
administrative access; coordinate rather than command, since the processes
belong to another user.

### S6 — extend coverage to the unprobed cells

Probe `w512/R5/s0` specifically, plus one `off` boot and one block-2 or
block-3 boot, to confirm the mitigation holds where it has not been tested.

Trigger: if a re-screen is deferred. If a re-screen runs, it supersedes this —
it covers all 108 cells by construction. Cost: ~2 boots.

## Recommended sequence

S1 and S2 before the next scored run; S3 offline in parallel since it is free.
Then re-screen. Hold S4, S5, and S6 unless S1 shows the residual is still
biting.

The single most valuable item is **S1**: every other question in this document
becomes easier to answer once each round carries its own host-load record.
