# G98-E design — D3, end-to-end selector value

Source: `w98_prereg.md` §D3, unchanged. Written 2026-08-15, after G98-C
(cost, scored) and G98-D (acceptance, scored). Nothing here is a
measurement.

**Gate letter.** D3 runs as **G98-E**, not G98-D. The registered gate order
predates the Round-2 amendment, which split cost into its own round and
shifted the executed letters by one; the drift is recorded in the phase
README. The CLAIM is unchanged.

## The registered claim

> On the declared workload mix, the selector's chosen configuration per
> regime (boot-static composition plus the live K/OFF ladder) achieves at
> least **90%** of the omniscient per-segment composite, and its LCB beats
> every static single configuration including OFF under the **+2% LCB**
> rule, in decode currency.

## The measurement is mix-independent; only the verdict is not

Everything D3 needs reduces to one grid:

    rate(regime, configuration)   in decode currency, under equal work

from which all three quantities are aggregations over a mix `v`:

| quantity | definition |
| --- | --- |
| selector | `sum_R v_R * rate(R, selector_choice(R))` |
| omniscient per-segment composite | `sum_R v_R * max_C rate(R, C)` |
| static single `C` | `sum_R v_R * rate(R, C)`, one `C` for every regime |

So the grid can be measured NOW and the verdict computed later, once the mix
is frozen. That ordering is deliberate: it makes it impossible to choose the
mix after seeing which configuration wins where.

## The blocking gap: the workload mix is not declared

`w98_prereg.md` says "the declared workload mix" but no artifact declares
it, and `data/prereg/` carries none. This is load-bearing, not cosmetic:
**the mix determines the answer.** Phase 82's omniscient per-regime switcher
scored only **+1.7% over static-OFF** because the single spec-favorable
regime carried 9% of trace tokens — a different mix over the same
measurements would have produced a different verdict.

Three routes, to be recorded before aggregation:

1. **Equal weight over the six regimes.** Neutral and precedented (Phase
   97 freezes "a scored, equal-weight W3 research objective"), but it
   asserts a workload no deployment has.
2. **Volume-weighted from a real trace**, as Phase 82 did. Strongest, and
   the only route that makes the 90% target mean what it says, but it needs
   a trace declared as the deployment target.
3. **Report the verdict as a function of the mix** — the frontier over
   which mixes clear 90% and which do not. This cannot be gamed after the
   fact and is arguably the most informative, but it answers "when does the
   selector pay" rather than the registered yes/no.

**Recommendation: freeze (1) as the registered headline for comparability
with the phase's own equal-weight precedent, and report (3) alongside it**,
since the frontier is free once the grid exists and directly sizes Phase
97's engineering — which is what §D3 says the composite gap is for.

## Grid to measure

**Configurations.** The selector's per-regime choices, every static single
lever, the compositions those choices draw from, and **OFF** (K=0, no
speculation) as the baseline the claim must beat. One boot measures ALL six
regimes, as in G98-C/G98-D, so the boot count is per configuration, not per
(configuration, regime).

**Currency: decode-only**, `T(1+N) - T(1)`. Phase 96's I1 found C2's map is
decode-only while Phase 95 measured wall clock, and mixing the two produced
a retracted claim; at R5 prefill is 46.6% of wall. D3 is registered in
decode currency and must not be mixed with wall.

**Equal work.** Phase 96 found that under natural EOS the speculative and
autoregressive arms generate different-length text, so the batch drains
asymmetrically; the screen found 140/726 cells contaminated and re-measuring
moved one MoE cell from 1.288 to 0.954. D3 runs fixed-length
(`ignore_eos`), which is what G98-C and G98-D already did.

**Dual arm (amendment 2).** The decisive comparison runs under BOTH
instruments — corrected (default) and legacy
(`VLLM_SELF_SPEC_PROFILE_LEGACY_SYNCS=1`) — and BOTH numbers appear wherever
the D3 result appears. This matters here more than anywhere else in the
phase: the inner-sync cost is paid only by ARMED configurations, and OFF is
one of the baselines D3 must beat, so the instrument choice moves exactly
the comparison being scored.

**The live K/OFF ladder** rides on top of the boot-static composition, as
registered. It is the one genuinely live axis in this phase.

## Box

**h103**, GPU 7, NUMA node 1 — verified bare metal (`systemd-detect-virt:
none`, no hypervisor flag, clocksource `tsc`), all eight GPUs idle, no
co-tenants, driver 580.65.06 matching h104 so the locally rebuilt
`_C_stable_libtorch` carries over.

`/h/v-sukmincho` is ONE NFS tree shared by both boxes, not a clone: the
repo, venv, checkpoints and output directories are the same files h104
sees. Two boxes must therefore never write the same campaign directory
concurrently. `/tmp` is box-local, so h103's compile cache starts cold and
is warmed exactly as h104's was.

## Decision criteria to leave the design stage

* The workload-mix route recorded (blocking the VERDICT only, not the grid).
* An h103 lane and a G98-E authorization issued, with the grid frozen
  before the first scored boot.
