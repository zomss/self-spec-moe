# When is switching worth anything? An exact decomposition, and the axis that would pay

Record of 2026-08-16. Analysis over the measured G98-E grid; no new
measurement. Script: `scripts/analyze_w98_switching_law.py`; record:
`data/g98_e/d3_switching_law.json`.

Companion to `results_d3_failclosed.md`, which raised per-regime switching
from +1.4% to +6.2% over the best static. +6.2% is still not much. This
record says exactly why, and what would have to be true for switching to be
worth more.

## The law

With time-weighted aggregation `1 / sum(v_R / rate_R)`, the gain of any
per-regime assignment over any fixed configuration is **exactly**

```
gain = sum_R  t_R * ( rate_sel(R) / rate_static(R) )

t_R  = (v_R / rate_sel(R)) / sum_R' (v_R' / rate_sel(R'))
```

where `t_R` is the selector's **time** share in regime R — the fraction of
wall time the mix spends there — not its token share. Verified numerically
on both instrument arms; identity residual `2.2e-16`.

The law is not deep, but it is decisive, because `t_R` is inversely
proportional to the regime's own throughput. **Slow regimes dominate the
time budget.** Switching can only pay in proportion to time spent in regimes
where the static is wrong.

## Applied: why +6.2% and not more

Fail-closed selector against the best static `w4a16-quantized/w1024/skip4`,
equal-weight mix, corrected instrument:

| regime | time share | ratio sel/static | excess contribution |
| --- | --- | --- | --- |
| **R1** (b1, short) | **0.5855** | 1.0024 | **+0.0014** |
| **R4** (b8, 8k) | 0.1523 | **1.3922** | **+0.0597** |
| R5 (b8, 14k) | 0.1036 | 1.0079 | +0.0008 |
| R5cot (b8, 14k) | 0.0912 | 1.0052 | +0.0005 |
| R6 (b32, short) | 0.0248 | 0.9504 | -0.0012 |
| R8 (b16, short) | 0.0426 | 1.0148 | +0.0006 |
| | | | **= 1.0618** |

**One regime carries 59% of the time budget and contributes +0.14%.** R1 is
batch 1 at 178 tok/s, so an equal token mix spends most of its wall clock
there — and at batch 1 every quantized configuration performs within 0.3% of
every other. The mix's dominant term offers nothing to switch between.

Meanwhile R4 delivers the entire gain (+5.97%) off 15% of the time, because
it is the one regime where the right answer is categorically different:
don't speculate.

The ceiling confirms there is nothing left to win here. The **omniscient**
per-regime switcher scores 1.0664x over the same static; the fail-closed
selector's 1.0618x captures **93% of all switching value the grid contains**.
The limitation is not the selector. It is the grid.

## The two conditions

From the law, switching gain requires a regime that is simultaneously

1. **time-dominant** — high `t_R`, i.e. slow, i.e. low batch or long context; and
2. **optimum-distinctive** — the best static is materially wrong there.

This grid satisfies (1) at R1 and (2) at R4, and never both at once. That
single sentence explains the whole "switching is marginal" finding, and it
also says the finding does not generalize: it is a property of which six
regimes were measured, not of switching.

## The axis that would pay: mixed feasibility

The law above assumes every configuration is deployable in every regime. It
is not. The `w4a16-quantized` draft is a **separate ~6.1 GB resident**, and
under KV pressure that memory is not available. When the lever set is not
uniformly feasible, no single static can serve the whole mix at all — and
switching stops being a marginal ranking improvement and becomes a
feasibility requirement.

Scoring that case on the measured grid, with 14k regimes modelled as unable
to host the quantized draft:

| | corrected | legacy |
| --- | --- | --- |
| best static feasible everywhere | `target-matching/w512/skip4` 460.5 | `target-matching/woff/skip0` 459.9 |
| per-regime selection | 601.0 | 574.6 |
| **switching gain** | **1.305x** | **1.249x** |

**5x the uniform-feasibility gain**, and for a structural reason rather than
a lucky mix: the best static must be feasible everywhere, so it surrenders
the quantized draft in every unpressured regime — giving up ~22% there —
while the selector takes `w4a16` where it fits and `target-matching` where
it does not.

Note this is the *opposite* of the usual argument for switching. It does not
depend on regimes disagreeing about which lever is best; it depends on the
lever set being unavailable in some regimes. That is a much more robust
premise, and it is exactly the deployment condition a serving system faces.

**This is an estimate, not a measurement.** The rates are measured; the
feasibility mask is modelled. KV pressure was never varied in this phase —
every boot ran at `gpu_memory_utilization=0.90` with the draft resident — so
this sizes the effect and does not establish it. Two things it does not
capture: under real pressure the rates themselves shift (less KV cache
changes achievable batching), and the switchover point in KV working-set
size is unknown.

## Recommendation

The next campaign should vary KV pressure, not content and not a second
model. It is the only axis in the record that is (a) well-posed, (b)
predicted to move the headline by a factor of 5 rather than a few percent,
and (c) tests a claim the paper actually needs — that a serving system must
switch because its lever set is not uniformly affordable, which is a
stronger and more defensible claim than "regimes prefer different levers".

Registered prediction, to be committed before that campaign runs: **at KV
working-set sizes where the quantized draft cannot be resident, per-regime
selection beats the best uniformly-feasible static by >= 1.20x**, against
this record's modelled 1.305x.
