# X17/X18 — skip16 smoke, and the host-load gate it forced

Diagnostic, not scored. Nothing here amends a frozen artifact. The Round-1
runner, cost model, and `koff_runtime.py` are hash-bound by the v6 authorization
and were verified unchanged (`19da2106…`, `408d4f8f…`, `fe111b8f…`).

## 1. Why skip16 was smoked before the campaign

Round 2 extends the skip axis to 16 layers. It is not a marginal addition:
skip16 appears in **4 of 15 fit cells and 3 of 8 held-out cells — 7 of 23 scored
boots.** Three things had never been exercised at that width: the extended
`W98_SKIP_COUNTS`, the `boot_scope`-gated capture assertions, and acceptance at
`keep_frac = 0.556`.

All three smoke cells are **fit-set** cells, so nothing was revealed. The
held-out skip16 cells (`tm/w512`, `tm/w1024`, `w4a16/w512`) were untouched.

## 2. skip16 passes every structural gate

| gate | result |
| --- | --- |
| 16-layer cut accepted | 3/3 boots |
| skip set reaches the engine intact | `[1,2,4,6,7,9,11,13,16,18,20,23,25,27,30,33]`, 20/36 layers survive |
| `boot_scope`-gated capture assertion | `koff_runtime_error: False` |
| shared-KV binding at reduced layer count | 20 draft layers bind target KV |
| Marlin resolves under quant | `MarlinLinearKernel` |
| armed steps per regime | 218–441, far above the 64-step floor |

This resolves **prereg §7 assumption 1** ("skip16 may collapse acceptance…but if
the configuration cannot boot or produces too few armed steps to measure, the
fit set loses the point that carries most of `F`'s identifiability"). It boots,
and it produces ample armed steps. The fit point stands.

## 3. Cost lands exactly on the keep_frac trend

Measured on quiet boots only (§4 explains why that qualifier is load-bearing):

| ladder | skip0 | skip4 | skip8 | skip16 | predicted skip16 |
| --- | --- | --- | --- | --- | --- |
| woff, R1 | 30.36 | 27.30 | 24.43 | **18.54** | 18.50 |
| w256, R1 | 31.63 | ~28.3 | 25.35 | **19.41** | 19.07 |

The `woff` prediction extrapolates the skip0→skip8 slope (−2.965 ms per 4
layers) and lands within **0.2%**; `w256` within 1.8%. skip16 is monotone,
on-trend, and extends `keep_frac` support from [0.778, 1.0] to [0.556, 1.0] —
more than doubling the lever arm available to separate `F` from the per-layer
term, which is the entire reason Round 2 exists.

## 4. Acceptance collapses — and that is fine

| cell | accepted draft tokens per armed request | of K=4 |
| --- | --- | --- |
| woff/skip0 | 3.987 | 99.7% |
| woff/skip4 | 3.746 | 93.6% |
| woff/skip8 | 3.183 | 79.6% |
| **woff/skip16** | **1.410** | **35.3%** |
| w256/skip8 | 2.921 | 73.0% |
| **w256/skip16** | **1.236** | **30.9%** |
| **w4a16/w256/skip16** | **1.063** | **26.6%** |

Note the counters are per-REQUEST: `koff_runtime` builds `D_armed` as
`sum(row["draft_armed"])`, so `A/D` is accepted tokens per armed request,
bounded by K. Treating both as token counts yields a "rate" above 1.

Working the economics at R1/w256, with verify+overhead ≈ 11.7 ms from the
measured step times:

* skip8 — 25.35 ms chain, 3.92 tokens/step → **9.45 ms/token**
* skip16 — 19.41 ms chain, 2.24 tokens/step → **13.9 ms/token**

**skip16 is ~47% worse per token despite being 23% cheaper per step.** The
knapsack will almost certainly never select it. That does not weaken its role:
D1' fits COST, and a cell can be operationally worthless while being the most
informative cell in the design. It should not be read as a recommendation.

## 5. Two of three smoke boots were contaminated

The first boot measured `woff/skip16` at 29.56 ms — a 5 ms *regression* from
skip8 while removing 8 more layers. It was not real. The tell was context
scaling:

| cell | R1 (ctx 431) | R4 (ctx 8853) | R5 (ctx 14357) |
| --- | --- | --- | --- |
| woff/skip16 (contaminated) | 29.56 | 29.65 | 32.66 |
| woff/skip16 (re-run, quiet) | **18.54** | **25.91** | **29.29** |
| w256/skip16 (quiet) | 19.41 | 19.88 | 19.84 |

Window-**off** attention must scale with context. The contaminated boot was flat
across a 33× context range because it was clamped at a host-bound floor that hid
the GPU work entirely. The re-run recovers the expected +10.75 ms of scaling.

The cause was co-tenancy: four unpinned `VLLM::EngineCore` processes
(`Cpus_allowed_list` 0-191) were free to schedule onto lane-a's pinned CPUs
0-15, and the draft chain under the pinned piecewise runtime is substantially
host-bound.

**A clamped boot is not noisy, it is wrong, and it is wrong in a plausible
direction** — it inflates cheap configurations toward a common floor, which is
exactly the "composition costs more than predicted" signature Round 1 spent this
phase chasing. Nothing in the record would have flagged it.

## 6. The gate

Every boot already logs two pure host-side quantities before measuring anything.
They separate cleanly, with a 38% gap and nothing in between:

| population | compile_s | capture_s |
| --- | --- | --- |
| v6 campaign, all 27 scored boots | 9.72–11.29 | 8–9 |
| X17/X18 quiet boots | 9.54–9.86 | 7–9 |
| X17 contaminated boots | 15.63, 16.15 | 13, 14 |

`compile_s` is the primary threshold because it barely moves with the levers
(~1.5 s across every skip count and both quant arms), while capture *rate*
tracks skip count strongly (6.5 it/s at skip0 → 9.75 at skip16) and is kept as a
diagnostic only. Thresholds sit between the populations: **12.5 s compile, 11 s
capture.**

`w98_host_load.py` implements this; `guarded_boot()` is the whole integration
surface for a campaign runner. Three properties worth stating:

1. **Runtime-aware.** The band is calibrated for piecewise. A legitimate
   whole-chain boot in this phase took **86 s** to capture against a normal
   10.35 s compile — that is its cut-point graph budget, not contention. Such
   boots return `not_applicable`, and the runtime is read from what the boot
   *did* (`Draft chain PIECEWISE:` in the log) rather than from the environment
   it was handed — the lesson from the inert whole-chain fix, where the
   environment claimed one runtime and eight boots ran another.
2. **Fails closed.** `unknown` and `not_applicable` both raise.
3. **Keeps rejects.** `guarded_boot` returns every attempt, discarded ones
   included. A runner that retries until it likes the number and reports only
   the survivor is selecting, not measuring.

Applied to the phase's whole history — 200 boots — it reads 151 quiet, 37
not_applicable (whole-chain), 9 unknown, and **3 loud**: the two X17 boots and
one arm of the X16 lm_head probe. **All 27 scored v6 boots are quiet.** The
Round-1 residual is therefore not a host-load artifact, and the earlier
misspecification analysis stands.

Signatures are distilled to `data/host_load_signatures.json` and regression-
tested from there, because `*.log` and `research/**/traces/` are both gitignored
— a test globbing the logs would pass here and fail on a fresh checkout.

## 7. Control: reproducible, with one unexplained transient

Re-measuring the two v6 reference points tonight, interleaved with a skip16
repeat, all three passing the gate:

| cell | R1 | R4 | R5 | R5cot | R6 | R8 |
| --- | --- | --- | --- | --- | --- | --- |
| skip8 vs v6 | −0.74% | −0.15% | −0.60% | −0.66% | −0.96% | −0.71% |
| skip0 vs v6 | **+10.15%** | +0.32% | −0.13% | +0.20% | +0.75% | **+5.57%** |

skip8 reproduces v6 to **under 1% on every regime, across sessions**. That
validates both the box and σ_repro's tightness.

skip0 did not, and X19 chased it.

### The shape of the anomaly

Sorting skip0's regimes by their v6 cost shows the delta shrinking monotonically
to zero as the baseline grows: R1 +3.08 ms, R8 +1.77, R6 +0.25, R4 +0.14,
R5cot +0.10, R5 −0.07. That is not an offset, it is a **floor at ~33.5 ms** —
every regime cheaper than the floor lifted to it, every regime above untouched.
The same clamp that made the contaminated skip16 boot read flat across a 33×
context range, only milder. Its within-boot signature agrees: on a quiet box the
cheap regimes separate by batch (v6 skip0: 30.36 / 31.75 / 33.92, spread
3.56 ms), and the anomalous boot compressed to **0.73 ms**.

Across all 15 v6 boots that spread ranges 1.31–3.72 ms and tracks the window
sensibly. **No v6 boot shows the compressed signature** — a second, independent
confirmation that the Round-1 residual is not a load artifact.

### X19: the clamp hypothesis does not reproduce

I predicted this was mid-measurement host starvation, invisible to a gate that
samples compile and capture at startup, and that skip0 and skip8 would therefore
both clamp to a common floor under load. Four boots, interleaved
(skip8, skip0, skip8, skip0), on a **busier** box than X18 ran on:

| boot | R1 | R8 | R6 | spread | Δ R1 vs v6 | runqueue wait | SM clock |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r0_skip0 | 30.15 | 31.56 | 33.80 | 3.65 | −0.69% | 0.01% | 1980 MHz |
| r1_skip0 | 30.30 | 31.73 | 33.97 | 3.67 | −0.20% | 0.07% | 1980 MHz |
| r0_skip8 | 24.25 | 25.30 | 26.59 | 2.34 | −0.74% | 0.01% | 1980 MHz |
| r1_skip8 | 24.27 | 25.29 | 26.60 | 2.33 | −0.65% | 0.02% | 1980 MHz |

**The prediction failed.** skip0 reproduces v6 to −0.69% and −0.20% with normal
spreads. Worst deviation across all six regimes of all four boots is 2.02%.
Runqueue wait — measured directly from `/proc/self/task/*/schedstat` field 2,
the kernel's own count of time runnable-but-not-running — never exceeds 0.07%,
so there was no starvation to find. SM clocks sat pinned at 1980/1980 MHz
throughout, eliminating thermal capping independently.

Initialisation was identical across the anomalous boot, X19's clean boots, and
v6: 53.89 GiB KV cache, 392,432 tokens, 9 s capture, 0.55 GiB of graphs.

### What this leaves

The X18 skip0 boot is a **single unexplained transient**, not a reproducible
property of the cell, the box under load, or the lever. Six boots of these two
cells across two sessions agree with v6 to within 0.74%; one did not, by 10%,
with a clamp-shaped signature and no identifiable cause.

Two claims from the first version of this section are **withdrawn**:

* That the effect is lever-dependent. skip0 reproduces fine; X18 confounded the
  cell with the moment it was measured.
* That σ_repro understates *cross-session* reproducibility and R2 numbers cannot
  be compared to v6. Cross-session agreement is ≤0.74% in five of six boots, so
  §3's ladder — which uses v6 `woff` reference points — stands as written.

What survives is narrower and still worth acting on: **a boot can be corrupted
in a way the startup gate cannot see**, since the anomalous boot passed it. The
runqueue instrument now exists but has only ever read ~0, so it is unvalidated
against a positive case. The cheap-regime spread did flag the bad boot, but it
varies 3× across cells for legitimate reasons and there is exactly one clamped
example to calibrate against — enough to **record** per boot as a diagnostic,
not enough to **gate** on.

## 8. Consequences for the campaign

1. skip16 stays in the R2 lattice. Prereg §7 assumption 1 resolves favourably.
2. Every R2 boot must pass the startup gate, with rejected attempts retained.
3. Record cheap-regime spread and runqueue wait per boot as diagnostics, so a
   transient like X18's is auditable after the fact. Do not gate on either yet.
4. **The Round-2 runner does not exist yet.** `run_w98_g98b_round1.py` is
   hash-bound and closed, and its package ID, prereg paths, and output directory
   are module constants. R2 needs its own runner, and the R2 model needs its own
   module — `w98_cost_model.py` is likewise hash-bound. This build sits between
   the G98-C authorization and the campaign; it was missing from the plan.
