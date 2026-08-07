# W13 results — τ* does NOT yet reproduce the dense selector

Pre-registration `w13_tau_star_validation.md` (1dbb193a9), scorer
`score_w13.py` committed before the data (a2f0d9…). Artifacts
`data/w13/`. **Scope: 3 of 5 compositions** — both skip variants
(`s-2,8/w512`, `s-2,8/w2048`) hung at Humming JIT prewarm, twice, each
burning a 90-minute timeout with the log ending silently after draft
load (the documented odd-width/prewarm wedge). Stopped rather than
spend three more hours; the s-none window sweep is complete and
decides the question.

## Verdicts

| # | prediction | outcome |
|---|---|---|
| P-W13a | predicted S within ±10% | **REFUTED** — 9/18 cells |
| P-W13b | ranking reproduced (**the decision test**) | **REFUTED** — 2/6 top-1, 1/6 exact |
| P-W13c | τ* content-free (R5 vs R5cot) | **CONFIRMED** — agree within ±8% |
| P-W13d | windowed D flat, full-KV D grows | **REFUTED as stated** — both flat at b1 |
| P-W13e | τ* spread ≥15% | **SPLIT** — 1–8% at b1, 30–49% at b8 |

**The redesign does not ship on this instrument.**

## The cause: the profiler, not the formulation

τ* over-estimates cost systematically, and the bias is **cell- and
composition-dependent**:

| comp | mean bias | range |
|---|---|---|
| s-none/w512 | +11.1% | +0.9 … +21.9% |
| s-none/w2048 | +17.3% | −5.9 … +28.5% |
| s-none/woff | +12.2% | −5.1 … +25.6% |

A *uniform* bias would be harmless — ranking compares compositions
within a cell, so any constant scale on τ* cancels. What breaks the
ranking is the **differential** bias at a single cell. At R1 b8:
w512 +7.7% vs w2048 +27.6% vs woff +25.6% — a 20-point spread, and the
model duly picks w512 where reality picks woff.

**Correction (2026-08-08): the "fixed per-region cost" explanation
first written here is REFUTED by its own test.** If the bias were a
constant additive overhead Δ per armed step, then Δ = bias_abs × T
would be constant across cells. Measured: mean 3.48 ms, sd 3.24,
**CV 93%**, range −4.15…+6.87 ms — not constant, and it changes sign.

What the data does show is that bias tracks the **GPU-work fraction**,
not step length per se:

| KV share of the draft step | mean bias | spread across comps |
|---|---|---|
| 3.4% (R1 b1) | +23.1% | 8.8 pts |
| 22.0% (R1 b8) | +20.3% | 19.9 pts |
| 31.7% (R5 b1) | +12.4% | 12.4 pts |
| 78.8% (R5 b8) | −3.4% | 6.9 pts |

Leading explanation: sync-bracketed regions **destroy CPU/GPU
overlap**, so summing them counts serially what the engine pipelines.
The over-count is proportional to the CPU-side work that would
otherwise be hidden — largest when the step is GPU-light, vanishing
(slightly negative) once GPU work dominates. Consistent with sync
semantics, with the cell-level trend, and with W8 §3b passing at b32.
**NOT explained: why the bias differs BETWEEN COMPOSITIONS at one
cell** — and that differential, not the level, is what breaks the
ranking. Recorded as open rather than replaced with an untested story.

### Mechanisms TESTED and REFUTED (2026-08-08)

| hypothesis | test | outcome |
|---|---|---|
| fixed overhead per armed step | Δ = bias_abs × T constant? | **REFUTED** — CV 93%, sign changes |
| over-counts the KV READ (user) | excess ∝ draft KV bytes? | **REFUTED** — corr **−0.44**, i.e. excess SHRINKS as KV grows |
| simple overlap, excess = min(CPU,GPU) | excess rises then plateaus with GPU work? | **REFUTED** — observed excess FALLS |

Extremes for the KV test: w512 @ R1 b1 reads 0.075 GB and over-counts
**+5.06 ms**; woff @ R5 b8 reads 17.16 GB and UNDER-counts **−4.15 ms**.

**Best-fitting hypothesis (not yet isolated): TWO OPPOSING ERRORS.**
(a) over-count from serialization — syncs expose CPU work (launches,
attention-metadata prep, sampling, bookkeeping) that serving pipelines
away; dominates when the step is GPU-light. (b) under-count from
UNINSTRUMENTED work — the two regions do not span the whole armed step
(scheduler, rejection sampling, output processing sit outside and grow
with batch); W8 measured exactly this on MoE at b32, where regions
covered 78% of the step. They cross over near R5 b8, which explains the
sign change and why no single variable correlates.

**The experiment that separates them** (small, do before the CUDA-event
rewrite): add ONE region spanning the entire armed step. Then
full-step-region vs sum-of-sub-regions measures the uninstrumented gap,
and CUDA-events vs syncs on the same regions measures the serialization
error — both directly, instead of inferred from residuals.

**What is NOT refuted:** the τ* formulation itself. τ* is algebraically
the denominator of the identity; it is correct by construction, and
τ*_true = τ/S_actual is a well-defined quantity that behaves exactly as
the design predicts (P-W13c: content-free to ±8%). What is refuted is
that the *current instrument* measures it accurately enough to decide.
The unperturbed profiler flagged as OPEN in `w12_search_spec.md` §7 is
now a **blocker**, with a measured magnitude and a demonstrated
consequence.

## Two findings that survive and change the design

**0. The deeper reason Round 1 fails at short-step cells: for THIS
pool there is almost no cost signal to use.** The window lever acts
only on the KV/attention term, whose share of the draft step's bytes
runs 3.4% (R1 b1, 1.1k total KV) → 22% (R1 b8) → 31.7% (R5 b1) →
78.8% (R5 b8, 116k total KV). At b1 the draft reads ~4.6 GB of weights
against ~0.16 GB of KV, so changing the window moves 3% of the step.
Round 1 cannot eliminate what does not differ, at any instrument
quality. This generalises: **window and kv-quant act on the KV term
(signal only at high total-KV); quant and skip act on the WEIGHT term
(signal at every batch, strongest at low batch where weights
dominate).** Round 1's power is therefore a function of which levers
are in the pool, evaluated per cell.

### LEVER LEVERAGE (user, 2026-08-08) — the governing principle

**A lever's maximum cost leverage = the share of the step's memory
traffic attributable to the term it modifies.**

| cell | weight share | KV share | quant caps | window caps | skip 8/36 caps |
|---|---|---|---|---|---|
| R1 b1 | **96.6%** | 3.4% | ~97% | ~3% | ~22% of both |
| R1 b8 | 77.9% | 22.1% | ~78% | ~22% | ~22% |
| R5 b8 | 21.2% | **78.8%** | ~21% | ~79% | ~22% |

Quantization acts on weight bytes; window and kv-quant on KV bytes;
layer-skip removes whole layers so it cuts BOTH — leverage k/L at every
cell, making it the universal cost lever. Measured D/T spread across
the three windows confirms it exactly: 4%/7%/2% at the low-KV cells
(noise) vs **97% and 110%** at R5 b8 and R5cot b8.

**Independent confirmation from W6.** R1's total context is ~1.1k and
W6's R1 winner uses w2048 — a window that never binds at 1.1k, i.e.
the winner is effectively skip-only with the window inert. R5 and
R5cot (14k, b8) both won with w512, where the window is active. W6's
measured winners already obey the leverage principle: low total-KV →
weight-term lever wins; high total-KV → KV-term lever wins.

### VERDICT RESTATED

This section first read "Round 1 fails at short-step cells". The honest
statement is **"the WRONG LEVERS were tested at short-step cells"**.
Three compounding errors, all in the experiment design:
1. the pool was copied from W6 to enable comparison, and both of its
   weight-term (skip) arms hung — execution stripped the informative half;
2. the intended skip variant was count-2 (5.5% leverage) when W6's own
   ladder had already shown count-8 (22%) is the R1 winner;
3. **quantization — the ~97%-leverage lever at b1 — was held FIXED**
   across all five compositions.
At b1 the experiment varied a 3% lever, weakly varied a 5.5% lever, and
held the 97% lever constant. No instrument quality could have rescued
that cell. The instrument bias (+23% at R1 b1) is real but was NOT the
binding constraint there.

The correct b1 re-run varies QUANTIZATION (W4A8 / W8A16 / FP8 /
bf16-self), where τ* differences are first-order.

**1. At b1, cost does not discriminate compositions at all.** τ* spread
across the three windows is 1–8% at b1 but 30–49% at b8/14k. The
compositions are nearly cost-identical at low batch, so Round 1 cannot
narrow the pool there — **at b1 the window is purely an ACCEPTANCE
lever, and Round 2 does all the work.** The "span τ*" pool criterion
only has content at high total-KV cells.

**2. The window's cost benefit scales with TOTAL KV tokens, not
context.** P-W13d predicted full-KV draft cost would grow with context
while windowed stayed flat. At b1 *both* are flat (w512 −3.6%, woff
+2.2% from ~1k to ~14.5k ctx) — because at b1 the KV read is negligible
against an 8B weight read. The separation appears only at b8/14k, where
D/T is 0.25 (w512) vs 0.50 (woff), a 2× difference. So the governing
variable is batch × ctx, which **supports the (requests, total-KV)
index adopted in W12 §0 over the (b, ctx) form** — and means the
closed-form window crossover is real but only above a total-KV
threshold, not at every cell.

## What this costs, and what to do

The τ*-based Round 1 cannot be validated on dense until the instrument
is replaced. Options, in order of cost:

1. **CUDA-event profiler** (no syncs; record events around each region
   and read elapsed asynchronously). Removes the fixed per-region cost
   that causes the bias. This is the real fix and it is not large.
2. **Long-step cells only** — τ* is already accurate where steps are
   long (b8/14k: bias −6…+9%). A Round 1 restricted to high-total-KV
   cells would work today, which is also exactly where finding (1) says
   cost discriminates. That is a defensible interim scope: *use τ* where
   it is measurable, use Round 2 everywhere else.*
3. Re-run the two skip compositions once the Humming prewarm wedge is
   worked around, to complete the 5-way ranking test.

Interim honest statement for the paper: Round 1's constructive form is
validated on long-step cells (W8 §3b at b32: ±1.5%) and **not** on
short-step cells (W13: differential bias up to 20 points, ranking
2/6). The two-round design still stands — Round 2 measured acceptance
and decided correctly in W6 — but the claim that Round 1 can be made
fully constructive is scoped to where the instrument is accurate.
