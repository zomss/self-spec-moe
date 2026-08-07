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

The bias tracks step length: worst at b1/short-context (+15…+28%, steps
≈5 ms), near zero at b8/14k-context (−6…+9%, long steps). That is the
signature of a **fixed per-region cost** — the profiler's CUDA syncs —
which is negligible on long steps and dominant on short ones. It
explains why W8 §3b passed at b32 on MLA/MoE (11.5 ms steps, +1.5%/
−0.3%) and fails here.

**What is NOT refuted:** the τ* formulation itself. τ* is algebraically
the denominator of the identity; it is correct by construction, and
τ*_true = τ/S_actual is a well-defined quantity that behaves exactly as
the design predicts (P-W13c: content-free to ±8%). What is refuted is
that the *current instrument* measures it accurately enough to decide.
The unperturbed profiler flagged as OPEN in `w12_search_spec.md` §7 is
now a **blocker**, with a measured magnitude and a demonstrated
consequence.

## Two findings that survive and change the design

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
