# No Universal Draft: A Measured, Audited Selector for Training-Free Self-Speculative Decoding

> Assembled draft (2026-07-18) from research/79_paper/sections/. Figures:
> research/79_paper/figures/fig1-fig8. Every number traces to a committed
> phase artifact (traceability table in outline.md).

## Abstract

Which self-speculative draft — quantized, windowed, layer-skipped,
expert-restricted, or none at all — should a deployment run? We treat the
question as a measurement problem. Speculative payoff factors into a cost
ratio R and an acceptance rate β; we measure both surfaces once, offline,
with calibration gates against ground truth, for seven lever families
across batch, context, and three architectures (dense, MoE, MLA). From
the surfaces, selection is exhaustive and free: a term-decomposed cost
model (held-out lever error 2.2%; one-anchor architecture transfer 9.3%)
prices every configuration including compositions (β_combo ≈ Πβᵢ, median
deviation 0.01, with mechanistic exceptions) and OFF. The map's structure
is non-trivial: crossovers inside architectures, almost nothing portable
across them (fp8 weight-quant excepted, with a measured QK-norm rule for
KV-cache quantization), OFF regions that survive exhaustive search over
the full combination space, and profiled lever forms that change
decisions — frequency-profiled expert selection flips MoE cells and the
profitable profiling axis is itself architecture-determined. Two
execution laws (captured attention schedules are replay-safe iff geometry
is constant; compacted decode steps must trim rejected-slot KV) remove
the draft-chain launch floor that gated composed configurations,
delivering 1.91× at the map's registered roofline (delivery ≈ 100%) and
2.77× at long context. The final selector prices REALIZATIONS under
per-source uncertainty (lower confidence bounds, feasibility, fitted
per-chain execution constants), emits its own measurement queue, and in
its first audited round the end-to-end layer corrected the offline layer
three times — including retiring one of our own levers whose gate proved
circular. A 91-GPU-minute profiling protocol recovers the map within
2.6% of a full profile. Profiles nominate, measurements confirm; the
audit loop is the contribution.

---


# §1 Introduction


A practitioner serving model M at batch b and context c faces a concrete
question: *should this deployment self-speculate — draft with a cheapened
copy of itself, no trained head, no second model — and if so, with which
cheapening?* The menu is real: weight quantization, windowed attention
over shared KV, draft-only KV-cache quantization, layer skip, restricted
expert routing. Each has a paper arguing for it. None of those papers
answers the practitioner, because the answer changes under their feet: a
lever that wins at batch 1 loses at batch 32; a lever that is free on one
architecture destroys acceptance on another; in whole regions of the space
the right answer is not to speculate at all.

This paper treats that question as a measurement problem. Speculative
payoff factors into two surfaces — a cost ratio R (how much cheaper the
draft's step is) and an acceptance rate β (how often the target keeps the
draft's tokens) — and both are measurable once, offline, per lever and
architecture, with calibration gates against ground truth. From the two
surfaces, selection is free: exhaustive search over priced configurations,
including composed ones and including OFF.

Contributions, each with its number:

- **C1 — the cost map** (§4): serve-mode R for 7–9 levers × 9 regime
  cells × dense/MoE (139 ratios, MLA by transfer), with two clean
  crossovers, an OFF region, and the finding that weight-quant — the
  community default — is a parity band on expert-parallel MoE on both
  kernel stacks.
- **C2 — the acceptance map** (§5): teacher-forced β for 12 levers × 3
  architectures (102 cells, anchor-gated to ≤2%), yielding a portability
  verdict — only fp8 weight-quant carries across architectures — and the
  QK-norm rule: KV-path normalization decides draft-only fp8-K viability,
  with a V-only design rule for un-normed models. Both verified on a
  second prompt distribution.
- **C3 — the selector** (§6): a term-decomposed cost model validated up a
  ladder (held-out lever 2.2%; forward 2.9%; one-anchor architecture
  transfer, R 9.3%), composed into a strategy map that matches 5/5
  end-to-end ground-truth cells — with OFF backed by exhaustive search
  over the full combination space, not lever exhaustion.
- **C4 — the composition law** (§7): β_combo ≈ Πβ_i (median deviation
  ~0.01, 48 paired cells) with two mechanistic exceptions; combo cost
  priced by term edits; and a profiling-budget backtest showing 91
  GPU-minutes of protocol recovers the map within 2.6% of optimal (§7.5).
- **C5 — the floor-free chain** (§8): two execution laws — FA3-family
  captured schedules are replay-safe iff attention geometry is constant;
  compacted decode steps must trim rejected-slot KV — that turn composed
  rooflines into delivered numbers: **1.91× measured exactly at the
  registered roofline (delivery ≈ 100%) at dense b32/16k, growing to
  2.77× at 32k context**, +12% on MoE with the identical stack, and an
  out-of-sample diagnosis of an MLA backend failure the laws were not
  derived from.
- **C6 — profiled levers and the audited selector** (§5.4, §6, §7.5–7.6,
  §8.6): offline profiling changes what the map says — frequency-profiled
  expert selection (+0.13 β from measured routing skew) flips MoE cells,
  one delivered end to end (1.03×) through a partial-replica loader that
  ships with the paper; placement-not-contiguity on dense skip; the
  architecture-split law (even profiling STRATEGY fails to port). The
  final selector prices REALIZATIONS under per-source uncertainty (LCB +
  feasibility + fitted execution constants), emits its own measurement
  queue, and in its first audited round the end-to-end layer corrected
  the offline layer three times — including retiring one of our own
  levers whose gate proved circular. The audit loop is the contribution.

The anti-contributions are load-bearing: three map regions say OFF (and
the exhaustive search plus a measured challenger say it stays OFF); the
envelope-extension property belongs to standalone configs, not shared-KV
self-speculation; and every trap our gates caught is reported as
methodology, because a map built without them inverts real decisions.

Everything is training-free, measured on one serving stack (the R-selector
transfers; absolute TPOT does not — §6), and every claim traces to a
surface measurement or an end-to-end run (§3).

# §2 Background and related work


## 2.1 Setting

Speculative decoding accepts draft tokens by rejection sampling against the
target, preserving the output distribution; per-cycle payoff is
τ_β(γ)/(γR + 1). SELF-speculation instantiates the draft from the target
itself — no trained head, no second model to deploy — by cheapening some
part of the forward pass. Each cheapening is a *lever*: weight quantization
(W4/fp8), sparse/window attention over shared KV, draft-only KV-cache
quantization, layer skip, and (on MoE) restricted expert routing. Every
lever has a literature that studies it alone; §4–5 measure them as one
portfolio.

## 2.2 The selection-axis taxonomy (Table 1)

Three selection axes exist in the 2026 literature. They compose rather than
compete, and ours is the third:

| axis | representative work | chooses | from what signal |
|---|---|---|---|
| speculation depth (γ) | SmartSpec, Learning-to-Draft (RL), PACER, DEL (exit layer) | how far to draft, per step | goodput / online reward |
| drafter identity | Not-a-Bandit (no-regret online) | which trained drafter, per query | online accept feedback |
| **lever × architecture (ours)** | — | which self-spec lever (incl. OFF), per deployment regime | **offline measured R and β surfaces** |

Depth adaptation lives INSIDE one lever (our γ* falls out of the same
composition formula); drafter selection assumes a zoo of trained drafts.
Neither answers the practitioner deciding what to deploy for model M at
(batch, context) — the lever question — and none carries an OFF region.

## 2.3 Related work by lever and the deltas

**The seed framing.** MagicDec posed the bottleneck-aware question — where
speculation pays as batch and sequence length move — for dense models with a
KV-sparse self-draft, including "select drafting strategy" language. Our
paper is MagicDec's regime question answered with a measured multi-lever,
multi-architecture selector; the bottleneck framing is theirs, the map,
the acceptance-portability results, the composition law, and the delivery
analysis are ours.

**Window/sparse attention.** SparseSpec develops our window lever into a
full system (PillarAttn, verification-informed token selection, serving
co-design) on dense reasoning models. We cite it as the lever's strongest
form: our fixed sinks+window arm is a LOWER BOUND on what their selection
buys, and our contribution is the lever's PLACE in the map — including
where it loses (short context: strict no-op; MLA: the cost×accept double
weakness of §5-F8), which single-lever treatments cannot see. Our
window-size β-insensitivity on GQA (win128 ≈ win2048 ⇒ smallest window
dominates) is likewise absent there.

**Weight quantization.** EfficientRollout (reproduced in our record as the
dense weight-quant win) and QuantSpec-adjacent work; our delta is the
parity verdict on sparse MoE (18 cells, two kernels — §4-F2) and fp8's
status as the ONLY architecture-portable β (§5-F5).

**KV-cache quantization.** The KIVI line studies K/V asymmetry for the
TARGET; our draft-only fp8 read flips sign across architectures, and the
QK-norm rule (§5-F6) gives the mechanism plus a design rule (V-only on
un-normed models) that KIVI-style rescaling cannot rescue (it makes β
worse).

**Layer skip.** LayerSkip/SWIFT/DEL search or adapt skip sets, and
KnapSpec (2602.20217) is the lever's strongest form — non-contiguous layer
SELECTION solved as a knapsack over offline-profiled per-layer
latency/acceptance. Our fixed middle-block skip is a lower bound on that
lever (the same relation our window arm has to PillarAttn). KnapSpec is
dense-only, single-lever, with no OFF notion and no regime axis — and its
knapsack-over-profiles methodology independently validates the
offline-profiling-then-search approach our §7 applies ACROSS levers. Our
measured β curve (§5-F7) is the caution its cost side needs: skip
collapses super-linearly on dense, runs +0.24 higher on MoE, and cliffs at
50% on MLA.

**MoE-specific mechanisms.** SS-MoE / MoE-Spec / SP-MoE and utility-driven
variants each build one MoE mechanism (expert-subset draft, verify-side
expert budgeting, prefetch). None selects among levers or maps regimes;
several compose with any draft lever our map picks (e.g., MoE-Spec's
verification budgeting).

**Serving-side overlap.** Speculative-speculative decoding overlaps
drafting with verification; our delivery(γ,R) analysis and the floor-free
chain (§8) address the complementary serving question — whether the
SELECTED config's roofline survives execution — and our two execution laws
(constant-geometry capture; compacted-step rejection trimming) are, to our
knowledge, unreported.

**Camera-ready check**: one login-walled OpenReview entry ("rethinking
high-throughput speculative decoding") remains unread; re-verify before
submission (flag carried from the duplicate check).

# §3 Measuring the two surfaces


## 3.1 Design principle

Every quantitative claim in this paper traces to one of two measured
surfaces — R (cost) and β (acceptance) — or to an end-to-end run. The two
surfaces are measured independently, by different methods, each carrying
its own calibration gate against ground truth; the selector composes them
and is then audited end-to-end. This section describes both instruments
and the traps each gate exists to catch — traps we hit, not hypothesize.

## 3.2 The cost surface R

R = TPOT_lever/TPOT_bf16 per (batch, context) cell, measured in SERVE mode
— one server per arm, decode-step time under steady occupancy — because
serve-mode is what the selector prices: paging, scheduling, and batching
effects are inside the number. Method gates:

- **Infra parity (preflight).** The numerator and denominator must run the
  same attention backend, kernel set, and CUDA-graph mode unless the lever
  itself changes them — and vLLM silently flips these under levers: fp8
  e5m2 KV flips the attention backend to FlashInfer; MLA+fp8-KV flips to
  FLASHMLA; TP-EP has NO dispatch collective at all (so a "local-route is
  free" measurement on TP-EP measures nothing). The preflight asserts the
  full backend/kernel/graph configuration per arm before any cell is
  recorded.
- **Warm pass per cell.** The serving benchmark reuses prompts per seed,
  so a first pass measures decode interleaved with chunked prefill — up to
  4.7× TPOT inflation at b32/16k, large enough to INVERT a kernel
  comparison (§4). Every recorded cell is a warmed re-run.
- **Over-capacity detection.** Cells where the bf16 denominator cannot
  hold the batch are marked, not ratio'd (vLLM v1 logs no preemption
  keyword; the signature is a persistent scheduler wait queue).
- **Cross-method anchor (S1).** Serve-mode TPOT agrees with an independent
  offline two-output-length slope measurement to 2% — two instruments, one
  number.
- **Noise flags (S4).** DP-placement bimodality at low batch on the MoE
  stack: medians of 4 runs, flags retained in the dataset rather than
  scrubbed.
- **Matched denominators.** Skip arms compare dummy-weight to dummy-weight
  (dummy ≈ real validated at 1.00 ± 0.02 dense, 0.95–1.15 MoE).

## 3.3 The acceptance surface β

β = per-token draft acceptance under rejection sampling, measured OFFLINE
by teacher forcing: cache the target's reference continuations and
per-position softmax once per (architecture, context), then score each
lever's draft distribution at the same positions. The draft shares the
target's KV exactly as the lever dictates (window = sliced pages with
original RoPE; draft-only KV-quant = quantized read of the same cache), so
β isolates the lever's distributional damage. Method gates:

- **The anchor gate.** Before the sweep is trusted, measured β composed
  geometrically into an accept length must reproduce real end-to-end
  spec-decoding accept lengths: −1.6% (dense, 16k, K=4) and +0.3% (MoE,
  32k, K=6). A method that cannot reproduce known τ does not get to
  produce new β. (The gate's scope is also a limitation we later paid for:
  it covered dense and MoE, and the un-gated MLA e2e path hid a harness
  defect — §9.)
- **Sample-size floor.** ≥12 prompts × 96 positions (~1,150) per cell:
  per-prompt β spreads 0.91–1.00 over one bank, and a 4-prompt estimate
  missed by −9%. Per-prompt β also jitters run-to-run at numerics
  boundaries (router argmax flips); only the position aggregate is stable,
  and we never quote per-prompt values.
- **Pairing.** All arms within a cell score against the SAME references at
  the SAME positions, so cross-lever deltas cancel prompt-sampling
  variance. The distribution-robustness check (§5) re-pairs everything on
  a second prompt bank.

## 3.4 Composition and audit

The selector composes the surfaces as speedup = τ_β(γ)/(γR+1) — a formula
independently validated at ~90% delivery (γ=3) before this work relied on
it — and §6's validation ladder then tests every use the model is put to:
in-grid fit, held-out lever, forward prediction, architecture transfer.
End-to-end runs audit the composed claims at seven cells across two
architectures (§7–8). The delivery term itself became a measured object
(the launch floor, §8) rather than a fudge factor.

[Table: the gate ledger — gate / what it catches / incident it caught /
where recorded. Fig: two-surface pipeline diagram.]

# §4 The cost map


## 4.1 What R measures and how

For each lever ℓ and regime cell (b, c) we report the draft-step cost ratio
R = TPOT_ℓ(b, c) / TPOT_bf16(b, c), measured in serve mode — one server per
arm, decode-step time (TPOT) under steady batch occupancy — rather than as
offline kernel microbenchmarks. Serve-mode R is the quantity the selector
actually consumes: it includes paging, scheduling, and batching effects that
microbenchmarks miss, and §3's preflight guarantees the numerator and
denominator run the same attention backend, CUDA-graph mode, and kernel set
unless the lever itself changes them. Each cell is a warm-pass median of 2–4
runs; denominators are matched (dummy-weight arms compare against
dummy-weight baselines — validated at 1.00 ± 0.02 dense, 0.95–1.15 MoE).
The method cross-checks against an independent offline slope measurement at
2% (S1 gate), and cells where the bf16 baseline cannot hold the batch are
marked over-capacity rather than reported as ratios (S5 gate).

We sweep two architectures fully — dense (Qwen2.5-7B, TP1) and sparse MoE
(Qwen3-30B-A3B, attention-DP4+EP4 over NVLink) — across batch {1(4), 8, 32}
× context {2k, 16k, 32k}, with 7–9 lever arms per architecture (139 ratio
rows). The MLA row is handled by the transfer protocol of §6.

## 4.2 Findings

**F1 — Crossovers exist; the selection problem is real (Fig 2).** On dense,
W4 weight-quant owns the short-context/low-batch corner (R = 0.60 at b1/2k)
— the regime where decode is weight-read-bound and int4 cuts the dominant
byte term. As context and batch grow, KV reads overtake weight reads and the
window lever inverts the ranking: a strict no-op at 2k (R = 1.00, as
pre-registered), it overtakes W4 at b8/32k (0.63 vs 0.69) and dominates at
b32/16k (0.54 vs 0.76). On MoE the crossover runs along context between
different levers: local expert routing is the best accept-preserving lever
at short context (0.83); window takes over from 16k (0.65–0.68) and reaches
R = 0.35 at b32/32k. The best lever is regime-dependent within a single
architecture, and the crossover PAIR differs between architectures — a
static per-model recommendation is wrong somewhere in its own deployment
envelope.

**F2 — Weight-quant is a parity band on sparse MoE, on both kernels.**
Across all 18 MoE cells and two independent kernel stacks (Marlin W8A16 and
native FI-CUTLASS block-fp8), weight-quant never leads a cell (core band
0.88–1.03). The per-expert GEMM M is tiny under EP; the weight-read term a
quant lever cuts is already amortized. This extends the parity verdict of
our earlier 6-cell probe to full coverage and both kernels, and it is the
map's first OFF-region boundary: on this architecture class, the
community's default lever buys nothing.

**F3 — Levers extend the serviceable envelope (standalone configs only).**
At dense b32/32k the bf16 baseline is over-capacity — the scheduler queues
7 of 32 requests and TPOT thrashes to 117–140 ms — while window (6.8 ms),
W4 (12–14 ms), and KV-fp8 (13 ms) all hold the batch cleanly. A draft
lever can therefore create operating points where the target alone cannot
even run, which no speedup-ratio table can express; we mark these cells
specially in the map. The scoping matters: this is a property of
STANDALONE lever configs, which allocate less. Shared-KV self-speculation
cannot extend the envelope by construction — the target keeps its full KV
allocation (a draft window cuts reads, not pages) and the draft's weights
add pressure; we verified the composed self-spec draft also thrashes at
b32/32k. Envelope extension via drafting requires an allocation-cutting
lever (the draft-only KV pool of §5-F6's design rule — unbuilt).

**F4 — A law refined: local routing is not free even on NVLink.** Prior
readings that expert-parallel width "doesn't matter on NVLink" measured
TP-EP, which has no dispatch collective at all (§3 preflight catalog).
Under attention-DP+EP the all-gather/reduce-scatter pair and the
remote-token expert compute are real even on NVLink: local-route measures
R = 0.74–0.94, not 1.0. The refined statement — local routing pays in
proportion to dispatch cost — predicts its largest wins on comm-bound
fabrics, which we defer (§9).

**Kernel choice is itself regime-dependent — and a measurement trap.** The
contaminated first pass (decode interleaved with chunked prefill; up to
4.7× TPOT inflation) showed Marlin losing to Machete at b8+/16k+. Clean
warm-pass decode shows Marlin ahead at every dense cell. We keep this in
the paper as a methodology exhibit: a cost map built without the §3 gates
inverts real decisions.

**Cost-only maps mislead (bridge to §5).** On a cost-only reading, 50%
layer-skip is the strongest lever everywhere (R = 0.44–0.64). §5 shows its
acceptance collapses super-linearly (β = 0.03 dense), which is exactly why
the map must be composed with the acceptance surface before selection —
speedup = τ_β(γ)/(γR + 1), not 1/R.

## 4.3 Measurement notes (sidebar)

Warm pass per cell is mandatory (prefill contamination, ×4.7, inverted a
kernel comparison); over-capacity is detected from scheduler wait-queue
stats, not preemption logs; DP4 placement is bimodal at low batch (medians
of 4, flagged cells retained); dummy-weight denominators validated against
real weights. Full gate table in §3.

# §5 The acceptance map


## 5.1 What β measures and how

For each lever and architecture we measure β, the per-token acceptance
probability of the lever's draft against its own target, by offline
teacher-forced rejection sampling: the target's reference continuations and
per-position softmax are cached once per (architecture, context) cell, then
each lever's draft distribution is scored at the same ~1,150 generation
positions (≥12 prompts × 96 positions — the floor below which per-prompt
sampling variance of ±9% contaminates the estimate). Draft and lever share
the target's KV where the lever dictates (window slices with original RoPE;
draft-only KV-quant reads a quantized copy), so β isolates the lever's
distributional damage, not implementation noise. All arms within a cell are
PAIRED — same references, same positions — so cross-lever deltas are free
of prompt-sampling variance.

The method is anchor-gated before use: composing measured β geometrically
into an expected accept length τ reproduces two independent end-to-end
spec-decoding measurements to −1.6% (dense, 16k, K=4) and +0.3% (MoE, 32k,
K=6). Every β in this section inherits that calibration.

Architectures: dense Qwen2.5-7B, MoE Qwen3-30B-A3B, and MLA
DeepSeek-V2-Lite — three points chosen to separate the two mechanisms §5.3
shows actually govern portability (KV-path normalization; expert structure).
102 cells total.

## 5.2 The three-architecture table

[Table: β at 2k/16k/32k for win128/512/2048, q_fp8, q_int4, kvq_fp8,
skip125/25/50, lr25/50 × three architectures — from data/beta.csv]

**F5 — The portability verdict: almost nothing is portable.** Of 12 levers
× 3 architectures, only fp8 weight-quant holds β across every regime and
architecture (.978 → .993 → 1.000 at 16k). Everything else moves:
draft-only KV-quant INVERTS (0.84 dense → 0.97 MoE → 0.997 MLA); layer
skip swings +0.5 across architectures at the same skip fraction (skip125:
.45 dense, .70 MoE, .97 MLA); the window lever — portable between dense and
GQA-MoE (±0.03) — breaks on MLA, where β decays with context and window
SIZE becomes decisive (win128: .976 dense vs .576 MLA, collapsing to 0.49
at 32k). A selector that borrows β across architectures gets the map's
regions right and its values wrong (we show this quantitatively in §6's
v1→v2 comparison); a selector that borrows across the wrong architecture
pair gets even the regions wrong.

**Robustness to prompt distribution.** We re-measured the claim-bearing
arms on a second, deliberately different distribution — competition-math
reasoning (AIME-derived bank, same harness, same 1,152 positions per cell)
— at the map's central context:

| arm | dense (gen → math) | MoE | MLA |
|---|---|---|---|
| q_fp8 | .978 → .991 | .993 → .995 | 1.000 → .995 |
| q_int4 | .924 → .974 | .972 → .980 | .997 → .983 |
| win512 | .978 → .995 | .971 → .990 | **.922 → .795** |
| kvq_fp8 | .841 → .905 | .971 → .986 | .997 → .993 |
| skip50 | .029 → .067 | .154 → .159 | .000 → .005 |
| lr25 | — | .693 → .797 | .936 → .897 |

Every structural claim survives: fp8 weight-quant remains the top, flat,
portable lever (|Δ| ≤ 0.012 across all three architectures); the kvq
inversion keeps its monotone QK-norm ordering (0.905 < 0.986 < 0.993);
skip stays dead with the same architecture ordering; and within-cell lever
RANKINGS are unchanged in all three architectures. Absolute β shifts are
small and mostly upward (median |Δ| ≈ 0.015 — math continuations are more
predictable). The one large mover is the one the map already flags as
fragile: MLA-window drops a further 0.13 on math (0.922 → 0.795) — the
distribution check AMPLIFIES the F8 double-weakness rather than
challenging it. Local-route is the noisiest lever (±0.10, opposite signs
on MoE/MLA), consistent with its routing-sensitivity history; its map
regions carry the widest error bars.

**F6 — The QK-norm rule (mechanism, not correlation).** The kvq inversion
has a measurable cause. K-cache outlier magnitude anti-correlates
monotonically with draft-only fp8-K viability across all three
architectures: dense (no KV-path normalization) K amax = 426 — at e4m3's
448 saturation, quantization spacing ~32 on exactly the outlier channels
that dominate attention logits — and β is 0.60–0.84 and unstable; QK-normed
MoE, amax 150, β 0.97–0.98; MLA behind kv_a_layernorm, amax 28, β
0.996–0.999. The ablation isolates the K side: quantizing V only is free
(β 0.99) on the un-normed model, quantizing K only reproduces the full
damage, and amax rescaling (per-token or KIVI-style per-channel) makes it
WORSE (β ~0.5) — the scale-1.0 cast is already optimal for this tensor
shape. Design rule: KV-path normalization decides whether a draft-only
fp8-K read is a lever or a wound; on un-normed models, quantize V only.

**F7 — The skip curve is decision-grade only with β (Fig 5).** Dense β at
12.5/25/50% middle-block skip is 0.47/0.09/0.03 — super-linear collapse,
every point below its cost-composed break-even. MoE runs ~0.24 higher at
every fraction and MLA tolerates shallow skip almost freely (0.97 at 12.5%)
before a cliff to 0.000 at 50%. Cost-only skip results (including our own
F2/§4 table, where skip is the "best" lever everywhere) are not
decision-grade; the (R, β) plane is.

**F8 — Window: size-insensitive where portable, doubly weak on MLA.** On
dense and GQA-MoE, window size barely moves β (win128 ≈ win2048), so the
smallest window strictly dominates — same β, better R (a revision of our
own earlier default of 512). On MLA, β is window-size- and
context-sensitive, and §4/§6's cost side shows MLA's compressed KV gives a
window little to cut: the lever is doubly disadvantaged on exactly one
architecture, a coherent story the selector encodes as a region boundary.

**Aside — early-exit vs middle-skip.** Early-exit skip (drop the last
layers) has β = 0.000 at every depth: same R as middle-skip, dead β. The
draft must keep the target's head-adjacent layers; every skip number above
uses SWIFT-style middle-block skip keeping the last two layers.

## 5.3 What §5 hands the selector

β must be measured per architecture, but not per cell: β is context-STABLE
for every lever except MLA-window (±0.01–0.03 across 2k→32k), so one β
column per (lever, architecture) — ~1 GPU-hour offline, anchor-gated —
prices the whole acceptance side of the map. Combined with §4's R surface,
the selector of §6 needs no end-to-end sweeps at all.

## 5.4 Profiled lever forms: the naive-instantiation check

Every arm above is a lever's NAIVE form — contiguous middle-block skip,
RTN quantization, a contiguous expert shard for local routing. The
single-lever literature (KnapSpec's layer knapsack; GPTQ/AWQ) shows
offline profiling recovers accuracy at equal cost, so we re-measured the
levers' PROFILED forms on the same paired references (Phase 83):

| lever, profiled form | naive β | profiled β | mechanism measured |
|---|---|---|---|
| expert selection, top-frequency sets (MoE) | .826 / .693 (50/25%) | **.953 / .823** | top-half experts carry 96.3% of routing mass |
| layer sets, iterative greedy (dense, budget 3/7) | .448 / .09 (contiguous middle) | **.869 / .557** | redundancy concentrated in EARLY blocks |
| layer sets (MoE, budget 6/48) | ~.70 | .742 | leave-one-out profile FLAT (.946–.959) |
| int4, GPTQ-calibrated (dense) | .924 (RTN) | .9427 | calibration near the lever's ceiling |
| vocab restriction (dense) | — | .85-.88 honest (live C4-keep coverage .80/.90) | first gate was CIRCULAR (keep built from the scored refs); caught by the e2e audit — the cautionary exhibit for profiled-artifact gating |

Three structural findings. **Placement, not contiguity**: dense iterative
greedy picks a nearly contiguous EARLY block — the middle-block
convention, not contiguity itself, is what the naive skip arm got wrong;
early layers drop in blocks almost freely. **The architecture-split law**:
profiling headroom lives where the architecture's redundancy lives —
layer placement pays on dense (+0.42 at equal budget) and almost nothing
on MoE (+0.04, flat profile: each layer's contribution is already diluted
across 128 experts), while expert selection pays on MoE (+0.13) and does
not exist elsewhere. Even the profitable profiling STRATEGY fails to port
across architectures, extending F5 from lever values to lever forms.
**Frequency-selected quarter ≈ contiguous half**: flr25 (0.823) matches
naive lr50 (0.826) at half the resident memory — a free 2× on the
residency axis. Selection consequences in §7.6; delivery in §8.6.

# §6 The selector


## 6.1 From two surfaces to a decision

The selector composes the two measured surfaces into expected speedup,

  speedup(ℓ, γ; b, c, arch) = τ_β(γ) / (γ·R + 1),

where τ_β(γ) is the expected accepted length from per-token acceptance β
(geometric composition, anchor-calibrated in §5 to ≤2% on both
architectures; delivery ~90% at γ=3, ~86% at γ=6 on the pre-§8 chain), and
R comes from §4 directly where measured. For unmeasured cells and levers, R
comes from a term-decomposed cost model fitted to §4's absolute TPOTs:

  T(b, c) = F(b) + [W_read + KV_read]/BW_eff + h·b·c + C_comm(b) + κ_kernel,

in which levers are TERM EDITS (window shrinks KV_read; W4 shrinks W_read;
skip scales the per-layer terms; local-route shrinks C_comm and the remote
expert read) and architectures are CONSTANTS (F, BW_eff, h, comm
coefficients, per-kernel κ). Two of the constants were discoveries, not
assumptions: the KV-management term h·b·c — draft-side read levers do not
remove block-table/metadata work that scales with ALLOCATED context, which
a pure byte model under-predicts by 33–49% on exactly the large-KV MoE
cells — and the per-kernel κ, which recovers §4's Marlin/Machete ordering
(κ_machete = 0.67 ms, κ_marlin ≈ 0) from the fit alone.

## 6.2 The validation ladder

We validate the model the way it will be used: predicting things it was not
fitted on, with gates registered before measurement.

| rung | question | result | gate |
|---|---|---|---|
| V0 in-grid | does the decomposition fit? | dense median err 3.5%, 8/8 crossover orderings; MoE 6.8%, 8/9 | dense PASS; MoE at its DP4 noise floor |
| V1 held-out lever | predict window's full R curve with ALL window data excluded | **2.2% dense / 6.9% MoE** | **PASS — the load-bearing rung** |
| V2 forward | price a never-measured arm (win128: 2.9%, 17 cells) and a never-swept cell (b16×8k: 8.0%) | registered → verified | PASS |
| V3 transfer | predict the MLA tier from config constants alone | R err 15.5% | **FAIL — reported** |
| V3b one-anchor | one 5-minute bf16 anchor cell; everything else transferred | **R 9.3% on 53 held-out cells**; TPOT 18.3% | R PASS / TPOT FAIL |

V1 is the claim that the decomposition is physics rather than
curve-fitting: window's cost curve is recoverable from KV-byte accounting
alone. The V3→V3b pair is the honest transfer statement: the R-SELECTOR
transfers to an unswept architecture with a single anchor measurement —the
V3 failure decomposed into a step-fixed vs per-layer floor split (the skip
arms prove it: measured skip R is 0.58–0.92, not the 0.50 a
layer-proportional floor implies) and architecture-specific κ at tiny GEMM
M — while absolute TPOT additionally needs 2–3 per-stack constants (h,
BW_eff) that are properties of the serving stack, not the architecture.
We report both halves; the selector only needs the half that passes.

Caveats we carry rather than hide: BW_eff fits above the hardware peak
(4.8 vs 3.35 TB/s) because weight reads partially overlap compute — the
model is EFFECTIVE, validated by prediction, and makes no bandwidth claim;
MoE's V0 residual concentrates in cells flagged for DP4 placement
bimodality in §4.

## 6.3 The strategy map and its e2e check

Composing measured R (dense, MoE, and the anchored MLA tier) with §5's
per-architecture β yields the full strategy map — (architecture, batch,
context) → lever + depth, including OFF:

- **Dense**: a W4 region (short ctx / low batch, 1.28–1.30×) and a window
  region (long ctx / high batch, up to 1.56× single-lever), with window-128
  tying window-512 — §5-F8's size-insensitivity surfacing in the decision.
- **MoE-GQA**: OFF-or-marginal at 2k (≤1.06× — §4-F2's parity band composed
  with β); a window region from 16k (1.27–2.22× roofline).
- **MLA**: OFF everywhere at NVLink — the doubly-weak window (§5-F8) and
  the anchored cost tier agreeing. MLA's measured β strengths
  (shared-expert local-route 0.95–0.99, shallow skip 0.97) are parked for
  the comm-bound fabric (§9).

**OFF is search-backed, not lever-exhaustion.** "OFF" would be a naive
verdict if it meant only that the levers we tried lose — some COMBINATION
might win. We therefore re-derive every OFF cell by exhaustive priced
search over the full combination space (53 configs per cell: all subsets
up to size 4 with ≤1 lever per class, including MLA's best acceptance
lever lr50 β 0.95–0.99, × γ ≤ 8), priced OPTIMISTICALLY — product-law β is
an upper bound (§7's measured exceptions are destructive), measured
combo/single R where available, and delivery = 1 (the §8 chain). Even this
upper bound stays ≤ 1.13× in every OFF cell, and — the structural finding
— NO composed configuration materially beats the best single lever in any
OFF region: composition cannot rescue OFF, because these regions are OFF
precisely where the base cost terms the levers cut are already small
(MLA's compressed KV mutes window; EP amortization mutes weight-quant;
NVLink mutes local-route). The residual challengers are all the portable
fp8 single lever at 1.05–1.13× optimistic; we report them as measured
marginal cells, not OFF flips. [Artifact: off_hardening.md]

All five end-to-end ground-truth cells match the map's winner (5/5), and
the map-vs-measured speedups agree to 2.5–16% pre-§8 (the residual being
delivery, which §7 diagnoses and §8 eliminates). Borrowed-β ablation:
recomposing the map with literature β values gets the REGIONS right and
the values wrong (dense b32/16k: 1.40× borrowed → 1.55× measured vs 1.54×
e2e) — measuring β is what makes the map quantitative.

**Selection under uncertainty (the final selector).** The map of record
(v6) selects by LOWER CONFIDENCE BOUND under per-source uncertainty
(measured R ±4%, term-model R ±10–30% by family, product-β ±0.02,
per-chain-tier execution constants φ/ψ from §8's fits), with residency
feasibility as an admission filter, Monte-Carlo P(winner) per cell, and
γ capped per lever semantics. Three properties follow by construction:
an under-measured realization cannot win (its wide bounds sink its LCB);
every cell prints winner + LCB + P(win) + provenance; and the map emits
its own measurement queue — the cells where uncertainty is
decision-relevant. In its first round that queue adjudicated all three
of its items against the offline layer and the offline layer lost twice
on realization constants and once on our own gate circularity (§7.5) —
without any confident cell moving.

The deployment recipe this section justifies: one R sweep per serving
stack (§4), one β column per (lever, architecture) (§5, ~1 GPU-hour), one
anchor cell per new architecture (5 minutes) — then every regime cell of
the map is a prediction, not a measurement.

# §7 Lever composition


## 7.1 Composition as offline profiling

Levers cut different terms of §6's cost model, so composing them should
compound R; the open question is what composition does to β. We treat this
the way the pruning and mixed-precision-quantization literature treats
configuration search: profile offline, fit a cheap composition rule,
search the composed space, and validate the argmax end to end. (SWIFT
searches WITHIN one lever — skip sets; ours is search ACROSS levers, with
the single-lever maps of §4–5 as the profile.)

## 7.2 F9 — the β-composition law

Across 48 combo cells (22 combinations × contexts × three architectures,
paired against §5's singles on identical references):

  β_combo ≈ Π β_i, median |deviation| 0.013 / 0.010 / 0.009
  (dense / MoE / MLA); 40/48 cells within ±0.03.

Acceptance errors from different levers are near-independent — the search
can price unmeasured combinations by multiplication. The two exceptions are
structured, and both teach mechanism:

**Constructive (dense): window RESCUES KV-quant** (+0.13…+0.26 over
product; kvq-alone 0.60–0.75 → win+kvq 0.84–0.87). The window slice stops
reading exactly the fp8-damaged K outliers that §5-F6 identified — a lever
that cuts a term also cuts a co-lever's ERROR EXPOSURE on that term.
(Cost-moot — quantizing 528 windowed tokens saves nothing — but it breaks
error-independence in a predictable direction: shared surface ⇒ shared
error.)

**Destructive (MLA-only, long-context-only): shallow skip composed with
any context-affecting lever collapses at 32k** (skip×lr50 −0.114,
skip×window −0.100; all ≤0.036 at 2k; the same pairs track product on
dense/MoE). Composition portability tracks §5's architecture verdict —
MLA is simultaneously the most skip-tolerant single-lever architecture and
the most destructive composer.

MoE composes mildly ABOVE product everywhere (+0.005…+0.027): hard tokens
are correlated across levers, pushing toward min()-like behavior. Notably,
the comm-free triple this project began from — window+local-route+fp8,
assembled — holds β = 0.82–0.83 at every context; §7.3 shows why it still
loses at NVLink.

## 7.3 Pricing composed cost by term edits

Combo R̂ comes from applying multiple term edits to §6's fitted model;
measured combos then grade the pricing (predictions registered first):

- **The falsification cell passes**: kvq+window is priced COST-MOOT
  (≈ window alone) precisely because the model says the window already
  removed the KV bytes kvq would quantize — measured +3.5%/−4.9%.
- **The dense composed bet exceeds its registration**: W4+window at
  b32/16k measured R = 0.311 (predicted 0.353) — composed with the
  measured β 0.917, a ~1.9× roofline where the best single lever offers
  1.56×.
- **A model refinement found by its own failure**: windowed-MoE combos
  measured R ≈ 0.35 vs predicted 0.70 — the h (KV-management) term should
  scale with ATTENDED length, not allocated context, for sliding-window
  arms. Reported as the model's current edge; dense pricing is unaffected.
- **MLA's best challenger fails on cost** (fp8 κ at tiny per-expert M —
  §6's per-architecture κ lesson), so MLA's OFF region survives
  composition.
- At NVLink, window-alone beats the comm-free triple (β 0.98 vs 0.83 at
  ~equal R): the triple's regime is the comm-bound fabric, deferred (§9).

The search over the composed space (product-β default, measured pairs
where available, measured-combo R as gold) upgrades the dense long-context
map: its winner becomes the composed W4+window configuration, with
measured-β × measured-R provenance.

## 7.4 F10 — composition end to end, and the delivery(γ, R) law

Running the composed draft live (W4 checkpoint as draft_model + window-KV
over the shared target cache):

| cell | roofline | measured | delivery |
|---|---|---|---|
| dense b8/16k K=4 | 1.55× | 1.41× | **91% — composition CONFIRMED e2e** |
| dense b32/16k K=6 | 1.91× | 1.48× | 77% |
| dense b32/16k K=4 | — | 1.48× | (flat vs K=6) |

The accept side delivered at both cells (implied β ≈ 0.90 vs measured-combo
0.917). The b32 miss is not a composition failure but a systems law:
**delivery is a function of (γ, R), not γ alone.** Composition cut draft
bytes to R = 0.31, but each draft step still pays a fixed launch cost,
which now DOMINATES a much cheaper step — K=4 ≈ K=6 in the measurement is
the floor's signature (more, cheaper steps buy nothing when each step pays
the same fixed toll). Composition succeeds by shifting the draft bottleneck
from bytes to launch — so delivery decays exactly when composition works
best, and the composed winner (1.48×) lands below the single-window 1.54×
despite far lower R.

This is the cliff-hanger §8 resolves: the floor is measurable, attributable
— and removable. With it removed, this table's b32 row becomes 1.91× at
~100% delivery.

## 7.5 The cost of the search: a profiling-budget backtest

The selection step is free — pricing all 53 configurations per cell is CPU
work. The search's real cost is PROFILING, so we ask: how few GPU-minutes
buy the same decisions? We backtest by replaying a minimal protocol
against the completed record, treating MLA as the new architecture (which
is how the project actually unfolded), revealing at each budget level only
what the protocol would have measured, and scoring every cell's argmax
against the full-information map. Regret = true speedup of the chosen
config vs the true best.

| budget level | GPU-min | near-opt (≤0.02) | median regret | max regret |
|---|---|---|---|---|
| L0: mechanism priors only | 0 | 1/9 | +0.077 | +0.223 |
| L1: + one bf16 R anchor | 5 | 1/9 | +0.077 | +0.223 |
| L2: + one κ anchor + checklist β at one ctx | 26 | 3/9 | +0.057 | +0.213 |
| L3: + deviation-triggered ctx reveal | 46 | 3/9 | +0.064 | +0.213 |
| **L4: + contested-cell R + conservatism rule** | **91** | **8/9** | **+0.000** | **+0.026** |
| full profile | 305 | 9/9 | 0 | 0 |

Three lessons. (1) **Priors alone fail** (max regret 0.22): per-kernel κ
and per-architecture context behavior are not config-derivable — the maps
are load-bearing, not decoration. (2) The last mile comes from DECISION
RULES, not more data: measure R only where the top-2 gap is inside the
model's validated error (±0.09), and on such ties prefer measured-R
configs over model-priced ones — that one conservatism rule alone cut max
regret from 0.213 to 0.026. (3) The protocol's cost is roughly
grid-independent (two anchors + one β column + a handful of contested
cells), so its advantage grows with map size: here 91 vs 305 GPU-minutes
(3.4×) for one 9-cell architecture. This is the search protocol we
recommend over exhaustive profiling: exhaustive SELECTION over measured
physics, with measurement itself allocated by decision uncertainty.

**The same allocation rule governs search WITHIN a lever.** Non-contiguous
layer-set selection (KnapSpec's lever) opens a 2^24 space our maps do not
enumerate; we backtest the rule on a measured 17-set pool (budgets 3/5/7)
with the 24 leave-one-out singles as the only prior. The product prior's
quality DEGRADES with depth — Spearman ρ = 0.94 / 0.80 / 0.60, absolute
error growing to −0.16..−0.28 at budget 7 — so pure profile-then-solve
(the knapsack recipe) is unreliable at depth in two distinct ways: its
RANKING decays, and even when its argmax is right its VALUE is wrong
(prior 0.663 vs measured 0.507 at budget 7 — a selector composing the
unverified estimate would over-price the config by 30%). Prior-guided
measurement fixes both at almost no cost: the pool-best set is found
within 1–2 measurements (~90 s) at every budget, and the winning set
enters the map with a MEASURED β. Set-selection itself is a real lever
upgrade — {2,4,5} holds β 0.855 where contiguous skip at the same budget
holds 0.448 — though even doubled, skip flips no winner on this
architecture (§6). The unified statement: profiles nominate, measurements
confirm, and the confirmation budget is 1–2 per decision at every level
of the hierarchy (lever, combination, set, cell).

**The confirmation layer catches the profiler's own bugs.** In the map's
first queue-and-correct round, end-to-end measurement contradicted the
offline layer three times and was right all three: a model-free lever's
realization cost (n-gram's CPU lookup, ψ = 2.4 target-steps), a
context-scaling assumption (the full-attention expert draft's R is
context-invariant), and — most instructive — a CIRCULAR gate (§3's
disjoint-artifact rule exists because the vocabulary-restriction keep
set was built from its own evaluation refs; honest coverage 0.80–0.90
retired the lever from the dense map and withdrew a headline candidate).
Every correction was absorbed as a constant or a rule, no
measured-provenance cell moved, and the loop closed in one round. This
is the paper's operational thesis: profiles nominate, measurements
confirm — and the confirmation layer is load-bearing precisely because
the profiling layer, including ours, has bugs.

## 7.6 Map v5: profiled levers change what the map says

Re-running the exhaustive search with the profiled columns of §5.4 flips
FOUR cells — all on MoE, all driven by frequency-profiled expert
selection: the short-context band (b4/b8/b32 at 2k) moves from
OFF/marginal to flr50+q_fp8 (1.12–1.15× priced), and b4/16k composes
flr50+q_fp8+win512 (1.11×). Dense and MLA winners are unchanged
(calibration lifts dense values +0.02–0.04× without moving any argmax) —
the architecture-split law carried through search: the only
decision-changing profiled lever is the one aligned with the
architecture's redundancy. Each flipped cell carries its e2e provenance
in the map (delivered / chain-blocked, §8.6) — a profiled winner enters
the map with a measured β (§7.5's rule) and exits to deployment only
through a realization whose R is also measured.

# §8 Cashing the frontier: the floor-free chain


## 8.1 Where §7 left the composed draft

Composition cut the dense draft's bytes to R = 0.31, and delivery
collapsed to 77% at b32 — because each draft step pays a fixed launch
cost that now dominates a byte-cheap step. Anatomy (Kineto traces of the
exact composed config) makes the floor concrete: the chain step is 6.47 ms
at only 61% GPU-active — 2.26 ms/step is launch idle spread across ~370
eager kernels — and cycle arithmetic bounds the recoverable payoff at
1.98×. CUDA-graph capture is the textbook remedy, and it is exactly what
the record says fails: naive FA3 chain capture collapses acceptance (5.69
→ 1.93 in our control), the known freeze trap. This section is about why
it fails, the two laws that fix it, and what the fixed chain delivers.

## 8.2 F11 — the constant-geometry capture law

FA3's host-side kernel schedule (work distribution, splits) is decided at
capture time. A captured graph replays that schedule verbatim, so capture
is replay-safe **iff the attention geometry is constant** — and a draft
chain's sequence GROWS each step, which is why every naive chain capture
in the record collapses acceptance (FA3 5.69→1.93; TRITON_ATTN 5.69→1.95
— refuting our own earlier hypothesis that a seq-len-independent grid
would be safe). The law's contrapositive is the fix: the window scratchpad
CLAMPS geometry (sinks+window = 528 keys, constant by construction), so
the entire chain step becomes capture-stable. One implementation subtlety
carries the win: the scratchpad's attention must be a fused paged-FA3 call
— torch SDPA at q_len=1 with a mask silently dispatches a mem-efficient
decomposition (~0.25 ms/layer of gemv soup where the fused kernel needs
~30 µs), which is also the post-mortem of our own earlier scratchpad
prototype (and, we suspect, of others' "graphs don't help" readings).
Result: chain step 6.47 → 3.70 ms at 94% GPU-active, acceptance
bit-preserved (5.655 vs 5.685).

## 8.3 F12 — the compacted step-0 law

With the chain fixed, profiling re-attributes the wall to step-0: the
draft re-ingests the K+1 verify-span tokens whose KV verify already wrote
(under shared KV) — a ~20 ms ragged forward whose outputs are discarded
except at one position. Semantically, step-0 is a q=1 decode of the
appended token. Compacting it that way, however, costs −0.55 acceptance
naively, and the cause is a law worth stating generally: **padded-path
seq_lens span rejected slots, which a causal ragged forward masks
implicitly and a q=1 decode does not.** The compacted token attends the
stale KV of the just-rejected draft tokens; under a KV window those stale
keys sit among the ~window most recent — maximal attention mass — so the
window regime AMPLIFIES the poisoning (−0.55) that full context dilutes
to noise (−0.04, matching an earlier shelved reading of the same
mechanism). The fix is one line of metadata: trim per-request seq_lens by
the cycle's rejection count. Acceptance restores bit-clean (5.672).

## 8.4 F13 — the payoff, end to end

Re-running §7's cells on the fixed chain (same method, same baselines):

| cell | roofline | broken chain | fixed chain | delivery |
|---|---|---|---|---|
| dense b32/16k K=6 | 1.91× | 1.48× (77%) | **4152 tok/s = 1.91×** | **≈100%** |
| dense b32/16k K=4 | — | 1.48× | 1.85× | |
| dense b8/16k K=4 | 1.55× | 1.41× (91%) | 1.64× | >100%* |
| dense b8/16k K=6 | — | — | 1.52× | |
| dense b16/32k K=4 | — | — | **2994 ±517 = 2.77×** | |
| dense b16/32k K=6 | — | — | 2517 ±340 = 2.33× | |

The registered roofline is measured EXACTLY at the headline cell — the
delivery discount was the launch floor and nothing else. The map's γ*
structure survives (b8: K=4 > K=6, as selected), and the baseline is not
handicapped: async-scheduling helps nospec by only 4% (headline 1.84×
against the async baseline). (*b8 slightly exceeds its registration
because the fixed chain's R is better than the standalone R the map used.)

The long-context rows are the composition thesis completing itself: the
scratchpad chain's cost is context-INDEPENDENT (a constant 528-key window)
while the target's per-token cost doubles from 16k to 32k — so the
composed advantage GROWS with context once the floor is gone, reaching
**2.77× at b16/32k** (acceptance still composing at the product law:
4.33/5 ≈ β 0.92). At b32/32k neither arm runs: the target itself is
over-capacity, and shared-KV self-spec cannot relieve residency (§4-F3
scoping) — the honest boundary of the envelope.

## 8.5 F14 — the laws travel

Architecture generality: the identical stack engages under DP4/EP4 MoE —
compaction on all four ranks, acceptance unchanged (3.766 vs 3.764) — and
lifts the MoE window cell 615 → 688 tok/s (+12%). The chain fix is not a
dense special case.

Prediction beyond the derivation set: the MLA challenger of §6's OFF
hardening initially collapsed (acceptance 2.00) on a path none of §8's
work touched. F11 called it: FLASH_ATTN_MLA is FA3-family, its captured
chain schedule freezes exactly as the law says, and the one-line
reclassification restores acceptance to 5.92 by default. A law that
diagnoses failures in code it was not derived from is doing the work we
claim for it.

Delivery(γ,R) closes as a design rule: delivery is a function of (γ, R),
the launch floor binds precisely when composition makes the draft
byte-cheap, and a constant-geometry chain removes the floor — after
which the selector may rank on raw τ_β(γ)/(γR+1). Remaining headroom is
known and bounded: verify still idles 31% of its forward, and step-0's
residual over a pure chain step is the last ~6% to the 1.98× ceiling.

## 8.6 The execution boundary, generalized: a delivered flip and its realization ladder

§7.6's flipped cells are where selection and execution meet, and we walk
one through to metal. The frequency-profiled expert draft admits several
REALIZATIONS, and the ladder separates the lever from its execution:

| realization (MoE 2k band, γ=2) | b4 | b8 | b32 | accept |
|---|---|---|---|---|
| naive contiguous shard (EP-local) | 0.55× | — | — | 2.17 |
| fp8 full replica + freq mask | 0.93× | — | 0.75× | 2.85 |
| **bf16 partial replica (ships with this paper)** | **1.03×** | 0.63× | 0.64× | **2.88** |

The acceptance flip transfers EXACTLY (offline β 0.953 → accept
2.81–2.88 everywhere — the profiling surface loses nothing at the
metal), and each throughput gap is attributable by the paper's own laws:
the bf16 full replica does not fit (73.25 GiB/rank — realization is a
memory problem first); the fp8 replica pays fp8's small-M κ (§6's
per-kernel lesson, biting our own realization); the partial replica —
per-layer frequency sets installed as the draft's expert_map, so the
loader skips non-resident experts (46.25 GiB measured) — removes κ and
DELIVERS the b4 flip at 1.03×, the first naive-map OFF cell measured
above baseline. At b8/b32 the 2k verify is too cheap to amortize the
draft chain (needed cycle 27 ms, measured 44 ms): delivery(γ,R) again,
now with the boundary measured inside one band. The division of labor is
the section's thesis in one exhibit: profiling changes what the map says;
execution decides what the deployment collects; and the gaps between
them are quantified, named, and lever-external.

# §9 Discussion, limitations, and open surface


**One box, one stack.** All measurements are single-node H100 ×4 on one
serving stack. The split §6 validated is the portable claim: the
R-selector transfers (one anchor cell, 9.3%), absolute TPOT does not (h,
BW_eff are stack constants). Deployments re-run the 91-minute protocol of
§7.5, not our sweep.

**Noise floors we report rather than hide.** MoE DP-placement bimodality
bounds V0 at ~6.8% (medians of 4, flags retained); the b16/32k spec cells
carry 17% run variance (error bars quoted); the over-capacity detector
false-positives on cold prefill (affected cells kept by manual inspection,
noted in the dataset).

**The anchor-gate lesson (paid for, then repaid).** The β anchor gate
covered dense and MoE; the un-gated MLA end-to-end path concealed a
harness defect that our own capture law later diagnosed (FLASH_ATTN_MLA's
captured chain is not replay-safe; fixed, acceptance 2.00 → 5.92 default).
The episode is a validation asymmetry worth stating as practice: every
architecture whose numbers a map carries needs its own end-to-end anchor,
even when the offline method is calibrated elsewhere. It also left OFF on
MLA measured three ways — priced (≤1.13× optimistic over 53 configs),
mechanistic, and observed (best challenger 0.54×; even β≈1 delivers
0.56× on the eager MLA chain).

**Harness lever fidelity.** The runtime fp8 draft is W8A8 while the map's
q_fp8 β is weight-only — a −2.1 accept gap on MLA at K=5. A weight-only
runtime draft option is plumbing, not research, but until it exists the
map's q_fp8 column prices a config the harness cannot yet run exactly.

**Retractions carried, not buried.** One Phase-84 gate (vocabulary
restriction) was circular and its map consequences were withdrawn after
the e2e audit (§7.5); the n-gram lever's dramatic MLA pricing died on
its realization (CPU-side lookup); both survive in the record with
mechanisms named, and both produced rules now applied paper-wide.

**Profiled flips: delivery boundary measured, not closed.** Of the four
map-v5 flips (§7.6), one is delivered (b4/2k, 1.03× — margins ±0.10, a
parity-to-modest win); the 2k band's higher batches are chain-blocked
(0.63–0.64×, needed cycle 27 ms vs measured 44 ms). The acceptance side
transfers exactly everywhere (2.81–2.88 vs offline 0.953), so what
remains is the windowless-MoE instance of the §8 execution program plus
the partial replica's memory rent (~half the expert bytes per rank) —
both quantified. Profiled sets also inherit a monitoring burden a naive
shard does not: routing frequencies can drift with workload; the
distribution-robustness check (§5.2) bounds this for our two banks, not
for all traffic.

**The deferred fabric.** The comm-bound tier (PCIe/multi-node) is where
the refined local-route law predicts its largest wins and where the MoE
comm-free triple (β 0.82–0.83, assembled and measured) is parked. The
original thesis of this project lives there, with its acceptance side now
fully measured and its cost side one sweep away.

**The draft-only KV pool.** Twice motivated and still unbuilt: it is the
lever that would extend the serviceable envelope (§4-F3's scoping — the
only claim shared-KV self-spec structurally cannot reach) and the QK-norm
rule prices exactly where it is safe (fp8-K on normed architectures;
V-only otherwise, 25% of KV bytes for free).

**Remaining headroom on the fixed chain.** Verify idles 31% of its
forward; step-0's residual over a pure chain step is the last ~6% to the
1.98× ceiling at 16k. Neither gates any claim; both are known dials.

**Search scope.** The backtest replays one architecture-as-new (MLA — the
one the record supports honestly); a second replay (MoE-as-new) would
strengthen §7.5. The searched combo space is bounded (≤4 levers, one per
class, fixed lever settings); per-layer assembly à la KnapSpec, per-lever
setting search à la SparseSpec, and online refinement à la Not-a-Bandit
all compose with the maps rather than compete with them.

**Runtime switching.** With delivery ≈ 1 and per-regime winners measured,
the natural system is one that switches levers as its regime shifts —
batch ramps, context growth — using the map as scheduler policy and §7.5's
amortization arithmetic for hysteresis. Scoped as follow-on work; this
paper establishes the maps, the selector, and that the selected configs
deliver.

# §10 Conclusion


"Should this deployment self-speculate, and with what?" is answerable by
measurement. Two calibrated surfaces — a serve-mode cost ratio and a
teacher-forced acceptance rate — price every lever, every composition,
and OFF, per regime and per architecture; a term-decomposed model carries
the map off-grid and to new architectures with one anchor cell; a
91-GPU-minute protocol recovers the decisions within 2.6% of a full
profile. The map's structure is real: crossovers inside architectures,
near-nothing portable across them (one lever excepted), a composition law
with mechanistic exceptions, and OFF regions that survive exhaustive
search and a measured challenger.

And the map's promises are collectable. Two execution laws — capture is
replay-safe iff attention geometry is constant; a compacted decode step
must trim the rejected tail — remove the launch floor that taxed exactly
the configs the map ranked highest, after which the selected
configurations deliver their rooflines: 1.91× at the registered value,
2.77× where context growth favors composition, on an unmodified sampling
semantics with acceptance bit-preserved. The laws travel: they fixed a
backend they were not derived from.

The instrument is the contribution: measure once, select everywhere, and
let OFF be an answer.


## Figures

1. **Fig 1** — the cost map: R per lever, regime, architecture (3-panel heatmap; over-capacity hatched).
2. **Fig 2** — crossovers: R vs context, faceted by batch.
3. **Fig 3** — strategy map (v5 rendering; v6.2 = the LCB map of record in §6).
4. **Fig 4** — the validation ladder: predicted vs measured with ±10% band.
5. **Fig 5** — the (R, β) plane with iso-speedup contours: why cost-only maps mislead.
6. **Fig 6** — β portability slopegraph: only fp8 weight-quant is flat.
7. **Fig 7** — the composition law and its measured limits (across levers; within-lever depth).
8. **Fig 8** — the floor-free chain: anatomy, delivered speedups vs rooflines, the capture law across five backends.

