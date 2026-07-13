# §4 The cost map (draft)

> Source: Phase 76 (`results_sweep.md`, `data/e1/summary.csv`, figures
> fig1/fig2). Numbers final; prose is submission-draft quality, figure refs
> symbolic. ~1.5 pages.

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
