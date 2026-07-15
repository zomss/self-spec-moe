# §5 The acceptance map (draft)

> Source: Phase 77 (`results_accept.md`, `data/beta.csv`, `kvq_probe.py`).
> Numbers final. Robustness check vs a second prompt distribution (math):
> `79_paper/data/beta.csv` — PENDING, slot reserved in 5.2. ~1.5 pages.

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
