# Phase 76 E1 results — the lever × regime draft-latency map (dense + MoE)

Box cloud-9ezI3Q, GPUs 0/1/6/7. Method: serve-mode TPOT (decode-step time),
one server per arm, warm pass per cell, median of 2-4 runs (4 on S4-flagged
cells). Denominators matched (dummy↔dummy; MLA/PCIe tiers deferred by user).
Full tables + gates: `data/e1/final_tables.md`; per-cell CSV `data/e1/summary.csv`
(139 ratio rows, over-capacity cells marked); runner/analyzer in `scripts/`.

**Gates: S1 cross-check PASS** (dense bf16 b1/2k TPOT 5.98 ms vs P75-E1 offline
slope 5.86 ms — two independent methods agree to 2%). S5 marks dense bf16
b32/32k (+dummy) OVER-CAPACITY. S4 flags remain on many MoE cells (see noise
note) — MoE ratios are medians with ~±10% cell-level uncertainty at b4.

## R = TPOT(lever)/TPOT(bf16) — DENSE Qwen2.5-7B (TP1)

| arm | b1/2k | b1/16k | b1/32k | b8/2k | b8/16k | b8/32k | b32/2k | b32/16k | b32/32k |
|---|---|---|---|---|---|---|---|---|---|
| W4 Marlin | **0.604** | 0.595 | 0.607 | **0.596** | 0.669 | 0.692 | **0.701** | 0.756 | (denom over-cap) |
| W4 Machete | 0.729 | 0.734 | 0.743 | 0.715 | 0.783 | 0.817 | 0.804 | 0.828 | " |
| fp8 W8A8 | 0.709 | 0.720 | 0.713 | 0.704 | 0.779 | 0.820 | 0.787 | 0.830 | " |
| KV fp8 | 1.017 | 1.010 | 0.996 | 0.981 | 0.904 | 0.860 | 0.943 | 0.871 | " |
| window 512 | 1.000 | 0.949 | 0.901 | 0.952 | 0.749 | **0.626** | 0.879 | **0.541** | " |
| skip50 | 0.570 | 0.545 | 0.533 | 0.566 | 0.512 | 0.463 | 0.541 | 0.438 | " |
| skip25 | 0.778 | — | 0.759 | — | — | — | 0.770 | — | " |

## R — MoE Qwen3-30B-A3B (attention-DP4+EP4, NVLink), batch is GLOBAL

| arm | b4/2k | b4/16k | b4/32k | b8/2k | b8/16k | b8/32k | b32/2k | b32/16k | b32/32k |
|---|---|---|---|---|---|---|---|---|---|
| fp8-Marlin (W8A16) | 0.952 | 0.994 | 0.659† | 0.904 | 0.902 | 1.345† | 0.900 | 0.967 | 0.929 |
| fp8-block (native) | 1.036 | 0.915 | 0.736† | 1.010 | 0.989 | 1.163† | 1.029 | 0.879 | 0.926 |
| KV fp8 | 1.093 | 0.925 | 0.693† | 1.085 | 0.828 | 0.882 | 1.059 | 0.943 | 0.706 |
| window 512 | 1.097 | 0.931 | 0.684 | 0.983 | 0.677 | 0.645 | 0.992 | 0.647 | **0.348** |
| local-route | **0.826** | 0.834 | 0.789 | **0.826** | 0.868 | 0.939 | 0.847 | 0.736 | 0.829 |
| skip-a2a (timing) | 0.788 | 0.762 | 0.721 | 0.761 | 0.715 | 1.030 | 0.882 | 0.805 | 0.763 |
| skip50 | 0.636 | 0.629 | 0.436 | 0.573 | 0.506 | 0.492 | 0.555 | 0.523 | 0.444 |

† = high-variance cells (S4, b4-b8 long-ctx); treat as parity-band, not signal.

## The map's headline: the best lever IS regime-dependent (≥2 clean crossovers)

1. **Dense crossover along (ctx × batch): W4-Marlin → window.** W4 owns the
   short-ctx/low-batch corner (0.60); window is a strict no-op at 2k (1.00,
   as pre-registered) but overtakes at b8/32k (0.626 vs 0.692) and dominates
   at b32/16k (0.541 vs 0.756). Weight-read share shrinks as KV grows —
   the byte-budget inversion, now as one continuous surface.
2. **MoE crossover along ctx: local-route → window.** Local routing is the best
   accept-preserving lever at short ctx (0.83); window takes over from 16k
   (0.65-0.68) and reaches **0.35** at b32/32k. Weight-quant NEVER leads a
   cell — P74's parity verdict, now at 18-cell coverage on TWO kernels
   (Marlin W8A16 and native FI-CUTLASS block-fp8).
3. **Kernel choice is itself regime-dependent — and the contaminated first
   pass had it wrong.** Clean decode TPOT: Marlin beats Machete at EVERY dense
   cell (M=1..32). The pre-fix data showed Marlin "losing" at b8+/16k+ — that
   was Marlin's slow PREFILL leaking into TPOT via chunked-prefill overlap,
   not decode. (P75-E1's Marlin>Machete at M=1 was right and extends to b32.)

## Prediction scorecard (pre-registered in README)

| # | prediction | verdict |
|---|---|---|
| 1 | L1a dense ~0.6 at b1, degrades with batch/ctx; MoE ≈1.0 | **CONFIRMED** (0.604→0.756; MoE core band 0.88-1.03) |
| 1b | L1b wins only where GEMMs MAC-bound; MoE ≈1 (tiny per-expert M) | **CONFIRMED** (dense b32/2k 0.787 modest; MoE 1.03 ≈1) |
| 1c | L1c improves with ctx, ~flat in batch | **CONFIRMED** (dense 1.02→0.86, MoE →0.71) — global-lever caveat stands (P74): standalone R shows what a draft-only KV pool WOULD buy |
| 2 | window monotone in ctx, no-op at 2k | **CONFIRMED** (both models; MLA deferred) |
| 3 | skip sub-linear at b1 (fixed floor), ~linear at scale | **CONFIRMED** (0.570 at b1 vs 0.5 ideal; 0.44-0.46 at b32 — *below* 0.5 because KV/comm halve too) |
| 4 | local-route ≈ no-op on NVLink | **REFUTED — law refined**: 0.74-0.94 measured. P12's "EP-width doesn't matter on NVLink" was TP-EP (no dispatch collective, E0). Under DP-EP the AgRs gather/RS + remote-token expert compute are real even on NVLink; local-route cuts both. The PCIe tier (deferred) should widen this. |
| 5 | no single lever R<0.8 in every cell | **REFUTED by skip50 (cost side only)**: 0.44-0.64 everywhere. Expected: layer-skip is the lever whose ACCEPTANCE collapses (P17, super-linear), so on a cost-only map it is trivially strong. This is precisely why the strategy map must be composed with τ — `speedup = τ/(γ·R+1)` — before selection (next phase). |

## Bonus finding — levers extend the serviceable envelope (residency)

At b32/32k the dense bf16 baseline is OVER-CAPACITY (scheduler queues 7/32
requests, KV 87%, TPOT 117-140 ms thrashing) while window (6.8 ms), KV-fp8
(13.1), W4 (12.2-13.6), skip (5.8-10.8) all run it cleanly. A draft lever can
therefore create operating points where the target alone cannot even hold the
batch — the P28 residency story surfacing spontaneously in the cost map.

## Measurement notes (hard-won, encoded in the runner)

- **Warm pass per cell is mandatory**: bench-serve reuses prompts per seed, so
  run 1 measured decode interleaved with chunked prefill (up to 4.7× inflation
  at b32/16k) — and it INVERTED the Marlin/Machete decode comparison.
- **Over-capacity detection**: vLLM v1 logs no 'preempt'; the signature is
  "Waiting: N reqs" in the periodic stats line (S5 gate).
- **DP4 placement bimodality**: per-run TPOT at b4-b8 long-ctx is bimodal
  (request placement across DP ranks; prefix-cache-aware routing can clump the
  warmed prompts). Median-of-4 quoted; S4 flags retained. Dense TP1 shows no
  such variance.
- bash trap: `GROUPS` is a reserved builtin array — assignments silently
  ignored (cost one repair round).
- dummy-weight ≈ real-weight cost on dense (1.00±0.02) and MoE (0.95-1.15,
  noisier) — validates dummy denominators for the skip arms.

## Deferred / next

- MLA + PCIe tiers (user-deferred; runner groups exist: `e1_sweep.sh mla|pcie`).
  PCIe is where local-route's refined law predicts its largest win.
- Compose R with measured τ per lever (window 4.75/5 @K=4 P74, fp8 β~0.95 P18,
  W4 β~0.92 P22, skip collapse P17, local-route β P24-29) via the P75-validated
  formula → the strategy-selection map, then fit the term-decomposed cost model
  (Phase 77) to `summary.csv`.
