# Phase 87 — three new levers: infra check first, then actual runs

Source: paper/DIRECTION.md + user directive 2026-07-18 (include as many
strategies as possible — we position the strategy space; each lever must
enter at FULL-POTENTIAL form; realizations built only on map selection).

## Levers and their entry criteria

L1 **Calibrated sparse pruning**: SparseGPT one-shot 2:4 (the fair form;
    magnitude was gated dead), and the PRIZE composition int4 x 2:4 via
    the marlin_24 kernel (~0.125x bf16 weight bytes). Entry: beta >= ~0.9
    at 32B-class; could beat our own 1.63x headline.
L2 **Activation quant**: isolate the A-quant beta factor (never measured
    alone); W4A8 as the compute-bound quant for BATCH rows (b1 is
    weight-bound: W-only wins there, measured).
L3 **Draft-only KV-quant pool**: beta already measured (.985-.997 on
    QK-normed); the missing piece is a REGIME where it wins — a
    retrieval/needle workload bank where window beta should collapse
    (new workload axis for the maps). Pool plumbing (dual-write fp8 KV +
    draft binding) built ONLY if the map selects it.

## E0 — infrastructure check matrix (do FIRST)

| # | check | how |
|---|---|---|
| C1 | SparseGPT modifier in lc_cuda_venv | import SparseGPTModifier |
| C2 | vLLM 2:4 kernels: marlin_24 (W4A16-2:4), sparse-only path | grep kernels / quant configs |
| C3 | llm-compressor recipe: 2:4 + GPTQ combined (sparsity then quant) | docs/modifier signature |
| C4 | offline beta path for 2:4-calibrated: ckpt -> decompressed scoring (GPTQ-scorer pattern) works for sparse ckpts | run_compressed=False semantics |
| C5 | activation fake-quant hooks in the 77 harness (pre-forward on Linear inputs, dynamic e4m3 per-token) | new ~30-line ctx manager |
| C6 | W4A8 scheme in llm-compressor + CutlassW4A8 kernel in vLLM (it is on our DISABLED list => exists) | imports/grep |
| C7 | retrieval bank: needle-style prompts (planted facts at controlled depths in the filler, question at end) buildable on the 77 ref pipeline | script |
| C8 | dual-write fp8 KV pool feasibility: where verify writes KV (reshape_and_cache) + P66 draft binding surface; effort estimate ONLY | code read |
| C9 | CPU-side KV requant + upload cost (the switching-cost path from DIRECTION 3.2) | one timed measurement |

## E1 — actual runs (per check outcome)

- R1 (L1): SparseGPT-2:4 ckpt Qwen3-8B (calib 512 C4) -> beta score
  (decompressed, paired refs). If beta >= .90: marlin_24 W4A16-2:4 ckpt
  -> e2e arm at b1 + batch. Then 32B.
- R2 (L2): beta arms actfp8 (A-only) and w4a8-proxy on dense refs; W4A8
  ckpt 8B -> e2e batch rows (b8/b16) where compute-bound.
- R3 (L3): build retrieval bank -> beta {baseline, win512, win128, kvq}
  on it (prediction: window collapses, kvq holds ~.98) -> map pricing
  with the workload axis -> pool build decision.

## Gates

- Disjoint-artifact rule everywhere (calibration data C4; eval refs
  ondist/math/retrieval banks).
- Each lever's e2e only for map-selected realizations.
- All results DRAFT-flagged like paper/TABLES_DRAFT.md.
