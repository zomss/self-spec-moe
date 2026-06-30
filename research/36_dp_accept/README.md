# Phase 36 — World-A self-spec DP acceptance drop (isolation + fix)

**Source phase.** `34_worldA_system` (World-A full-replica self-spec) +
`35_gqa_fullcg_debug` (draft FULL-CG GQA fix). Symptom carried in: Qwen3-30B-A3B
(GQA-MoE, FP8 draft) full-replica self-spec accept ~1.65 at DP=2, ~1.0 at DP=8,
under BOTH PIECEWISE and FULL-CG — while DeepSeek-V2-Lite (MLA-MoE, bf16) is
healthy (~2.7 at K=2).

**Objective.** Isolate the cause on a small GQA-MoE that fits at DP=1 and DP=2
(`Qwen/Qwen1.5-MoE-A2.7B`), fix it if it is a bug, validate on Qwen3-30B.

**Decision criterion.** Isolation 3-cell table (DP1-bf16 / DP2-bf16 / DP2-FP8)
pinpoints DP-bug vs FP8 vs intrinsic. Fix gate: Qwen1.5-MoE accept ≈ the eager
(uncompiled) reference (~4.8); then Qwen3-30B DP=2 accept jumps ~1.65 -> ~4-5.

See `results_W7_dp_accept.md` for the full writeup.

## Scripts
- `scripts/accept_isolate.py` — DP-rank multiprocessing accept_len harness.
  Knobs: AI_MODEL/AI_DP/AI_TP/AI_QUANT/AI_K/AI_BATCH/AI_OUTLEN/AI_EP/AI_EAGER/
  AI_FULLREP/AI_LOCALROUTE/AI_CGMODE/AI_TAG.
- `scripts/run_cell.sh` — env hygiene wrapper (forced-PCIe, DEEP_GEMM off).
- `scripts/qwen30b_validate.sh` — Qwen3-30B DP=2 before/after the fix.
