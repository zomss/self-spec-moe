# Phase 35 — GQA (FLASH_ATTN) draft FULL-CG accept-length collapse

**Source phase.** `34_worldA_system/results_W7_fg2.md` (MLA fix) +
`results_W7_qwen30b.md` §3a (GQA collapse observed, root cause unconfirmed).

**Objective.** Reproduce the draft FULL-CG (`VLLM_SELF_SPEC_DRAFT_FULL_CG=1`)
accept-length collapse on a GQA / FLASH_ATTN model fast (no EP), find the
GQA-specific root cause the MLA fix doesn't cover, fix it, validate.

**Decision criterion (gate).** Qwen3-8B FULL-CG accept_len ≈ the PIECEWISE
reference (~4.9). Then Qwen3-30B DP=2 accept jumps from ~1.9 to ~4–5.

See `results_W7_gqa_debug.md` for the full writeup.

## Scripts
- `scripts/repro_qwen3_8b.py` — single-GPU A/B harness (MODE=piecewise|fullcg).
- `scripts/diag_steps.py` — per-step draft-token dump (W7_GQA_DEBUG_STEPS).

## Commands (single GPU)
```
HF_HOME=/home/smcho/.cache/huggingface CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/data/smcho/ssm-gqa VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 \
MODE=piecewise|fullcg MB_BATCH=16 MB_OUTLEN=128 MB_K=4 \
/data/smcho/self-spec-moe/.venv/bin/python scripts/repro_qwen3_8b.py
```
