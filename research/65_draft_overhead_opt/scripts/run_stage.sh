#!/bin/bash
# Phase 65: run one stage's full 16k measurement set sequentially:
#   w512k4 at b8,12 (headline) then w512k2 at b8 (fixed/marginal split).
# Usage: run_stage.sh {base|f1|f12|f123}
set -u
STAGE="${1:?base|f1|f12|f123}"
PHASE=/h/v-sukmincho/self-spec-moe/research/65_draft_overhead_opt
echo "[p65] ===== stage $STAGE: w512k4 b8,12 ====="
bash "$PHASE/scripts/run_arm.sh" w512k4 "$STAGE" 8,12 2 2400 || exit 1
echo "[p65] ===== stage $STAGE: w512k2 b8 ====="
bash "$PHASE/scripts/run_arm.sh" w512k2 "$STAGE" 8 2 1800 || exit 1
echo "[p65] ===== stage $STAGE COMPLETE ====="
