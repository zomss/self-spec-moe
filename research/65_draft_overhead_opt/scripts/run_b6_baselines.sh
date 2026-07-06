#!/bin/bash
# Phase 65: b6 baselines for the f123 resident point — f12 at b6 (K4+K2)
# isolates Fix 3 at the same batch, and no-spec b6 gives the speedup
# denominator (16k is batch-flat: 386.8/395.7/379.0 at b8/12/32, so b6
# should land ~385-395; measured for rigor).
set -u
PHASE=/h/v-sukmincho/self-spec-moe/research/65_draft_overhead_opt
echo "[p65] ===== f12 w512k4 b6 ====="
bash "$PHASE/scripts/run_arm.sh" w512k4 f12 6 2 2400 _b6 || exit 1
echo "[p65] ===== f12 w512k2 b6 ====="
bash "$PHASE/scripts/run_arm.sh" w512k2 f12 6 2 1800 _b6 || exit 1
echo "[p65] ===== nospec b6 ====="
bash "$PHASE/scripts/run_arm.sh" nospec f12 6 2 1800 _b6 || exit 1
echo "[p65] ===== b6 baselines COMPLETE ====="
