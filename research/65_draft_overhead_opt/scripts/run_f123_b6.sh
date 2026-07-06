#!/bin/bash
# Phase 65: f123 RESIDENT diagnostic point. The fp8 replica shrinks the pool
# to 115,968 tokens/rank at EP16 (22 GiB replica), so protocol b8 = 114%
# over-pool (waves; 43.7 tok/s, accept 4.548) and b12 = 171%. b6 = 85% is
# the largest clean resident batch: measure K4 + K2 there.
set -u
PHASE=/h/v-sukmincho/self-spec-moe/research/65_draft_overhead_opt
echo "[p65] ===== stage f123: w512k4 b6 (resident) ====="
bash "$PHASE/scripts/run_arm.sh" w512k4 f123 6 2 2400 _b6 || exit 1
echo "[p65] ===== stage f123: w512k2 b6 (resident) ====="
bash "$PHASE/scripts/run_arm.sh" w512k2 f123 6 2 1800 _b6 || exit 1
echo "[p65] ===== f123 b6 COMPLETE ====="
