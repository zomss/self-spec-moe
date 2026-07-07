#!/bin/bash
# Phase 67: arm-B 16k with the step-0 decode compaction ON. Runs the clean
# F/D pair (K=2,4 b8) then the FINE decomposition (K=4 b8). Compare F/D and the
# draft_forward_first region to the baseline (F=54.4 D=16.9; step0 fwd 14.6).
set -u
S=/h/v-sukmincho/self-spec-moe/research/67_propose_fixed_opt/scripts
DEC="VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1"
echo "=== arm-B + step0-decode: clean K2,4 b8 (F/D) ==="
bash "$S/run_prof.sh" armb_dec_clean 2,4 8 off 16960 "$DEC"
echo "=== arm-B + step0-decode: FINE K4 b8 (decomposition) ==="
bash "$S/run_prof.sh" armb_dec_fine 4 8 1 16990 "$DEC"
echo "=== armb_decode done ==="
