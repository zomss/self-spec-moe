#!/bin/bash
# Phase 70 probe: single-node DP8 16k arm-B + FULLCG with the descriptor debug
# (W7_FULLCG_DBG=1) ON. Lean (1 iter, 0 warmup) -- we only need the one-time
# [fullcg-dbg] lines that print on the first chain propose. Usage: probe.sh K
set -u
K="${1:-4}"; PORT="${2:-17400}"
PHASE=/h/v-sukmincho/self-spec-moe/research/70_fullcg_coverage
export W69_ITERS=1 W69_WARMUP=0
"$PHASE/scripts/run_1node.sh" probe_fullcg "$K" 8 off "$PORT" \
  "VLLM_SELF_SPEC_DRAFT_FULLCG=1 W7_FULLCG_DBG=1"
