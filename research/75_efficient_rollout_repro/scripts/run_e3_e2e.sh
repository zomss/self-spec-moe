#!/bin/bash
# Phase 75 -- E3: end-to-end confirmation. Real tok/s, self-spec vs autoregressive.
#
# E1 times a standalone CUDA-graphed forward; the REAL self-spec draft chain runs
# PIECEWISE/eager (Phase 35/72). So E1's Tq/Tp is a LOWER BOUND -- optimistic. E3 is the
# truth. The decisive comparison is:
#
#     measured E2E speedup   vs   tau / (gamma * (Tq/Tp)_E1 + 1)
#
#   close        -> the lever survives the harness. Paper reproduced.
#   E2E much worse -> the lever is REAL but OUR harness eats it (PIECEWISE draft chain).
#                     That is a systems bug on our side, NOT a refutation of the paper.
#                     Exactly the Phase-74 fp8 story: ideal +6.5% -> measured -2%.
#
# Reference points come from E1/E2 logs; pass GAMMA (tau-optimal) via env.
# Run BY PATH: `bash scripts/run_e3_e2e.sh`
set -u
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PHASE/../.." && pwd)"
P52="$REPO/research/52_two_node_e2e"
PY="$REPO/.venv/bin/python"
CKPT_ROOT="${CKPT_ROOT:-$HOME/ckpts}"
NAME="$(basename "${E75_MODEL:-Qwen/Qwen2.5-7B-Instruct}")"
W4="$CKPT_ROOT/${NAME}-W4A16-INT4-sym"
PROMPTS="$PHASE/prompts/math_prompts.txt"
GAMMA="${E75_GAMMA:-5}"
mkdir -p "$PHASE/logs" "$PHASE/data"
ME="$(whoami)"
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 4; }

# run <mode> <arm> <gamma> <temp> <batch>
run(){
  local mode=$1 arm=$2 g=$3 t=$4 b=$5
  local tag="${arm}_g${g}_T${t}_b${b}"
  local LOG="$PHASE/logs/e3_${tag}.log"
  kill_mine
  ( source "$PHASE/scripts/env_e75.sh"
    export VLLM_SELF_SPEC_SHARED_KV=0
    # E1: Marlin beats Machete at b1 decode (Tq/Tp 0.58 vs 0.72), and Marlin is the
    # paper's kernel + the one the 1.29x prediction assumed. Force it for the draft
    # (bf16 target has no int4 kernel, so this only affects the spec arm).
    e75_force_marlin
    [ "$arm" = spec ] && export W7_SPEC_MODEL="$W4"
    export W7_KS="$g" W7_TEMP="$t" W7_TOPP=1.0
    export W7_BATCHES="$b" W7_ITERS=4 W7_WARMUP=1 W7_TAG="p75_e3_${tag}"
    export W7_CTX_TOKENS=2048 W7_MAX_MODEL_LEN=4096
    export W7_PROMPT_FILE="$PROMPTS" W7_CHAT=1
    export W7_OUT="$PHASE/data" W7_MASTER_PORT=$((19400 + g + b + RANDOM % 30))
    W7_NODE_RANK=0 timeout 700 "$PY" "$P52/scripts/w7_2node.py" "$mode" ) > "$LOG" 2>&1
  echo "[e3] ${tag}: $(grep -hE 'W7-2N (nospec|K=)' "$LOG" | tail -1)"
  grep -hE 'Traceback|out of memory' "$LOG" | tail -1 | sed 's/^/       ERR: /'
  kill_mine
}

[ -d "$W4" ] || { echo "[e3] ABORT: missing $W4 (run make_ckpts.sh)"; exit 1; }

echo "[e3] gamma=$GAMMA  (set E75_GAMMA to E2's tau-optimal value)"
echo "[e3] --- rollout setting: temperature 1.0, b1 (their point) ---"
run nospec nospec "$GAMMA" 1.0 1
run spec   spec   "$GAMMA" 1.0 1

echo "[e3] --- greedy control ---"
run nospec nospec "$GAMMA" 0.0 1
run spec   spec   "$GAMMA" 0.0 1

echo "[e3] --- batch sensitivity (self-spec decays as batch grows) ---"
for b in 4 8; do
  run nospec nospec "$GAMMA" 1.0 "$b"
  run spec   spec   "$GAMMA" 1.0 "$b"
done

cat <<EOF

[e3] DONE ($(date +%H:%M:%S))
Compute speedup = spec_tok/s / nospec_tok/s at each point, then compare to
  predicted = tau / ($GAMMA * (Tq/Tp)_from_E1 + 1)
Paper's idealized point: tau=5.18, Tq/Tp=0.360, gamma=5 -> 1.85x (A100).
Predicted on H100 (F~0.5ms): Tq/Tp~0.386 -> ~1.77x.

If E1 passed but E3 << predicted: the lever is real, the PIECEWISE draft chain is
eating it. Report that as a harness finding, not as a refutation of the paper.
EOF
