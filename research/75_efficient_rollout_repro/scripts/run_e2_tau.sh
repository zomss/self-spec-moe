#!/bin/bash
# Phase 75 -- E2 (KEY #2): block efficiency tau of a weight-quantized self-draft.
#
# Paper Tab.4 (Qwen2.5-7B-Instruct, b1, seq 2k, rollout temperature 1.0):
#     RTN W4: tau = 3.59 / 5.18 / 6.70   at gamma = 3 / 5 / 7
#     RTN W8: tau = 3.94 / 5.87 / 7.79
# PASS: within +-0.3.
#
# This is ALSO the accept-vs-temperature de-risk Phase 74 deferred: every accept number
# we have is greedy; theirs is at T=1.0. We run both.
#
# tau is a per-sequence statistic and does not depend on batch, so the sweep runs at
# b8 (8 distinct math prompts -> real prompt-to-prompt statistics) and a b1 control
# confirms batch-independence. The harness indexes the prompt bank by batch, so at b1
# every run would otherwise reuse prompt[0] (hence W7_PROMPT_OFFSET).
#
# DEVIATIONS recorded here, not hidden:
#  * our 2k context is filler-prose padding + the question; theirs is a genuine rollout
#    prefix (the model's own reasoning). Control: CTX=0 (unpadded, short) arm.
#  * SHARED_KV defaults to 0 (faithful: the drafter computes its OWN KV). SHARED_KV=1
#    would let the quantized draft attend the target's exact KV and INFLATE tau.
#
# Run BY PATH: `bash scripts/run_e2_tau.sh`
set -u
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PHASE/../.." && pwd)"
P52="$REPO/research/52_two_node_e2e"
PY="$REPO/.venv/bin/python"
CKPT_ROOT="${CKPT_ROOT:-$HOME/ckpts}"
NAME="$(basename "${E75_MODEL:-Qwen/Qwen2.5-7B-Instruct}")"
PROMPTS="$PHASE/prompts/math_prompts.txt"
mkdir -p "$PHASE/logs" "$PHASE/data"
ME="$(whoami)"
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 4; }

W4="$CKPT_ROOT/${NAME}-W4A16-INT4-sym"
W8="$CKPT_ROOT/${NAME}-W8A16-INT8-sym"
W4A="$CKPT_ROOT/${NAME}-W4A16-INT4-asym"

# ---------------------------------------------------------------- correctness gate
echo "[e2] GATE: greedy spec must be token-identical to greedy no-spec"
if [ ! -d "$W4" ]; then echo "[e2] ABORT: missing $W4 (run make_ckpts.sh)"; exit 1; fi
kill_mine
( source "$PHASE/scripts/env_e75.sh"
  "$PY" "$PHASE/scripts/check_lossless.py" --mode nospec --out "$PHASE/data/lossless_nospec.json"
) > "$PHASE/logs/e2_gate_nospec.log" 2>&1 || { echo "[e2] gate nospec FAILED -> logs/e2_gate_nospec.log"; exit 1; }
kill_mine
( source "$PHASE/scripts/env_e75.sh"
  "$PY" "$PHASE/scripts/check_lossless.py" --mode spec --draft "$W4" --gamma 5 \
        --out "$PHASE/data/lossless_spec.json"
) > "$PHASE/logs/e2_gate_spec.log" 2>&1 || { echo "[e2] gate spec FAILED -> logs/e2_gate_spec.log"; exit 1; }
kill_mine
if ! "$PY" "$PHASE/scripts/check_lossless.py" --compare \
      "$PHASE/data/lossless_nospec.json" "$PHASE/data/lossless_spec.json"; then
  echo "[e2] ABORT: losslessness gate failed. Do not report tau until this passes."
  exit 1
fi

# ---------------------------------------------------------------- tau sweep
# tau <draft-name> <draft-path> <gamma> <temp> <batch> <ctx> <shared_kv>
tau(){
  local dn=$1 dp=$2 g=$3 t=$4 b=$5 ctx=$6 skv=$7
  [ -d "$dp" ] || { echo "[e2] SKIP $dn (missing $dp)"; return; }
  local tag="${dn}_g${g}_T${t}_b${b}_ctx${ctx}_skv${skv}"
  local LOG="$PHASE/logs/e2_${tag}.log"
  kill_mine
  ( source "$PHASE/scripts/env_e75.sh"
    export E75_SHARED_KV="$skv"; export VLLM_SELF_SPEC_SHARED_KV="$skv"
    export W7_SPEC_MODEL="$dp" W7_KS="$g" W7_TEMP="$t" W7_TOPP=1.0
    export W7_BATCHES="$b" W7_ITERS=2 W7_WARMUP=1 W7_TAG="p75_e2_${tag}"
    export W7_CTX_TOKENS="$ctx" W7_MAX_MODEL_LEN=$((ctx > 0 ? ctx + 1024 : 4096))
    export W7_PROMPT_FILE="$PROMPTS" W7_CHAT=1 W7_PROMPT_OFFSET=0
    export W7_OUT="$PHASE/data" W7_MASTER_PORT=$((19100 + g + b + RANDOM % 30))
    W7_NODE_RANK=0 timeout 2400 "$PY" "$P52/scripts/w7_2node.py" spec ) > "$LOG" 2>&1
  local res; res=$(grep -hE 'W7-2N K=' "$LOG" | tail -1)
  local q;   q=$(grep -hoE 'Draft model quantization: [a-z0-9_-]+' "$LOG" | head -1)
  echo "[e2] ${tag}: ${res:-NO RESULT}"
  echo "        ${q:-WARNING: draft quantization NOT logged -- draft may be bf16!}"
  grep -hE 'Traceback|out of memory' "$LOG" | tail -1 | sed 's/^/        ERR: /'
  kill_mine
}

echo "[e2] tau sweep (primary: T=1.0, b8, ctx2k, SHARED_KV=0)"
for g in 3 5 7; do tau w4sym "$W4" "$g" 1.0 8 2048 0; done
for g in 3 5 7; do tau w8sym "$W8" "$g" 1.0 8 2048 0; done

echo "[e2] paper-fidelity: ASYMMETRIC RTN (they use asym; ours above is sym)"
for g in 3 5 7; do tau w4asym "$W4A" "$g" 1.0 8 2048 0; done

echo "[e2] controls"
tau w4sym "$W4" 5 0.0 8 2048 0     # greedy control: isolates the temperature effect
tau w4sym "$W4" 5 1.0 1 2048 0     # b1 control: tau should be batch-independent
tau w4sym "$W4" 5 1.0 8 0    0     # unpadded control: no filler prose
tau w4sym "$W4" 5 1.0 8 2048 1     # SHARED_KV=1: expect INFLATED tau (not the paper's setup)

echo "[e2] DONE ($(date +%H:%M:%S))"
echo "[e2] compare accept_len to paper: W4 3.59/5.18/6.70, W8 3.94/5.87/7.79 (g=3/5/7), +-0.3"
