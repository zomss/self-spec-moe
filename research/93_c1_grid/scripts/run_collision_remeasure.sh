#!/bin/bash
# Re-measure the C1 dense arms implicated in the cache-key collision, now
# that SpeculativeConfig.compute_hash() includes draft_model_config.
#
#   fp8dyn + w4a8cut : shared key 7cd76306f0 -- both showed plausible
#                      ACCEPTANCE, so any corruption is in COST (R).
#   w4a16            : CONTROL. It shared a key with w8int8 but was the
#                      primer (its own numbers looked right); it must come
#                      back unchanged, otherwise the re-measure itself is
#                      suspect.
#
# Runs on ONE SHARED cache root on purpose: with the fix, sharing must be
# safe. Env for each arm is copied verbatim from run_stage_a.sh so the only
# difference from C1 is the code fix.
set -uo pipefail
ARM=${1:?usage: run_collision_remeasure.sh fp8dyn|fp8dyn_nc|w4a8cut|w4a16}
REPO=/data/smcho/self-spec-moe
PHASE=$REPO/research/93_c1_grid
P82DATA=$REPO/research/82_runtime_switching/data
COMPILE=$PHASE/scripts/compile_cells_93.py
export HF_HOME=/data/smcho/huggingface
export PATH="$REPO/.venv/bin:$PATH"
cd "$REPO"
GPU="${CR_GPU:-0}"
MODEL="Qwen/Qwen3-8B"; KVLIM=260000
BATCHES="1,4,8,16,32,64,128"; CTXS="2000,8000,14000"; KS="k2 k4 k6"
COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"
case "$ARM" in
  # C1 ran fp8dyn with TORCHINDUCTOR_FORCE_DISABLE_CACHES=1, a workaround
  # for the collision. Post-fix that env FAILS outright ("Cannot precompile
  # with force_disable_caches=True"), which proves fp8dyn never compiled its
  # own graph in C1 -- it only ever booted by loading a foreign cached one.
  fp8dyn)  DRAFT="$HOME/ckpts/Qwen3-8B-FP8-dynamic";   EXTRA="$SHARED TORCHINDUCTOR_FORCE_DISABLE_CACHES=1";;
  fp8dyn_nc) DRAFT="$HOME/ckpts/Qwen3-8B-FP8-dynamic"; EXTRA="$SHARED";;
  w4a8cut) DRAFT="$HOME/ckpts/Qwen3-8B-W4A8-gptq";     EXTRA="$SHARED";;
  w4a16)   DRAFT="$HOME/ckpts/Qwen3-8B-W4A16-INT4";    EXTRA="$SHARED";;
esac
gpu_cleanup() {
  local u; u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
  for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ',' | awk -v u="$u" '$2==u{print $1}'); do
    [ "$(ps -o user= -p "$p" 2>/dev/null)" = "smcho" ] || continue
    kill "$p" 2>/dev/null; sleep 2; kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null
  done; sleep 3
}
csv="cells_93v2_dense_${ARM}.csv"
for karm in $KS; do
  if [ -f "$P82DATA/$csv" ] && grep -q "^${karm}," "$P82DATA/$csv"; then
    echo "[CR:$ARM] skip $karm"; continue
  fi
  echo "[CR:$ARM] measure $karm"
  env CUDA_VISIBLE_DEVICES=$GPU COMPILE_MODEL="$MODEL" COMPILE_DRAFT="$DRAFT" \
      COMPILE_TP=1 COMPILE_BATCHES="$BATCHES" COMPILE_CTXS="$CTXS" \
      COMPILE_KV_LIMIT=$KVLIM COMPILE_CELLS="$csv" \
      COMPILE_TABLE="table_93v2_dense_${ARM}.json" \
      VLLM_CACHE_ROOT="/data/smcho/vllm_cache_postfix_shared" \
      $EXTRA timeout 5400 .venv/bin/python "$COMPILE" --measure "$karm" \
      >> "$PHASE/logs/collision_remeasure.log" 2>&1 || echo "[CR:$ARM] FAIL $karm"
  gpu_cleanup
done
cp -f "$P82DATA"/cells_93v2_dense_*.csv "$PHASE/data/" 2>/dev/null
echo "[CR:$ARM] COLLISION-REMEASURE-${ARM}-DONE"
