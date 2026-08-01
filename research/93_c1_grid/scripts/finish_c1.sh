#!/bin/bash
# Phase 93 C1 finisher: drive the last two columns to completion on a
# flaky shared box via bounded resumable retries. Each Stage-B driver
# skips arms with a "complete": true marker, so re-running only re-does
# what transient/co-tenant/wedge failures left unmarked.
set -uo pipefail
PHASE=/data/smcho/self-spec-moe/research/93_c1_grid
LOG=$PHASE/logs

wait_free() {  # both GPUs 6+7 quiet
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6)" -lt 2000 ] \
     && [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7)" -lt 2000 ]; do
    sleep 120
  done
}

complete() {  # $1=arch : all its Stage-B arm jsons carry the marker?
  local arch=$1 want got
  want=$(bash -c "source $PHASE/scripts/run_stage_b.sh $arch 2>/dev/null" 2>/dev/null; true)
  # count expected arms from the driver's ARMLIST by dry-run grep
  got=$(grep -l '"complete": true' $PHASE/data/stageb_${arch}_*.json 2>/dev/null | wc -l)
  echo "$got"
}

# 32B Stage B: up to 6 passes (Humming K4 wedge + co-tenant transients)
for pass in 1 2 3 4 5 6; do
  n=$(grep -l '"complete": true' $PHASE/data/stageb_q3_32b_*.json 2>/dev/null | wc -l)
  echo "[finish] q3_32b pass $pass: $n/5 arms complete"
  [ "$n" -ge 5 ] && break
  wait_free
  STAGEB_GPU=6,7 bash $PHASE/scripts/run_stage_b.sh q3_32b >> $LOG/stage_b_q3_32b.log 2>&1 || true
done
echo "Q32B-STAGEB-FINAL: $(grep -l '"complete": true' $PHASE/data/stageb_q3_32b_*.json 2>/dev/null | wc -l)/5"

# Llama AR re-measure: 3 cleared cells (off + 2 arms), GPU 6, up to 4 passes
for pass in 1 2 3 4; do
  n=$(grep -l '"complete": true' $PHASE/data/stageb_llama_off.json $PHASE/data/stageb_llama_w4a16_k4.json $PHASE/data/stageb_llama_win2048_k4.json 2>/dev/null | wc -l)
  echo "[finish] llama-remeasure pass $pass: $n/3 cells complete"
  [ "$n" -ge 3 ] && break
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6)" -lt 2000 ]; do sleep 120; done
  STAGEB_GPU=6 STAGEB_ARMS="off,w4a16_k4,win2048_k4" bash $PHASE/scripts/run_stage_b.sh llama >> $LOG/stage_b_llama.log 2>&1 || true
done
echo "C1-FINISH-DONE"
