#!/bin/bash
# G98-G cost-stage ratchet.
#
# The cost stage is gated (a clamped boot is rejected, not recorded) and
# resumable (an accepted cell is never re-measured), so repeated passes
# accumulate cells across whatever clean windows the box grants. This box
# flips state on a scale of tens of minutes, so one pass is never enough.
#
# Usage: run_g98g_cost_loop.sh [passes] [sleep_seconds] [gpu]
set -u
PASSES=${1:-40}
SLEEP_S=${2:-120}
GPU=${3:-0}
REPO_ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
cd "$REPO_ROOT" || exit 2
WANT=$(taskset -c 32-63 .venv/bin/python -c "
import sys; sys.path.insert(0,'research/98_selector_demo/scripts')
import run_w98_g98g_e2e as g; print(len(g.cost_configs()))")
for i in $(seq 1 "$PASSES"); do
  taskset -c 32-63 .venv/bin/python \
    research/98_selector_demo/scripts/run_w98_g98g_e2e.py --stage cost --gpu "$GPU"
  have=$(find research/98_selector_demo/data/g98_g/cost -maxdepth 1 -name '*.json' \
    ! -name '*telemetry*' 2>/dev/null | wc -l)
  echo "[costloop] pass $i: $have/$WANT cells accepted $(date -Is)"
  if [ "$have" -ge "$WANT" ]; then
    echo "[costloop] COST STAGE COMPLETE $(date -Is)"
    exit 0
  fi
  sleep "$SLEEP_S"
done
echo "[costloop] exhausted $PASSES passes with $have/$WANT cells"
exit 1
