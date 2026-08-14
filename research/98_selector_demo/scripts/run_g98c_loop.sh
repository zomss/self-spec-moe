#!/bin/bash
# G98-C v6 campaign resume loop.
#
# Each attempt runs the authorized runner; the host-load and measurement
# gates reject clamped boots (3 tries per cell) and the runner aborts, so a
# bad box costs one gated attempt and no data. v6 resumes past retained
# attempt traces, so no cleanup is needed between attempts. The interval
# doubles as a clamp probe cadence: each attempt's first boot is a gate
# verdict on the real workload (see probe_w98_flip_sentinel.py for the
# continuous instrument this pairs with).
#
# Usage: run_g98c_loop.sh [attempts] [sleep_seconds]
#   fast ratchet:  run_g98c_loop.sh 40 120
#   slow cadence:  run_g98c_loop.sh 16 1800
set -u
ATTEMPTS=${1:-16}
SLEEP_S=${2:-1800}
REPO_ROOT=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
cd "$REPO_ROOT" || exit 2
# h104 CPU map, all NUMA node 1 (GPU 4's node; NUMA 0 carries other tenants):
# loop+runner parent 56-63, hash-bound telemetry set 64-95, engine lane 96-111
# (set by the runner's own sched_setaffinity). membind=1 keeps first-touch
# allocations off the contended node-0 pool.
for i in $(seq 1 "$ATTEMPTS"); do
  echo "[loop] attempt $i start $(date -Is)"
  numactl --physcpubind=56-63 --membind=1 .venv/bin/python \
    research/98_selector_demo/scripts/run_w98_g98c_round2.py
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[loop] campaign COMPLETE on attempt $i $(date -Is)"
    exit 0
  fi
  echo "[loop] attempt $i failed rc=$rc $(date -Is); next in ${SLEEP_S}s"
  sleep "$SLEEP_S"
done
echo "[loop] exhausted $ATTEMPTS attempts without a clean window"
exit 1
