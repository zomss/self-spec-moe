#!/usr/bin/env bash
# The full factorial: every lattice configuration, every cell, every batch.
#
# Sections 39-43 scored the selector against a "best static" chosen from four
# common arms of which exactly ONE carried a window. `w1024/skip4` won that
# comparison by being the only windowed arm competing at cells where windows
# dominate, not by beating a field -- so "no static policy is best" was never
# tested. Of the 20 window x skip combinations per weight version, the refined
# grid measured five, and the untested region is the one D2(a) measured to be
# constructive (interaction ratio up to 1.670): windowed arms at deep skip,
# where the window has already discarded the context that layer skipping would
# have degraded.
#
# 42 arms (40 lattice + off + stock) x 14 (cell, batch) points = 588 boots,
# about 21 GPU-hours. Cheap cells run first so the lattice question is
# answered before LO's 15 hours are spent.
#
# Resumable: the artifact layer skips a boot whose record already exists and
# refuses one whose file holds a different configuration, so re-running after
# an interruption costs nothing and cannot silently mix arms.
#
# Serial, and gated on a quiet box. This is a timing measurement, and section
# 34 caught a co-tenant corrupting one by 47.6% while section 43's first
# launch OOM'd five boots against a neighbour holding 77 GB.
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=research/98_selector_demo/scripts/run_w98_refined_lo.py
GPU="${GPU:-0}"

# Gates on OUR lane only, not the whole box. CASYS runs auxiliary filler load
# on unreserved GPUs continuously, so a whole-box gate would never open. That
# it is safe to ignore is measured, not assumed: with filler running on seven
# GPUs at 27-70% util, a canary re-measurement on a reserved GPU 0 reproduced
# the quiet-box numbers to 0.03% (stock 399.2 tok/s against 399.3 and 399.0)
# and 0.42% (an armed arm). Section 34's contamination was a co-tenant on the
# SAME device, which this still refuses.
wait_for_quiet() {
  local streak=0 total
  for _ in $(seq 1 480); do
    total=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            -i "${GPU}")
    if [ "$total" -lt 2000 ]; then
      streak=$((streak+1))
      [ "$streak" -ge 2 ] && return 0
    else
      streak=0
    fi
    sleep 30
  done
  echo "[lat] box never went quiet; aborting" >&2
  return 1
}

# cell:batches -- the registered Phase-100 sweep
PLAN="${PLAN:-LI:1,8,16 LIO:1,8,16 SS:1,8,32,64 LO:1,8,16,32}"

for entry in $PLAN; do
  cell="${entry%%:*}"
  for b in $(echo "${entry#*:}" | tr ',' ' '); do
    out="research/98_selector_demo/data/g98_lat_${cell,,}_b${b}"
    mkdir -p "$out"
    echo "[lat] ${cell} b${b} -- waiting for a quiet box $(date -Is)"
    wait_for_quiet || exit 1
    echo "[lat] ${cell} b${b} START $(date -Is)"
    W98_LO_NOINSTRUMENT=1 W98_LO_LATTICE=1 W98_LO_CELL="$cell" W98_LO_BATCH="$b" \
      .venv/bin/python "$R" --output-dir "$out" --gpu "$GPU" >/dev/null \
      || echo "[lat] ${cell} b${b} PARTIAL -- resume by re-running"
    n=$(ls "$out"/*.json 2>/dev/null | grep -vc summary || true)
    echo "[lat] ${cell} b${b} DONE ${n}/42 arms $(date -Is)"
  done
done
echo "[lat] COMPLETE $(date -Is)"
