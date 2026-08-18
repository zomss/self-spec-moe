#!/usr/bin/env bash
# Step 3 -- does "speculation loses at long input" reproduce on this box?
#
# Campaign 1 measured every armed arm losing to stock at LI and LIO (0.61-0.94
# of stock), which is the arming decision's whole case on the refined grid and
# the reason the fail-closed rule matters there.
#
# G98-F is why this must be re-measured rather than inherited. The phase's
# other "speculation loses" result -- D3's R4 regression, 522 against OFF's
# 685 -- did NOT reproduce: on this box R4's armed arm WINS at 1.129 against
# h103's 0.763, with R1 reproducing h103 to 1% as the control, and six
# independent lines pointing at h103's R4 armed column having been measured
# under contention. Its line 6 named the mechanism: the most bandwidth-hungry
# armed steps are the exposed ones.
#
# LI and LIO are exactly that shape -- 9-16K of context, unwindowed draft
# attention, batch 8 -- so the same artifact would produce the same verdict.
# Whether "do not arm at long input" is a property of the workload or of that
# box is unsettled, and it decides whether the fail-closed rule has any
# firing set on the grid we are moving to.
#
# Scored protocol: natural EOS, registered per-cell caps (LI 2K, LIO 4K).
# `woff/skip0` is included deliberately as the most bandwidth-hungry armed
# arm -- the one G98-F's mechanism would hit hardest.
set -euo pipefail
cd "$(dirname "$0")/../../.."
RUNNER=research/98_selector_demo/scripts/run_w98_refined_lo.py
BATCH="${BATCH:-8}"
GPU="${GPU:-0}"
ARMS="${ARMS:-off,base,skip4,w512skip4,w1024skip4}"

for cell in ${CELLS:-LI LIO}; do
  out="research/98_selector_demo/data/g98_arm_${cell,,}_b${BATCH}${SUFFIX:-}"
  mkdir -p "$out"
  echo "[arm] cell ${cell} batch ${BATCH} -- ${ARMS} -- $(date -Is)"
  W98_LO_CELL="$cell" W98_LO_QUANT=1 W98_LO_BATCH="$BATCH" W98_LO_ARMS="$ARMS" \
    .venv/bin/python "$RUNNER" --output-dir "$out" --gpu "$GPU" >/dev/null \
    || echo "[arm] cell ${cell} FAILED"
  echo "[arm] ${cell} done $(date -Is)"
done
echo "[arm] COMPLETE $(date -Is)"
