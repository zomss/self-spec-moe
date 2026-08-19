#!/usr/bin/env bash
# Acceptance on LI, LIO and SS content -- the critical path.
#
# Every u-resolved acceptance curve this phase owns was measured on LO
# content. Sections 31-34 measured the refined grid's THROUGHPUT at four
# cells, which is the ceiling the selector is scored against, but with no
# acceptance for three of them there is no prediction map, so the selector
# can be handed a ceiling and never scored against it. This closes that.
#
# `ignore_eos` is used deliberately and is the researcher's registered
# position: the scored protocol keeps natural EOS, calibration uses equal
# work. Section 25 gave that teeth -- at LO lengths the registered context
# correction leaves up to 22 points of confound standing.
#
# Generation budgets are matched to each cell's natural output so the curve
# covers the range the workload occupies: LI 0.3-1K, LIO 1-2K, SS 0.1-0.3K.
# The registered u-edges (256, 1024, 3072, 8192) are unchanged, so SS
# populates one bucket and LI two -- which is the honest resolution for a
# generation that short, not a defect.
#
# The two cells run on separate GPUs concurrently. Acceptance is a COUNTING
# measurement, not a timing one, and G98-F measured bit-identical accept
# patterns across boxes -- so unlike sections 29-34 this campaign is immune
# to the co-tenancy that section 34 caught corrupting a throughput boot.
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=research/98_selector_demo/scripts/probe_w98_long_u.py

run_cell() {   # cell gen gpu
  local cell="$1" gen="$2" gpu="$3"
  local out="research/98_selector_demo/data/g98_acc_${cell,,}_q4"
  mkdir -p "$out"
  echo "[acc] ${cell} gen=${gen} gpu=${gpu} $(date -Is)"
  W98_LONGU_QUANT=1 W98_LONGU_CELL="$cell" W98_LONGU_TOKENS="$gen" \
    W98_LONGU_BATCH=8 \
    .venv/bin/python "$P" --output-dir "$out" --gpu "$gpu" >/dev/null \
    || echo "[acc] ${cell} FAILED"
  echo "[acc] ${cell} done $(date -Is)"
}

run_cell LIO 2048 1 &
run_cell LI  1024 0
wait
run_cell SS   512 0
echo "[acc] COMPLETE $(date -Is)"
