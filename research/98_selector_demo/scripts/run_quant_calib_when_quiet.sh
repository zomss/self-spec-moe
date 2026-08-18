#!/bin/bash
# Quant-axis calibration, gated on a quiet box.
#
# The six w4a16 arms must be comparable to the ten target-matching arms
# already measured, so this refuses to measure beside a co-tenant. Each pass:
#   1. require enough free GPU memory for the standard 0.9 utilisation;
#   2. re-measure target-matching/woff/skip4 and require it within 2% of its
#      quiet-box value (32.933 ms) -- the same control that reproduced at
#      0.9% when the box was briefly free;
#   3. only then run the quant arms, which are resumable.
set -u
REPO=$(cd -- "$(dirname -- "$0")/../../.." && pwd); cd "$REPO" || exit 2
REF=32.933; TOL=0.02; NEED=73000   # MiB free required on GPU 0
OUT=research/98_selector_demo/data/g98_equalwork_q4
CTL=research/98_selector_demo/data/g98_ew_control
for i in $(seq 1 60); do
  free=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits \
         | head -1 | awk -F', ' '{print $1-$2}')
  if [ "$free" -lt "$NEED" ]; then
    echo "[q4] pass $i: only ${free}MiB free, waiting $(date -Is)"; sleep 300; continue
  fi
  rm -f "$CTL"/*.json "$CTL"/*.FAILED
  W98_EW_CELL=LO W98_EW_TOKENS=8192 taskset -c 32-63 .venv/bin/python -c "
import sys; sys.path.insert(0,'research/98_selector_demo/scripts')
import run_w98_equalwork_cost as m
from pathlib import Path
cfg={'action':'armed','quant':'target-matching','window':'off','skip_count':4}
m.run_all.__globals__['arms']=lambda: [cfg]
m.run_all(Path('$CTL'), 0)" >/dev/null 2>&1
  ctl=$(taskset -c 32-63 .venv/bin/python -c "
import json,glob
f=[x for x in glob.glob('$CTL/*.json') if 'summary' not in x]
print(json.load(open(f[0]))['draft_chain_ms'] if f else 0)" 2>/dev/null)
  ok=$(taskset -c 32-63 .venv/bin/python -c "print(1 if abs($ctl/$REF-1)<=$TOL and $ctl>0 else 0)")
  if [ "$ok" != "1" ]; then
    echo "[q4] pass $i: control $ctl ms vs $REF -- box not comparable, waiting"; sleep 300; continue
  fi
  echo "[q4] pass $i: control $ctl ms OK, running quant arms $(date -Is)"
  W98_EW_QUANT=1 taskset -c 32-63 .venv/bin/python \
    research/98_selector_demo/scripts/run_w98_equalwork_cost.py --output-dir "$OUT" --gpu 0
  have=$(find "$OUT" -maxdepth 1 -name '*.json' ! -name 'summary.json' | wc -l)
  echo "[q4] pass $i: $have/6 quant arms measured"
  [ "$have" -ge 6 ] && { echo "[q4] COMPLETE $(date -Is)"; exit 0; }
  sleep 300
done
echo "[q4] exhausted passes"; exit 1
