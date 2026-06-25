#!/bin/bash
set -e
cd /data/smcho/self-spec-moe/research/20_injection_emulation
PY=/data/smcho/self-spec-moe/.venv/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3
for D in 0 163 320; do
  echo "=== delay ${D}us ==="
  $PY bench_injection_sweep.py --delay-us $D \
      --tensor-parallel-size 4 --batch-sizes 1,8,32,64 \
      --output-json data/ep4_d${D}.json > logs/ep4_d${D}.log 2>&1
  echo "  -> done ${D}us"
done
echo "=== ALL DONE ==="
