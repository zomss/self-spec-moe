#!/bin/bash
set -e
cd /data/smcho/self-spec-moe/research/20_injection_emulation
PY=/data/smcho/self-spec-moe/.venv/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3
for D in 0 163 320; do
  echo "=== gptoss delay ${D}us ==="
  timeout 500 $PY bench_dp_injection.py --model openai/gpt-oss-20b --delay-us $D \
      --data-parallel-size 4 --tensor-parallel-size 1 --batch-sizes 1,8,32,64 \
      --iters 5 --warmup 2 --output-json data/gptoss_dp4_d${D}.json > logs/gptoss_dp4_d${D}.log 2>&1
  echo "  -> done ${D}us (exit $?)"
done
echo "=== GPTOSS ALL DONE ==="
