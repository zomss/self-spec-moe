#!/usr/bin/env bash
# Section 36's registered fix: cost calibration at long context.
#
# Every cost coefficient this phase owns was fitted on 156-token prompts, so
# `kappa_kv` and `f_win` were identified where KV is a minor term. Section 24
# measured the consequence at calibration context -- the quantized family's
# window-size axis carried no resolved signal, 1.9 sigma -- and section 36
# measured it where it bites: the resulting negative coefficients say that
# reading 15,000 KV positions is 4.2 ms CHEAPER than reading 4,252, and the
# prediction over-states the unwindowed arms by 41-42% at LI.
#
# Same design as section 24's sweep -- three windows at three keeps, which
# identifies f_win from arms sharing a window while differing in keep, and
# kappa_kv from arms sharing a keep while differing in window -- moved to
# 12-19K prompts where the refined grid actually operates.
#
# Deliberately SERIAL on one GPU. This is a timing measurement, and section
# 34 caught a co-tenant corrupting one by 47.6%; running our own two lanes
# concurrently would court the same thing. The acceptance campaign could be
# parallel because counting is immune to contention. This cannot.
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=research/98_selector_demo/scripts/run_w98_equalwork_cost.py
GPU="${GPU:-0}"
GEN="${GEN:-1024}"

echo "[ewli] quantized family $(date -Is)"
W98_EW_QUANT=1 W98_EW_CELL=LI W98_EW_BATCH=8 W98_EW_TOKENS="$GEN" \
  .venv/bin/python "$P" --output-dir research/98_selector_demo/data/g98_equalwork_li_q4 \
  --gpu "$GPU" >/dev/null || echo "[ewli] quantized FAILED"
echo "[ewli] bf16 family $(date -Is)"
W98_EW_CELL=LI W98_EW_BATCH=8 W98_EW_TOKENS="$GEN" \
  .venv/bin/python "$P" --output-dir research/98_selector_demo/data/g98_equalwork_li \
  --gpu "$GPU" >/dev/null || echo "[ewli] bf16 FAILED"
echo "[ewli] COMPLETE $(date -Is)"
