#!/usr/bin/env bash
# Step 1 -- the equal-work re-score of the full seven-arm lattice, both
# draft weight versions, at one batch.
#
# Sections 15 and 17 compared the two families across seven arms under
# NATURAL EOS, and reported that the lattice reorders (Spearman +0.286) and
# that the prediction ranks `skip8` fifth where the cell measures it second.
# Section 21 then established that any comparison holding the machine fixed
# while letting the workload move is measuring both -- and under natural EOS
# each arm emits a different token count (section 15's own scope note: w128
# bf16 emitted 159K with 3 cap hits against its quantized twin's 125K with
# 1). The equal-work data that settled sections 18-20 covers TWO arms, not
# seven, so neither the reorder nor the misranking has ever been re-taken
# under the discipline that retired them.
#
# This does that. `ignore_eos` with a fixed budget makes every arm in both
# families emit exactly `GEN * BATCH` tokens over the same prompts, so the
# context trajectory is identical across arms and the Campaign-1 context
# correction becomes a common factor -- raw and corrected rankings coincide
# by construction, which is why the ranking can be read directly.
#
# Calibration, not a scored run: the scored protocol keeps natural EOS.
set -euo pipefail
cd "$(dirname "$0")/../../.."
GEN="${GEN:-16384}"
BATCH="${BATCH:-8}"
GPU="${GPU:-0}"
RUNNER=research/98_selector_demo/scripts/run_w98_refined_lo.py
BF16="${BF16:-research/98_selector_demo/data/g98_lo_ew_bf16}"
Q4="${Q4:-research/98_selector_demo/data/g98_lo_ew_q4}"
mkdir -p "$BF16" "$Q4"

echo "[ew] gen ${GEN} batch ${BATCH} gpu ${GPU} -- $(date -Is)"

# The bf16 family has no single mode covering ARMS and SWEEP together, so it
# takes two invocations into one directory; the artifact layer's claim/skip
# makes that safe and the second run re-summarises the whole directory.
echo "[ew] bf16 base set"
W98_LO_FIXED="$GEN" W98_LO_BATCH="$BATCH" \
  .venv/bin/python "$RUNNER" --output-dir "$BF16" --gpu "$GPU" >/dev/null
echo "[ew] bf16 window sweep"
W98_LO_SWEEP=1 W98_LO_FIXED="$GEN" W98_LO_BATCH="$BATCH" \
  .venv/bin/python "$RUNNER" --output-dir "$BF16" --gpu "$GPU" >/dev/null

# The quantized mode already unions the two sets, and carries its own OFF so
# the two families' denominators are checked rather than assumed equal.
echo "[ew] w4a16 family"
W98_LO_QUANT=1 W98_LO_FIXED="$GEN" W98_LO_BATCH="$BATCH" \
  .venv/bin/python "$RUNNER" --output-dir "$Q4" --gpu "$GPU" >/dev/null

echo "[ew] COMPLETE $(date -Is)"
