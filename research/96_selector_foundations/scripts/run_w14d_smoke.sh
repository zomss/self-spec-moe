#!/usr/bin/env bash
# W14/D0 — five NON-SCORED smoke boots for the live half of Gate D0.
# AR + piecewise K2/K4 (w512) + piecewise-nowindow K2/K4 (w-off).
# Smoke observations can never become training anchors: every output is
# tagged smoke=true and written under a smoke_ prefix.
set -uo pipefail
cd /data/smcho/self-spec-moe
PHASE=research/96_selector_foundations
export W14D_CTX=2048 W14D_RID=R5,R5cot W14D_BATCH=1,8 W14D_ITERS=2 W14D_SMOKE=1
ok=0; wedge=0
for spec in "AR 0" "w512 2" "w512 4" "w-off 2" "w-off 4"; do
  set -- $spec; cfg=$1; kk=$2
  tag="w14d_smoke_${cfg}_K${kk}"
  out="$PHASE/data/w14/${tag}.json"
  for att in 1 2 3; do
    [ -s "$out" ] && grep -q '"complete": true' "$out" && break
    echo "[SMOKE] $tag attempt $att"
    bash "$PHASE/scripts/run_w14d_boot.sh" "$cfg" "$kk" "$tag" "$out"
    rc=$?
    [ $rc -eq 75 ] && { wedge=$((wedge+1)); continue; }
    [ $rc -eq 0 ] && break
  done
  [ -s "$out" ] && grep -q '"complete": true' "$out" && ok=$((ok+1))
done
echo "[SMOKE] complete=$ok/5 wedges=$wedge"
