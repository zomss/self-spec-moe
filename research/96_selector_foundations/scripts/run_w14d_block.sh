#!/usr/bin/env bash
# W14/D — run one COMPLETE registered boot block on one GPU.
#
# Block-per-GPU is the safe unit of parallelism: q = tau/(rate_spec/rate_AR)
# cancels a lane-wide scale factor only if the AR anchor shares the lane
# with the speculative boots it anchors, and each block carries its own
# fresh AR anchor. Lanes calibrated at 0.45-0.49% (prereg
# lane_assignment), so absolute-latency surface fits are safe too.
#
# usage: run_w14d_block.sh <block 1|2|3>
set -uo pipefail
cd /data/smcho/self-spec-moe
PHASE=research/96_selector_foundations
BLOCK="${1:?block 1|2|3}"
REG="$PHASE/data/w14/w14d_prereg.json"

GPU=$(.venv/bin/python -c "import json;print(json.load(open('$REG'))['lane_assignment']['blocks']['$BLOCK'])")
ORDER=$(.venv/bin/python -c "import json;print(' '.join(json.load(open('$REG'))['block_order'][$BLOCK-1]))")
CTX=$(.venv/bin/python -c "import json;print(','.join(map(str,json.load(open('$REG'))['ctx_order'][$BLOCK-1])))")

echo "[D] block $BLOCK -> GPU $GPU"
echo "[D] boot order: $ORDER"
echo "[D] ctx order:  $CTX"

ok=0; wedge=0
for item in $ORDER; do
  if [ "$item" = "AR" ]; then cfg="AR"; kk=0; else
    cfg="${item%-K*}"; kk="${item##*-K}"
  fi
  tag="w14d_block${BLOCK}_${item}"
  out="$PHASE/data/w14/${tag}.json"
  # K4 draft-chain capture wedges materially more often than K2 (smoke:
  # w512_K4 0/3; B at K4: w2048 3/3 but woff and w512 1/3 first-pass).
  # With 9 K4 boots in D, a 4-attempt cap loses at least one boot ~43% of
  # the time at a 50% wedge rate; 8 attempts drops that to ~3.5%. Each
  # wedge costs only ~7 min under the 400 s watchdog, so the budget is
  # cheap insurance against losing a registered slot.
  for att in 1 2 3 4 5 6 7 8; do
    if [ -s "$out" ] && grep -q '"complete": true' "$out"; then break; fi
    echo "[D] block$BLOCK $item attempt $att"
    # retry on the SAME gpu: a wedge must not migrate the lane
    W14_GPU="$GPU" W14D_CTX="$CTX" W14D_RID="R5,R5cot" W14D_BATCH="1,8" \
      W14D_ITERS=4 \
      bash "$PHASE/scripts/run_w14d_boot.sh" "$cfg" "$kk" "$tag" "$out"
    rc=$?
    [ $rc -eq 75 ] && { wedge=$((wedge+1)); continue; }
    [ $rc -eq 0 ] && break
  done
  if [ -s "$out" ] && grep -q '"complete": true' "$out"; then
    ok=$((ok+1))
    # stamp the lane into the artifact for post-hoc lane-effect detection
    .venv/bin/python - "$out" "$GPU" "$BLOCK" <<'PY'
import json,sys
p,g,b=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
d=json.load(open(p)); d["gpu"]=g; d["block"]=b
json.dump(d,open(p,"w"),indent=1)
PY
  fi
done
echo "[D] block $BLOCK done: $ok/7 boots, $wedge wedges"
