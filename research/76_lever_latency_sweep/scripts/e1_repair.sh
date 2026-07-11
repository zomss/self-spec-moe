#!/bin/bash
# E1 repair pass: rerun ONLY the S4-flagged (noisy) cells with more runs.
#
# Parses `e1_analyze.py` S4 lines ("S4 NOISY <group>/<arm> b<B>/c<C>k runs=[..]"),
# groups flagged cells per (group, arm), and re-invokes e1_sweep.sh with an
# E1_CELLS override and E1_RUNS=4 -- one server relaunch per affected arm.
# Fresh warm pass + r1..r4 REPLACE the old r1/r2 for those cells (same names),
# so the analyzer's mean is over 4 clean runs afterwards.
#
# ctx label -> input-token mapping is group-dependent: the dense 32k cell uses
# input 32512 (Qwen2.5 max_position_embeddings cap), moe/mla use 32768.
#
# Usage: bash scripts/e1_repair.sh [groups...]   (default: dense moe)
set -u
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PHASE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
D="$PHASE/data/e1"
RGROUPS="${*:-dense moe}"

FLAGS=$("$PY" "$PHASE/scripts/e1_analyze.py" --data "$D" 2>/dev/null | grep -oE "S4 NOISY [a-z]+/[a-z0-9_]+ b[0-9]+/c[0-9]+k")
[ -n "$FLAGS" ] || { echo "[repair] no S4-flagged cells -- nothing to do"; exit 0; }

ctx_of(){  # <group> <klabel>
  case "$1:$2" in
    dense:32) echo 32512 ;;
    *:2) echo 2048 ;; *:16) echo 16384 ;; *:32) echo 32768 ;;
    *) echo $(( $2 * 1024 )) ;;
  esac
}

# group flagged cells per group/arm
declare -A CELLMAP
while read -r _ _ ga cell; do
  g="${ga%%/*}"; a="${ga##*/}"
  echo " $RGROUPS " | grep -q " $g " || continue
  b=$(echo "$cell" | grep -oE "^b[0-9]+" | tr -d b)
  k=$(echo "$cell" | grep -oE "c[0-9]+k" | tr -dc 0-9)
  CELLMAP["$g/$a"]+="$b:$(ctx_of "$g" "$k") "
done <<< "$FLAGS"

[ "${#CELLMAP[@]}" -gt 0 ] || { echo "[repair] no flagged cells in groups: $RGROUPS"; exit 0; }

for ga in "${!CELLMAP[@]}"; do
  g="${ga%%/*}"; a="${ga##*/}"
  echo "[repair] $g/$a cells: ${CELLMAP[$ga]}"
  E1_RUNS=4 E1_CELLS="${CELLMAP[$ga]}" bash "$PHASE/scripts/e1_sweep.sh" "$g" "^${a}\$"
done

echo "[repair] done; re-analyzing"
"$PY" "$PHASE/scripts/e1_analyze.py" --data "$D"
