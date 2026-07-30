#!/bin/bash
# Kill a stalled Stage-B run for ONE arch: if its log gains no [G93]
# line for STALL_MIN minutes while its driver has a live run_grid
# descendant, kill that descendant only (two-hop parent check:
# python <- timeout <- run_stage_b.sh <arch>).
ARCH=${1:?arch}; STALL_MIN=${2:-45}
LOG=/data/smcho/self-spec-moe/research/93_c1_grid/logs/stage_b_${ARCH}.log
last=-1; still=0
while true; do
  sleep 300
  victims=""
  for p in $(pgrep -u smcho -f "run_grid.p[y]" || true); do
    t=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    d=$(ps -o ppid= -p "$t" 2>/dev/null | tr -d ' ')
    ps -o cmd= -p "$d" 2>/dev/null | grep -q "run_stage_b.sh $ARCH" && victims="$victims $p"
  done
  [ -z "$victims" ] && { last=-1; still=0; continue; }
  n=$(grep -c "^\[G93\]" "$LOG" 2>/dev/null || echo 0)
  if [ "$n" = "$last" ]; then
    still=$((still+5))
    if [ "$still" -ge "$STALL_MIN" ]; then
      echo "$(date) [watchdog:$ARCH] stalled ${STALL_MIN}m -> kill$victims"
      for p in $victims; do kill "$p" 2>/dev/null; done
      still=0
    fi
  else
    still=0
  fi
  last=$n
done
