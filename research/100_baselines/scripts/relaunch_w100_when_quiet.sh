#!/usr/bin/env bash
# Relaunch campaign 1 on the first box whose lane comes genuinely quiet.
# Polls h103 then h104 every 10 min; the cleanliness test replicates the
# boot gate (GPU 7 idle AND no >=50%cpu foreign process whose
# Cpus_allowed_list overlaps 96-111). Launches the resumable runner and
# exits. Never touches foreign processes.
set -u
REPO=/h/v-sukmincho/self-spec-moe
CHECK='
import subprocess, re, sys
from pathlib import Path
rows = subprocess.run(["nvidia-smi","--query-gpu=index,memory.used",
    "--format=csv,noheader,nounits"],capture_output=True,text=True).stdout
gpu7 = [r for r in rows.splitlines() if r.startswith("7,")]
if not gpu7 or int(gpu7[0].split(",")[1]) >= 1024: sys.exit(1)
lane = set(range(96,112))
ps = subprocess.run(["ps","-eo","pid,pcpu,comm","--no-headers"],
    capture_output=True,text=True).stdout
for line in ps.splitlines():
    f = line.split(None,2)
    try: pid, pcpu = int(f[0]), float(f[1])
    except (ValueError,IndexError): continue
    if pcpu < 50: continue
    try: st = Path(f"/proc/{pid}/status").read_text()
    except OSError: continue
    m = re.search(r"^Cpus_allowed_list:\s*(\S+)", st, re.M)
    if not m: continue
    ids=set()
    for part in m.group(1).split(","):
        if "-" in part:
            lo,hi=part.split("-"); ids|=set(range(int(lo),int(hi)+1))
        else: ids.add(int(part))
    if ids & lane: sys.exit(1)
sys.exit(0)
'
while true; do
  for host in h103 h104; do
    if timeout 60 ssh -o BatchMode=yes "$host" \
        "$REPO/.venv/bin/python - <<'EOF'
$CHECK
EOF" 2>/dev/null; then
      echo "LANE CLEAN on $host; relaunching campaign"
      timeout 60 ssh -o BatchMode=yes "$host" \
        "{ nohup setsid $REPO/.venv/bin/python \
           $REPO/research/100_baselines/scripts/run_w100_campaign1.py \
           >> $REPO/research/100_baselines/data/campaign1_run.log 2>&1 \
           < /dev/null & }"
      echo "RELAUNCHED on $host"
      exit 0
    fi
  done
  sleep 600
done
