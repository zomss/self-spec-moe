#!/usr/bin/env python3
"""Global cross-checkpoint cache-sharing screen (the "collision screen"
of paper/data/c1_corruption_ledger.md), over every retained phase log.

A boot is exposed iff its draft_model compile-cache DIRECTORY (full
path, root INCLUDED — the same pre-fix hash under two isolated roots is
harmless) was also used by a run with a DIFFERENT draft checkpoint.
Pairs each "Using cache directory: .../draft_model" line with the most
recent SpeculativeConfig line of the same EngineCore pid.

2026-08-05 run: 923 draft boots, 574 cache dirs; sharing found only in
the documented pre-fix 93/94 campaigns plus p75 (W4asym/W4sym/W8 on one
dir) and p92 (drafter-refresh ckpts <-> static W4A16).
"""
import glob
import re
from collections import defaultdict

CFG = re.compile(r"speculative_config=SpeculativeConfig\(method='draft_model',"
                 r" model='([^']+)'")
DIR = re.compile(r"Using cache directory: (\S+?)/rank_\d+_\d+/draft_model")
PID = re.compile(r"pid=(\d+)")

dir_ckpts = defaultdict(set)
dir_logs = defaultdict(set)
n = 0
for log in sorted(set(glob.glob("research/*/logs/**/*.log", recursive=True))):
    cur = {}
    try:
        fh = open(log, errors="replace")
    except OSError:
        continue
    for line in fh:
        m = CFG.search(line)
        if m:
            cur[(PID.search(line) or [None, "main"])[1]] = m.group(1)
            continue
        m = DIR.search(line)
        if m:
            pid = (PID.search(line) or [None, "main"])[1]
            ck = cur.pop(pid, None) or cur.pop("main", None)
            if ck:
                dir_ckpts[m.group(1)].add(ck)
                dir_logs[m.group(1)].add(
                    log.split("research/")[1].split("/logs")[0])
                n += 1
    fh.close()

print(f"{n} draft boots, {len(dir_ckpts)} cache dirs")
bad = {d: c for d, c in dir_ckpts.items() if len(c) > 1}
if not bad:
    print("no cross-checkpoint sharing found")
for d in sorted(bad):
    print(f"COLLISION {d}")
    for c in sorted(bad[d]):
        print(f"    {c.rsplit('/', 1)[-1]}")
    print(f"    phases: {', '.join(sorted(dir_logs[d]))}")
