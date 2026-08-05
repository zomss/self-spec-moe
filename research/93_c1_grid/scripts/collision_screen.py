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

DAMAGE BOUNDING (the compile-vs-load order): the FIRST boot to write a
dir compiles its own graph and is CLEAN (the ledger's "primer"
precedent); only boots that LOAD after a different checkpoint compiled
are exposed. Each event below is classified from the lines following
its cache-directory line: "Compiling a graph ... takes" = COMPILED,
a direct-load line = LOADED; a LOADED boot after a different ckpt's
compile is marked EXPOSED.
"""
import glob
import re
from collections import defaultdict

CFG = re.compile(r"speculative_config=SpeculativeConfig\(method='draft_model',"
                 r" model='([^']+)'")
DIR = re.compile(r"Using cache directory: (\S+?)/rank_\d+_\d+/draft_model")
TS = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d)")
COMPILE = re.compile(r"Compiling a graph .* takes [0-9.]+ s")
LOAD = re.compile(r"irectly load|oad the compiled|Loaded compiled")
PID = re.compile(r"pid=(\d+)")

events = defaultdict(list)   # dir -> [(time, log, ckpt, verdict)]
n = 0
for log in sorted(set(glob.glob("research/*/logs/**/*.log", recursive=True))):
    try:
        lines = open(log, errors="replace").read().splitlines()
    except OSError:
        continue
    cur = {}
    for i, line in enumerate(lines):
        m = CFG.search(line)
        if m:
            cur[(PID.search(line) or [None, "main"])[1]] = m.group(1)
            continue
        m = DIR.search(line)
        if not m:
            continue
        pid = (PID.search(line) or [None, "main"])[1]
        ck = cur.pop(pid, None) or cur.pop("main", None)
        if not ck:
            continue
        verdict = "?"
        for j in range(i + 1, min(i + 40, len(lines))):
            nxt = lines[j]
            if pid != "main" and f"pid={pid}" not in nxt:
                continue
            if "Using cache directory" in nxt:
                break
            if COMPILE.search(nxt):
                verdict = "COMPILED"
                break
            if LOAD.search(nxt):
                verdict = "LOADED"
                break
        ts = TS.search(line)
        events[m.group(1)].append(
            (ts.group(1) if ts else "?", log.split("research/")[1], ck, verdict))
        n += 1

print(f"{n} draft boots, {len(events)} cache dirs")
bad = {d: e for d, e in events.items() if len({x[2] for x in e}) > 1}
if not bad:
    print("no cross-checkpoint sharing found")
for d in sorted(bad):
    print(f"\nCOLLISION {d}")
    writer = None
    for t, log, ck, v in sorted(bad[d]):
        tag = ""
        if v == "LOADED" and writer and writer != ck:
            tag = f"   <== EXPOSED (graph from {writer.rsplit('/', 1)[-1]})"
        if v == "COMPILED":
            writer = ck
        print(f"  {t}  {log.split('/logs')[0]:30s} "
              f"{ck.rsplit('/', 1)[-1]:38s} {v:9s}{tag}")
