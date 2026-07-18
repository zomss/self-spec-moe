#!/usr/bin/env python3
"""Compare two chrome traces: per-stream GPU busy/idle, top ops, and
main-compute-stream gap histogram. Usage: analyze_trace.py A.json B.json"""
import json
import sys
from collections import defaultdict


def load(path):
    d = json.load(open(path))
    evs = d["traceEvents"] if isinstance(d, dict) else d
    return [e for e in evs if isinstance(e, dict) and e.get("ph") == "X"]


def summarize(path):
    evs = load(path)
    kernels = [e for e in evs if e.get("cat") in ("kernel", "gpu_memcpy",
                                                  "gpu_memset")]
    if not kernels:
        cats = defaultdict(int)
        for e in evs:
            cats[e.get("cat")] += 1
        print(f"{path}: no kernel events; cats={dict(cats)}")
        return
    t0 = min(e["ts"] for e in kernels)
    t1 = max(e["ts"] + e["dur"] for e in kernels)
    span = t1 - t0
    by_stream = defaultdict(list)
    for e in kernels:
        by_stream[(e.get("pid"), e.get("tid"))].append(e)
    print(f"\n== {path}")
    print(f"  window {span/1e3:.1f} ms")
    for sid, ks in sorted(by_stream.items(),
                          key=lambda kv: -sum(e['dur'] for e in kv[1]))[:4]:
        busy = sum(e["dur"] for e in ks)
        ks.sort(key=lambda e: e["ts"])
        gaps = []
        for a, b in zip(ks, ks[1:]):
            g = b["ts"] - (a["ts"] + a["dur"])
            if g > 0:
                gaps.append((g, a["name"][:60], b["name"][:60]))
        gaps.sort(reverse=True)
        print(f"  stream {sid}: busy {busy/1e3:.1f} ms "
              f"({100*busy/span:.0f}%), n={len(ks)}, "
              f"top gaps(us): {[round(g[0]) for g in gaps[:5]]}")
        for g, an, bn in gaps[:3]:
            print(f"    gap {g/1e3:.2f} ms after [{an}] before [{bn}]")
    byname = defaultdict(float)
    for e in kernels:
        byname[e["name"][:70]] += e["dur"]
    print("  top kernels (ms):")
    for n, d_ in sorted(byname.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {d_/1e3:8.1f}  {n}")
    cpu = [e for e in evs if e.get("cat") == "cpu_op"]
    byname = defaultdict(float)
    for e in cpu:
        byname[e["name"][:70]] += e["dur"]
    print("  top cpu ops (ms):")
    for n, d_ in sorted(byname.items(), key=lambda kv: -kv[1])[:6]:
        print(f"    {d_/1e3:8.1f}  {n}")


for p in sys.argv[1:]:
    summarize(p)
