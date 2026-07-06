"""Phase 64: decompose the traced spec-decode cycle from a torch trace.

Buckets the runner's named spans (gpu_model_runner: preprocess / forward /
sample / draft / bookkeep / ...) per execute_model step, then digs into the
`draft` spans (the K-step propose): runtime-launch mix (graph vs kernel
launches), explicit sync calls, and the top CPU ops by total time inside
draft vs everywhere.

Usage: analyze_trace16k.py TRACE_DIR_OR_FILE
"""

import glob
import gzip
import json
import os
import sys
from collections import defaultdict


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def pick(path):
    if os.path.isfile(path):
        return path
    paths = sorted(glob.glob(f"{path}/**/*.json*", recursive=True))
    if not paths:
        sys.exit(f"no trace files under {path}")
    return max(paths, key=os.path.getsize)


def main(path):
    path = pick(path)
    print(f"# {path}")
    ev = load(path)["traceEvents"]
    X = [e for e in ev if e.get("ph") == "X"]

    spans = defaultdict(list)
    for e in X:
        n = e.get("name", "")
        if n.startswith("gpu_model_runner: "):
            spans[n.split(": ", 1)[1]].append(e)

    print("\n## runner span totals (rank 0, traced window)")
    print(f"{'span':<18}{'count':>7}{'total_ms':>10}{'mean_ms':>9}")
    order = sorted(spans, key=lambda k: -sum(x["dur"] for x in spans[k]))
    for k in order:
        v = spans[k]
        tot = sum(x["dur"] for x in v) / 1e3
        print(f"{k:<18}{len(v):>7}{tot:>10.1f}{tot / len(v):>9.2f}")

    def inside(windows, e):
        t = e["ts"]
        return any(t0 <= t < t1 for t0, t1 in windows)

    draft_w = [(e["ts"], e["ts"] + e["dur"]) for e in spans.get("draft", [])]
    runtime = [
        e for e in X
        if e.get("cat") in ("cuda_runtime", "cuda_driver")
    ]
    cpu_ops = [e for e in X if e.get("cat") == "cpu_op" and e.get("dur", 0) > 0]

    print("\n## runtime calls inside `draft` spans "
          f"(n={len(draft_w)} propose calls)")
    agg = defaultdict(lambda: [0, 0.0])
    for r in runtime:
        if inside(draft_w, r):
            a = agg[r["name"]]
            a[0] += 1
            a[1] += r["dur"] / 1e3
    for n, (c, tot) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:12]:
        per = c / max(1, len(draft_w))
        print(f"  {tot:8.1f} ms  n={c:<6} ({per:6.1f}/propose)  {n}")

    print("\n## top cpu ops inside `draft` spans (total ms)")
    agg = defaultdict(lambda: [0, 0.0])
    for e in cpu_ops:
        if inside(draft_w, e):
            a = agg[e["name"]]
            a[0] += 1
            a[1] += e["dur"] / 1e3
    for n, (c, tot) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:15]:
        print(f"  {tot:8.1f} ms  n={c:<6} {n[:80]}")

    print("\n## top cpu ops overall (total ms)")
    agg = defaultdict(lambda: [0, 0.0])
    for e in cpu_ops:
        a = agg[e["name"]]
        a[0] += 1
        a[1] += e["dur"] / 1e3
    for n, (c, tot) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:15]:
        print(f"  {tot:8.1f} ms  n={c:<6} {n[:80]}")

    print("\n## longest individual cpu ops")
    for e in sorted(cpu_ops, key=lambda e: -e["dur"])[:10]:
        print(f"  {e['dur'] / 1e3:8.2f} ms  {e['name'][:80]}")


if __name__ == "__main__":
    main(sys.argv[1])
