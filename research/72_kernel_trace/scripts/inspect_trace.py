"""Phase 72: inspect a torch-profiler trace to map its structure.

Dumps: stream tids + kernel counts, user_annotation / gpu_user_annotation
names + counts, and per-'gpu_model_runner: draft' GPU span the kernel count
and the moe_align landmark count (48/forward) so we can segment K chain steps.
"""
import gzip
import json
import sys
import collections

path = sys.argv[1]
op = gzip.open if path.endswith(".gz") else open
d = json.load(op(path))
ev = d["traceEvents"]
print("num events:", len(ev))

cats = collections.Counter(e.get("cat") for e in ev)
print("cats:", cats.most_common())

# GPU device events: kernel / gpu_memset / gpu_memcpy carry pid=0 (device).
gpu_cats = {"kernel", "gpu_memset", "gpu_memcpy"}
by_tid = collections.Counter()
for e in ev:
    if e.get("cat") in gpu_cats:
        by_tid[(e.get("pid"), e.get("tid"))] += 1
print("GPU device (pid,tid) kernel-ish counts (top 12):")
for k, c in by_tid.most_common(12):
    print("   ", k, c)

for name, cat in [("user_annotation", "user_annotation"),
                  ("gpu_user_annotation", "gpu_user_annotation")]:
    cc = collections.Counter(e["name"] for e in ev if e.get("cat") == cat)
    print(f"=== {name} names (top 25) ===")
    for n, c in cc.most_common(25):
        print(f"   {c:5d}  {n[:70]}")


def spans(cat, name):
    out = []
    for e in ev:
        if e.get("cat") == cat and e.get("name") == name:
            out.append((e["ts"], e["ts"] + e.get("dur", 0), e.get("dur", 0)))
    out.sort()
    return out


for cat in ["gpu_user_annotation", "user_annotation"]:
    ds = spans(cat, "gpu_model_runner: draft")
    print(f"=== '{cat}' gpu_model_runner: draft spans: {len(ds)} ===")
    for ts0, ts1, dur in ds[:8]:
        print(f"   dur={dur/1000:.3f} ms  [{ts0:.1f},{ts1:.1f}]")
