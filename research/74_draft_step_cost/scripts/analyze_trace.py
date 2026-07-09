#!/usr/bin/env python
"""Capture the EXACT draft-step overhead from a kineto trace.

Segments the GPU timeline into 48-layer forwards (reshape_and_cache landmark,
cut at the largest inter-forward gap -- Phase-72 method). For each forward
reports wall (first->last kernel), GPU-active (union of kernel intervals),
idle-gap fraction, and kernel count. The base arm runs both draft (B*1 tok)
and verify (B*(K+1) tok) forwards through FlashAttn; classify by the count of
attention kernels' work is unreliable, so we CLUSTER forwards by wall time
(draft chain = K short forwards, verify = 1 longer) and report both clusters.

Key question: is the ~10ms draft overhead GPU-IDLE (launch gaps -> CUDA graph
fixes) or GPU-ACTIVE (unfused eager work -> graphs won't help)?
Usage: analyze_trace.py <trace.json[.gz]>
"""
import gzip, json, sys, statistics as st

path = sys.argv[1]
op = gzip.open if path.endswith(".gz") else open
ev = json.load(op(path))["traceEvents"]
GPU = {"kernel", "gpu_memset", "gpu_memcpy"}
gk = sorted([e for e in ev if e.get("cat") in GPU and e.get("pid") == 0],
            key=lambda e: e["ts"])
print(f"total GPU kernels in trace: {len(gk)}")

LAYERS = 48
R = [i for i, e in enumerate(gk) if "reshape_and_cache" in e["name"].lower()]
nf = len(R) // LAYERS
print(f"reshape landmarks={len(R)} => forwards={nf}")
if nf < 2:
    print("too few forwards; trace may be malformed"); sys.exit(0)

def gap_before(i): return gk[i]["ts"] - (gk[i-1]["ts"] + gk[i-1].get("dur", 0))

cuts = [R[0] - 1]
for f in range(1, nf):
    a, b = R[LAYERS*f - 1], R[LAYERS*f]
    best, bg = a, -1.0
    for i in range(a+1, b+1):
        g = gap_before(i)
        if g > bg: bg, best = g, i
    cuts.append(best - 1)
cuts.append(min(len(gk)-1, R[LAYERS*nf - 1] + (R[1]-R[0]) + 8))

def stats(seg):
    if not seg: return None
    wall = (seg[-1]["ts"] + seg[-1].get("dur", 0)) - seg[0]["ts"]
    # union of busy intervals
    iv = sorted((e["ts"], e["ts"]+e.get("dur", 0)) for e in seg)
    active = 0.0; cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce: active += ce-cs; cs, ce = s, e
        else: ce = max(ce, e)
    active += ce-cs
    return dict(wall=wall/1000, active=active/1000, n=len(seg),
                idle_pct=100*(1-active/wall) if wall else 0)

forwards = [stats(gk[cuts[f]+1:cuts[f+1]+1]) for f in range(nf)]
forwards = [f for f in forwards if f and f["wall"] > 0]
walls = sorted(f["wall"] for f in forwards)
med = st.median(walls)
short = [f for f in forwards if f["wall"] <= med*1.15]   # draft chain steps
long = [f for f in forwards if f["wall"] > med*1.15]      # verify (fewer, longer)

def show(tag, group):
    if not group: print(f"{tag}: none"); return
    w = st.median([f["wall"] for f in group]); a = st.median([f["active"] for f in group])
    n = st.median([f["n"] for f in group]); idle = st.median([f["idle_pct"] for f in group])
    print(f"{tag:20} n_fwd={len(group):3}  wall={w:6.2f}ms  GPU-active={a:6.2f}ms  "
          f"idle={idle:5.1f}%  kernels/fwd={int(n)}")

print()
show("DRAFT (short fwds)", short)
show("VERIFY (long fwds)", long)
print(f"\n=> If DRAFT GPU-active ~= the graphed no-spec forward (~9.7ms @2k/b8) and")
print(f"   idle is large, the overhead is LAUNCH-BOUND (CUDA graph removes it).")
print(f"   If GPU-active ~= wall (idle small), the overhead is real GPU work.")
