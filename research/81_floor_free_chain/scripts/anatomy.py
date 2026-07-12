#!/usr/bin/env python3
"""E0 anatomy: decompose the composed dense draft chain from a kineto trace.

P72/P74 segmentation (reshape_and_cache landmarks -> forwards, largest-gap
cuts), extended with the BETWEEN-forward gaps (inter-step orchestration:
chain metadata rebuild, python, launch trains) that in-forward stats miss.

Per cluster (draft chain = K short forwards/cycle; verify = long):
  wall, GPU-active, in-forward idle %, kernel count.
Plus: median inter-forward gap inside the draft chain, and the CYCLE
decomposition {draft GPU-active, draft in-forward idle, inter-step gaps,
verify}. The E1 gate: (in-forward idle + inter-step gaps) >= 50% of the
chain-step cost.

Usage: anatomy.py <trace.json.gz> [--layers 28]
"""
import argparse
import gzip
import json
import statistics as st
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--layers", type=int, default=28)
    a = ap.parse_args()
    op = gzip.open if a.trace.endswith(".gz") else open
    ev = json.load(op(a.trace))["traceEvents"]
    GPU = {"kernel", "gpu_memset", "gpu_memcpy"}
    gk = sorted([e for e in ev if e.get("cat") in GPU and e.get("pid") == 0],
                key=lambda e: e["ts"])
    print(f"total GPU kernels: {len(gk)}")
    R = [i for i, e in enumerate(gk) if "reshape_and_cache" in e["name"].lower()]
    L = a.layers
    nf = len(R) // L
    print(f"reshape landmarks={len(R)} => forwards={nf} (layers={L})")
    if nf < 4:
        print("too few forwards")
        return 1

    def gap_before(i):
        return gk[i]["ts"] - (gk[i - 1]["ts"] + gk[i - 1].get("dur", 0))

    cuts = [R[0] - 1]
    for f in range(1, nf):
        lo, hi = R[L * f - 1], R[L * f]
        best, bg = lo, -1.0
        for i in range(lo + 1, hi + 1):
            g = gap_before(i)
            if g > bg:
                bg, best = g, i
        cuts.append(best - 1)
    cuts.append(min(len(gk) - 1, R[L * nf - 1] + (R[1] - R[0]) + 8))

    def stats(seg):
        if not seg:
            return None
        wall = (seg[-1]["ts"] + seg[-1].get("dur", 0)) - seg[0]["ts"]
        iv = sorted((e["ts"], e["ts"] + e.get("dur", 0)) for e in seg)
        act, (cs, ce) = 0.0, iv[0]
        for s, e in iv[1:]:
            if s > ce:
                act += ce - cs
                cs, ce = s, e
            else:
                ce = max(ce, e)
        act += ce - cs
        return dict(wall=wall / 1e3, active=act / 1e3, n=len(seg),
                    start=seg[0]["ts"], end=seg[-1]["ts"] + seg[-1].get("dur", 0))

    fw = []
    for f in range(nf):
        s = stats(gk[cuts[f] + 1:cuts[f + 1] + 1])
        if s and s["wall"] > 0:
            fw.append(s)
    walls = sorted(f["wall"] for f in fw)
    thresh = walls[len(walls) // 2] * 1.7  # verify forwards are the long tail
    draft = [f for f in fw if f["wall"] < thresh]
    verify = [f for f in fw if f["wall"] >= thresh]

    def rep(name, fs):
        if not fs:
            print(f"{name}: none")
            return
        w = st.median(f["wall"] for f in fs)
        ac = st.median(f["active"] for f in fs)
        n = st.median(f["n"] for f in fs)
        print(f"{name}: forwards={len(fs)} wall={w:.2f}ms active={ac:.2f}ms "
              f"in-forward idle={100 * (1 - ac / w):.0f}% kernels={n:.0f}")

    rep("draft ", draft)
    rep("verify", verify)

    # inter-forward gaps between consecutive DRAFT forwards (chain orchestration)
    fw_sorted = sorted(fw, key=lambda f: f["start"])
    dgaps, cycle_gaps = [], []
    for x, y in zip(fw_sorted, fw_sorted[1:]):
        g = (y["start"] - x["end"]) / 1e3
        if x in draft and y in draft:
            dgaps.append(g)
        else:
            cycle_gaps.append(g)
    if dgaps:
        print(f"draft->draft inter-step gap: median={st.median(dgaps):.2f}ms "
              f"(n={len(dgaps)})")
    if cycle_gaps:
        print(f"other transitions (draft<->verify): median={st.median(cycle_gaps):.2f}ms")

    if draft and dgaps:
        dw = st.median(f["wall"] for f in draft)
        da = st.median(f["active"] for f in draft)
        gap = st.median(dgaps)
        step = dw + gap
        floor = (dw - da) + gap
        print(f"\nCHAIN STEP = {step:.2f}ms of which GPU-active {da:.2f}ms "
              f"({100 * da / step:.0f}%), in-forward idle {dw - da:.2f}ms, "
              f"inter-step gap {gap:.2f}ms")
        print(f"E1 GATE (floor share = idle+gap): {100 * floor / step:.0f}% "
              f"{'>= 50% -> PROCEED to E1' if floor / step >= 0.5 else '< 50% -> CG will not pay'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
