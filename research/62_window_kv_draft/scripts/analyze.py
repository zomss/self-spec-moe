"""Phase 62: collect per-arm accept_len from data JSONs and solve per-token r.

accept_len ~= 1 + r + r^2 + ... + r^K (geometric, truncated at K+1); r is
solved by bisection. Usage: analyze.py [data_dir]
"""
import glob
import json
import os
import sys

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)


def solve_r(al: float, k: int) -> float:
    def f(r):
        s, p = 1.0, 1.0
        for _ in range(k):
            p *= r
            s += p
        return s

    lo, hi = 0.0, 1.0
    if al >= f(1.0):
        return 1.0
    if al <= 1.0:
        return 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) < al:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


rows = []
for path in sorted(glob.glob(os.path.join(DATA, "w72n_w62_*_K*.json"))):
    with open(path) as f:
        d = json.load(f)
    k = int(d.get("this_k", 0))
    tag = d.get("tag", "?")
    for res in d.get("results", []):
        if "error" in res:
            rows.append((tag, k, res.get("batch"), None, None, None, res["error"]))
            continue
        al = res.get("accept_len")
        r = solve_r(al, k) if al else None
        rows.append(
            (tag, k, res.get("batch"), al, r, res.get("tok_s_mean"), None)
        )

hdr = f"{'arm':22s} {'K':>2s} {'b':>3s} {'accept_len':>10s} {'r':>6s} {'tok/s':>8s}"
print(hdr)
print("-" * len(hdr))
for tag, k, b, al, r, tps, err in rows:
    arm = tag.replace("w62_", "")
    if err:
        print(f"{arm:22s} {k:2d} {b!s:>3s} ERROR: {err[:60]}")
        continue
    al_s = f"{al:.3f}" if al else "n/a"
    r_s = f"{r:.3f}" if r is not None else "n/a"
    tps_s = f"{tps:.1f}" if tps else "n/a"
    print(f"{arm:22s} {k:2d} {b!s:>3s} {al_s:>10s} {r_s:>6s} {tps_s:>8s}")
