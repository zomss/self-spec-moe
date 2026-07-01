"""Phase 46 analysis: build accept_len(C), g(C), and speedup(C,f) tables.

Reads research/46_topc_probe/data/tc_<tag>_b<B>_K<K>_C<C>.json for a model tag
and computes:
  - accept_len(C)      : mean accept length per C (sanity: C=top_k == full).
  - draft_ms(C)        : profiler draft_forward mean_ms per C.
  - g(C) = draft_ms(C)/draft_ms(top_k)   (compute-saving ratio; floor where flat)
  - speedup(C,f) = accept_len(C) / (K*g(C)*(1-f) + 1)  for f in {0.4,0.62,0.8}
  - optimal C per f and whether any C beats C=top_k (the full draft).

Usage: analyze_topc.py <tag> <top_k> [K] [f1 f2 f3]
"""
import glob
import json
import os
import sys

DATA = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data"


def load(tag):
    rows = {}
    for f in sorted(glob.glob(os.path.join(DATA, f"tc_{tag}_b*_K*_C*.json"))):
        if "_nospec" in f:
            continue
        j = json.load(open(f))
        if j.get("nospec"):
            continue
        C = j["C"]
        rr = j.get("run_result", {}) or {}
        prof = (j.get("rank0_profile") or {}).get("summary", {}) or {}

        def _ms(lbl):
            d = prof.get(lbl, {})
            return d.get("mean_ms")
        rows[C] = {
            "C": C,
            "accept_len": rr.get("accept_len"),
            "sys_tok_s": rr.get("sys_tok_s"),
            "draft_ms": _ms("draft_forward"),
            "draft_first_ms": _ms("draft_forward_first"),
            "chain_ms": _ms("draft_chain"),
            "verify_ms": _ms("verify"),
            "K": j["K"],
        }
    return rows


def main():
    tag = sys.argv[1]
    top_k = int(sys.argv[2])
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    fs = [float(x) for x in sys.argv[4:]] if len(sys.argv) > 4 else [0.4, 0.62, 0.8]

    rows = load(tag)
    Cs = sorted(rows.keys(), reverse=True)
    if not Cs:
        print(f"no data for tag {tag}")
        return
    base = rows.get(top_k)
    base_ms = base["draft_ms"] if base else None

    print(f"\n=== {tag}  top_k={top_k}  K={K} ===")
    print(f"{'C':>3} {'accept_len':>11} {'draft_ms':>9} {'g(C)':>7} "
          f"{'verify_ms':>9} {'chain_ms':>9} {'sys_tok_s':>10}")
    for C in Cs:
        r = rows[C]
        al = r["accept_len"]
        dm = r["draft_ms"]
        g = (dm / base_ms) if (dm and base_ms) else None
        print(f"{C:>3} {(f'{al:.4f}' if al else 'NA'):>11} "
              f"{(f'{dm:.3f}' if dm else 'NA'):>9} "
              f"{(f'{g:.4f}' if g else 'NA'):>7} "
              f"{(f'{r['verify_ms']:.3f}' if r['verify_ms'] else 'NA'):>9} "
              f"{(f'{r['chain_ms']:.3f}' if r['chain_ms'] else 'NA'):>9} "
              f"{(f'{r['sys_tok_s']:.1f}' if r['sys_tok_s'] else 'NA'):>10}")

    # speedup(C,f) = accept_len(C) / (K*g(C)*(1-f) + 1)
    print(f"\n=== projected speedup(C,f) = accept_len(C)/(K*g(C)*(1-f)+1), K={K} ===")
    hdr = f"{'C':>3} " + " ".join(f"f={f:<5}" for f in fs)
    print(hdr)
    speed = {f: {} for f in fs}
    for C in Cs:
        r = rows[C]
        al = r["accept_len"]
        dm = r["draft_ms"]
        g = (dm / base_ms) if (dm and base_ms) else None
        cells = []
        for f in fs:
            if al is None or g is None:
                cells.append(f"{'NA':>7}")
                continue
            sp = al / (K * g * (1 - f) + 1)
            speed[f][C] = sp
            cells.append(f"{sp:>7.4f}")
        print(f"{C:>3} " + " ".join(cells))

    print("\n=== optimal C per f (vs full draft C=top_k) ===")
    for f in fs:
        d = speed[f]
        if not d:
            continue
        bestC = max(d, key=lambda c: d[c])
        full = d.get(top_k)
        gain = (d[bestC] / full - 1) * 100 if full else None
        note = ("(== full draft)" if bestC == top_k
                else f"beats full by {gain:+.1f}%" if gain is not None else "")
        print(f"  f={f:.2f}: best C={bestC}  speedup={d[bestC]:.4f}  "
              f"full(C={top_k})={full:.4f}  {note}")


if __name__ == "__main__":
    main()
