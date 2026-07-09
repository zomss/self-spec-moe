#!/usr/bin/env python3
"""Build the fp8-ceiling table from the harness JSONs.

Reports per-iteration tok/s (so stragglers are visible), the mean+-std, the
harness's own `suspect` flag (rel-std > 15%), and the ratio to the same-box
nospec denominator for that (ctx, batch) cell. Cross-box ratios are never
computed: only arms from this session's data/ dir are read.

Usage: python scripts/analyze_fp8_ceiling.py [data_dir]
"""

import glob
import json
import os
import re
import statistics
import sys

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)

# w72n_p74_fc<gen>_<arm>_K<k>_ctx<ctx>_b<b>_<mode>_...json  (gen = "", 2, 3, ...)
PAT = re.compile(r"w72n_p74_fc\d?_(?P<arm>[a-z0-9]+)_K(?P<k>\d+)_ctx(?P<ctx>\d+)_b(?P<b>\d+)_")


def rows():
    for f in sorted(glob.glob(os.path.join(DATA, "w72n_p74_fc*.json"))):
        m = PAT.search(os.path.basename(f))
        if not m:
            continue
        d = json.load(open(f))
        for r in d.get("results", []):
            ls, ss = r.get("long_s_all"), r.get("short_s_all")
            if not ls or not ss:
                continue
            ntok = r["out_tokens_decode"]
            per_iter = [ntok / (a - b) for a, b in zip(ls, ss)]
            yield {
                "file": os.path.basename(f),
                "arm": m["arm"], "k": int(m["k"]),
                "ctx": int(m["ctx"]), "b": int(m["b"]),
                "iters": r.get("iters"),
                "mean": r["tok_s_mean"], "std": r["tok_s_std"],
                "suspect": r.get("suspect", False),
                "accept": r.get("accept_len"),
                "per_iter": per_iter,
            }


def main():
    data = list(rows())
    if not data:
        print(f"no data in {DATA}")
        return
    # Keep the highest-iters run per (arm, ctx, b, k): a re-run supersedes.
    best = {}
    for r in data:
        key = (r["arm"], r["ctx"], r["b"], r["k"])
        if key not in best or r["iters"] > best[key]["iters"]:
            best[key] = r

    denom = {(r["ctx"], r["b"]): r["mean"]
             for r in best.values() if r["arm"] == "nospec"}

    hdr = (f"{'cell':>14} {'arm':>8} {'it':>3} {'tok/s (mean)':>16} {'median':>8} "
           f"{'vs nospec':>10} {'accept':>7}  per-iter")
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(best, key=lambda t: (-t[1], t[2], t[0])):
        r = best[key]
        cell = f"ctx{r['ctx']//1024}k/b{r['b']}"
        ref = denom.get((r["ctx"], r["b"]))
        rel = f"{r['mean']/ref:.2f}x" if ref else "-"
        acc = f"{r['accept']:.3f}" if r["accept"] else "-"
        flag = " SUSPECT" if r["suspect"] else ""
        med = statistics.median(r["per_iter"])
        pi = "[" + ", ".join(f"{x:.0f}" for x in r["per_iter"]) + "]"
        print(f"{cell:>14} {r['arm']:>8} {r['iters']:>3} "
              f"{r['mean']:8.1f}+-{r['std']:<5.1f} {med:8.1f} {rel:>10} {acc:>7}  {pi}{flag}")

    print("\nNote: 'suspect' = harness guardrail, rel-std > 15%. Do not cite a "
          "suspect row on its mean alone.")
    print("Where per-iter is BIMODAL (two tight clusters, e.g. fp8blk @16k), the "
          "mean is a mixture, not an estimate of either mode: quote median +\n"
          "the stall separately and explain the stall.")


if __name__ == "__main__":
    main()
