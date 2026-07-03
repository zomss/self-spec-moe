"""Phase 52 analysis: 2-node spec vs no-spec + f-bound from the 1-node ref.

Reads (from ../data):
  w72n_qwen30b_2node_nospec_cg_nospec.json   (2-node DP16 no-spec)
  w72n_qwen30b_2node_spec_cg_K2.json         (2-node DP16 spec, FP8 draft, K=2)
  w7fp8_qwen30b_1node_nvlink_nospec_cg_nospec.json  (1-node DP8 native ref)

f estimate: per-rank batch matched, f = 1 - decode_s(1node)/decode_s(2node)
(lower bound: EP16 halves per-rank expert weight read vs EP8, so the true
2-node compute floor is slightly below the 1-node step).
"""

import json
import os
import sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def rows(path):
    with open(path) as f:
        d = json.load(f)
    return {r["batch"]: r for r in d["results"] if "error" not in r}


def main():
    ns2 = rows(os.path.join(DATA, "w72n_qwen30b_2node_nospec_cg_nospec.json"))
    sp2 = rows(os.path.join(DATA, "w72n_qwen30b_2node_spec_cg_K2.json"))
    try:
        ns1 = rows(os.path.join(
            DATA, "w7fp8_qwen30b_1node_nvlink_nospec_cg_nospec.json"))
    except FileNotFoundError:
        ns1 = {}

    print("batch | 1n-nospec tok/s | 2n-nospec tok/s | 2n-spec tok/s | "
          "accept | speedup | f_lb")
    for b in sorted(ns2):
        n2, s2 = ns2[b], sp2.get(b)
        n1 = ns1.get(b)
        f_lb = (1 - n1["decode_s_mean"] / n2["decode_s_mean"]) if n1 else None
        spd = (s2["tok_s_mean"] / n2["tok_s_mean"]) if s2 else None
        cells = [
            f"{b:5d}",
            f"{n1['tok_s_mean']:8.1f}" if n1 else "     n/a",
            f"{n2['tok_s_mean']:8.1f}",
            f"{s2['tok_s_mean']:8.1f}" if s2 else "     n/a",
            f"{s2.get('accept_len'):5.2f}" if s2 and s2.get("accept_len")
            else "  n/a",
            f"{spd:6.3f}" if spd else "   n/a",
            f"{f_lb:5.2f}" if f_lb is not None else "  n/a",
        ]
        print(" | ".join(cells))
        for tag, r in (("2n-nospec", n2), ("2n-spec", s2)):
            if r and r.get("suspect"):
                print(f"      ^ WARNING {tag} b={b} suspect timing "
                      f"(std/mean or slope check failed)")


if __name__ == "__main__":
    sys.exit(main())
