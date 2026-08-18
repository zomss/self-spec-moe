#!/usr/bin/env python3
"""Per-request divergence pattern between two campaign-1 boot records.

Usage: analyze_w100_divergence.py <arm_a> <arm_b> <batch>

For every shared cell label: how many requests match exactly (sha256),
and for the differing ones, the output-length deltas. Distinguishes the
two hypotheses for T=0 divergence: scattered greedy tie-flips from
batch-composition numerics (most requests identical, a few diverge with
assorted length deltas) versus a systematic engine-path difference
(every request diverges).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "campaign1"


def main() -> None:
    arm_a, arm_b, batch = sys.argv[1], sys.argv[2], int(sys.argv[3])
    a = json.loads((OUT / f"{arm_a}__b{batch}.json").read_text())
    b = json.loads((OUT / f"{arm_b}__b{batch}.json").read_text())
    for label in a["cells"]:
        if label not in b["cells"]:
            continue
        ra = {r["prompt_index"]: r for r in a["cells"][label]["requests"]}
        rb = {r["prompt_index"]: r for r in b["cells"][label]["requests"]}
        same = [i for i in ra if ra[i]["sha256"] == rb[i]["sha256"]]
        diff = [i for i in ra if i not in same]
        deltas = {i: rb[i]["out_toks"] - ra[i]["out_toks"] for i in diff}
        lens_equal_but_diff = [i for i in diff if deltas[i] == 0]
        print(f"[{label}] n={len(ra)} identical={len(same)} "
              f"diverged={len(diff)}")
        if diff:
            print(f"    out_tok deltas ({arm_b}-{arm_a}): "
                  f"{dict(sorted(deltas.items()))}")
            if lens_equal_but_diff:
                print(f"    same-length-but-different-text: "
                      f"{lens_equal_but_diff}")


if __name__ == "__main__":
    main()
