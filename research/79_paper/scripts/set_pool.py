#!/usr/bin/env python3
"""Set-level ground truth: measure beta for a diverse pool of layer sets.

Pool design (budgets 3/5/7): top-product picks, spread, random (seeded),
late-heavy -- diverse enough that a search backtest over the pool is
non-trivial. Appends to data/beta_menu_ext.csv (arms ls_p_<ids>).
Usage: set_pool.py --half {0,1}   # split across two GPUs
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beta_menu_ext as ME  # noqa: E402
import score_accept as SA  # noqa: E402

POOL = [
    # budget 3 (have: greedy {2,5,6}=0.849)
    [2, 4, 5], [3, 11, 20], [8, 17, 23], [20, 22, 24], [4, 12, 19],
    # budget 5
    [2, 3, 4, 5, 6], [2, 4, 5, 6, 12], [6, 10, 15, 19, 24],
    [3, 7, 13, 18, 22], [19, 21, 22, 23, 25],
    # budget 7 (have: greedy {2-7,12}=0.507, nonadj {2,5,7,9,12,14,16}=0.435)
    [3, 6, 10, 14, 17, 21, 24], [2, 4, 8, 11, 15, 19, 23],
    [17, 19, 20, 21, 22, 23, 24], [2, 6, 9, 13, 16, 20, 24],
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", type=int, required=True, choices=[0, 1])
    a = ap.parse_args()
    done = ME.done_arms()
    todo = [s for i, s in enumerate(POOL) if i % 2 == a.half]
    model = SA.load_model(ME.CFG)
    for drop in todo:
        arm = "ls_p_" + "-".join(map(str, drop))
        if arm in done:
            print(f"skip {arm}", flush=True)
            continue
        with ME.LayerSet(model, drop):
            ME.append_row(SA.score_arm(model, ME.CFG, arm, ME.REFDIR, ME.Args()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
