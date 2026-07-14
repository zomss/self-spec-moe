#!/usr/bin/env python3
"""E1: interaction-aware iterative-greedy layer profiling (dense).

Round 1 = 79's measured leave-one-out singles (reused, not re-measured).
Each later round re-MEASURES beta(current_set + candidate) for the top-POOL
unchosen candidates ranked by the previous round's column (79: interactions
break the product at depth -- the singles ranking is only a prior). Accept
the best candidate; record the frontier point; stop at budget 7.

Writes data/iter_greedy.csv (every evaluation) and prints the frontier.
Naive references (dense): contiguous skip125=0.448, skip25=0.09;
singles-greedy {2,5,6}@3=0.849, {2..7,12}@7=0.507.
"""

import csv
import sys
from pathlib import Path

import torch

PHASE = Path(__file__).resolve().parents[1]
P79 = PHASE.parent / "79_paper"
P77 = PHASE.parent / "77_acceptance_map"
sys.path.insert(0, str(P77 / "scripts"))
sys.path.insert(0, str(P79 / "scripts"))
import score_accept as SA  # noqa: E402
from beta_menu_ext import CFG, REFDIR, Args, LayerSet  # noqa: E402

OUT = PHASE / "data/iter_greedy.csv"
(PHASE / "data").mkdir(parents=True, exist_ok=True)
POOL = 8            # candidates re-measured per round (cost cap)
BUDGET = 7


def append_row(row, rnd):
    row = dict(row)
    row["round"] = rnd
    exists = OUT.exists()
    with OUT.open("a") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    # round-1 column: 79's measured singles
    singles = {}
    for r in csv.DictReader((P79 / "data/beta_menu_ext.csv").open()):
        if r["arm"].startswith("ls_") and r["arm"].count("_") == 1:
            singles[int(r["arm"][3:])] = float(r["beta_greedy"])
    done = {}
    if OUT.exists():
        for r in csv.DictReader(OUT.open()):
            done[r["arm"]] = float(r["beta_greedy"])

    model = SA.load_model(CFG)
    chosen = []
    col = dict(singles)                      # candidate -> beta(set+cand)
    frontier = []
    for rnd in range(1, BUDGET + 1):
        best_c = max(col, key=col.get)
        chosen.append(best_c)
        frontier.append((len(chosen), tuple(chosen), col[best_c]))
        print(f"[E1] round {rnd}: +layer {best_c} -> set {sorted(chosen)} "
              f"beta={col[best_c]:.4f}", flush=True)
        if rnd == BUDGET:
            break
        cands = [c for c in sorted(col, key=col.get, reverse=True)
                 if c not in chosen][:POOL]
        col = {}
        for c in cands:
            drop = sorted(chosen + [c])
            arm = "ig_" + "-".join(map(str, drop))
            if arm in done:
                col[c] = done[arm]
                continue
            with LayerSet(model, drop):
                row = SA.score_arm(model, CFG, arm, REFDIR, Args())
            append_row(row, rnd + 1)
            col[c] = float(row["beta_greedy"])
    print("\n[E1] FRONTIER (budget, set, beta):", flush=True)
    for b, s, v in frontier:
        print(f"  {b}: {sorted(s)} -> {v:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
