#!/usr/bin/env python3
"""What actually determines the chosen configuration -- and how much it matters.

Two questions this answers, both over the measured G98-E grid:

1. **Is the strategy genuinely different per regime?** Reported as each
   regime's own optimum against the single best GLOBAL static, at that
   regime. If those ratios are ~1.00 the answer is no.
2. **What does getting it wrong cost?** A transfer matrix: run row-regime's
   optimum at column-regime, relative to column-regime's own optimum.

The two are not the same question and the grid answers them differently,
which is the point. Specialization can be worth almost nothing while
MIS-specialization is expensive -- that happens when the global static is a
good compromise and a config tuned for a different regime is a bad one.

No new measurement. Script for the numbers quoted in
`results_strategy_map.md`.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import score_w98_g98e_d3 as scorer

PHASE = Path(__file__).resolve().parent.parent
OFF_KEY = "off"
QUANT_PREFIX = "w4a16"

# (batch, input context) per regime, from the phase's regime definitions.
SHAPE = {
    "R1": (1, "short"),
    "R4": (8, "8k"),
    "R5": (8, "14k"),
    "R5cot": (8, "14k"),
    "R6": (32, "short"),
    "R8": (16, "short"),
}


def per_regime(
    grid: Mapping[str, Mapping[str, float]],
    regimes: Sequence[str],
    weights: Mapping[str, float],
) -> dict[str, Any]:
    best = {r: max(grid, key=lambda c: grid[c][r]) for r in regimes}
    global_static = max(
        grid, key=lambda c: scorer.aggregate(grid, {r: c for r in regimes}, weights)
    )
    rows = {}
    for regime in regimes:
        quant_best = max(
            (c for c in grid if c.startswith(QUANT_PREFIX)),
            key=lambda c: grid[c][regime],
        )
        rows[regime] = {
            "batch": SHAPE.get(regime, (None, None))[0],
            "input_context": SHAPE.get(regime, (None, None))[1],
            "own_optimum": best[regime],
            "vs_global_static": round(
                grid[best[regime]][regime] / grid[global_static][regime], 6
            ),
            "vs_best_quantized": round(
                grid[best[regime]][regime] / grid[quant_best][regime], 6
            ),
            "own_optimum_is_quantized": best[regime].startswith(QUANT_PREFIX),
        }
    return {"global_static": global_static, "regimes": rows, "best": best}


def transfer(
    grid: Mapping[str, Mapping[str, float]],
    regimes: Sequence[str],
    best: Mapping[str, str],
) -> dict[str, Any]:
    matrix = {
        src: {
            dst: round(grid[best[src]][dst] / grid[best[dst]][dst], 6)
            for dst in regimes
        }
        for src in regimes
    }
    off = [r for r in regimes if best[r] == OFF_KEY]
    pairs = [(s, d) for s in regimes for d in regimes if s != d]
    armed_pairs = [(s, d) for s, d in pairs if s not in off and d not in off]
    return {
        "matrix": matrix,
        "worst_off_diagonal": min(matrix[s][d] for s, d in pairs),
        "worst_off_diagonal_pair": min(pairs, key=lambda p: matrix[p[0]][p[1]]),
        "regimes_whose_optimum_is_off": off,
        "worst_off_diagonal_excluding_off_regimes": (
            min(matrix[s][d] for s, d in armed_pairs) if armed_pairs else None
        ),
        "worst_pair_excluding_off_regimes": (
            min(armed_pairs, key=lambda p: matrix[p[0]][p[1]])
            if armed_pairs
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=PHASE / "data/g98_e/grid")
    parser.add_argument(
        "--out", type=Path, default=PHASE / "data/g98_e/strategy_map.json"
    )
    args = parser.parse_args()

    grids = scorer.load_grid(args.grid)
    record: dict[str, Any] = {
        "record_type": "w98_strategy_map",
        "schema_version": 1,
        "arms": {},
    }

    for arm, grid in sorted(grids.items()):
        regimes = scorer.regimes_of(grid)
        complete = {
            cell: by_regime
            for cell, by_regime in grid.items()
            if all(r in by_regime for r in regimes)
        }
        weights = {r: 1.0 / len(regimes) for r in regimes}
        summary = per_regime(complete, regimes, weights)
        moved = transfer(complete, regimes, summary["best"])
        record["arms"][arm] = {
            "global_static": summary["global_static"],
            "per_regime": summary["regimes"],
            "transfer": moved,
        }

        if arm != "corrected":
            continue
        print(f"best GLOBAL static: {summary['global_static']}\n")
        print("regime  b   ctx     own optimum                    "
              "vs global static  vs best quantized")
        for regime in regimes:
            row = summary["regimes"][regime]
            print(
                f"{regime:6s} {str(row['batch']):3s} {row['input_context']:6s}  "
                f"{row['own_optimum']:30s} {row['vs_global_static']:15.4f}  "
                f"{row['vs_best_quantized']:16.4f}"
            )
        print("\n=== transfer: ROW regime's optimum run at COLUMN regime ===")
        print("        " + "".join(f"{c:>9s}" for c in regimes))
        for src in regimes:
            cells = "".join(f"{moved['matrix'][src][d]:9.3f}" for d in regimes)
            print(f"{src:7s}{cells}")
        s, d = moved["worst_off_diagonal_pair"]
        print(f"\nworst off-diagonal: {moved['worst_off_diagonal']:.3f} ({s} -> {d})")
        if moved["worst_pair_excluding_off_regimes"]:
            s2, d2 = moved["worst_pair_excluding_off_regimes"]
            print(
                "worst excluding OFF-optimum regimes: "
                f"{moved['worst_off_diagonal_excluding_off_regimes']:.3f} "
                f"({s2} -> {d2})"
            )

    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
