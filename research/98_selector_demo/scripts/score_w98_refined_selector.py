# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The two-round selector, scored end to end on the refined grid.

Sections 31-34 measured the grid against a stock baseline and sections 36-38
built a cost model that ranks it. Neither ran the selector's own decision.
This does: the pick is the argmax of PREDICTED value, the grid supplies only
the realized rate, and the selector is therefore free to be wrong -- the same
construction `score_w98_g98e_d3.selector_choice` uses, and for the same
reason. Without it the comparison scores a relabelled ceiling.

Three quantities per workload mix, in D3's time-weighted currency:

    selector    sum_c v_c * rate(c, argmax_a predicted[c][a])
    omniscient  sum_c v_c * max_a rate(c, a)
    static A    sum_c v_c * rate(c, A)              one A for every cell

Rates are against each cell's own **stock** boot, which is what a deployment
replaces -- not against `off`, which is our runtime parked and costs 11-31%
by itself (section 30). Every arm is instrument-free (section 32).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import w98_artifacts as artifacts  # noqa: E402

DATA = SCRIPT_DIR.parent / "data"
STOCK = "stock"
OFF = "off"
CELL_DIRS = {
    "LI": "g98_noinstr_li_b8",
    "LIO": "g98_noinstr_lio_b8",
    "SS": "g98_noinstr_ss_b8",
    "LO": "g98_noinstr_lo_b8",
}


def measured_grid() -> dict[str, dict[str, float]]:
    """cell -> arm -> throughput relative to that cell's own stock boot."""
    grid: dict[str, dict[str, float]] = {}
    for cell, directory in CELL_DIRS.items():
        records = {}
        for path in sorted((DATA / directory).glob("*.json")):
            try:
                record = artifacts.read(path)
            except artifacts.ArtifactError:
                continue
            if record.get("record_type") != "w98_refined_lo":
                continue
            records[record["cell"]] = record
        if STOCK not in records:
            raise RuntimeError(f"{cell}: no stock boot")
        ref = records[STOCK]["wall_s"] / records[STOCK]["total_out_tokens"]
        grid[cell] = {
            (arm if arm in (STOCK, OFF) else arm.split("/", 1)[1]): ref
            / (r["wall_s"] / r["total_out_tokens"])
            for arm, r in records.items()
        }
    return grid


def aggregate(
    grid: Mapping[str, Mapping[str, float]],
    choice: Mapping[str, str],
    weights: Mapping[str, float],
) -> float:
    """Time-weighted, not an arithmetic mean of rates.

    To emit a token share `v_c` from each cell the run spends `v_c / rate_c`
    of its time there, so the mix's throughput is `1 / sum(v_c / rate_c)`.
    Averaging rates arithmetically would be an error, not a style choice: the
    cells' absolute rates differ by 2x and the verdict would follow whichever
    was picked.
    """
    seconds = 0.0
    for cell, share in weights.items():
        if share <= 0:
            continue
        rate = grid[cell][choice[cell]]
        seconds += share / rate
    return 0.0 if seconds <= 0 else 1.0 / seconds


def main() -> int:
    predictions = json.loads(
        (DATA / "g98_fit" / "refined_predictions.json").read_text()
    )
    grid = measured_grid()
    cells: Sequence[str] = [c for c in CELL_DIRS if c in predictions]
    armed = {c: {a for a in grid[c] if a not in (STOCK, OFF)} for c in cells}
    common = sorted(set.intersection(*armed.values()))

    selector = {c: max(predictions[c], key=lambda a: predictions[c][a]) for c in cells}
    omniscient = {c: max(armed[c], key=lambda a: grid[c][a]) for c in cells}
    weights = {c: 1.0 / len(cells) for c in cells}

    statics = {
        a: aggregate(grid, dict.fromkeys(cells, a), weights) for a in common + [OFF]
    }
    sel = aggregate(grid, selector, weights)
    omni = aggregate(grid, omniscient, weights)
    best_static = max(statics, key=lambda a: statics[a])

    result = {
        "record_type": "w98_refined_selector_score",
        "cells": list(cells),
        "common_arms": common,
        "selector_choice": selector,
        "omniscient_choice": omniscient,
        "per_cell": {
            c: {
                "selector_arm": selector[c],
                "selector_rate": round(grid[c][selector[c]], 4),
                "omniscient_arm": omniscient[c],
                "omniscient_rate": round(grid[c][omniscient[c]], 4),
                "share_of_omniscient": round(
                    grid[c][selector[c]] / grid[c][omniscient[c]], 4
                ),
                "stock": 1.0,
            }
            for c in cells
        },
        "equal_mix": {
            "selector": round(sel, 4),
            "omniscient": round(omni, 4),
            "share_of_omniscient": round(sel / omni, 4),
            "statics": {a: round(v, 4) for a, v in sorted(statics.items())},
            "best_static": best_static,
            "selector_over_best_static": round(sel / statics[best_static], 4),
            "selector_over_stock": round(sel, 4),
        },
    }
    (DATA / "g98_fit" / "refined_selector_score.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
