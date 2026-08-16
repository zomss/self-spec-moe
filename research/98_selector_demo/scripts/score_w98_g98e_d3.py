# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Score D3 from the G98-E grid: selector vs omniscient vs every static.

The grid is `rate(regime, configuration)` in decode currency. Every D3
quantity is an aggregation of it over a workload mix `v`, so scoring is
arithmetic on measured data and adds no measurement of its own.

Registered mix: **equal weight** over the six regimes (decided 2026-08-15,
before any D3 boot). The **mix frontier** is reported alongside it, per the
same decision: it is free once the grid exists, cannot be gamed after the
fact, and is the quantity that sizes Phase 97's switching engineering.

The selector's choice is the configuration Rounds 1-2 would pick per regime.
It is computed from the MEASURED grid restricted to what the selector could
know, never from hindsight -- the omniscient arm is the hindsight one, and
keeping them separate is the entire point of the comparison.

Both amendment-2 arms are scored and BOTH are reported, per the registered
reporting rule: a dual-arm run is not licence to quote the kinder arm.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

OFF_KEY = "off"
OMNISCIENT_TARGET = 0.90
LCB_MARGIN = 0.02


def load_grid(grid_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    """arm -> cell -> regime -> decode tokens/s."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for path in sorted(Path(grid_dir).glob("*.json")):
        record = json.loads(path.read_text())
        arm = record["arm"]
        cell = record["cell"]
        for regime_id, obs in record["observations"].items():
            out.setdefault(arm, {}).setdefault(cell, {})[regime_id] = float(
                obs["decode_tokens_per_s"]
            )
    return out


def regimes_of(grid: Mapping[str, Mapping[str, float]]) -> list[str]:
    common: set[str] | None = None
    for by_regime in grid.values():
        keys = set(by_regime)
        common = keys if common is None else (common & keys)
    return sorted(common or set())


def aggregate(
    grid: Mapping[str, Mapping[str, float]],
    cell_for_regime: Mapping[str, str],
    weights: Mapping[str, float],
) -> float:
    """Aggregate decode throughput of a per-regime assignment.

    TIME-WEIGHTED, not an arithmetic mean of rates. To emit a token share
    `v_R` from each regime the run spends `v_R / rate_R` of its time there,
    so the throughput of the mix is `1 / sum(v_R / rate_R)`.

    Averaging rates arithmetically would be a straightforward error here, not
    a stylistic one: measured OFF rates span 125 tok/s at R1 (batch 1) to
    3448 at R6 (batch 32), so an arithmetic mean is dominated by the
    high-batch regimes and corresponds to no workload anyone runs. At equal
    weight the two differ by ~7x, and the D3 verdict would follow whichever
    was chosen.
    """
    seconds = 0.0
    for regime, share in weights.items():
        if share <= 0:
            continue
        rate = grid[cell_for_regime[regime]][regime]
        if rate <= 0:
            raise ValueError(f"non-positive rate for {regime}")
        seconds += share / rate
    return 0.0 if seconds <= 0 else 1.0 / seconds


def omniscient_choice(
    grid: Mapping[str, Mapping[str, float]], regimes: Sequence[str]
) -> dict[str, str]:
    """Best cell per regime, with hindsight. The ceiling, not the selector."""
    return {
        regime: max(grid, key=lambda cell: grid[cell][regime]) for regime in regimes
    }


def selector_choice(
    grid: Mapping[str, Mapping[str, float]],
    regimes: Sequence[str],
    predicted: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, str]:
    """What the two-round selector picks per regime.

    With a prediction map, the pick is the argmax of PREDICTED value and the
    grid supplies only its realized rate — the honest construction, in which
    the selector can be wrong. Without one, the selector degenerates to the
    omniscient pick and the comparison becomes vacuous, so that case is
    reported explicitly rather than silently.
    """
    if not predicted:
        return omniscient_choice(grid, regimes)
    out = {}
    for regime in regimes:
        candidates = [c for c in grid if regime in predicted.get(c, {})]
        if not candidates:
            out[regime] = max(grid, key=lambda cell: grid[cell][regime])
            continue
        out[regime] = max(candidates, key=lambda cell: predicted[cell][regime])
    return out


def score_arm(
    grid: Mapping[str, Mapping[str, float]],
    weights: Mapping[str, float],
    predicted: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    regimes = list(weights)
    omni_pick = omniscient_choice(grid, regimes)
    sel_pick = selector_choice(grid, regimes, predicted)
    omni = aggregate(grid, omni_pick, weights)
    sel = aggregate(grid, sel_pick, weights)
    statics = {
        cell: aggregate(grid, dict.fromkeys(regimes, cell), weights)
        for cell in grid
        if all(regime in grid[cell] for regime in regimes)
    }
    best_static = max(statics, key=statics.get) if statics else None
    off = statics.get(OFF_KEY)
    beaten = {
        cell: sel >= value * (1.0 + LCB_MARGIN) for cell, value in statics.items()
    }
    return {
        "selector_rate": round(sel, 6),
        "omniscient_rate": round(omni, 6),
        "omniscient_fraction": round(sel / omni, 6) if omni else None,
        "meets_90pct": bool(omni and sel / omni >= OMNISCIENT_TARGET),
        "off_rate": None if off is None else round(off, 6),
        "selector_over_off": round(sel / off, 6) if off else None,
        "best_static_cell": best_static,
        "best_static_rate": (
            None if best_static is None else round(statics[best_static], 6)
        ),
        "beats_every_static_by_lcb_margin": all(beaten.values()),
        "statics_not_beaten": sorted(c for c, ok in beaten.items() if not ok),
        "selector_choice": sel_pick,
        "omniscient_choice": omni_pick,
        "selector_is_omniscient": sel_pick == omni_pick,
    }


def mix_frontier(
    grid: Mapping[str, Mapping[str, float]],
    regimes: Sequence[str],
    predicted: Mapping[str, Mapping[str, float]] | None,
    steps: int = 21,
) -> list[dict[str, Any]]:
    """How the verdict moves as one regime's share sweeps 0 -> 1.

    Phase 82's lesson in one number: its omniscient switcher cleared only
    +1.7% over static-OFF because the spec-favorable regime carried 9% of
    tokens. The frontier says, for THIS grid, which mixes clear the target
    and which do not, so the answer is not hostage to one declared mix.
    """
    out = []
    others = len(regimes) - 1
    for regime in regimes:
        for index in range(steps):
            share = index / (steps - 1)
            rest = (1.0 - share) / others if others else 0.0
            weights = {r: (share if r == regime else rest) for r in regimes}
            if abs(sum(weights.values()) - 1.0) > 1e-9:
                continue
            scored = score_arm(grid, weights, predicted)
            out.append(
                {
                    "concentrated_regime": regime,
                    "share": round(share, 4),
                    "omniscient_fraction": scored["omniscient_fraction"],
                    "meets_90pct": scored["meets_90pct"],
                    "selector_over_off": scored["selector_over_off"],
                    "beats_every_static": scored["beats_every_static_by_lcb_margin"],
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-dir",
        type=Path,
        default=Path("research/98_selector_demo/data/g98_e/grid"),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--predicted",
        type=Path,
        default=None,
        help="optional cell -> regime -> predicted rate, for the honest pick",
    )
    args = parser.parse_args()
    grid = load_grid(args.grid_dir)
    if not grid:
        raise SystemExit(f"no grid cells under {args.grid_dir}")
    predicted = json.loads(args.predicted.read_text()) if args.predicted else None

    arms: dict[str, Any] = {}
    for arm, by_cell in sorted(grid.items()):
        regimes = regimes_of(by_cell)
        complete = {c: r for c, r in by_cell.items() if set(regimes) <= set(r)}
        weights = dict.fromkeys(regimes, 1.0 / len(regimes)) if regimes else {}
        arms[arm] = {
            "cells": len(complete),
            "regimes": regimes,
            "equal_weight": score_arm(complete, weights, predicted),
            "frontier": mix_frontier(complete, regimes, predicted),
        }
    record = {
        "schema_version": 1,
        "record_type": "w98d3_result",
        "mix": "equal weight over the six regimes (registered 2026-08-15)",
        "omniscient_target": OMNISCIENT_TARGET,
        "lcb_margin": LCB_MARGIN,
        "selector_pick_source": "predicted map" if predicted else "MEASURED (vacuous)",
        "arms": arms,
    }
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text)
    print(
        json.dumps(
            {
                arm: {
                    k: v["equal_weight"][k]
                    for k in (
                        "omniscient_fraction",
                        "meets_90pct",
                        "selector_over_off",
                        "beats_every_static_by_lcb_margin",
                    )
                }
                for arm, v in arms.items()
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
