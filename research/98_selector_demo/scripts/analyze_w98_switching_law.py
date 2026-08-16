#!/usr/bin/env python3
"""When is per-regime switching worth anything? An exact decomposition.

D3 reported per-regime switching at +1.4% over the best static and concluded
the gain being bought was small. That number is now known to be an artifact
of the missing fail-closed rule (`analyze_w98_failclosed.py` -> +6.2%), but
the deeper question stands: +6.2% is still not much. This script answers WHY
exactly, and what would have to be true for switching to be worth more.

## The law

With time-weighted aggregation `1 / sum(v_R / rate_R)`, the gain of a
per-regime assignment over any fixed configuration is EXACTLY

    gain = sum_R  t_R * (rate_sel(R) / rate_static(R))

where `t_R` is the selector's **time** share in regime R,

    t_R = (v_R / rate_sel(R)) / sum_R' (v_R' / rate_sel(R'))

i.e. the fraction of wall time the mix spends in R, not its token share. The
identity is checked numerically here rather than merely asserted.

Two consequences, and they are the whole story:

1. Switching pays in proportion to time spent in regimes where the static is
   wrong. A regime that is fast contributes little no matter how badly the
   static loses there.
2. The regime that dominates the time budget is the SLOWEST one — batch 1.
   If every configuration performs about the same at batch 1, switching
   cannot pay, however much the other regimes disagree.

## The feasibility term

The above assumes every configuration is deployable in every regime. It is
not: the `w4a16-quantized` draft is a separate ~6.1 GB resident, and under KV
pressure that memory is not available. When the lever set is not uniformly
feasible, no single static can serve the whole mix, and the switching gain
stops being marginal.

`--feasibility` scores that case. It applies a feasibility MASK to measured
rates from the existing grid; the rates are measured, the mask is modelled.
KV pressure was never varied as an axis in this phase (every boot ran at
`gpu_memory_utilization=0.90` with the draft resident), so this SIZES the
effect and does not establish it.
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

# Regimes whose KV working set plausibly excludes a separate 6.1 GB draft.
# 14k context at batch 8 is where KV actually competes for HBM in this grid.
PRESSURED_REGIMES = frozenset({"R5", "R5cot"})
QUANTIZED_PREFIX = "w4a16"


def equal_weights(regimes: Sequence[str]) -> dict[str, float]:
    return {r: 1.0 / len(regimes) for r in regimes}


def best_static(
    grid: Mapping[str, Mapping[str, float]],
    regimes: Sequence[str],
    weights: Mapping[str, float],
    cells: Sequence[str],
) -> str:
    return max(
        cells, key=lambda c: scorer.aggregate(grid, {r: c for r in regimes}, weights)
    )


def decompose(
    grid: Mapping[str, Mapping[str, float]],
    regimes: Sequence[str],
    weights: Mapping[str, float],
    choice: Mapping[str, str],
    static_cell: str,
) -> dict[str, Any]:
    """The identity above, term by term, with a numerical check."""
    total_time = sum(weights[r] / grid[choice[r]][r] for r in regimes)
    terms = []
    reconstructed = 0.0
    for regime in regimes:
        share = (weights[regime] / grid[choice[regime]][regime]) / total_time
        ratio = grid[choice[regime]][regime] / grid[static_cell][regime]
        reconstructed += share * ratio
        terms.append(
            {
                "regime": regime,
                "time_share": round(share, 6),
                "selector_rate": round(grid[choice[regime]][regime], 4),
                "static_rate": round(grid[static_cell][regime], 4),
                "ratio": round(ratio, 6),
                "contribution": round(share * ratio, 6),
                "excess_contribution": round(share * (ratio - 1.0), 6),
            }
        )
    actual = scorer.aggregate(grid, choice, weights) / scorer.aggregate(
        grid, {r: static_cell for r in regimes}, weights
    )
    return {
        "static_cell": static_cell,
        "gain": round(actual, 8),
        "gain_reconstructed": round(reconstructed, 8),
        "identity_residual": abs(actual - reconstructed),
        "terms": terms,
    }


def feasible(cell: str, regime: str) -> bool:
    if cell == OFF_KEY:
        return True
    if regime in PRESSURED_REGIMES and cell.startswith(QUANTIZED_PREFIX):
        return False
    return True


def feasibility_study(
    grid: Mapping[str, Mapping[str, float]],
    regimes: Sequence[str],
    weights: Mapping[str, float],
) -> dict[str, Any]:
    cells = list(grid)
    uniform = [c for c in cells if all(feasible(c, r) for r in regimes)]
    constrained_static = best_static(grid, regimes, weights, uniform)
    unconstrained_static = best_static(grid, regimes, weights, cells)
    choice = {
        r: max((c for c in cells if feasible(c, r)), key=lambda c: grid[c][r])
        for r in regimes
    }
    rate_choice = scorer.aggregate(grid, choice, weights)
    rate_constrained = scorer.aggregate(
        grid, {r: constrained_static for r in regimes}, weights
    )
    rate_unconstrained = scorer.aggregate(
        grid, {r: unconstrained_static for r in regimes}, weights
    )
    return {
        "model": (
            "measured rates with a MODELLED feasibility mask: the quantized "
            "draft's separate ~6.1 GB is assumed unaffordable at 14k context. "
            "KV pressure was never varied in this phase, so this sizes the "
            "effect rather than establishing it."
        ),
        "pressured_regimes": sorted(PRESSURED_REGIMES),
        "best_static_feasible_everywhere": constrained_static,
        "best_static_feasible_everywhere_rate": round(rate_constrained, 4),
        "best_static_ignoring_feasibility": unconstrained_static,
        "best_static_ignoring_feasibility_rate": round(rate_unconstrained, 4),
        "note_on_unconstrained": "not deployable across the mix; shown for scale",
        "selection_choice": dict(choice),
        "selection_rate": round(rate_choice, 4),
        "switching_gain": round(rate_choice / rate_constrained, 6),
        "decomposition": decompose(
            grid, regimes, weights, choice, constrained_static
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=PHASE / "data/g98_e/grid")
    parser.add_argument(
        "--failclosed", type=Path, default=PHASE / "data/g98_e/d3_failclosed.json"
    )
    parser.add_argument(
        "--out", type=Path, default=PHASE / "data/g98_e/d3_switching_law.json"
    )
    args = parser.parse_args()

    failclosed = json.loads(args.failclosed.read_text())
    grids = scorer.load_grid(args.grid)

    arms: dict[str, Any] = {}
    for arm, grid in sorted(grids.items()):
        regimes = scorer.regimes_of(grid)
        complete = {
            cell: by_regime
            for cell, by_regime in grid.items()
            if all(r in by_regime for r in regimes)
        }
        weights = equal_weights(regimes)
        payload = failclosed["arms"][arm]
        fc_choice = payload["fail_closed"]["choice"]
        argmax_choice = payload["as_measured_argmax"]["choice"]
        static_cell = payload["fail_closed"]["best_static_cell"]
        omni = scorer.omniscient_choice(complete, regimes)

        arms[arm] = {
            "as_measured": decompose(
                complete, regimes, weights, argmax_choice, static_cell
            ),
            "fail_closed": decompose(
                complete, regimes, weights, fc_choice, static_cell
            ),
            "omniscient_ceiling": decompose(
                complete, regimes, weights, omni, static_cell
            ),
            "feasibility_study": feasibility_study(complete, regimes, weights),
        }

    record = {
        "record_type": "w98d3_switching_law",
        "schema_version": 1,
        "identity": "gain = sum_R t_R * (rate_sel(R)/rate_static(R)), t_R = time share",
        "arms": arms,
    }
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    for arm, payload in sorted(arms.items()):
        print(f"=== {arm} ===")
        for name in ("as_measured", "fail_closed", "omniscient_ceiling"):
            d = payload[name]
            print(
                f"  {name:20s} gain {d['gain']:.4f}  "
                f"(identity residual {d['identity_residual']:.2e})"
            )
        print()
        d = payload["fail_closed"]
        print("  fail-closed decomposition vs", d["static_cell"])
        print("  regime  time-share   ratio    excess contribution")
        for t in d["terms"]:
            print(
                f"  {t['regime']:6s} {t['time_share']:10.4f} {t['ratio']:8.4f} "
                f"{t['excess_contribution']:+18.4f}"
            )
        f = payload["feasibility_study"]
        print()
        print("  mixed feasibility (modelled):")
        print(
            f"    best static feasible everywhere : "
            f"{f['best_static_feasible_everywhere']} "
            f"({f['best_static_feasible_everywhere_rate']:.1f} tok/s)"
        )
        print(
            f"    per-regime selection            : "
            f"{f['selection_rate']:.1f} tok/s"
        )
        print(f"    switching gain                  : {f['switching_gain']:.4f}x")
        print()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
