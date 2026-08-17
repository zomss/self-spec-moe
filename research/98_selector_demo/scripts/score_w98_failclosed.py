# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Score the fail-closed rule against the measured D3 grid.

This reuses `score_w98_g98e_d3`'s loader and aggregator rather than
reimplementing them, so the corrected selector is judged in exactly the
currency D3 was judged in -- time-weighted decode throughput over the mix,
which the D3 record notes differs from an arithmetic mean of rates by ~7x.

The comparison is IN-SAMPLE by construction: the rule was written after
seeing R4 fail on this grid. It is reported as the arithmetic consequence of
a rule committed beforehand (`data/g98_f/commitment.json`), not as
independent evidence. The out-of-sample check is G98-F's fresh interleaved
measurement.

Nothing here writes to the D3 artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import score_w98_g98e_d3 as d3  # noqa: E402
from w98_failclosed import (  # noqa: E402
    OFF_KEY,
    apply_rule,
    commitment_digest,
    firing_set,
    load_inputs,
)

DATA = SCRIPT_DIR.parent / "data"
COMMITMENT = DATA / "g98_f" / "commitment.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_commitment(decisions: dict[str, Any]) -> dict[str, Any]:
    """Refuse to score unless the rule matches what was committed."""
    _require(
        COMMITMENT.is_file(),
        f"no commitment at {COMMITMENT}; the rule must be committed before "
        "it is scored against measured data",
    )
    committed = json.loads(COMMITMENT.read_text(encoding="utf-8"))
    _require(
        committed.get("committed_before_scoring") is True,
        "commitment does not claim pre-scoring commitment",
    )
    _require(
        committed["digest"] == commitment_digest(decisions),
        "the rule's decisions no longer match the committed digest: it was "
        "retuned after commitment",
    )
    return committed


def score(grid_dir: Path, weights: dict[str, float] | None = None) -> dict[str, Any]:
    """Aggregate the D3 grid under the baseline and fail-closed selectors."""
    grids = d3.load_grid(grid_dir)
    predicted, envelopes = load_inputs(
        DATA / "g98_e" / "d3_predictions.json", DATA / "g98_c" / "d1p_fits.json"
    )
    decisions = apply_rule(predicted, envelopes)
    committed = require_commitment(decisions)

    out: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "w98_failclosed_result",
        "in_sample": True,
        "in_sample_note": (
            "the rule was written after observing R4 in this grid; these "
            "figures are the consequence of a pre-committed rule, not "
            "independent confirmation"
        ),
        "commitment_digest": committed["digest"],
        "firing_set": firing_set(decisions),
        "arms": {},
    }
    for arm, grid in sorted(grids.items()):
        regimes = d3.regimes_of(grid)
        share = {r: 1.0 / len(regimes) for r in regimes}
        base_pick = d3.selector_choice(grid, regimes, predicted)
        fc_pick = {r: decisions[r]["choice"] for r in regimes}
        omni_pick = d3.omniscient_choice(grid, regimes)
        base = d3.aggregate(grid, base_pick, share)
        fixed = d3.aggregate(grid, fc_pick, share)
        omni = d3.aggregate(grid, omni_pick, share)
        off = d3.aggregate(grid, {r: OFF_KEY for r in regimes}, share)
        statics = {
            cell: d3.aggregate(grid, dict.fromkeys(regimes, cell), share)
            for cell in grid
        }
        best_static = max(statics, key=statics.__getitem__)
        out["arms"][arm] = {
            "baseline_selector": base,
            "failclosed_selector": fixed,
            "omniscient": omni,
            "static_off": off,
            "best_static": best_static,
            "best_static_rate": statics[best_static],
            "baseline_frac_of_omniscient": base / omni,
            "failclosed_frac_of_omniscient": fixed / omni,
            "failclosed_over_baseline": fixed / base,
            "failclosed_over_off": fixed / off,
            "failclosed_over_best_static": fixed / statics[best_static],
            "per_regime": {
                r: {
                    "baseline_pick": base_pick[r],
                    "failclosed_pick": fc_pick[r],
                    "baseline_rate": grid[base_pick[r]][r],
                    "failclosed_rate": grid[fc_pick[r]][r],
                    "omniscient_rate": grid[omni_pick[r]][r],
                    "changed": base_pick[r] != fc_pick[r],
                }
                for r in regimes
            },
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", type=Path, default=DATA / "g98_e" / "grid")
    parser.add_argument("--out", type=Path, default=DATA / "g98_f" / "result.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = score(args.grid_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for arm, row in sorted(result["arms"].items()):
        print(f"=== {arm} ===")
        print(
            f"  omniscient frac: {row['baseline_frac_of_omniscient']:.3f}"
            f" -> {row['failclosed_frac_of_omniscient']:.3f}"
        )
        print(f"  over OFF       : {row['failclosed_over_off']:.3f}x")
        print(f"  over best static: {row['failclosed_over_best_static']:.3f}x")
        for regime, cell in sorted(row["per_regime"].items()):
            if cell["changed"]:
                print(
                    f"  {regime}: {cell['baseline_pick']} "
                    f"{cell['baseline_rate']:.1f} -> {cell['failclosed_pick']} "
                    f"{cell['failclosed_rate']:.1f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
