# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure the batch-shared / per-request split from a batch sweep.

Section 7 split verify into a batch-shared and a per-request part and
calibrated it on the parked arm at ONE batch, where the split is not
identifiable: a single per-token number cannot say how much of a step is paid
once and how much is paid per sequence. It was fixed by taking the
per-request level from the profiler's verify and solving the shared level for
the residual, which is an assumption, not a measurement.

Batch identifies it. For the parked arm, with `tau = 1`::

    per_token(B) = Vs * H(B) / T(B) + Vp

where `H` is the run's horizon (its longest generation) and `T` its total
output tokens, both of which every record already carries. `H/T` falls
roughly as `1/B`, so a sweep traces a straight line whose slope is the shared
cost per step and whose intercept is the per-request cost -- the two numbers
section 7 had to assume.

The residual matters as much as the coefficients: a clean line says the
two-term split is the right shape, and curvature says the model is missing a
term that scales with batch some other way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import w98_artifacts as artifacts  # noqa: E402

OFF_KEY = "off"


def observations(directories: list[Path], arm: str = OFF_KEY) -> list[dict[str, Any]]:
    rows = []
    for directory in directories:
        for path in sorted(Path(directory).glob("*.json")):
            try:
                record = artifacts.read(path)
            except Exception:
                continue
            if record.get("record_type") != "w98_refined_lo":
                continue
            if record["cell"] != arm:
                continue
            lengths = [float(v) for v in record["out_tokens_by_request"].values()]
            rows.append(
                {
                    "batch": record["batch"],
                    "per_token_s": record["wall_s"] / record["total_out_tokens"],
                    "horizon": max(lengths),
                    "total_tokens": float(sum(lengths)),
                    "shape": max(lengths) / float(sum(lengths)),
                }
            )
    return sorted(rows, key=lambda r: r["batch"])


def fit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    if len(rows) < 2:
        raise RuntimeError("a split needs at least two batches")
    design = np.array([[r["shape"], 1.0] for r in rows])
    target = np.array([r["per_token_s"] for r in rows])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ beta
    residual = predicted / target - 1.0
    return {
        "record_type": "w98_verify_split_fit",
        "batches": [r["batch"] for r in rows],
        "verify_shared_ms": float(beta[0]) * 1000,
        "verify_per_request_ms": float(beta[1]) * 1000,
        "per_token_ms": [round(r["per_token_s"] * 1000, 4) for r in rows],
        "shape_h_over_t": [round(r["shape"], 6) for r in rows],
        "relative_residual": [round(float(v), 4) for v in residual],
        "mean_abs_residual": float(abs(residual).mean()),
        "physical": bool((beta >= 0).all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--arm", default=OFF_KEY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = observations(args.directories, args.arm)
    result = fit(rows)
    result["arm"] = args.arm
    if args.output:
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
