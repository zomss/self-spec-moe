# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refit the draft-cost coefficients ON the refined cell, not on the R-regimes.

Every prediction in `results_refined_lo.md` inherited its coefficients from an
R5cot fit -- batch 8, a FIXED 14K context, 640-token generations -- and applied
them to a cell whose context sweeps from 120 to 33000 tokens under a decaying
batch. The window sweep left ~12% of residual that no functional form removed,
and `kappa_kv` carrying the wrong regime is the obvious suspect.

The refined cell can identify the coefficients itself. Writing the drain
integral out, one arm's measured time is linear in the unknowns::

    time = (v_shared + F) * S
         + keep * A * S
         + keep * f_win * [w>0] * S
         + keep * kv_bytes * kappa_kv * R
         + v_per * Q

    S = integral du / tau(u)                  steps
    Q = integral B(u) / tau(u) du             per-request verify exposure
    R = integral B(u) * positions(u) / tau(u) du    KV exposure
    N = integral B(u) du                      tokens emitted

so seven armed arms give seven equations in four unknowns, solved by least
squares. The design identifies them because the arms vary the right things:
`keep` moves over {1.0, 0.889, 0.778} at fixed window, which separates the
per-layer term `A` from the floor `F` (the reason skip16 was admitted in
Round 2), and the window moves over {128, 256, 512, 1024, off} at fixed keep,
which is what pins `kappa_kv`.

`A` absorbs the weight term: every arm here shares the target's weights, so
`W * kappa_w` is constant across arms and collinear with the per-layer
constant. That is a property of this arm set, not a modelling choice, and it
is why the fit reports `A` rather than pretending to separate them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import w98_artifacts as artifacts  # noqa: E402
from w98_cost_u import SEGMENT, kv_positions, survival  # noqa: E402
from w98_tau_u import bucket_bounds  # noqa: E402

DATA = SCRIPT_DIR.parent / "data"
EDGES = (256, 1024, 3072, 8192)
BATCH = 8
KV_BYTES_PER_TOKEN = 2 * 2 * 8 * 128 * 36  # k+v, bf16, 8 kv heads, 128 dim, 36 layers
PROMPT = 118.0
KEEP = {0: 1.0, 4: 32 / 36, 8: 28 / 36}
# Verify's per-request growth, shape taken from the batch-8 R-regimes. Only
# the LEVEL is refit here; the slope is not identifiable from one cell.
VERIFY_SLOPE = (11.505e-3 - 9.54e-3) / (14000 - 8800)


def load_curves() -> dict[Any, dict[str, float]]:
    """Measured acceptance curves, keyed by window, at skip 0/4/8."""
    out: dict[Any, dict[str, float]] = {}
    for directory in ("g98_longu", "g98_longu_windows"):
        for path in sorted((DATA / directory).glob("*.json")):
            if path.name == "summary.json":
                continue
            record = artifacts.read(path)
            cfg = record["config"]
            tau = {}
            for name, entry in record["tau_profile"]["buckets"].items():
                armed = int(entry.get("armed_steps", 0))
                if armed:
                    positions = entry.get("pos_accepted") or []
                    tau[str(name)] = 1.0 + sum(positions[:4]) / armed
            out[(cfg["window"], cfg["skip_count"])] = tau
    return out


def load_runs() -> dict[Any, dict[str, Any]]:
    """Measured LO throughput and per-request generation lengths."""
    out: dict[Any, dict[str, Any]] = {}
    for directory in ("g98_lo_b8", "g98_lo_sweep"):
        for path in sorted((DATA / directory).glob("*.json")):
            if path.name == "summary.json":
                continue
            record = artifacts.read(path)
            cfg = record["config"]
            key = (cfg["window"], cfg["skip_count"], cfg["action"])
            out[key] = {
                "per_token": record["wall_s"] / record["total_out_tokens"],
                "lengths": sorted(record["out_tokens_by_request"].values()),
            }
    return out


def exposures(
    tau: dict[str, float], window: Any, lengths: list[float], segment: int = SEGMENT
) -> dict[str, float]:
    """S, Q, R, N for one arm, by direct integration over its own run."""
    bounds = bucket_bounds(EDGES)

    def tau_at(u: float) -> float:
        last = None
        for index, (lo, hi) in enumerate(bounds):
            value = tau.get(str(index))
            if value:
                last = value
            if lo <= u < hi:
                return value or last or 0.0
        return last or 0.0

    horizon = float(max(lengths))
    s = q = r = n = 0.0
    u = 0.0
    while u < horizon:
        width = min(float(segment), horizon - u)
        centre = u + width / 2.0
        active = survival(lengths, centre)
        if active:
            t = tau_at(centre)
            steps = width / t
            s += steps
            q += active * steps
            r += active * steps * kv_positions(window, PROMPT + centre)
            n += active * width
        u += width
    return {"S": s, "Q": q, "R": r, "N": n}


def fit() -> dict[str, Any]:
    import numpy as np

    curves = load_curves()
    runs = load_runs()
    parked = runs[("off", 0, "off")]
    park = exposures({"0": 1.0}, "off", parked["lengths"])
    # verify level from the parked arm, whose draft cost is zero by definition
    v_shared = (parked["per_token"] * park["N"] - VERIFY_SLOPE * park["Q"]) / park["S"]

    rows, targets, labels, workloads = [], [], [], []
    for (window, skip, action), run in sorted(runs.items(), key=lambda kv: str(kv[0])):
        if action != "armed":
            continue
        tau = curves.get((window, skip))
        if tau is None:
            continue
        keep = KEEP[skip]
        ex = exposures(tau, window, run["lengths"])
        windowed = 0.0 if window in ("off", 0, None) else 1.0
        rows.append(
            [
                ex["S"],  # F
                keep * ex["S"],  # A
                keep * windowed * ex["S"],  # f_win
                keep * KV_BYTES_PER_TOKEN * ex["R"],  # kappa_kv
            ]
        )
        targets.append(
            run["per_token"] * ex["N"] - v_shared * ex["S"] - VERIFY_SLOPE * ex["Q"]
        )
        labels.append(f"w{window}/skip{skip}")
        workloads.append(sum(run["lengths"]))
    design = np.array(rows)
    target = np.array(targets)
    # Unconstrained least squares returns a NEGATIVE kappa_kv and a negative
    # floor here -- reading more KV making the draft faster -- so the solution
    # is reported under a non-negativity constraint, and the conditioning that
    # produced the degenerate answer is reported beside it rather than hidden.
    scaled = design / np.abs(design).max(axis=0)
    condition = float(np.linalg.cond(scaled))
    correlation = np.corrcoef(scaled.T)
    from scipy.optimize import nnls

    solution, _ = nnls(design, target)
    free, *_ = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ solution
    residual = predicted / target - 1.0
    return {
        "record_type": "w98_refined_cost_fit",
        "cell": "LO",
        "batch": BATCH,
        "arms": labels,
        "verify_shared_s": v_shared,
        "verify_per_request_s_per_token": VERIFY_SLOPE,
        "F": solution[0],
        "A_per_layer_shared": solution[1],
        "f_win": solution[2],
        "kappa_kv": solution[3],
        "unconstrained_solution_is_unphysical": bool((free < 0).any()),
        "unconstrained": {
            name: float(value)
            for name, value in zip(("F", "A", "f_win", "kappa_kv"), free)
        },
        "design_condition_number": condition,
        "f_win_kappa_kv_correlation": float(correlation[2][3]),
        "realized_workload_spread": max(workloads) / min(workloads),
        "relative_residual": {k: round(float(v), 4) for k, v in zip(labels, residual)},
        "mean_abs_residual": float(abs(residual).mean()),
    }


def main() -> int:
    result = fit()
    out = DATA / "g98_fit" / "refined_cost_fit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
