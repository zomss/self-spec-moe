# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Predict the refined-LO arms, per weight-version family.

Assembles the pieces the phase measured separately -- equal-work cost
coefficients, u-resolved acceptance, realized generation lengths -- into the
drain-integrated prediction of section 7, and scores it against the run.

Three things this makes explicit that the ad-hoc version left implicit:

* **Acceptance is re-derived at the deployed depth.** The u-curves are
  measured at depth 8 and the scored runs speculate at K=4, so a depth-8
  ``tau`` of 7.9 is not a number of tokens any scored step could commit. It
  is recomputed from ``pos_accepted``, which the profile carries per bucket.
* **Cost coefficients are fitted per family and constrained non-negative.**
  Section 14 refuted the additive model ACROSS weight versions while it holds
  within one, so each family gets its own fit; NNLS keeps the quantized
  block's window inversion in the residual, where a model that cannot express
  it belongs, rather than in a negative coefficient.
* **The comparison is against the RAW per-token ratio**, not the
  context-corrected score. Campaign 1's correction exists to remove context
  growth from a model-free comparison; this model integrates context growth
  explicitly, so correcting the measurement too would charge it twice.

``--acceptance-from`` swaps in another family's curves, which is the ablation
that isolates how much of the prediction depends on measuring acceptance on
the family being deployed.
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
from fit_w98_equalwork import KV_BYTES_PER_TOKEN, mean_kv_positions  # noqa: E402
from w98_cost_u import integrated_with_drain  # noqa: E402

U_EDGES = (256, 1024, 3072, 8192)
K_DEPLOYED = 4
DRAFT_LAYERS = 36
W_LAYER_BYTES = {"target-matching": 13.892e9, "w4a16-quantized": 3.581e9}
OFF_KEY = "off"


def fit_cost(directories: list[Path] | Path, batched: bool = False) -> dict[str, float]:
    """Per-family draft-cost coefficients, constrained physical.

    With `batched`, the KV column carries per-STEP bytes (`keep * kv * B`)
    rather than per-sequence bytes, so `kappa_kv` comes out as a true
    per-request coefficient. That distinction is invisible in a design where
    every arm shares one batch -- which every cost arm in this phase did --
    and it is exactly the defect section 18 found in the verify split: a
    shared/per-request division that was assumed at the fitting batch rather
    than measured. Measured directly, the draft chain's batch slope is twice
    what dividing a batch-8 fit by 8 implies.
    """
    import numpy as np
    from scipy.optimize import nnls

    if isinstance(directories, Path):
        directories = [directories]
    rows, targets = [], []
    for directory in directories:
        for path in sorted(Path(directory).glob("*.json")):
            if path.name == "summary.json":
                continue
            record = artifacts.read(path)
            cfg = record["config"]
            keep = float(record["keep_frac"])
            windowed = 0.0 if cfg["window"] in ("off", 0, None) else 1.0
            kv = mean_kv_positions(
                cfg["window"],
                float(record["prompt_tokens"]),
                float(record["gen_tokens"]),
            )
            scale = float(record["batch"]) if batched else 1.0
            rows.append(
                [keep, keep * windowed, keep * kv * KV_BYTES_PER_TOKEN * scale, 1.0]
            )
            targets.append(float(record["draft_chain_ms"]) / 1000.0)
    design, target = np.array(rows), np.array(targets)
    beta, _ = nnls(design, target)
    residual = abs((design @ beta) / target - 1.0)
    return {
        # The weight column is folded into `c_layer`: within one family every
        # arm carries the same bytes, so only their sum is identifiable, and
        # section 14 is where the two are separated.
        "kappa_w": 0.0,
        "c_layer": float(beta[0]),
        "f_win": float(beta[1]),
        "kappa_kv": float(beta[2]),
        "F": float(beta[3]),
        "fit_mean_residual": float(residual.mean()),
        "fit_arms": len(targets),
        "kappa_kv_is_per_request": batched,
    }


def tau_at_depth(profile: dict[str, Any], depth: int) -> dict[str, float]:
    """Committed tokens per armed step at `depth`, from the per-position counts.

    A depth-8 curve credits acceptances at positions the deployed K never
    drafts. Truncating the position counts is exact, not an approximation:
    position i is only reached when every earlier position was accepted.
    """
    out = {}
    for index, bucket in profile["buckets"].items():
        steps = bucket["armed_steps"]
        if not steps:
            continue
        accepted = sum(bucket["pos_accepted"][:depth])
        out[index] = 1.0 + accepted / steps
    return out


def pool_below_1k(profile: dict[str, Any], depth: int) -> dict[str, float]:
    """The scalar the selector actually consumes: buckets 0 and 1 pooled.

    G98-D generated 640 tokens, so its acceptance measurement lives entirely
    in the first two buckets and enters the selector as one number. Pooling by
    armed steps reproduces what that campaign would have reported, and
    returning it as a single-bucket curve makes the integrator carry it flat
    across the whole generation -- which is the assumption under test.
    """
    steps = accepted = 0
    for index in ("0", "1"):
        bucket = profile["buckets"].get(index)
        if not bucket or not bucket["armed_steps"]:
            continue
        steps += bucket["armed_steps"]
        accepted += sum(bucket["pos_accepted"][:depth])
    return {"0": 1.0 + accepted / steps} if steps else {}


def load_curves(
    directories: list[Path], depth: int, scalar: bool = False
) -> dict[str, dict[str, float]]:
    curves = {}
    for directory in directories:
        for path in sorted(Path(directory).glob("*.json")):
            record = json.loads(Path(path).read_text(encoding="utf-8"))
            if record.get("record_type") != "w98_long_u":
                continue
            arm = record["cell"].split("/", 1)[1]
            profile = record["tau_profile"]
            curves[arm] = (
                pool_below_1k(profile, depth)
                if scalar
                else tau_at_depth(profile, depth)
            )
    return curves


def load_runs(directories: list[Path]) -> dict[str, dict[str, Any]]:
    runs = {}
    for directory in directories:
        for path in sorted(Path(directory).glob("*.json")):
            try:
                record = artifacts.read(path)
            except Exception:
                continue
            if record.get("record_type") != "w98_refined_lo":
                continue
            key = record["cell"]
            runs[key if key == OFF_KEY else key.split("/", 1)[1]] = record
    return runs


def verify_split(off: dict[str, Any], verify_ms: float) -> tuple[float, float]:
    """Split verify into batch-shared and per-request, calibrated on OFF.

    The per-request level is the profiler's measured verify divided by the
    nominal batch. The shared level is then whatever the parked arm's own
    per-token cost requires -- on this box that is large, because the clamp
    is a fixed per-step host delay and the parked step is the one it dominates.
    """
    lengths = list(off["out_tokens_by_request"].values())
    horizon = float(max(lengths))
    tokens = float(sum(lengths))
    per_request = (verify_ms / 1000.0) / off["batch"]
    measured = off["wall_s"] / off["total_out_tokens"]
    shared = (measured - per_request) * tokens / horizon
    return shared, per_request


def predict(
    fit: dict[str, float],
    runs: dict[str, dict[str, Any]],
    curves: dict[str, dict[str, float]],
    weight_bytes: float,
    verify_ms: float,
    prompt_tokens: float,
    gamma: float,
    split: tuple[float, float] | None = None,
) -> dict[str, Any]:
    off = runs[OFF_KEY]
    shared, per_request = split or verify_split(off, verify_ms)
    off_per_token = off["wall_s"] / off["total_out_tokens"]
    rows = {}
    for arm, record in sorted(runs.items()):
        if arm == OFF_KEY:
            continue
        if arm not in curves:
            continue
        window, skip = arm.split("/")
        cfg = {
            "keep_frac": 1.0 - int(skip.replace("skip", "")) / DRAFT_LAYERS,
            "window": "off" if window == "woff" else int(window[1:]),
            "weight_bytes": weight_bytes,
        }
        lengths = [float(v) for v in record["out_tokens_by_request"].values()]
        result = integrated_with_drain(
            fit,
            cfg,
            curves[arm],
            U_EDGES,
            prompt_tokens,
            lengths,
            shared,
            per_request,
            KV_BYTES_PER_TOKEN,
            fit_batch=1.0
            if fit.get("kappa_kv_is_per_request")
            else float(record["batch"]),
            gamma=gamma,
        )
        measured = off_per_token / (record["wall_s"] / record["total_out_tokens"])
        predicted = off_per_token / result["per_token_s"]
        rows[arm] = {
            "predicted": round(predicted, 4),
            "measured": round(measured, 4),
            "relative_error": round(predicted / measured - 1.0, 4),
        }
    errors = [abs(r["relative_error"]) for r in rows.values()]
    return {
        "verify_shared_ms": round(shared * 1000, 4),
        "verify_per_request_ms": round(per_request * 1000, 4),
        "arms": rows,
        "mean_abs_error": round(sum(errors) / len(errors), 4) if errors else None,
    }


def ranking(rows: dict[str, Any], key: str) -> list[str]:
    return [a for a, _ in sorted(rows.items(), key=lambda kv: -kv[1][key])]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equalwork", required=True, help="comma-separated")
    parser.add_argument(
        "--batched-fit",
        action="store_true",
        help="treat the KV column as per-step, which needs batch to vary "
        "across the equal-work directories given",
    )
    parser.add_argument("--runs", required=True, help="comma-separated directories")
    parser.add_argument("--curves", required=True, help="comma-separated directories")
    parser.add_argument(
        "--acceptance-from",
        help="ablation: take acceptance from these directories instead",
    )
    parser.add_argument("--quant", required=True, choices=sorted(W_LAYER_BYTES))
    parser.add_argument("--verify-ms", type=float, required=True)
    parser.add_argument("--prompt-tokens", type=float, default=156.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--scalar-tau",
        action="store_true",
        help="ablation: pool acceptance below 1K and carry it flat, as the "
        "selector's current input does",
    )
    parser.add_argument(
        "--split-ms",
        help="measured verify split as SHARED,PER_REQUEST in ms; without it "
        "the split is assumed from the profiler and solved on the parked arm",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    fit = fit_cost(
        [Path(p) for p in args.equalwork.split(",")], batched=args.batched_fit
    )
    runs = load_runs([Path(p) for p in args.runs.split(",")])
    source = args.acceptance_from or args.curves
    curves = load_curves(
        [Path(p) for p in source.split(",")], K_DEPLOYED, args.scalar_tau
    )
    result = predict(
        fit,
        runs,
        curves,
        W_LAYER_BYTES[args.quant],
        args.verify_ms,
        args.prompt_tokens,
        args.gamma,
        tuple(float(v) / 1000 for v in args.split_ms.split(","))
        if args.split_ms
        else None,
    )
    result.update(
        {
            "record_type": "w98_refined_lo_prediction",
            "quant": args.quant,
            "k_deployed": K_DEPLOYED,
            "gamma": args.gamma,
            "cost_fit": fit,
            "acceptance_from": source,
            "acceptance_scalar": args.scalar_tau,
            "predicted_ranking": ranking(result["arms"], "predicted"),
            "measured_ranking": ranking(result["arms"], "measured"),
        }
    )
    if args.output:
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"== {args.quant}  acceptance from {source}")
    print(
        f"   cost fit: {fit['fit_arms']} arms, residual {fit['fit_mean_residual']:.4f}"
    )
    print(f"{'arm':14s}{'predicted':>11}{'measured':>10}{'error':>9}")
    for arm, row in sorted(result["arms"].items(), key=lambda kv: -kv[1]["measured"]):
        print(
            f"{arm:14s}{row['predicted']:11.3f}{row['measured']:10.3f}"
            f"{100 * row['relative_error']:+8.1f}%"
        )
    print(f"{'mean abs error':14s}{result['mean_abs_error']:11.4f}")
    print(f"   predicted rank: {' > '.join(result['predicted_ranking'])}")
    print(f"   measured  rank: {' > '.join(result['measured_ranking'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
