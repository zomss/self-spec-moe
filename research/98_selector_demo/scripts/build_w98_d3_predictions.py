# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build D3's prediction map — what the two-round selector expects, in advance.

D3 asks whether the selector's per-regime choice reaches 90% of the
omniscient composite. That is only a question if the selector can be WRONG,
so its pick must come from what Rounds 1-2 predicted, never from D3's own
grid. This builds that prediction and is committed as a barrier BEFORE the
grid is scored.

Predicted decode rate for a cell at a regime:

    rate_hat = tau_hat / (verify + D_hat)          armed
    rate_hat = 1 / verify                          OFF (no draft, one token)

with

* **D_hat** — the draft chain cost from Round 2's FITTED model
  (`d1p_fits.json`), evaluated on the same design row the campaign used. This
  is the half the phase showed is identifiable offline.
* **tau_hat** — acceptance at the DEPLOYED depth K=4, read from G98-D's
  position-resolved KMAX=8 stream via `1 + sum(pos_accepted[:4])/armed`.
  The preregistration registered K in {2,4} as readable "from one KMAX = 8
  stream" precisely so this needs no new measurement.
* **verify** — the target verify cost per regime, measured in Round 2. No
  lever touches the target, so it is a per-regime constant.

Two properties make this an honest test rather than a formality:

1. **The acceptance map was measured on content seeds 4-5; D3 runs on seeds
   2-3.** The selector therefore predicts unseen content, which is the
   generalization the claim needs.
2. **A cell with no acceptance measurement gets no prediction**, rather than
   an optimistic default. The selector may only choose what it actually knows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98c_round2 as g98c  # noqa: E402
from w98r2_cost_model import R2CostModel, R2LeverPoint  # noqa: E402

FITS = "research/98_selector_demo/data/g98_c/d1p_fits.json"
G98C_DIR = "research/98_selector_demo/data/g98_c"
G98D_DIR = "research/98_selector_demo/data/g98_d"
OUT = "research/98_selector_demo/data/g98_e/d3_predictions.json"
DEPLOY_K = 4
OFF_KEY = "off"


def _load(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _cells(directory: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted(Path(directory).rglob("*.json")):
        if path.name.endswith(".telemetry.json"):
            continue
        record = _load(path)
        if isinstance(record, dict) and "config" in record and "observations" in record:
            out.append(record)
    return out


def verify_and_context(repo: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Per-regime target verify cost and context length, from Round 2."""
    verify: dict[str, list[float]] = {}
    context: dict[str, list[float]] = {}
    for record in _cells(repo / G98C_DIR):
        for regime, obs in record["observations"].items():
            if obs.get("verify_s"):
                verify.setdefault(regime, []).append(float(obs["verify_s"]))
            if obs.get("context_tokens"):
                context.setdefault(regime, []).append(float(obs["context_tokens"]))
    return (
        {r: sum(v) / len(v) for r, v in verify.items()},
        {r: sum(v) / len(v) for r, v in context.items()},
    )


def tau_at_deploy_depth(repo: Path) -> dict[str, dict[str, float]]:
    """cell -> regime -> tau at K=DEPLOY_K, from the KMAX=8 stream."""
    acc: dict[str, dict[str, list[float]]] = {}
    for record in _cells(repo / G98D_DIR):
        cfg = record["config"]
        cell = g98c._config_key(cfg)
        for regime, obs in record["observations"].items():
            profile = obs.get("tau_profile") or {}
            armed = 0
            positions = [0] * DEPLOY_K
            for entry in (profile.get("buckets") or {}).values():
                armed += int(entry.get("armed_steps", 0))
                pos = entry.get("pos_accepted") or []
                for index in range(min(DEPLOY_K, len(pos))):
                    positions[index] += int(pos[index])
            if armed:
                acc.setdefault(cell, {}).setdefault(regime, []).append(
                    1.0 + sum(positions) / armed
                )
    return {c: {r: sum(v) / len(v) for r, v in by.items()} for c, by in acc.items()}


def build(repo: Path) -> dict[str, Any]:
    fits = _load(repo / FITS)["fits"]
    verify, context = verify_and_context(repo)
    tau = tau_at_deploy_depth(repo)

    predicted: dict[str, dict[str, float]] = {}
    detail: dict[str, dict[str, Any]] = {}

    # OFF: no draft, one committed token per target step.
    predicted[OFF_KEY] = {r: 1.0 / v for r, v in verify.items() if v > 0}
    detail[OFF_KEY] = {"basis": "1 / verify (no draft chain)"}

    skipped: list[str] = []
    for cell, by_regime in sorted(tau.items()):
        cfg = _config_from_key(cell)
        for regime, tau_hat in sorted(by_regime.items()):
            fit = fits.get(regime)
            if not fit or not fit.get("fitted") or regime not in verify:
                skipped.append(f"{cell}/{regime}")
                continue
            model = R2CostModel(
                kappa_w=fit["kappa_w"],
                kappa_kv=fit["kappa_kv"],
                c_layer=fit["c_layer"],
                f_win=fit["f_win"],
                f_fixed=fit["F"],
                log_envelope=fit.get("envelope", 0.0),
            )
            point = g98c.lever_point(cfg, context.get(regime, 0.0), 1.0)
            d_hat = model.predict(
                R2LeverPoint(
                    weight_layer_bytes=point.weight_layer_bytes,
                    kv_bytes=point.kv_bytes,
                    keep_frac=point.keep_frac,
                    windowed=point.windowed,
                    d_measured=1.0,
                )
            )
            denominator = verify[regime] + d_hat
            if denominator <= 0:
                skipped.append(f"{cell}/{regime}")
                continue
            predicted.setdefault(cell, {})[regime] = tau_hat / denominator
            detail.setdefault(cell, {})[regime] = {
                "tau_hat_at_k4": round(tau_hat, 6),
                "d_hat_s": round(d_hat, 6),
                "verify_s": round(verify[regime], 6),
            }

    record = {
        "schema_version": 1,
        "record_type": "w98d3_predictions",
        "deploy_k": DEPLOY_K,
        "basis": {
            "cost": "Round-2 fitted model (d1p_fits.json)",
            "acceptance": "G98-D KMAX=8 stream read at K=4",
            "verify": "Round-2 measured per-regime target verify cost",
            "acceptance_seeds": [4, 5],
            "d3_evaluation_seeds": [2, 3],
            "generalization_note": (
                "acceptance was measured on content the D3 grid does not use, "
                "so the selector predicts unseen content"
            ),
        },
        "cells_predicted": len(predicted),
        "unpredicted": sorted(set(skipped)),
        "predicted_decode_tokens_per_s": {
            c: {r: round(v, 6) for r, v in by.items()} for c, by in predicted.items()
        },
        "detail": detail,
    }
    record["digest"] = hashlib.sha256(
        json.dumps(
            record["predicted_decode_tokens_per_s"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return record


def _config_from_key(cell: str) -> dict[str, Any]:
    quant, window, skip = cell.split("/")
    return {
        "quant": quant,
        "window": "off" if window == "woff" else int(window[1:]),
        "skip_count": int(skip.replace("skip", "")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    record = build(args.repo)
    out = args.out or (args.repo / OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"{out} exists; the prediction barrier is immutable")
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "cells_predicted": record["cells_predicted"],
                "unpredicted": len(record["unpredicted"]),
                "digest": record["digest"][:16],
                "out": str(out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
