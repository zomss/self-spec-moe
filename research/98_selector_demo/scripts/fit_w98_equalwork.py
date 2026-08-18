# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fit the draft-cost coefficients from the equal-work sweep.

Consumes `run_w98_equalwork_cost`, where every arm emitted the same tokens
over the same contexts and the draft chain was read from the profiler rather
than inverted out of throughput. The regression is then linear and clean::

    D_arm = keep * (A + f_win * [w>0] + kappa_kv * kv_bytes * KVmean(w))
            + F

`KVmean(w)` is the mean KV position count over the run, integrated the same
way the cost model integrates it -- unwindowed it tracks the growing context,
windowed it saturates at `window + sinks` once the context passes it.

The design is what the previous attempt lacked: the window varies at THREE
keeps, so `f_win` is identified by arms sharing a window while differing in
keep, and `kappa_kv` by arms sharing a keep while differing in window. The
reported condition number and the f_win/kappa_kv correlation say whether that
worked, rather than leaving it to be assumed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import w98_artifacts as artifacts  # noqa: E402
from w98_cost_u import kv_positions  # noqa: E402

DATA = SCRIPT_DIR.parent / "data"
KV_BYTES_PER_TOKEN = 2 * 2 * 8 * 128 * 36
SEGMENT = 64.0


def mean_kv_positions(window: Any, prompt: float, gen: float) -> float:
    """Mean KV positions over an equal-work run, by direct integration."""
    total = 0.0
    u = 0.0
    while u < gen:
        width = min(SEGMENT, gen - u)
        total += width * kv_positions(window, prompt + u + width / 2.0)
        u += width
    return total / gen


def fit(directory: Path) -> dict[str, Any]:
    import numpy as np
    from scipy.optimize import nnls

    rows, targets, labels, steps = [], [], [], []
    for path in sorted(directory.glob("*.json")):
        if path.name == "summary.json":
            continue
        record = artifacts.read(path)
        cfg = record["config"]
        keep = float(record["keep_frac"])
        window = cfg["window"]
        windowed = 0.0 if window in ("off", 0, None) else 1.0
        kv = mean_kv_positions(
            window, float(record["prompt_tokens"]), float(record["gen_tokens"])
        )
        rows.append([keep, keep * windowed, keep * kv * KV_BYTES_PER_TOKEN, 1.0])
        targets.append(record["draft_chain_ms"] / 1000.0)
        labels.append(record["cell"])
        steps.append(record["engine_steps"])
    design = np.array(rows)
    target = np.array(targets)
    scaled = design / np.abs(design).max(axis=0)
    correlation = np.corrcoef(scaled.T)
    free, *_ = np.linalg.lstsq(design, target, rcond=None)
    constrained, _ = nnls(design, target)
    predicted = design @ free
    residual = predicted / target - 1.0
    return {
        "record_type": "w98_equalwork_cost_fit",
        "arms": labels,
        "equal_work_steps": {k: v for k, v in zip(labels, steps)},
        "A_per_layer_shared_s": float(free[0]),
        "f_win_s": float(free[1]),
        "kappa_kv": float(free[2]),
        "F_s": float(free[3]),
        "unconstrained_is_physical": bool((free >= 0).all()),
        "nnls_agrees": bool(np.allclose(free, constrained, rtol=0.05)),
        "design_condition_number": float(np.linalg.cond(scaled)),
        "f_win_kappa_kv_correlation": float(correlation[1][2]),
        "relative_residual": {k: round(float(v), 4) for k, v in zip(labels, residual)},
        "mean_abs_residual": float(abs(residual).mean()),
    }


# --- the quant axis: separating kappa_w from c_layer -------------------

# Registered body-bytes constants (`w98r2_cost_model.W_LAYER_BYTES`), used
# rather than re-derived so a fitted kappa_w is in the units the Round-1
# model already consumes.
W_LAYER_BYTES = {"target-matching": 13.892e9, "w4a16-quantized": 3.581e9}


def fit_split(directories: list[Path]) -> dict[str, Any]:
    """Fit across both weight versions, so `A` can be decomposed.

    Every arm measured before the quant sweep shared the target's weights, so
    `W * kappa_w` was constant across the design and perfectly collinear with
    the per-layer constant: only their sum `A` was identifiable. A second
    weight version breaks that collinearity -- the same keep now buys two
    different byte counts -- which is the only way to say how much of the
    draft step is weight traffic.
    """
    import numpy as np
    from scipy.optimize import nnls

    rows, targets, labels, quants = [], [], [], []
    for directory in directories:
        for path in sorted(Path(directory).glob("*.json")):
            if path.name == "summary.json":
                continue
            record = artifacts.read(path)
            cfg = record["config"]
            keep = float(record["keep_frac"])
            window = cfg["window"]
            windowed = 0.0 if window in ("off", 0, None) else 1.0
            kv = mean_kv_positions(
                window, float(record["prompt_tokens"]), float(record["gen_tokens"])
            )
            rows.append(
                [
                    keep * W_LAYER_BYTES[cfg["quant"]],
                    keep,
                    keep * windowed,
                    keep * kv * KV_BYTES_PER_TOKEN,
                    1.0,
                ]
            )
            targets.append(float(record["draft_chain_ms"]) / 1000.0)
            labels.append(record["cell"])
            quants.append(cfg["quant"])
    design = np.array(rows)
    target = np.array(targets)
    scaled = design / np.abs(design).max(axis=0)
    free, *_ = np.linalg.lstsq(design, target, rcond=None)
    constrained, _ = nnls(design, target)
    residual = (design @ free) / target - 1.0
    names = ["kappa_w", "c_layer_s", "f_win_s", "kappa_kv", "F_s"]
    decomposed = {}
    for quant, weights in sorted(W_LAYER_BYTES.items()):
        body = float(free[0]) * weights
        decomposed[quant] = {
            "weight_traffic_s": body,
            "c_layer_s": float(free[1]),
            "A_total_s": body + float(free[1]),
            "weight_share_of_A": body / (body + float(free[1])),
        }
    return {
        "record_type": "w98_equalwork_quant_fit",
        "arms": labels,
        "n_arms": len(labels),
        "weight_versions": sorted(set(quants)),
        "coefficients": {n: float(v) for n, v in zip(names, free)},
        "A_decomposed_s": decomposed,
        "unconstrained_is_physical": bool((free >= 0).all()),
        "nnls_agrees": bool(np.allclose(free, constrained, rtol=0.05)),
        "design_condition_number": float(np.linalg.cond(scaled)),
        "relative_residual": {k: round(float(v), 4) for k, v in zip(labels, residual)},
        "mean_abs_residual": float(abs(residual).mean()),
        "mean_abs_residual_by_quant": {
            q: float(abs(residual[[i for i, x in enumerate(quants) if x == q]]).mean())
            for q in sorted(set(quants))
        },
    }


def matched_pairs(directories: list[Path]) -> dict[str, Any]:
    """What quantization is worth, from arms that differ only in it.

    The regression attributes the whole bf16-to-w4a16 difference to a byte
    column, which is sound only if the two weight versions differ in nothing
    else. This is the check that does not assume it: for every (window, keep)
    measured under BOTH weight versions, the raw difference and the implied
    per-byte coefficient. Were quantization a pure reduction in weight
    traffic those coefficients would agree; their spread is the size of what
    a byte column cannot carry.
    """
    import numpy as np

    table: dict[tuple[str, float], dict[str, float]] = {}
    for directory in directories:
        for path in sorted(Path(directory).glob("*.json")):
            if path.name == "summary.json":
                continue
            record = artifacts.read(path)
            cfg = record["config"]
            key = (str(cfg["window"]), round(float(record["keep_frac"]), 3))
            table.setdefault(key, {})[cfg["quant"]] = float(record["draft_chain_ms"])
    delta_bytes = W_LAYER_BYTES["target-matching"] - W_LAYER_BYTES["w4a16-quantized"]
    rows, coefficients = [], []
    for (window, keep), arms in sorted(table.items()):
        if len(arms) < 2:
            continue
        saved = arms["target-matching"] - arms["w4a16-quantized"]
        coefficients.append((saved / 1000.0) / (keep * delta_bytes))
        rows.append(
            {
                "window": window,
                "keep": keep,
                "bf16_ms": round(arms["target-matching"], 3),
                "w4a16_ms": round(arms["w4a16-quantized"], 3),
                "quant_saves_ms": round(saved, 3),
                "implied_kappa_w": coefficients[-1],
            }
        )
    values = np.array(coefficients)
    spread = float(values.max() / values.min() - 1.0)
    return {
        "record_type": "w98_quant_matched_pairs",
        "pairs": rows,
        "kappa_w_mean": float(values.mean()),
        "kappa_w_spread_pct": spread * 100.0,
        "additive_in_bytes": bool(spread < 0.05),
    }


def main() -> int:
    out_dir = DATA / "g98_fit"
    out_dir.mkdir(parents=True, exist_ok=True)
    if "--with-quant" in sys.argv:
        directories = [DATA / "g98_equalwork", DATA / "g98_equalwork_q4"]
        result: dict[str, Any] = {
            "fit": fit_split(directories),
            "matched_pairs": matched_pairs(directories),
        }
        (out_dir / "equalwork_quant_fit.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
    else:
        result = fit(DATA / "g98_equalwork")
        (out_dir / "equalwork_cost_fit.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
