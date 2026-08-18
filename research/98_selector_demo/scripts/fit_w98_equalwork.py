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


def main() -> int:
    directory = DATA / "g98_equalwork"
    result = fit(directory)
    out = DATA / "g98_fit" / "equalwork_cost_fit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
