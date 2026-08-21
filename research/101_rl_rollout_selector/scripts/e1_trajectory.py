# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E1 -- the trajectory simulation: omniscient schedule vs best static.

Prices intra-rollout switching from existing phase-98 data, no boots. The
model is `design_round1_continuous.md`: total decode wall for a schedule
sigma over a draining rollout is

    T(sigma) = SUM_u  [ t_step(sigma(u), u, B(u)) / tau_sigma(u) ] du

with `B(u)` the empirical survival of a measured natural-EOS length
distribution, cost coefficients from the two-batch long-context fit, the
verify split solved on the parked arm, and tau from the measured depth-8
prefix profiles (PCHIP across bucket centres, carried flat past the last
bucket). Switch cost enters as a per-switch stall of the whole batch and the
optimal schedule under it comes from dynamic programming over a u-grid.

Everything here is model-space: the gain reported is T(best static) /
T(sigma*) under ONE model, so model level error largely cancels; E1a checks
the model's ranking against the measured b32 grid first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

HERE = Path(__file__).resolve()
P98 = HERE.parents[2] / "98_selector_demo"
sys.path.insert(0, str(P98 / "scripts"))

from predict_w98_refined_lo import fit_cost, verify_split  # noqa: E402
from w98_cost_u import kv_positions, window_binds  # noqa: E402

DATA98 = P98 / "data"
OUT = HERE.parents[1] / "data"
KV_BYTES = 2 * 2 * 8 * 128 * 36
K = 4
U_EDGES = (256, 1024, 3072, 8192)
BUCKET_CENTRES = (128.0, 640.0, 2048.0, 5632.0)
PROMPT = 156.0
VERIFY_MS = 8.135  # parked verify at the LO operating point (s17 pipeline)
DU = 64.0

WEIGHT_BYTES = {"target-matching": 13.892e9, "w4a16-quantized": 3.581e9}


def load_taus() -> dict[str, PchipInterpolator]:
    """tau(u) per arm at deployed K, monotone-interpolated over bucket centres."""
    curves = {}
    for path in sorted((DATA98 / "g98_latacc_lo_q4").glob("*.json")):
        if path.name == "summary.json":
            continue
        rec = json.loads(path.read_text())
        arm = rec["cell"].split("/", 1)[1]
        if arm.endswith("skip16"):
            continue  # never top-5 anywhere (s45); excluded from the lattice
        xs, ys = [], []
        for idx, centre in enumerate(BUCKET_CENTRES):
            b = rec["tau_profile"]["buckets"].get(str(idx))
            if not b or not b.get("armed_steps"):
                continue
            xs.append(centre)
            ys.append(1.0 + sum(b["pos_accepted"][:K]) / b["armed_steps"])
        if len(xs) >= 2:
            curves[arm] = PchipInterpolator(xs, ys, extrapolate=False)
    return curves


def tau_at(curve: PchipInterpolator, u: float) -> float:
    lo, hi = curve.x[0], curve.x[-1]
    return float(curve(min(max(u, lo), hi)))


def lengths_from(run: Path) -> list[float]:
    rec = json.loads(run.read_text())
    return sorted(float(v) for v in rec["out_tokens_by_request"].values())


def arm_cfg(arm: str) -> dict:
    w, s = arm.split("/")
    return {
        "keep_frac": 1.0 - int(s[4:]) / 36.0,
        "window": "off" if w == "woff" else int(w[1:]),
        "weight_bytes": WEIGHT_BYTES["w4a16-quantized"],
    }


def step_cost(fit, cfg, u: float, batch: float, vs: float, vp: float) -> float:
    """Wall seconds for one engine step at position u with `batch` active."""
    context = PROMPT + u
    keep = cfg["keep_frac"]
    windowed = window_binds(cfg["window"], context)
    draft_shared = (
        keep
        * (
            cfg["weight_bytes"] * fit["kappa_w"]
            + fit["c_layer"]
            + fit["f_win"] * windowed
        )
        + fit["F"]
    )
    kv = kv_positions(cfg["window"], context) * KV_BYTES
    draft_req = fit["kappa_kv"] * kv  # per-request per step (batched fit)
    return (vs + draft_shared) + batch * (vp + draft_req)


def simulate(fit, curves, lengths, vs, vp, switch_cost_s: float):
    """DP over the u-grid; returns (T_by_static, T_schedule, schedule)."""
    arms = sorted(curves)
    gmax = max(lengths)
    grid = np.arange(0.0, gmax, DU)
    batch_at = [float(sum(1 for g in lengths if g > u)) for u in grid]

    # per-arm integrand on the grid: seconds to advance DU tokens/request
    seg = {}
    for a in arms:
        cfg = arm_cfg(a)
        seg[a] = [
            DU / tau_at(curves[a], u) * step_cost(fit, cfg, u, b, vs, vp)
            if b > 0
            else 0.0
            for u, b in zip(grid, batch_at)
        ]
    statics = {a: sum(seg[a]) for a in arms}

    # DP with per-switch stall
    prev = {a: seg[a][0] for a in arms}
    back: list[dict[str, str]] = [{a: a for a in arms}]
    for i in range(1, len(grid)):
        best_prev = min(prev, key=lambda a: prev[a])
        nxt, links = {}, {}
        for a in arms:
            stay = prev[a]
            move = prev[best_prev] + switch_cost_s
            if stay <= move:
                nxt[a] = stay + seg[a][i]
                links[a] = a
            else:
                nxt[a] = move + seg[a][i]
                links[a] = best_prev
        prev, back = nxt, back + [links]
    end = min(prev, key=lambda a: prev[a])
    path = [end]
    for links in reversed(back[1:]):
        path.append(links[path[-1]])
    path.reverse()

    segments = []
    for u, a, b in zip(grid, path, batch_at):
        if not segments or segments[-1][2] != a:
            segments.append([u, u + DU, a, b])
        else:
            segments[-1][1] = u + DU
    return statics, prev[end], [(s[0], s[1], s[2], s[3]) for s in segments]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fit = fit_cost(
        [DATA98 / "g98_equalwork_li_q4", DATA98 / "g98_equalwork_li_q4_b16"],
        batched=True,
    )
    curves = load_taus()
    off = json.loads((DATA98 / "g98_lat_lo_b32" / "off.json").read_text())
    vs, vp = verify_split(off, VERIFY_MS)
    lengths = lengths_from(DATA98 / "g98_lat_lo_b32" / "stock.json")

    result = {
        "record_type": "e1_trajectory",
        "arms": sorted(curves),
        "n_requests": len(lengths),
        "lengths": {
            "min": min(lengths),
            "median": lengths[len(lengths) // 2],
            "max": max(lengths),
        },
        "sweeps": {},
    }
    for sw_ms in (0.0, 50.0, 500.0, 5000.0):
        statics, t_sched, segments = simulate(
            fit, curves, lengths, vs, vp, sw_ms / 1000.0
        )
        best_static = min(statics, key=lambda a: statics[a])
        result["sweeps"][f"{sw_ms:g}ms"] = {
            "best_static": best_static,
            "t_best_static_s": round(statics[best_static], 3),
            "t_schedule_s": round(t_sched, 3),
            "gain_pct": round((statics[best_static] / t_sched - 1.0) * 100, 2),
            "n_switches": len(segments) - 1,
            "schedule": [
                {"u_from": int(s0), "u_to": int(s1), "arm": a, "batch_at_entry": int(b)}
                for s0, s1, a, b in segments
            ],
        }
    (OUT / "e1_trajectory.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
