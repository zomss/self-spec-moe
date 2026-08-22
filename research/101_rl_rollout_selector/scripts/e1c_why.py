# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E1c -- why intra-rollout switching is worth so little: the decomposition.

Four analyses over the same instantiated model as E1, no boots:
  1. regret bands: where along the trajectory the potential gain lives,
     as (local margin) x (time weight);
  2. the arm-independent floor: what share of a step no lever can touch;
  3. frontier flatness: cost and acceptance spreads vs their RATIO;
  4. counterfactuals: short outputs, long prompts, and a mixed-content
     iteration with per-group arm assignment.
"""

# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "98_selector_demo" / "scripts"))

import e1_trajectory as e1
import numpy as np
from predict_w98_refined_lo import fit_cost, verify_split
from scipy.interpolate import PchipInterpolator

DATA98 = e1.DATA98


def curves_for(cell_dir: str) -> dict[str, PchipInterpolator]:
    saved = e1.DATA98 / "g98_latacc_lo_q4"
    target = e1.DATA98 / cell_dir
    if target == saved:
        return e1.load_taus()
    # reuse the loader against another cell's curves
    import types

    module = types.SimpleNamespace(**vars(e1))
    original = e1.DATA98
    try:
        e1.DATA98 = e1.DATA98  # loader reads the global; patch path directly
        curves = {}
        for path in sorted(target.glob("*.json")):
            if path.name == "summary.json":
                continue
            rec = json.loads(path.read_text())
            arm = rec["cell"].split("/", 1)[1]
            if arm.endswith("skip16"):
                continue
            xs, ys = [], []
            for idx, centre in enumerate(e1.BUCKET_CENTRES):
                b = rec["tau_profile"]["buckets"].get(str(idx))
                if not b or not b.get("armed_steps"):
                    continue
                xs.append(centre)
                ys.append(1.0 + sum(b["pos_accepted"][: e1.K]) / b["armed_steps"])
            if len(xs) >= 2:
                curves[arm] = PchipInterpolator(xs, ys, extrapolate=False)
        return curves
    finally:
        e1.DATA98 = original
        del module


def main() -> int:
    fit = fit_cost(
        [DATA98 / "g98_equalwork_li_q4", DATA98 / "g98_equalwork_li_q4_b16"],
        batched=True,
    )
    lo_curves = e1.load_taus()
    off = json.loads((DATA98 / "g98_lat_lo_b32" / "off.json").read_text())
    vs, vp = verify_split(off, e1.VERIFY_MS)
    lengths = e1.lengths_from(DATA98 / "g98_lat_lo_b32" / "stock.json")
    arms = sorted(lo_curves)

    statics, _, _ = e1.simulate(fit, lo_curves, lengths, vs, vp, 1e9)
    best_static = min(statics, key=lambda a: statics[a])

    gmax = max(lengths)
    grid = np.arange(0.0, gmax, e1.DU)
    batch_at = [float(sum(1 for g in lengths if g > u)) for u in grid]

    def per_token(arm: str, u: float, b: float) -> float:
        cfg = e1.arm_cfg(arm)
        return e1.step_cost(fit, cfg, u, b, vs, vp) / e1.tau_at(lo_curves[arm], u)

    # --- 1. regret bands -------------------------------------------------
    bands = [(0, 1024), (1024, 4096), (4096, 13056), (13056, int(gmax))]
    print(f"1) WHERE THE GAIN LIVES  (static = {best_static})")
    print(
        f"{'band':>15s} {'time share':>11s} {'mean local margin':>18s} "
        f"{'contribution':>13s}"
    )
    t_total = statics[best_static]
    for lo_u, hi_u in bands:
        t_band = m_band = 0.0
        for u, b in zip(grid, batch_at):
            if not (lo_u <= u < hi_u) or b == 0:
                continue
            ts = (
                e1.DU
                / e1.tau_at(lo_curves[best_static], u)
                * e1.step_cost(fit, e1.arm_cfg(best_static), u, b, vs, vp)
            )
            t_local = min(per_token(a, u, b) for a in arms)
            margin = per_token(best_static, u, b) / t_local - 1.0
            t_band += ts
            m_band += ts * margin
        share = t_band / t_total
        mean_m = (m_band / t_band) if t_band else 0.0
        print(
            f"{f'{lo_u}-{hi_u}':>15s} {share * 100:10.1f}% {mean_m * 100:17.2f}% "
            f"{m_band / t_total * 100:12.3f}%"
        )

    # --- 2. the arm-independent floor -----------------------------------
    print("\n2) THE FLOOR NO LEVER TOUCHES  (verify + F as share of the step)")
    for u_probe in (512.0, 4096.0, 16384.0):
        b = float(sum(1 for g in lengths if g > u_probe))
        if b == 0:
            continue
        step = e1.step_cost(fit, e1.arm_cfg(best_static), u_probe, b, vs, vp)
        floor = (vs + b * vp) + fit["F"]
        print(
            f"   u={int(u_probe):6d} B={int(b):3d}: floor {floor * 1000:7.2f} ms "
            f"of {step * 1000:7.2f} ms = {floor / step * 100:5.1f}%"
        )

    # --- 3. frontier flatness -------------------------------------------
    print("\n3) FRONTIER FLATNESS at u=4096 (top-6 by throughput)")
    b = float(sum(1 for g in lengths if g > 4096))
    rows = []
    for a in arms:
        t = e1.step_cost(fit, e1.arm_cfg(a), 4096.0, b, vs, vp) * 1000
        tau = e1.tau_at(lo_curves[a], 4096.0)
        rows.append((t / tau, a, t, tau))
    rows.sort()
    t_lo = min(r[2] for r in rows[:6])
    t_hi = max(r[2] for r in rows[:6])
    tau_lo = min(r[3] for r in rows[:6])
    tau_hi = max(r[3] for r in rows[:6])
    for tt, a, t, tau in rows[:6]:
        print(f"   {a:14s} step {t:7.2f} ms  tau {tau:5.3f}  ms/token {tt:6.3f}")
    print(
        f"   spreads: step {(t_hi / t_lo - 1) * 100:5.1f}%  tau "
        f"{(tau_hi / tau_lo - 1) * 100:5.1f}%  ms/token "
        f"{(rows[5][0] / rows[0][0] - 1) * 100:5.1f}%"
    )

    # --- 4. counterfactuals ----------------------------------------------
    print("\n4) COUNTERFACTUALS (switch cost 50 ms)")
    for label, ls in (
        ("outputs x1/8 (median 1.6K)", [g / 8 for g in lengths]),
        ("outputs x1/4 (median 3.3K)", [g / 4 for g in lengths]),
        ("outputs x1/2 (median 6.6K)", [g / 2 for g in lengths]),
        ("base (median 13.1K)", lengths),
    ):
        st, ts, seg = e1.simulate(fit, lo_curves, ls, vs, vp, 0.05)
        bb = min(st, key=lambda a: st[a])
        print(
            f"   {label:28s} gain {(st[bb] / ts - 1) * 100:+6.2f}%  "
            f"({len(seg) - 1} switches, static {bb})"
        )

    print("\n   long prompts (p=12,160; LO curves as proxy):")
    saved_prompt = e1.PROMPT
    e1.PROMPT = 12160.0
    st, ts, seg = e1.simulate(fit, lo_curves, lengths, vs, vp, 0.05)
    bb = min(st, key=lambda a: st[a])
    print(
        f"   {'p=12K, outputs 13K med':28s} gain {(st[bb] / ts - 1) * 100:+6.2f}%  "
        f"({len(seg) - 1} switches, static {bb})"
    )
    e1.PROMPT = saved_prompt

    # mixed-content iteration: half LI-shaped, half LO-shaped sub-batches
    li_curves = curves_for("g98_latacc_li_q4")
    li_lengths = e1.lengths_from(DATA98 / "g98_lat_li_b8" / "stock.json") * 4
    common = sorted(set(lo_curves) & set(li_curves))

    def total_for(curves, ls, prompt, arm):
        saved = e1.PROMPT
        e1.PROMPT = prompt
        st, _, _ = e1.simulate(fit, {arm: curves[arm]}, ls, vs, vp, 1e9)
        e1.PROMPT = saved
        return st[arm]

    print("\n   mixed-content iteration (LO group + LI group, sub-batch split):")
    best_single, best_single_t = None, None
    for a in common:
        t = total_for(lo_curves, lengths, 156.0, a) + total_for(
            li_curves, li_lengths, 14611.0, a
        )
        if best_single_t is None or t < best_single_t:
            best_single, best_single_t = a, t
    lo_best = min(common, key=lambda a: total_for(lo_curves, lengths, 156.0, a))
    li_best = min(common, key=lambda a: total_for(li_curves, li_lengths, 14611.0, a))
    t_group = total_for(lo_curves, lengths, 156.0, lo_best) + total_for(
        li_curves, li_lengths, 14611.0, li_best
    )
    print(f"   single static for both groups: {best_single} ({best_single_t:.1f}s)")
    print(f"   per-group: LO->{lo_best}, LI->{li_best} ({t_group:.1f}s)")
    print(f"   PER-GROUP GAIN {(best_single_t / t_group - 1) * 100:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
