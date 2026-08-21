# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E1 variants: integrator validation and trajectory-scale sensitivity."""

# ruff: noqa: E402  -- sys.path must precede the phase-98 imports

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import sys

sys.path.insert(0, "research/101_rl_rollout_selector/scripts")
sys.path.insert(0, "research/98_selector_demo/scripts")
import e1_trajectory as e1
from predict_w98_refined_lo import fit_cost, verify_split

fit = fit_cost(
    [e1.DATA98 / "g98_equalwork_li_q4", e1.DATA98 / "g98_equalwork_li_q4_b16"],
    batched=True,
)
curves = e1.load_taus()
off = json.loads((e1.DATA98 / "g98_lat_lo_b32" / "off.json").read_text())
vs, vp = verify_split(off, e1.VERIFY_MS)
base = e1.lengths_from(e1.DATA98 / "g98_lat_lo_b32" / "stock.json")

print("E1a: fixed-arm ranking of THIS integrator vs measured LO b32 grid")
statics, _, _ = e1.simulate(fit, curves, base, vs, vp, 1e9)
model_rank = sorted(statics, key=lambda a: statics[a])
meas = {}
for p in (e1.DATA98 / "g98_lat_lo_b32").glob("*.json"):
    r = json.loads(p.read_text())
    if r.get("record_type") != "w98_refined_lo" or "/" not in r.get("cell", ""):
        continue
    if not r["cell"].startswith("w4a16"):
        continue
    arm = r["cell"].split("/", 1)[1]
    if arm in curves:
        meas[arm] = r["wall_s"] / r["total_out_tokens"]
meas_rank = sorted(meas, key=lambda a: meas[a])
from scipy.stats import spearmanr

rho, _ = spearmanr(
    [model_rank.index(a) for a in sorted(curves)],
    [meas_rank.index(a) for a in sorted(curves)],
)
print(
    f"  spearman {rho:.3f}; model top3 {model_rank[:3]}; measured top3 {meas_rank[:3]}"
)
print(f"  measured best in model top-3: {meas_rank[0] in model_rank[:3]}")

print("\nE1b sensitivity: gain of omniscient schedule vs best static (switch 50ms)")
print(f"{'scenario':28s} {'best static':14s} {'gain':>7s} {'switches':>9s}")
for label, lengths in (
    ("base: B0=32, G~13K", base),
    ("longer: G x2 (~26K med)", [g * 2 for g in base]),
    ("longer: G x4 (~52K med)", [g * 4 for g in base]),
    ("bigger: B0=128", base * 4),
    ("bigger+longer: B0=128,Gx2", [g * 2 for g in base] * 4),
):
    st, ts, seg = e1.simulate(fit, curves, lengths, vs, vp, 0.05)
    b = min(st, key=lambda a: st[a])
    print(f"{label:28s} {b:14s} {(st[b] / ts - 1) * 100:+6.2f}% {len(seg) - 1:9d}")
