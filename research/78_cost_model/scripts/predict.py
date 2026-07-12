#!/usr/bin/env python3
"""V2 forward predictions — REGISTERED BEFORE MEASUREMENT.

Predicts from data/fit.json (additive form):
  1. R(win128) at every grid cell, dense + moe -- 77 measured win128's beta,
     76 never measured its COST. The model says: KV capped at 144 tokens.
  2. TPOT at the never-swept interpolation cell b16 x 8k for bf16 and win512.

Writes data/v2_predictions.json; e1 then measures exactly these cells and
verify_v2.py compares (gate: <=10% median).
"""

import json
import sys
from pathlib import Path

PHASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PHASE / "scripts"))
import fit_cost_model as M  # noqa: E402

# inject the win128 arm (KV capped at 128+16) and the 8k ctx point
M.ARMS[("dense", "d_win128")] = dict(win=128 + 16)
M.ARMS[("moe", "m_win128")] = dict(win=128 + 16, comm=True)
for g in ("dense", "moe"):
    M.CONST[g]["ctx_of"][8] = 8192


def params_vec(g, blob):
    p = blob[f"{g}/additive"]["params"]
    names = (["f0", "f1", "BW_GBs", "h_ms_per_Mtok"]
             + (["c0", "c1"] if g == "moe" else [])
             + [f"kappa_{k}" for k in M.KAPPAS[g]])
    return [p[n] for n in names]


def main() -> int:
    blob = json.loads((PHASE / "data/fit.json").read_text())
    out = {"win128_R": {}, "interp_b16_8k": {}}

    for g, warm in (("dense", "d_win128"), ("moe", "m_win128")):
        pv = params_vec(g, blob)
        cells = [(b, c) for b in ((1, 8, 32) if g == "dense" else (4, 8, 32))
                 for c in (2, 16, 32)]
        for (b, c) in cells:
            t_arm = M.predict(g, pv, warm, b, c, "additive")
            t_bf = M.predict(g, pv, "bf16", b, c, "additive")
            out["win128_R"][f"{g}_b{b}_c{c}k"] = dict(
                tpot=round(t_arm, 2), bf16=round(t_bf, 2),
                R=round(t_arm / t_bf, 3))
        for arm in ("bf16", ("d_win" if g == "dense" else "m_win")):
            t = M.predict(g, pv, arm, 16, 8, "additive")
            out["interp_b16_8k"][f"{g}_{arm}"] = round(t, 2)

    f = PHASE / "data/v2_predictions.json"
    f.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"REGISTERED -> {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
