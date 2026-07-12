#!/usr/bin/env python3
"""E1 combo-cost predictions — REGISTERED BEFORE MEASUREMENT.

R_combo from the 78 cost model with STACKED term edits (this is the point:
combos cost nothing to price). Arms chosen = the E3 candidates + one
falsification pair:

  d_w4win    W4-Marlin + window   (weight cut AND KV cut -- the dense combo bet)
  d_kvqwin   kv-quant + window    (FALSIFICATION cell: model says the combo is
                                   cost-MOOT -- window already capped the KV
                                   bytes, so R_combo ~ R_win)
  m_winlr    window + local-route (KV cut + comm zero)
  m_winlrq   + fp8-Marlin         (the original comm-free draft, full triple)
  ds_skip125q  skip12.5% + fp8    (MLA's OFF-region entry; V3b floor split:
                                   f_step NOT layer-scaled)

Writes data/e1_combo_predictions.json. Gate at verify: <=10% median on R.
"""

import json
import sys
from pathlib import Path

PHASE = Path(__file__).resolve().parent.parent
P78 = PHASE.parent / "78_cost_model"
sys.path.insert(0, str(P78 / "scripts"))
import fit_cost_model as M  # noqa: E402
import predict_v3_mla  # noqa: E402,F401  (installs mla CONST/ARMS)

WIN = 512 + 16
COMBO_ARMS = {
    ("dense", "d_w4win"): dict(w=0.28, win=WIN, kappa="marlin"),
    ("dense", "d_kvqwin"): dict(kvs=0.5, win=WIN),
    ("moe", "m_winlr"): dict(win=WIN, comm=False, local=True),
    ("moe", "m_winlrq"): dict(win=WIN, comm=False, local=True, w=0.5, kappa="fp8m"),
    ("mla", "ds_skip125q"): dict(ls=0.875, w=0.5, kappa="fp8m", comm=True),
}
M.ARMS.update(COMBO_ARMS)
CELLS = {"dense": [(8, 16), (32, 16)], "moe": [(8, 16), (32, 32)],
         "mla": [(8, 16), (32, 32)]}


def mla_pred(pv, f_step, arm, b, c):
    spec = M.ARMS[("mla", arm)]
    ls = spec.get("ls", 1.0)
    nonfloor = M.predict("mla", pv, arm, b, c, "additive")
    return nonfloor + f_step + (1.78 + 0.0 * b / 4) * ls  # f_layer from V3b


def main() -> int:
    blob = json.loads((P78 / "data/fit.json").read_text())
    v3b = json.loads((P78 / "data/v3b_verify.json").read_text())
    out = {}
    for (g, arm), spec in COMBO_ARMS.items():
        # mla has no own fit: it transfers moe params (V3b rule)
        p = blob[f"{'moe' if g == 'mla' else g}/additive"]["params"]
        kap = [f"kappa_{k}" for k in M.KAPPAS[g]]
        if g == "mla":
            pv = [0.0, 0.0, p["BW_GBs"], p["h_ms_per_Mtok"], p["c0"], p["c1"],
                  p["kappa_fp8m"], p["kappa_fib"], p["kappa_a2atile"]]
            f_step = v3b["f_step"]
            for (b, c) in CELLS[g]:
                t = mla_pred(pv, f_step, arm, b, c)
                den = mla_pred(pv, f_step, "ds_bf16", b, c)
                out[f"{g}_{arm}_b{b}_c{c}k"] = dict(tpot=round(t, 2),
                                                    R=round(t / den, 3))
        else:
            names = (["f0", "f1", "BW_GBs", "h_ms_per_Mtok"]
                     + (["c0", "c1"] if g == "moe" else []) + kap)
            pv = [p[n] for n in names]
            for (b, c) in CELLS[g]:
                t = M.predict(g, pv, arm, b, c, "additive")
                den = M.predict(g, pv, "bf16", b, c, "additive")
                out[f"{g}_{arm}_b{b}_c{c}k"] = dict(tpot=round(t, 2),
                                                    R=round(t / den, 3))
    f = PHASE / "data/e1_combo_predictions.json"
    f.write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(f"{k:26s} T̂={v['tpot']:6.2f}ms  R̂={v['R']:.3f}")
    print(f"REGISTERED -> {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
