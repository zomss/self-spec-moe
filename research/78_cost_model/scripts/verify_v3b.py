#!/usr/bin/env python3
"""V3b: the pre-declared refinement -- one-anchor-cell floor calibration.

V3's registered transfer FAILED with a clean decomposition: the floor does not
transfer by layer ratio, and skip arms prove it needs a STEP-FIXED vs
PER-LAYER split (measured skip R 0.58-0.92, not 0.50). Refinement:

  floor(arm) = f_step + f_layer * ls        (ls = layer multiplier)
  f_layer    = transferred moe floor * 27/48 (per-layer kernels)
  f_step     = calibrated from ONE measured bf16 cell (b4/2k, ~5 min on a new
               architecture) = meas - model_nonfloor - f_layer

Everything else transfers untouched. Verified on the 53 REMAINING cells
(calibration cell excluded). Claim if this passes: predicting a new
architecture costs ONE anchor cell, not a sweep.
"""

import json
import statistics
import sys
from pathlib import Path

PHASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PHASE / "scripts"))
import predict_v3_mla as P3  # noqa: E402  (installs mla CONST/ARMS)
import fit_cost_model as M  # noqa: E402
from verify_v3 import DENOM, overcap, tpot  # noqa: E402


def main() -> int:
    blob = json.loads((PHASE / "data/fit.json").read_text())
    p = blob["moe/additive"]["params"]
    lr = 27 / 48
    f_layer0, f_layer1 = p["f0"] * lr, p["f1"] * lr
    base = [0.0, 0.0, p["BW_GBs"], p["h_ms_per_Mtok"], p["c0"], p["c1"],
            p["kappa_fp8m"], p["kappa_fib"], p["kappa_a2atile"]]

    def pred(arm, b, c, f_step):
        spec = M.ARMS[("mla", arm)]
        ls = spec.get("ls", 1.0)
        nonfloor = M.predict("mla", base, arm, b, c, "additive")
        n = b / 4
        return nonfloor + f_step + (f_layer0 + f_layer1 * n) * ls

    cal_meas = tpot("ds_bf16", 4, 2)
    f_step = cal_meas - pred("ds_bf16", 4, 2, 0.0)
    print(f"calibration: bf16 b4/2k meas {cal_meas:.2f}ms -> f_step = {f_step:.2f}ms "
          f"(f_layer = {f_layer0:.2f} + {f_layer1:.3f}n)")

    oc = overcap()
    t_err, r_err, win_meas = [], [], []
    predj = json.loads((PHASE / "data/v3_predictions_mla.json").read_text())
    for key in sorted(predj["cells"]):
        arm, bs, cs = key.rsplit("_", 2)
        b, ck = int(bs[1:]), int(cs[1:-1])
        if (arm, b, ck) == ("ds_bf16", 4, 2):
            continue  # calibration cell
        if (arm, b, ck) in oc or (DENOM.get(arm, "ds_bf16"), b, ck) in oc:
            continue
        t, den = tpot(arm, b, ck), tpot(DENOM.get(arm, "ds_bf16"), b, ck)
        if t is None or den is None:
            continue
        pt = pred(arm, b, ck, f_step)
        pR = pt / pred("ds_bf16", b, ck, f_step)
        R = t / den
        t_err.append(abs((pt - t) / t))
        if arm != "ds_bf16":
            r_err.append(abs((pR - R) / R))
        if arm == "ds_win":
            win_meas.append(R)
    mt, mr = statistics.median(t_err), statistics.median(r_err)
    print(f"V3b on {len(t_err)} held-out cells:")
    print(f"  TPOT median |err| {100*mt:.1f}%  {'PASS' if mt <= 0.15 else 'FAIL'} @15%")
    print(f"  R    median |err| {100*mr:.1f}%  {'PASS' if mr <= 0.15 else 'FAIL'} @15%")
    print(f"  window R measured {min(win_meas):.3f}-{max(win_meas):.3f} "
          f"(claim band 0.80-1.10: {'CONFIRMED' if min(win_meas) >= 0.80 else 'one cell below'})")
    (PHASE / "data/v3b_verify.json").write_text(json.dumps(dict(
        f_step=round(f_step, 3), tpot_median=round(mt, 4), r_median=round(mr, 4),
        n_cells=len(t_err)), indent=1))
    return 0 if (mt <= 0.15 and mr <= 0.15) else 1


if __name__ == "__main__":
    sys.exit(main())
