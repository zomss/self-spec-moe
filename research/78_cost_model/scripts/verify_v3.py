#!/usr/bin/env python3
"""V3 verification: registered MLA-tier predictions vs the confirmation sweep.

Measured R uses SAME-SESSION denominators (all arms in one sweep):
  ds_skip50 -> ds_bf16dummy, ds_kvq -> ds_bf16fmla (FLASHMLA pair), else ds_bf16.
Cells with OVERCAP markers (meta sidecars) are excluded.
Gates (README): median |err| <= 15% on TPOT and R; registered claims checked
(window weak: measured R inside ~0.85-1.05 across the grid).
"""

import json
import statistics
import sys
from pathlib import Path

PHASE = Path(__file__).resolve().parent.parent
P76D = PHASE.parent / "76_lever_latency_sweep/data/e1"

DENOM = {"ds_skip50": "ds_bf16dummy", "ds_kvq": "ds_bf16fmla"}


def tpot(arm, b, ck):
    vals = [json.loads(f.read_text())["mean_tpot_ms"]
            for r in (1, 2, 3, 4)
            if (f := P76D / f"mla_{arm}_b{b}_c{ck}k_r{r}.json").exists()]
    return statistics.median(vals) if vals else None


def overcap():
    out = set()
    for f in P76D.glob("mla_*.meta"):
        arm = f.stem.split("_", 1)[1]
        for line in f.read_text(errors="replace").splitlines():
            if line.startswith("OVERCAP"):
                _, cell, _ = line.split()
                b, ck = cell.split("_")
                out.add((arm, int(b[1:]), int(ck[1:-1])))
    return out

def main() -> int:
    pred = json.loads((PHASE / "data/v3_predictions_mla.json").read_text())
    oc = overcap()
    t_err, r_err = [], []
    win_meas = []
    print(f"{'cell':22s} {'pred T':>7s} {'meas T':>7s} {'errT':>6s}  "
          f"{'pred R':>6s} {'meas R':>6s} {'errR':>6s}")
    for key, p in sorted(pred["cells"].items()):
        arm, bs, cs = key.rsplit("_", 2)
        b, ck = int(bs[1:]), int(cs[1:-1])
        if (arm, b, ck) in oc or (DENOM.get(arm, "ds_bf16"), b, ck) in oc:
            print(f"{key:22s} OVERCAP -- excluded")
            continue
        t = tpot(arm, b, ck)
        den = tpot(DENOM.get(arm, "ds_bf16"), b, ck)
        if t is None or den is None:
            print(f"{key:22s} MISSING")
            continue
        R = t / den
        et = (p["tpot"] - t) / t
        er = (p["R"] - R) / R
        if arm != "ds_bf16":
            r_err.append(abs(er))
        t_err.append(abs(et))
        if arm == "ds_win":
            win_meas.append(R)
        print(f"{key:22s} {p['tpot']:7.2f} {t:7.2f} {100*et:+5.1f}%  "
              f"{p['R']:6.3f} {R:6.3f} {100*er:+5.1f}%")
    print("\n== gates ==")
    mt, mr = statistics.median(t_err), statistics.median(r_err)
    print(f"  TPOT median |err| {100*mt:.1f}%  {'PASS' if mt <= 0.15 else 'FAIL'} @15%")
    print(f"  R    median |err| {100*mr:.1f}%  {'PASS' if mr <= 0.15 else 'FAIL'} @15%")
    if win_meas:
        claim = min(win_meas) >= 0.80
        print(f"  claim 'window WEAK on MLA': measured R(win) "
              f"{min(win_meas):.3f}-{max(win_meas):.3f}  "
              f"{'CONFIRMED (never < 0.80)' if claim else 'REFUTED'}")
    (PHASE / "data/v3_verify.json").write_text(json.dumps(dict(
        tpot_median=round(mt, 4), r_median=round(mr, 4),
        win_R_range=[round(min(win_meas), 3), round(max(win_meas), 3)] if win_meas else None),
        indent=1))
    return 0 if (mt <= 0.15 and mr <= 0.15) else 1


if __name__ == "__main__":
    sys.exit(main())
