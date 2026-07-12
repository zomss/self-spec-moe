#!/usr/bin/env python3
"""V2 verification: registered predictions vs the new measurements.

- win128 R: measured = median TPOT(win128 runs) / bf16(cell) with bf16
  reconstructed from 76's committed summary.csv (t/ratio median) -- the
  original denominators; the win128 files are new-session but S1 showed ~2%
  method stability.
- interp b16x8k: measured TPOT vs predicted TPOT directly (new cells).
Gate: median |rel err| <= 10% per family.
"""

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PHASE = Path(__file__).resolve().parent.parent
P76D = PHASE.parent / "76_lever_latency_sweep/data/e1"


def tpot(group, arm, b, ck):
    vals = []
    for r in (1, 2, 3, 4):
        f = P76D / f"{group}_{arm}_b{b}_c{ck}k_r{r}.json"
        if f.exists():
            vals.append(json.loads(f.read_text())["mean_tpot_ms"])
    return statistics.median(vals) if vals else None


def bf16_orig():
    recon = defaultdict(list)
    for r in csv.DictReader((P76D / "summary.csv").open()):
        if r["denom"].endswith("_bf16") and not int(r["overpool"]):
            recon[(r["group"], int(r["batch"]), int(r["ctx"]))].append(
                float(r["tpot_ms"]) / float(r["ratio"]))
    return {k: statistics.median(v) for k, v in recon.items()}


def main() -> int:
    pred = json.loads((PHASE / "data/v2_predictions.json").read_text())
    bf16 = bf16_orig()
    errs = {"win128_R": [], "interp": []}
    print("== win128 R: predicted vs measured ==")
    for key, p in pred["win128_R"].items():
        g, bs, cs = key.split("_")
        b, ck = int(bs[1:]), int(cs[1:-1])
        arm = "d_win128" if g == "dense" else "m_win128"
        t = tpot(g, arm, b, ck)
        den = bf16.get((g, b, ck))
        if t is None or den is None:
            print(f"  {key}: MISSING")
            continue
        R = t / den
        e = (p["R"] - R) / R
        errs["win128_R"].append(abs(e))
        print(f"  {key:18s} pred R={p['R']:.3f}  meas R={R:.3f}  err {100*e:+.1f}%")
    print("== interp b16/8k: predicted vs measured TPOT ==")
    for key, pt in pred["interp_b16_8k"].items():
        g, arm = key.split("_", 1)
        name = arm if arm != "bf16" else ("d_bf16" if g == "dense" else "m_bf16")
        t = tpot(g, name, 16, 8)
        if t is None:
            print(f"  {key}: MISSING")
            continue
        e = (pt - t) / t
        errs["interp"].append(abs(e))
        print(f"  {key:18s} pred {pt:.2f}ms  meas {t:.2f}ms  err {100*e:+.1f}%")
    print("== gates ==")
    ok = True
    for fam, es in errs.items():
        if not es:
            print(f"  {fam}: NO DATA"); ok = False
            continue
        med = statistics.median(es)
        verdict = "PASS" if med <= 0.10 else "FAIL"
        ok &= med <= 0.10
        print(f"  {fam}: median |err| {100*med:.1f}%  {verdict} @10%")
    out = dict(errs={k: [round(x, 4) for x in v] for k, v in errs.items()})
    (PHASE / "data/v2_verify.json").write_text(json.dumps(out, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
