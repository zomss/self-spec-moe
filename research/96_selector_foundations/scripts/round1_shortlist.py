#!/usr/bin/env python3
"""W5 Round 1: tie-sets under measured uncertainty + the W6 prediction.

Implements w5_round1.md exactly. R inversion mirrors compile_from_c2
(mean of the two oracle replications on the arch's 93 off anchor).
"""
import csv
import json
from pathlib import Path

ROOT = Path("/data/smcho/self-spec-moe")
C1 = ROOT / "research/93_c1_grid/data"
C2 = ROOT / "research/94_composition/data"
W2D = ROOT / "research/96_selector_foundations/data/w2"
OUT = ROOT / "research/96_selector_foundations/data/w5"

PAIRS = {
    "dense": ("oracle_dense_", "oracleR2_dense_"),
    "llama": ("oracle_llamafix_", "oracle_llamaR2_"),
    "mla": ("oracle_mla_", "oracleR2_mla_"),
    "moe": ("oracle_moe_", "oracleR2_moe_"),
}
ANCHOR = {"llama": "cells_93_llama_off.csv"}
QUANTS = {"dense": ["none", "hum", "w4a16"],
          "llama": ["none", "w4a16", "w8int8"],
          "mla": ["none", "w4a16", "w8chan"],
          "moe": ["none", "w4a16", "w8chan"]}
DEPLOYED_Q = {"dense": "hum", "llama": "w4a16"}

REGIME_CELL = {"R4": (8, 8000), "R5": (8, 14000), "R5cot": (8, 14000),
               "R8": (16, 2000), "R1": (1, 2000), "R6": (32, 2000)}
SHORT_DECODE = {"R4", "R5"}        # w5_round1.md section 2
PARKED = {1: 0.0, 8: 0.0, 16: 0.006, 32: 0.006}
NOISE = 0.01
NBIAS_LO = 0.88                    # serving R = 0.88..1.00 x compile R
NBIAS_LO_SHORT = 0.85              # widened for short-decode regimes
F_BAND = 0.05                      # rho spread floor + mixed-K allowance

REGIMES = ["R4", "R5", "R5cot", "R8", "R1", "R6"]


def read_cells(path):
    out = {}
    if not path.exists():
        return out
    for r in csv.DictReader(open(path)):
        out[(int(r["K"]), int(r["batch"]), int(r["ctx"]))] = (
            float(r["decode_toks"]), float(r["accept"]))
    return out


def invert(cells, anchor, k, b, c):
    """(R, f) via the compile_from_c2 identity, or None."""
    row = cells.get((k, b, c))
    off = anchor.get((0, b, c))
    if not row or not off or row[0] <= 0:
        return None
    s = row[0] / off[0]
    tau = row[1]
    r = (tau / s - 1) / k
    return (r, (tau - 1) / k) if r > 0 else None


def rf_maps(arch):
    """config -> (K, b, c) -> (R_mean, f_mean, R_halfspread_rel)."""
    p1, p2 = PAIRS[arch]
    anchor = read_cells(C1 / ANCHOR.get(arch, f"cells_93_{arch}_off.csv"))
    out = {}
    for q in QUANTS[arch]:
        for w in ("none", "512", "2048"):
            for s in ("none", "b2"):
                cfg = f"q-{q}_w-{w}_s-{s}"
                c1 = read_cells(C2 / f"{p1}{cfg}.csv")
                c2 = read_cells(C2 / f"{p2}{cfg}.csv")
                m = {}
                for k in (2, 4):
                    for b in (1, 8, 32):
                        for cx in (2000, 8000, 14000):
                            i1 = invert(c1, anchor, k, b, cx)
                            i2 = invert(c2, anchor, k, b, cx)
                            vals = [v for v in (i1, i2) if v]
                            if not vals:
                                continue
                            rm = sum(v[0] for v in vals) / len(vals)
                            fm = sum(v[1] for v in vals) / len(vals)
                            spread = (abs(vals[0][0] - vals[1][0]) / (2 * rm)
                                      if len(vals) == 2 else 0.02)
                            m[(k, b, cx)] = (rm, fm, spread)
                if m:
                    out[cfg] = m
    return out


def cell_rf(m, k, regime):
    """(R, f, band) at the regime's cell; R8 interpolates b16."""
    b, c = REGIME_CELL[regime]
    if b != 16:
        return m.get((k, b, c))
    lo, hi = m.get((k, 8, c)), m.get((k, 32, c))
    if not lo or not hi:
        return None
    r = lo[0] + 0.5 * (hi[0] - lo[0])           # log2-linear 3->4->5
    f = (lo[1] + hi[1]) / 2
    # interpolation residual: predict b8 from b1,b32
    b1 = m.get((k, 1, c))
    resid = 0.03
    if b1 and hi:
        pred8 = b1[0] + 0.6 * (hi[0] - b1[0])
        resid = abs(pred8 - lo[0]) / lo[0]
    return (r, f, max(lo[2], hi[2]) + resid)


def live_f(arch):
    """regime -> window -> live f (as-if K4; policy-mixed disclosed)."""
    out = {}
    for w in ("512", "2048"):
        p = W2D / f"w2_{arch}_w{w}_notune_s0.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for rid, reg in d["regimes"].items():
            accs = [r["accept"] for r in reg["rounds"] if r.get("accept")]
            if accs:
                out.setdefault(rid, {})[w] = (
                    sorted(accs)[len(accs) // 2] - 1) / 4
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for arch in ("dense", "llama"):
        maps = rf_maps(arch)
        lf = live_f(arch)
        dq = DEPLOYED_Q[arch]
        # --- separability: rho per (regime, window) at the deployed comp
        rho, sep = {}, {}
        for rid in REGIMES:
            rr = {}
            for w in ("512", "2048"):
                cfg = f"q-{dq}_w-{w}_s-b2"
                got = maps.get(cfg) and cell_rf(maps[cfg], 4, rid)
                if got and rid in lf and w in lf[rid]:
                    rr[w] = lf[rid][w] / got[1]
            if rr:
                vals = list(rr.values())
                rho[rid] = sum(vals) / len(vals)
                sep[rid] = (abs(vals[0] - vals[1]) / rho[rid]
                            if len(vals) == 2 else None)
        # --- tie-sets
        tiesets = {}
        for rid in REGIMES:
            if rid not in rho:
                continue
            b, _ = REGIME_CELL[rid]
            nb = NBIAS_LO_SHORT if rid in SHORT_DECODE else NBIAS_LO
            fb = F_BAND + (sep.get(rid) or 0.0)
            cands = [{"comp": "OFF", "K": 0,
                      "lo": 1 - PARKED.get(b, 0) - 0.005,
                      "hi": 1 - PARKED.get(b, 0) + 0.005,
                      "mid": 1 - PARKED.get(b, 0)}]
            for cfg, m in maps.items():
                for k in (2, 4):
                    got = cell_rf(m, k, rid)
                    if not got:
                        continue
                    r, fc4, band = got
                    fh = fc4 * rho[rid]
                    r_lo = r * nb * (1 - band - NOISE)
                    r_hi = r * (1 + band + NOISE)
                    f_lo, f_hi = fh * (1 - fb), min(fh * (1 + fb), 1.0)
                    s_lo = (1 + f_lo * k) / (k * r_hi + 1)
                    s_hi = (1 + f_hi * k) / (k * r_lo + 1)
                    s_mid = (1 + fh * k) / (k * r * (nb + 1) / 2 + 1)
                    cands.append({"comp": cfg, "K": k, "lo": round(s_lo, 4),
                                  "hi": round(s_hi, 4),
                                  "mid": round(s_mid, 4)})
            best_lo = max(c["lo"] for c in cands)
            keep = [c for c in cands if c["hi"] >= best_lo]
            keep.sort(key=lambda c: -c["mid"])
            tiesets[rid] = keep
        # --- W6 prediction
        combos = sorted({(c["comp"], c["K"]) for ts in tiesets.values()
                         for c in ts if c["comp"] != "OFF"})
        def score(comp, k, rid):
            for c in tiesets[rid]:
                if c["comp"] == comp and c["K"] == k:
                    return c
            return None
        best_static, b_val = None, -1
        for comp, k in combos:
            vals = []
            for rid in tiesets:
                c = score(comp, k, rid)
                vals.append(max(c["mid"], 1.0) if c else 1.0)
            mv = sum(vals) / len(vals)
            if mv > b_val:
                best_static, b_val = (comp, k), mv
        t_vals, b_vals, per_regime = [], [], {}
        for rid in tiesets:
            t = max(max(c["mid"], 1.0) for c in tiesets[rid])
            cb = score(*best_static, rid)
            bv = max(cb["mid"], 1.0) if cb else 1.0
            t_vals.append(t)
            b_vals.append(bv)
            per_regime[rid] = {"T": round(t, 4), "B": round(bv, 4),
                               "delta": round(t - bv, 4)}
        mean_gain = sum(t_vals) / len(t_vals) - sum(b_vals) / len(b_vals)
        single = max(v["delta"] for v in per_regime.values())
        gate = mean_gain >= 0.02 or (single >= 0.05 and mean_gain >= 0)
        summary[arch] = {
            "rho": {k: round(v, 4) for k, v in rho.items()},
            "separability_relspread": {k: (round(v, 4) if v is not None
                                           else None)
                                       for k, v in sep.items()},
            "tieset_sizes": {k: len(v) for k, v in tiesets.items()},
            "best_static": best_static, "B_mean": round(b_val, 4),
            "per_regime": per_regime,
            "mean_gain_T_minus_B": round(mean_gain, 4),
            "best_single_regime_gain": round(single, 4),
            "W6_gate": "PASS" if gate else "FAIL",
        }
        (OUT / f"tiesets_{arch}.json").write_text(
            json.dumps({"tiesets": tiesets, "summary": summary[arch]},
                       indent=1))
        print(f"\n=== {arch.upper()} ===")
        print("rho:", summary[arch]["rho"])
        print("separability rel-spread:",
              summary[arch]["separability_relspread"])
        print("tie-set sizes:", summary[arch]["tieset_sizes"])
        for rid in REGIMES:
            if rid in tiesets:
                top = tiesets[rid][:3]
                print(f"  {rid:6s} top: " + "; ".join(
                    f"{c['comp']}/K{c['K']} {c['mid']:.3f} "
                    f"[{c['lo']:.3f},{c['hi']:.3f}]" for c in top))
        print(f"best static: {best_static} B_mean={b_val:.4f}")
        print("per-regime T vs B:", json.dumps(per_regime))
        print(f"mean(T-B) = {mean_gain:+.4f}  "
              f"best single = {single:+.4f}  -> W6 gate: "
              f"{summary[arch]['W6_gate']}")
    (OUT / "w6_prediction.json").write_text(json.dumps(summary, indent=1))
    print("\nsaved ->", OUT)


if __name__ == "__main__":
    main()
