#!/usr/bin/env python3
"""Phase 94 P3: does COST (R) compose additively?

Free test -- no GPU. Inverts R from measured cells via the speedup
identity S = (1 + f*K) / (K*R + 1)  =>  R = ((1 + f*K)/S - 1) / K,
where f = (accept - 1)/K is the accepted FRACTION (accept = 1 + f*K).

Then compares the measured composed R against two predictions:
  additive   : R_comp = R_a + R_b - R_bf16      (cost SAVINGS add, i.e.
               each lever removes its own term from the bf16 baseline)
  product    : R_comp = R_a * R_b / R_bf16      (multiplicative savings)

Stage-1 pruning in the search rests on the additive model being close;
if it is badly off, nomination needs a measured correction term.
"""
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

C1 = Path("/data/smcho/self-spec-moe/research/93_c1_grid/data")
C2 = Path("/data/smcho/self-spec-moe/research/94_composition/data")
OUT = Path("/data/smcho/self-spec-moe/paper/data")

# composition -> its constituent single arms, per arch
PARTS = {
    "dense": {
        "hum_x_win512": ("w4a8hum", "win512"),
        "hum_x_win2048": ("w4a8hum", "win2048"),
        "hum_x_skipb2": ("w4a8hum", "skipb2"),
        "w4a16_x_win512": ("w4a16", "win512"),
        "win512_x_skipb2": ("win512", "skipb2"),
    },
    "llama": {
        "w4a16_x_win512": ("w4a16", "win512"),
        "w4a16_x_win2048": ("w4a16", "win2048"),
        "w4a16_x_skipb2": ("w4a16", "skipb2"),
    },
    "q3_32b": {
        "w4gptq_x_win512": ("w4gptq", "win512"),
        "w4gptq_x_win2048": ("w4gptq", "win2048"),
        "w4gptq_x_skipb2": ("w4gptq", "skipb2"),
    },
}


def load(pattern, tag):
    cells = defaultdict(dict)
    for f in glob.glob(str(pattern)):
        name = Path(f).stem.split(tag)[1]
        for r in csv.DictReader(open(f)):
            cells[(int(r["batch"]), int(r["ctx"]))][(name, int(r["K"]))] = (
                float(r["decode_toks"]), float(r["accept"] or 0))
    return cells


def invert_R(toks, ar_toks, accept, K):
    """R from S and accept. accept = 1 + f*K (tokens per draft)."""
    S = toks / ar_toks
    if S <= 0 or K <= 0:
        return None
    tau = accept          # = 1 + f*K
    return ((tau / S) - 1.0) / K


def main():
    rows = []
    for arch, parts in PARTS.items():
        singles = load(C1 / f"cells_93_{arch}_*.csv", f"cells_93_{arch}_")
        comps = load(C2 / f"cells_94_{arch}_*.csv", f"cells_94_{arch}_")
        for cell in sorted(set(singles) & set(comps)):
            off = singles[cell].get(("off", 0))
            if not off:
                continue
            for cname, (a, b) in parts.items():
                for K in (2, 4, 6):
                    c = comps[cell].get((cname, K))
                    ra_ = singles[cell].get((a, K))
                    rb_ = singles[cell].get((b, K))
                    if not (c and ra_ and rb_):
                        continue
                    Rc = invert_R(c[0], off[0], c[1], K)
                    Ra = invert_R(ra_[0], off[0], ra_[1], K)
                    Rb = invert_R(rb_[0], off[0], rb_[1], K)
                    if None in (Rc, Ra, Rb) or min(Ra, Rb, Rc) <= 0:
                        continue
                    # bf16 self-draft reference R ~ 1.0 (draft == target cost)
                    R_add = max(Ra + Rb - 1.0, 1e-3)
                    R_mul = max(Ra * Rb, 1e-3)
                    rows.append({
                        "arch": arch, "cell": f"b{cell[0]}/c{cell[1]}",
                        "comp": cname, "K": K,
                        "R_meas": round(Rc, 4),
                        "R_add": round(R_add, 4), "R_mul": round(R_mul, 4),
                        "err_add_pct": round((R_add / Rc - 1) * 100, 1),
                        "err_mul_pct": round((R_mul / Rc - 1) * 100, 1),
                        "accept_comp": c[1]})

    if not rows:
        print("no paired rows found")
        return

    def summ(key):
        v = [abs(r[key]) for r in rows]
        v.sort()
        return (sum(v) / len(v), v[len(v) // 2], v[int(len(v) * 0.9)])

    ma, mda, p90a = summ("err_add_pct")
    mm, mdm, p90m = summ("err_mul_pct")
    print(f"paired (composition, cell, K) rows: {len(rows)}\n")
    print("PREDICTING COMPOSED COST R:")
    print(f"  additive (R_a + R_b - 1): |err| mean {ma:5.1f}%  median {mda:5.1f}%  p90 {p90a:5.1f}%")
    print(f"  product  (R_a * R_b)    : |err| mean {mm:5.1f}%  median {mdm:5.1f}%  p90 {p90m:5.1f}%")
    better = sum(1 for r in rows
                 if abs(r["err_add_pct"]) < abs(r["err_mul_pct"]))
    print(f"  additive closer in {better}/{len(rows)} rows\n")

    # bias direction: does additive over- or under-predict cost?
    bias = sum(r["err_add_pct"] for r in rows) / len(rows)
    print(f"  additive bias (signed mean): {bias:+.1f}% "
          f"({'over' if bias > 0 else 'under'}-predicts R -> "
          f"{'pessimistic' if bias > 0 else 'OPTIMISTIC (unsafe for pruning)'})")

    print("\nworst 6 rows by additive error:")
    for r in sorted(rows, key=lambda r: -abs(r["err_add_pct"]))[:6]:
        print(f"  {r['arch']:7s} {r['cell']:11s} {r['comp']:22s} K{r['K']} "
              f"R_meas {r['R_meas']:.3f} R_add {r['R_add']:.3f} "
              f"({r['err_add_pct']:+.1f}%)")

    OUT.mkdir(exist_ok=True)
    (OUT / "c2_p3_cost_additivity.json").write_text(json.dumps(
        {"n_rows": len(rows),
         "additive_abs_err": {"mean": ma, "median": mda, "p90": p90a},
         "product_abs_err": {"mean": mm, "median": mdm, "p90": p90m},
         "additive_signed_bias_pct": bias,
         "additive_closer_frac": better / len(rows),
         "rows": rows}, indent=1))
    print("\nwrote", OUT / "c2_p3_cost_additivity.json")


if __name__ == "__main__":
    main()
