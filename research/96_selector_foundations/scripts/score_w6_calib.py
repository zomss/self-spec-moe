#!/usr/bin/env python3
"""W6 item-2 scorer: measured f + measured S per (comp, regime, seed);
gate arithmetic in S_dec (W3 currency), S_e2e alongside."""
import json
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "data" / "w6"
COMPS = [("2_8", "512"), ("2_8", "2048"), ("none", "512"), ("none", "2048")]
REG = ["R5", "R5cot", "R1"]


def med_rej(vals):
    ref = max(vals)
    keep = sorted(v for v in vals if v >= 0.95 * ref)
    return keep[len(keep) // 2], len(vals) - len(keep)


anchor = json.load(open(D / "w6cal_dense_s-none_woff.json"))["cells"]
data = {}
for s, w in COMPS:
    data[(s, w)] = json.load(
        open(D / f"w6cal_dense_s-{s}_w{w}.json"))["cells"]

print(f"{'comp':12s} {'reg':6s} | {'f s0':>6s} {'f s1':>6s} | "
      f"{'S_dec s0':>8s} {'S_dec s1':>8s} | {'S_e2e s0':>8s} "
      f"{'S_e2e s1':>8s} {'rej':>3s}")
table = {}
for (s, w) in COMPS:
    for rid in REG:
        row = {"f": {}, "S_dec": {}, "S_e2e": {}, "rej": 0}
        for seed in (0, 1):
            key = f"{rid}_s{seed}"
            rounds = data[(s, w)][key]
            a_rounds = anchor[key]
            accs = [r["accept"] for r in rounds if r["accept"]]
            row["f"][seed] = round((sum(accs) / len(accs) - 1) / 4, 4)
            for basis, name in (("dec_rate_req", "S_dec"),
                                ("e2e_rate", "S_e2e")):
                sv = [r[basis] for r in rounds if r.get(basis)]
                av = [r[basis] for r in a_rounds if r.get(basis)]
                sm, rej = med_rej(sv)
                am, arej = med_rej(av)
                row[name][seed] = round(sm / am, 4)
                if name == "S_e2e":
                    row["rej"] += rej + arej
        table[(s, w, rid)] = row
        print(f"s-{s:5s}/w{w:4s} {rid:6s} | {row['f'][0]:6.3f} "
              f"{row['f'][1]:6.3f} | {row['S_dec'][0]:8.3f} "
              f"{row['S_dec'][1]:8.3f} | {row['S_e2e'][0]:8.3f} "
              f"{row['S_e2e'][1]:8.3f} {row['rej']:>3d}")

# gate arithmetic in S_dec, mean over seeds
def sdec(s, w, rid):
    r = table[(s, w, rid)]["S_dec"]
    return (r[0] + r[1]) / 2


print("\nper-regime measured winners (S_dec, seed-mean):")
T = {}
for rid in REG:
    best = max(COMPS, key=lambda c: sdec(*c, rid))
    T[rid] = max(sdec(*best, rid), 1.0)
    print(f"  {rid:6s} -> s-{best[0]}/w{best[1]}  S_dec {T[rid]:.4f}   "
          f"(all: " + ", ".join(f"{c[0]}/{c[1]}={sdec(*c, rid):.3f}"
                                for c in COMPS) + ")")

best_static, bval = None, -1
for c in COMPS:
    m = sum(max(sdec(*c, rid), 1.0) for rid in REG) / len(REG)
    if m > bval:
        best_static, bval = c, m
B = {rid: max(sdec(*best_static, rid), 1.0) for rid in REG}
mean_gain = sum(T.values()) / len(T) - sum(B.values()) / len(B)
single = max(T[rid] - B[rid] for rid in REG)
print(f"\nbest static: s-{best_static[0]}/w{best_static[1]} "
      f"(mean {bval:.4f})")
for rid in REG:
    print(f"  {rid:6s} T={T[rid]:.4f} B={B[rid]:.4f} "
          f"delta={T[rid] - B[rid]:+.4f}")
print(f"mean(T-B) = {mean_gain:+.4f}   best single = {single:+.4f}")
print("NOTE: 3-regime slice; R4/R8/R6 are {OFF}-tied for both T and B "
      "(tie-sets), contributing delta 0 -- 6-regime mean = 3-regime "
      "mean x 0.5.")
gate = (mean_gain * 0.5 >= 0.02) or (single >= 0.05 and mean_gain >= 0)
print(f"W3 gate (6-regime basis): {'PASS' if gate else 'FAIL'}")

out = {f"s-{s}/w{w}/{rid}": v for (s, w, rid), v in
       ((k, table[k]) for k in table)}
(D / "w6cal_scored.json").write_text(json.dumps(
    {"cells": out, "T": T, "B": B, "best_static": best_static,
     "mean_gain_3reg": round(mean_gain, 4),
     "single_best": round(single, 4)}, indent=1))
print("saved ->", D / "w6cal_scored.json")
