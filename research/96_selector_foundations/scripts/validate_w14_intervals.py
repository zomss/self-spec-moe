#!/usr/bin/env python3
"""Validate the W14 interval fix by LEAVE-ONE-OUT on B's data.

For each transfer unit, the transfer distribution is estimated from the
OTHER units only, so no unit informs its own interval. This is a
genuine out-of-sample test of the fix, not a rescoring of P-W14b (which
stands as a registered FAIL).

Reports, for both the old (sampling-only) and fixed (convolved)
intervals:
  - directional containment of the actual S_dec
  - interval width
  - the SAFETY quantity: whether the elimination rule would fire
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w14_intervals import (  # noqa: E402
    loo_deltas_stratified, mean, q_interval, s_interval, s_max)

PHASE = Path(__file__).resolve().parents[1]
W14 = PHASE / "data" / "w14"
CONFIGS = ["w512", "w2048", "woff"]
K = 4


def load(pref):
    return [json.load(open(p))
            for p in sorted(glob.glob(str(W14 / f"w14b_{pref}_boot*.json")))
            if json.load(open(p)).get("complete")]


def rates(boots):
    r, t = {}, {}
    for d in boots:
        per = {}
        for c in d["cells"]:
            k = (c["rid"], c["n_active"])
            per.setdefault(k, {"r": [], "t": []})
            per[k]["r"] += [x["dec_rate_req"] for x in c["rounds"]
                            if x["dec_rate_req"]]
            per[k]["t"] += [x["tau"] for x in c["rounds"] if x["tau"]]
        for k, v in per.items():
            v["r"].sort(); v["t"].sort()
            r.setdefault(k, []).append(v["r"][len(v["r"]) // 2])
            if v["t"]:
                t.setdefault(k, []).append(v["t"][len(v["t"]) // 2])
    return r, t


def main():
    units = json.load(open(W14 / "w14b_scored.json"))["units"]
    ar_r, _ = rates(load("ar"))
    S = {c: rates(load(c)) for c in CONFIGS}

    print("LEAVE-ONE-OUT validation of the interval fix")
    print("(transfer delta for each unit comes ONLY from the other units)\n")
    print(f"{'unit':<12}{'dir':<12}{'old CI':>16}{'fixed CI':>18}"
          f"{'actual':>9}{'old':>6}{'fix':>6}")
    old_hit = fix_hit = tot = 0
    widths_old, widths_fix = [], []
    rows = []
    for c in CONFIGS:
        sr, st = S[c]
        for b in (1, 8):
            key = (c, b)
            dl = loo_deltas_stratified(units, key, lambda k: k[1])
            for src, dst in (("R5", "R5cot"), ("R5cot", "R5")):
                ks, kd = (src, b), (dst, b)
                if ks not in sr or kd not in sr:
                    continue
                tot += 1
                # old: sampling only
                _, qlo_o, qhi_o = q_interval(sr[ks], ar_r[ks], st[ks])
                # fixed: convolved with LOO transfer distribution
                _, qlo_f, qhi_f = q_interval(sr[ks], ar_r[ks], st[ks],
                                             deltas=dl)
                tau_d = mean(st[kd])
                act = mean(sr[kd]) / mean(ar_r[kd])
                slo_o, shi_o = s_interval(tau_d, qlo_o, qhi_o)
                slo_f, shi_f = s_interval(tau_d, qlo_f, qhi_f)
                ho = slo_o <= act <= shi_o
                hf = slo_f <= act <= shi_f
                old_hit += ho; fix_hit += hf
                widths_old.append((shi_o - slo_o) / act)
                widths_fix.append((shi_f - slo_f) / act)
                rows.append(dict(unit=f"{c}_b{b}", dir=f"{src}->{dst}",
                                 old=[slo_o, shi_o], fixed=[slo_f, shi_f],
                                 actual=act, old_hit=ho, fix_hit=hf,
                                 n_deltas=len(dl)))
                print(f"{c+'_b'+str(b):<12}{src+'->'+dst:<12}"
                      f"[{slo_o:.3f},{shi_o:.3f}]"
                      f"  [{slo_f:.3f},{shi_f:.3f}]"
                      f"{act:>9.3f}{'OK' if ho else 'MISS':>6}"
                      f"{'OK' if hf else 'MISS':>6}")

    print(f"\ncontainment: old {old_hit}/{tot}  ->  FIXED {fix_hit}/{tot}"
          f"   (registered bar was >=11/12)")
    print(f"mean interval width (relative): old "
          f"{100*mean(widths_old):.2f}%  ->  fixed {100*mean(widths_fix):.2f}%")

    # SAFETY: does the wider interval change any elimination decision?
    print("\nSAFETY — elimination rule S_max=(K+1)/q_lo vs 1+eps_arm:")
    flips = 0
    for c in CONFIGS:
        sr, st = S[c]
        for b in (1, 8):
            dl = loo_deltas_stratified(units, (c, b), lambda k: k[1])
            for rid in ("R5", "R5cot"):
                k = (rid, b)
                _, qlo_o, _ = q_interval(sr[k], ar_r[k], st[k])
                _, qlo_f, _ = q_interval(sr[k], ar_r[k], st[k], deltas=dl)
                so, sf = s_max(qlo_o, K), s_max(qlo_f, K)
                if (so < 1.015) != (sf < 1.015):
                    flips += 1
                    print(f"  {c}_b{b} {rid}: DECISION FLIP "
                          f"S_max {so:.3f} -> {sf:.3f}")
    print(f"  elimination decisions changed: {flips} "
          f"(none expected here: every unit is far from the bound)")
    print("  the fix widens q_lo DOWNWARD, so S_max rises and elimination")
    print("  becomes strictly LESS eager -- the safe direction.")

    out = {"containment_old": [old_hit, tot], "containment_fixed":
           [fix_hit, tot], "mean_width_old": mean(widths_old),
           "mean_width_fixed": mean(widths_fix), "rows": rows,
           "elimination_flips": flips}
    (W14 / "w14_interval_fix.json").write_text(json.dumps(out, indent=1))
    print("\nsaved ->", W14 / "w14_interval_fix.json")


if __name__ == "__main__":
    main()
