#!/usr/bin/env python3
"""W14 item B scorer — matched-cell cost-transfer pilot.

    q = tau / S_dec,   S_dec = rate_spec / rate_AR   (decode currency)

Resampling unit is the BOOT (rounds are not independent replicates), so
every interval is a boot-level percentile bootstrap.

Scores, per the pre-registration:
  P-W14a  for >=5/6 transfer units, the 95% CI for (q_R5cot/q_R5 - 1)
          lies inside +/-5%; no unit's point estimate exceeds +/-10%.
  P-W14b  transferred q intervals x held-out measured tau contain the
          actual S_dec for >=11/12 directional predictions; in all four
          selector evaluations the predicted eps-optimal tie-set contains
          the measured best and simple regret <= 1.5%.

Certification is LOCAL to (configuration, batch). A unit whose scored
total-KV ranges differ by more than 1% is INVALID, not a failure.
"""
import json
import random
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
W14 = PHASE / "data" / "w14"
CONFIGS = ["w512", "w2048", "woff"]
EPS = 0.015          # arming rent == selector epsilon
NBOOT = 4000
random.seed(20260808)


def load(tag_prefix):
    out = []
    for p in sorted(W14.glob(f"w14b_{tag_prefix}_boot*.json")):
        d = json.load(open(p))
        if d.get("complete"):
            out.append(d)
    return out


def cell_key(c):
    return (c["rid"], c["n_active"])


def rates(boots):
    """(rid, batch) -> list over boots of the boot's median dec_rate,
    pooling seeds; plus tau the same way."""
    r, t, kv = {}, {}, {}
    for d in boots:
        per = {}
        for c in d["cells"]:
            k = cell_key(c)
            vals = [x["dec_rate_req"] for x in c["rounds"]
                    if x["dec_rate_req"]]
            taus = [x["tau"] for x in c["rounds"] if x["tau"]]
            per.setdefault(k, {"r": [], "t": []})
            per[k]["r"] += vals
            per[k]["t"] += taus
            kv.setdefault(k, []).append(
                (c["total_kv_lo"] + c["total_kv_hi"]) / 2)
        for k, v in per.items():
            v["r"].sort(); v["t"].sort()
            r.setdefault(k, []).append(v["r"][len(v["r"]) // 2])
            if v["t"]:
                t.setdefault(k, []).append(v["t"][len(v["t"]) // 2])
    return r, t, kv


def boot_ci(fn, *samples, n=NBOOT, lo=2.5, hi=97.5):
    """Percentile bootstrap resampling each sample list independently."""
    vals = []
    for _ in range(n):
        res = [[random.choice(s) for _ in s] for s in samples]
        try:
            vals.append(fn(*res))
        except ZeroDivisionError:
            pass
    vals.sort()
    if not vals:
        return None, None, None
    m = fn(*samples)
    return m, vals[int(len(vals) * lo / 100)], vals[int(len(vals) * hi / 100)]


def mean(x):
    return sum(x) / len(x)


def main():
    ar = load("ar")
    if not ar:
        print("no AR anchor boots yet"); return
    ar_r, _, ar_kv = rates(ar)
    spec = {c: load(c) for c in CONFIGS}
    missing = [c for c, v in spec.items() if not v]
    if missing:
        print(f"incomplete: {missing} (have "
              f"{ {c: len(v) for c, v in spec.items()} }, AR {len(ar)})")
        return

    print(f"boots: AR={len(ar)}, " +
          ", ".join(f"{c}={len(v)}" for c, v in spec.items()))
    print("\nq = tau/S_dec per (config, regime, batch), boot-level 95% CI\n")
    Q = {}
    for c in CONFIGS:
        sr, st, skv = rates(spec[c])
        for k in sorted(sr, key=lambda z: (z[0], z[1])):
            if k not in ar_r:
                continue
            def qf(rs, ra, ts):
                return mean(ts) / (mean(rs) / mean(ra))
            m, lo, hi = boot_ci(qf, sr[k], ar_r[k], st[k])
            Q[(c,) + k] = dict(q=m, lo=lo, hi=hi,
                               tau=mean(st[k]),
                               s_dec=mean(sr[k]) / mean(ar_r[k]),
                               kv=mean(skv[k]))
            print(f"  {c:<6}{k[0]:>6} b{k[1]:<3} q={m:6.3f} "
                  f"[{lo:6.3f},{hi:6.3f}]  tau={mean(st[k]):.3f} "
                  f"S_dec={mean(sr[k])/mean(ar_r[k]):.3f}")

    # ---- P-W14a: 6 transfer units ----
    print("\nP-W14a — matched transfer, 95% CI on (q_R5cot/q_R5 - 1):")
    units, ok_a = [], 0
    for c in CONFIGS:
        for b in (1, 8):
            a, z = Q.get((c, "R5", b)), Q.get((c, "R5cot", b))
            if not a or not z:
                continue
            dkv = abs(z["kv"] - a["kv"]) / a["kv"]
            sr_a, st_a = rates(spec[c])[0][("R5", b)], rates(spec[c])[1][("R5", b)]
            sr_z, st_z = rates(spec[c])[0][("R5cot", b)], rates(spec[c])[1][("R5cot", b)]

            def ratio(ra1, rr1, tt1, ra2, rr2, tt2):
                q1 = mean(tt1) / (mean(ra1) / mean(rr1))
                q2 = mean(tt2) / (mean(ra2) / mean(rr2))
                return q2 / q1 - 1
            m, lo, hi = boot_ci(ratio, sr_a, ar_r[("R5", b)], st_a,
                                sr_z, ar_r[("R5cot", b)], st_z)
            valid = dkv <= 0.01
            passed = valid and abs(lo) <= .05 and abs(hi) <= .05 and abs(m) <= .10
            ok_a += passed
            units.append(dict(config=c, batch=b, point=m, lo=lo, hi=hi,
                              kv_mismatch=dkv, valid=valid, passed=passed))
            print(f"  {c:<6} b{b:<3} {100*m:+6.1f}% "
                  f"[{100*lo:+6.1f},{100*hi:+6.1f}]  "
                  f"dKV={100*dkv:.2f}% "
                  f"{'INVALID' if not valid else ('PASS' if passed else 'fail')}")
    print(f"  -> {ok_a}/{len(units)} units certified (need >=5/6)")

    # ---- P-W14b: 12 directional predictions + 4 selector evaluations ----
    print("\nP-W14b — transfer q + held-out tau contains actual S_dec:")
    hits = tot = 0
    for c in CONFIGS:
        for b in (1, 8):
            for src, dst in (("R5", "R5cot"), ("R5cot", "R5")):
                s, d = Q.get((c, src, b)), Q.get((c, dst, b))
                if not s or not d:
                    continue
                tot += 1
                pred_lo, pred_hi = d["tau"] / s["hi"], d["tau"] / s["lo"]
                hit = pred_lo <= d["s_dec"] <= pred_hi
                hits += hit
                print(f"  {c:<6} b{b:<3} {src}->{dst}: "
                      f"S_pred [{pred_lo:.3f},{pred_hi:.3f}] "
                      f"actual {d['s_dec']:.3f} {'OK' if hit else 'MISS'}")
    print(f"  -> {hits}/{tot} directional predictions contain actual "
          f"(need >=11/12)")

    print("\nP-W14b — selector evaluations (tie-set contains measured best):")
    sel_ok = 0; sel_tot = 0
    for b in (1, 8):
        for src, dst in (("R5", "R5cot"), ("R5cot", "R5")):
            cand = [(c, Q[(c, dst, b)]["tau"] / Q[(c, src, b)]["lo"],
                     Q[(c, dst, b)]["s_dec"])
                    for c in CONFIGS
                    if (c, src, b) in Q and (c, dst, b) in Q]
            if len(cand) < 2:
                continue
            sel_tot += 1
            best_pred = max(x[1] for x in cand)
            tie = [x[0] for x in cand if x[1] >= best_pred * (1 - EPS)]
            true_best = max(cand, key=lambda x: x[2])
            regret = (true_best[2] - max(x[2] for x in cand if x[0] in tie)) \
                / true_best[2]
            good = true_best[0] in tie or regret <= EPS
            sel_ok += good
            print(f"  b{b} {src}->{dst}: tie-set {tie} measured-best "
                  f"{true_best[0]} regret {100*regret:+.2f}% "
                  f"{'OK' if good else 'MISS'}")
    print(f"  -> {sel_ok}/{sel_tot} selector evaluations OK")

    res = {"units": units, "certified": ok_a, "directional": [hits, tot],
           "selector": [sel_ok, sel_tot],
           "transfer_mask": [f"{u['config']}_b{u['batch']}"
                             for u in units if u["passed"]]}
    (W14 / "w14b_scored.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"\nTRANSFER MASK (certified units): {res['transfer_mask']}")
    print("saved ->", W14 / "w14b_scored.json")


if __name__ == "__main__":
    main()
