#!/usr/bin/env python3
"""W13 scorer: does tau* reproduce the dense selector?

Per-cell terms come from DIFFING consecutive profiler snapshots. The
profiler accumulates across a boot, so snapshot k holds cumulative
(n, mean) per region and the per-cell mean over interval k is
  (n_k*mean_k - n_{k-1}*mean_{k-1}) / (n_k - n_{k-1}).

    tau*(comp,cell) = K*D/T + V + C/T        [cost only, Round 1]
    S_pred          = tau_measured / tau*    [Round 2 supplies tau]

Scored against the UNPROFILED uncond arms (actual S). Reports
P-W13a..e.
"""
import json
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
W13 = PHASE / "data" / "w13"
K = 4
COMPS = ["s-none_w512", "s-none_w2048", "s-none_woff",
         "s-2_8_w512", "s-2_8_w2048"]


def cells(p):
    d = json.load(open(p))
    return {(c["rid"], c["batch"]): c for c in d["cells"] if "toks" in c}


def cell_order(p):
    return [(c["rid"], c["batch"]) for c in json.load(open(p))["cells"]
            if "toks" in c]


def per_cell_regions(pdir, order):
    """Diff cumulative snapshots -> per-cell region means [ms]."""
    out, prev = {}, {}
    for key in order:
        rid, b = key
        snaps = sorted(Path(pdir).glob(
            f"snap_{rid}_b{b}_self_spec_profile_*.json"))
        if not snaps:
            continue
        # TP1: one worker file. Take the one with samples.
        cum = {}
        for sp in snaps:
            s = json.load(open(sp))["summary"]
            if not (s.get("verify", {}).get("n") or 0):
                continue
            for lab, v in s.items():
                if v["n"]:
                    cum[lab] = (v["n"], v["mean_ms"])
        if not cum:
            continue
        cell = {}
        for lab, (n, m) in cum.items():
            pn, pm = prev.get(lab, (0, 0.0))
            dn = n - pn
            cell[lab] = ((n * m - pn * pm) / dn) if dn > 0 else m
        out[key] = cell
        prev = cum
    return out


def main():
    offp = W13 / "w13_off.json"
    if not offp.exists():
        print("missing off arm"); return
    off = cells(offp)
    order = cell_order(offp)
    rows, by_cell = [], {}
    for comp in COMPS:
        pdir = W13 / f"prof_{comp}"
        uncp = W13 / f"w13_unc_{comp}.json"
        if not pdir.exists() or not uncp.exists():
            print(f"[{comp}] MISSING"); continue
        regs = per_cell_regions(pdir, order)
        unc = cells(uncp)
        for key in order:
            if key not in regs or key not in unc or key not in off:
                continue
            b = key[1]
            T = b * 1e3 / off[key]["toks"]
            r = regs[key]
            dff = r.get("draft_forward_first", 0.0)
            dfs = r.get("draft_forward", 0.0)
            fwd = dff + (K - 1) * dfs
            dc = r.get("draft_chain", fwd)
            V = r.get("verify", 0.0) / T
            D = fwd / K
            Cov = (dc - fwd) / T
            tau_star = K * D / T + V + Cov
            tau = unc[key]["accept"]
            S_pred = tau / tau_star if tau_star > 0 else None
            S_act = unc[key]["toks"] / off[key]["toks"]
            row = {"comp": comp, "rid": key[0], "batch": b,
                   "T_ms": round(T, 3), "D_ms": round(D, 3),
                   "D_over_T": round(D / T, 3), "V": round(V, 3),
                   "C_over_T": round(Cov, 3),
                   "tau_star": round(tau_star, 3), "tau": tau,
                   "S_pred": round(S_pred, 4) if S_pred else None,
                   "S_actual": round(S_act, 4),
                   "err_pct": round(100 * (S_pred - S_act) / S_act, 1)
                   if S_pred else None}
            rows.append(row)
            by_cell.setdefault(key, []).append(row)

    print(f"{'comp':<14}{'cell':<11}{'D/T':>7}{'V':>7}{'C/T':>7}"
          f"{'tau*':>8}{'tau':>7}{'S_pred':>8}{'S_act':>8}{'err%':>7}")
    for r in rows:
        print(f"{r['comp']:<14}{r['rid']+' b'+str(r['batch']):<11}"
              f"{r['D_over_T']:>7.2f}{r['V']:>7.2f}{r['C_over_T']:>7.2f}"
              f"{r['tau_star']:>8.2f}{r['tau']:>7.3f}"
              f"{r['S_pred']:>8.3f}{r['S_actual']:>8.3f}{r['err_pct']:>7.1f}")

    # P-W13a
    ok = [r for r in rows if r["err_pct"] is not None
          and abs(r["err_pct"]) <= 10]
    print(f"\nP-W13a  |err| <= 10%: {len(ok)}/{len(rows)} cells "
          f"(need >=5 of 6 per comp basis)")
    # P-W13b ranking per cell
    print("P-W13b  ranking per cell (predicted vs actual):")
    nb_ok = 0
    for key, rs in sorted(by_cell.items()):
        pr = [x["comp"] for x in sorted(rs, key=lambda z: -z["S_pred"])]
        ac = [x["comp"] for x in sorted(rs, key=lambda z: -z["S_actual"])]
        same_top = pr[0] == ac[0]
        exact = pr == ac
        nb_ok += exact
        print(f"   {key[0]} b{key[1]}: top1 {'OK' if same_top else 'MISS'}"
              f"  exact {'OK' if exact else 'no'}   pred={pr[0]} act={ac[0]}")
    print(f"   exact-order cells: {nb_ok}/{len(by_cell)}")
    # P-W13c cost content-free: R5 vs R5cot at b8
    print("P-W13c  tau* R5 b8 vs R5cot b8 (same batch, ~same ctx):")
    for comp in COMPS:
        a = next((r for r in rows if r["comp"] == comp
                  and (r["rid"], r["batch"]) == ("R5", 8)), None)
        c = next((r for r in rows if r["comp"] == comp
                  and (r["rid"], r["batch"]) == ("R5cot", 8)), None)
        if a and c:
            print(f"   {comp:<14} {a['tau_star']:.2f} vs {c['tau_star']:.2f}"
                  f"  ({100*(c['tau_star']-a['tau_star'])/a['tau_star']:+.1f}%)")
    # P-W13d windowed D flat in ctx
    print("P-W13d  D_ms at R1 b1 (~1k ctx) vs R5 b1 (~14k ctx):")
    for comp in COMPS:
        a = next((r for r in rows if r["comp"] == comp
                  and (r["rid"], r["batch"]) == ("R1", 1)), None)
        c = next((r for r in rows if r["comp"] == comp
                  and (r["rid"], r["batch"]) == ("R5", 1)), None)
        if a and c:
            print(f"   {comp:<14} {a['D_ms']:.2f} -> {c['D_ms']:.2f} ms "
                  f"({100*(c['D_ms']-a['D_ms'])/a['D_ms']:+.1f}%)")
    # P-W13e spread
    print("P-W13e  tau* spread per cell:")
    for key, rs in sorted(by_cell.items()):
        ts = [r["tau_star"] for r in rs]
        print(f"   {key[0]} b{key[1]}: {min(ts):.2f}-{max(ts):.2f} "
              f"({100*(max(ts)-min(ts))/min(ts):.0f}%)")

    (W13 / "w13_scored.json").write_text(json.dumps(rows, indent=1))
    print("\nsaved ->", W13 / "w13_scored.json")


if __name__ == "__main__":
    main()
