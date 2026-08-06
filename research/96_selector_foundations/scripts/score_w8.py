#!/usr/bin/env python3
"""W8 scorer: assemble the constructive cost model (w8_cost_model.md).

Per (arch, b, K) from profiler regions:
  D    mean draft forward step [ms]  (draft_forward_first / draft_forward)
  ovh  draft_chain minus its forwards = per-armed-step stack overhead
  V    verify / T  = verify cost in target-step units
T(b) from W7 off-arm certified R2 means (disclosed: e2e includes
prefill -> T overestimated -> D/T, V, ovh/T underestimated -> biased
TOWARD spec viability; OFF verdicts a fortiori sound).

Two Stage-0 floors (both sound over f, since S <= (1+K)/denom):
  THEORETICAL  D from the HBM bytes roofline at hardware peak BW and
               V >= 1, ovh >= 0. Sound against any implementation.
  EMPIRICAL    D, V, ovh measured for the cheapest realizable draft in
               the lever pool. Sound w.r.t. levers we can actually build.
"""
import json
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
W8 = PHASE / "data" / "w8"

PEAK_BW = 3.35e12          # H100 80GB HBM3, bytes/s
BF16 = 2.0                 # target weight bytes/param

# T(b) [ms/token] = b*1e3 / (W7 off certified R2 rate), results_w7.md
T_MS = {"mla": {1: 1e3 / 257.0, 8: 8e3 / 817.1, 32: 32e3 / 2423.1},
        "moe": {1: 1e3 / 252.6, 8: 8e3 / 575.1, 32: 32e3 / 1182.0}}
# W7 uncond R2 measured denominator K*R_hat+1 = tau/S at K=4
W7_DENOM = {"mla": {1: 4.953 / 0.5765, 8: 4.963 / 0.9091,
                    32: 4.985 / 1.1291},
            "moe": {1: 4.306 / 0.5325, 8: 4.188 / 0.6468,
                    32: 4.43 / 1.0868}}
# E, k routed experts; L moe layers; pe params/expert; dense = non-expert
# params; shared = always-on shared-expert params; qb = draft bytes/param
ARCH = {
    "mla": {"E": 64, "k": 6, "L": 26, "pe": 3 * 2048 * 1408,
            "dense": 1.3e9, "shared": 26 * 2 * 3 * 2048 * 1408,
            "qb": 1.0},                      # W8A16 -> 1 byte/param
    "moe": {"E": 128, "k": 8, "L": 48, "pe": 3 * 2048 * 768,
            "dense": 1.6e9, "shared": 0.0,
            "qb": 0.5},                      # W4A16 -> 0.5 byte/param
}


def cov(arch, n):
    """Expected unique experts/layer for n routed tokens (i.i.d. law)."""
    a = ARCH[arch]
    return a["E"] * (1 - (1 - a["k"] / a["E"]) ** n)


def wbytes(arch, n_tok, bpp):
    """Weight bytes read by one step processing n_tok tokens."""
    a = ARCH[arch]
    return (a["dense"] + a["shared"] + a["L"] * cov(arch, n_tok) * a["pe"]) * bpp


def prof(tag):
    d = W8 / f"prof_{tag}"
    best, best_n = None, -1
    for p in d.glob("self_spec_profile_*.json"):
        s = json.load(open(p))["summary"]
        n = s.get("verify", {}).get("n") or 0
        if n > best_n:
            best, best_n = s, n
    if not best:
        return None
    return {k: v["mean_ms"] for k, v in best.items() if v["n"]}


def main():
    out = {"peak_bw": PEAK_BW, "T_ms": T_MS, "rows": [], "stage0": []}
    hdr = (f"{'cell':<14}{'T_ms':>7}{'D/T':>7}{'V':>7}{'ovh/T':>7}"
           f"{'recon':>8}{'W7':>8}{'err%':>7}{'D/roof':>8}{'Vpred':>7}")
    print(hdr)
    for arch in ("mla", "moe"):
        for b in (1, 8, 32):
            T = T_MS[arch][b]
            for K in (1, 4):
                p = prof(f"{arch}_b{b}_K{K}")
                if not p or "verify" not in p:
                    print(f"{arch}_b{b}_K{K:<9} MISSING")
                    continue
                dff = p.get("draft_forward_first", 0.0)
                dfs = p.get("draft_forward", 0.0)
                fwd_tot = dff + (K - 1) * dfs
                d_fwd = fwd_tot / K
                dc = p.get("draft_chain") or fwd_tot
                V = p["verify"] / T
                ovh = (dc - fwd_tot) / T
                recon = (dc + p["verify"]) / T
                roof_ms = wbytes(arch, b, ARCH[arch]["qb"]) / PEAK_BW * 1e3
                v_pred = wbytes(arch, b * (K + 1), BF16) / wbytes(arch, b, BF16)
                row = {"arch": arch, "b": b, "K": K, "T_ms": round(T, 3),
                       "D_ms": round(d_fwd, 3), "D_over_T": round(d_fwd / T, 3),
                       "V": round(V, 3), "ovh_over_T": round(ovh, 3),
                       "recon_denom": round(recon, 3),
                       "D_roofline_ms": round(roof_ms, 3),
                       "D_vs_roofline": round(d_fwd / roof_ms, 2),
                       "V_bytes_pred": round(v_pred, 3)}
                if K == 4:
                    w7 = W7_DENOM[arch][b]
                    row["w7_denom"] = round(w7, 3)
                    row["recon_err_pct"] = round(100 * (recon - w7) / w7, 1)
                out["rows"].append(row)
                print(f"{arch}_b{b}_K{K:<9}{T:7.2f}{row['D_over_T']:7.2f}"
                      f"{V:7.2f}{ovh:7.2f}{recon:8.2f}"
                      f"{row.get('w7_denom', float('nan')):8.2f}"
                      f"{row.get('recon_err_pct', float('nan')):7.1f}"
                      f"{row['D_vs_roofline']:8.2f}{v_pred:7.2f}")

    print("\nStage-0 cheap-target test (K=4; S_max = 5/denom):")
    for arch in ("mla", "moe"):
        for b in (1, 8, 32):
            T = T_MS[arch][b]
            # THEORETICAL: bytes roofline at peak BW, V>=1, ovh>=0
            d_th = wbytes(arch, b, ARCH[arch]["qb"]) / PEAK_BW * 1e3 / T
            s_th = 5.0 / (4 * d_th + 1.0)
            # EMPIRICAL: cheapest profiled draft step for this cell
            cands = [r for r in out["rows"]
                     if r["arch"] == arch and r["b"] == b]
            row = {"arch": arch, "b": b,
                   "theory_D_over_T": round(d_th, 3),
                   "theory_S_max": round(s_th, 3),
                   "theory": "OFF-proven" if s_th <= 1 else "undecided"}
            k4 = [r for r in cands if r["K"] == 4]
            if k4:
                r4 = k4[0]
                d_emp = min(r["D_over_T"] for r in cands)
                denom = 4 * d_emp + r4["V"] + r4["ovh_over_T"]
                s_emp = 5.0 / denom
                row.update(emp_D_over_T=round(d_emp, 3),
                           emp_denom=round(denom, 3),
                           emp_S_max=round(s_emp, 3),
                           empirical="OFF-proven" if s_emp <= 1
                           else "cell survives")
            out["stage0"].append(row)
            print(f"  {arch} b{b:<3} theory: D/T={d_th:.2f} "
                  f"S_max={s_th:.2f} [{row['theory']}]   "
                  f"empirical: D/T={row.get('emp_D_over_T', float('nan')):.2f} "
                  f"denom={row.get('emp_denom', float('nan')):.2f} "
                  f"S_max={row.get('emp_S_max', float('nan')):.2f} "
                  f"[{row.get('empirical', '-')}]")

    (W8 / "w8_scored.json").write_text(json.dumps(out, indent=1))
    print("\n[W8] saved ->", W8 / "w8_scored.json")


if __name__ == "__main__":
    main()
