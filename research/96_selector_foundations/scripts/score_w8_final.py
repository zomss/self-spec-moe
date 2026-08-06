#!/usr/bin/env python3
"""W8 final scorer: constant-batch-width protocol, both arches.

Inputs (all G93_FIXED_LEN=256, notune, R2):
  w8c_mla_off_b*.json / w8b_moe_off_b*.json   -> T(b) = b*1e3/rate
  prof2_{arch}_b*_K4/{profile dumps, serving.json}

Three independent quantities, in increasing dependence on assumptions:
  1. COVERAGE  P_ms = draft_chain+verify  vs  b*tau*1e3/rate_uncond
     (same boot, no T, no off arm) -> what fraction of the armed step
     the two profiled regions actually contain.
  2. TERMS     D/T, V, ovh/T -> the cost mechanism (needs T).
  3. RECON     (draft_chain+verify)/T vs tau/S_ss -> P-W8c. Expected to
     understate by exactly the coverage gap; the test is whether those
     two independently measured numbers agree.
"""
import json
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
W8 = PHASE / "data" / "w8"
PEAK_BW = 3.35e12
BF16 = 2.0
ARCH = {
    "mla": {"E": 64, "k": 6, "L": 26, "pe": 3 * 2048 * 1408,
            "dense": 1.3e9, "shared": 26 * 2 * 3 * 2048 * 1408, "qb": 1.0},
    "moe": {"E": 128, "k": 8, "L": 48, "pe": 3 * 2048 * 768,
            "dense": 1.6e9, "shared": 0.0, "qb": 0.5},
}
OFF = {"mla": "w8c_mla_off_b{b}.json", "moe": "w8b_moe_off_b{b}.json"}


def cov(a, n):
    A = ARCH[a]
    return A["E"] * (1 - (1 - A["k"] / A["E"]) ** n)


def wbytes(a, n, bpp):
    A = ARCH[a]
    return (A["dense"] + A["shared"] + A["L"] * cov(a, n) * A["pe"]) * bpp


def regions(d):
    agg, ranks = {}, 0
    for p in d.glob("self_spec_profile_*.json"):
        s = json.load(open(p))["summary"]
        if not (s.get("verify", {}).get("n") or 0):
            continue
        ranks += 1
        for k, v in s.items():
            if v["n"]:
                agg[k] = max(agg.get(k, 0.0), v["mean_ms"])
    return agg, ranks


def cell_of(path, b):
    d = json.load(open(path))
    assert d.get("fixed_len"), f"{path.name}: not a fixed-length artifact"
    for c in d["cells"]:
        if c["batch"] == b and "toks" in c:
            return c
    return None


def main():
    out = {"protocol": "G93_FIXED_LEN=256, R2, notune", "rows": []}
    print(f"{'cell':<11}{'T_ms':>7}{'D/T':>6}{'V':>6}{'ovh/T':>7}"
          f"{'cover':>7}{'recon':>7}{'meas':>7}{'err%':>7}"
          f"{'D/roof':>8}{'Vpred':>7}")
    for arch in ("mla", "moe"):
        for b in (1, 8, 32):
            offp = W8 / OFF[arch].format(b=b)
            pdir = W8 / f"prof2_{arch}_b{b}_K4"
            if not offp.exists() or not pdir.exists():
                print(f"{arch}_b{b:<8} MISSING")
                continue
            oc = cell_of(offp, b)
            sc = cell_of(pdir / "serving.json", b)
            reg, ranks = regions(pdir)
            if not oc or not sc or "verify" not in reg:
                print(f"{arch}_b{b:<8} INCOMPLETE")
                continue
            T = b * 1e3 / oc["toks"]
            tau = sc["accept"]
            dff = reg.get("draft_forward_first", 0.0)
            dfs = reg.get("draft_forward", 0.0)
            fwd = dff + 3 * dfs
            dc = reg.get("draft_chain", fwd)
            P = dc + reg["verify"]
            step_serv = b * tau * 1e3 / sc["toks"]     # no T, same boot
            coverage = P / step_serv
            S_ss = sc["toks"] / oc["toks"]
            meas = tau / S_ss
            recon = P / T
            roof = wbytes(arch, b, ARCH[arch]["qb"]) / PEAK_BW * 1e3
            vpred = wbytes(arch, b * 5, BF16) / wbytes(arch, b, BF16)
            row = {"arch": arch, "b": b, "T_ms": round(T, 3),
                   "tau": tau, "S_ss": round(S_ss, 4),
                   "D_ms": round(fwd / 4, 3), "D_over_T": round(fwd / 4 / T, 3),
                   "V": round(reg["verify"] / T, 3),
                   "ovh_over_T": round((dc - fwd) / T, 3),
                   "armed_step_prof_ms": round(P, 2),
                   "armed_step_serv_ms": round(step_serv, 2),
                   "instrument_coverage": round(coverage, 3),
                   "recon_denom": round(recon, 3),
                   "meas_denom": round(meas, 3),
                   "recon_err_pct": round(100 * (recon - meas) / meas, 1),
                   "D_vs_roofline": round(fwd / 4 / roof, 2),
                   "V_bytes_pred": round(vpred, 3), "tp_ranks": ranks}
            out["rows"].append(row)
            print(f"{arch}_b{b:<8}{T:7.2f}{row['D_over_T']:6.2f}"
                  f"{row['V']:6.2f}{row['ovh_over_T']:7.2f}"
                  f"{coverage:7.2f}{recon:7.2f}{meas:7.2f}"
                  f"{row['recon_err_pct']:7.1f}"
                  f"{row['D_vs_roofline']:8.2f}{vpred:7.2f}")

    # RETRACTED CHECK (kept per T11). "Coverage-corrected reconstruction"
    # was reported as a PASS at -0.0% on all six cells. It is an ALGEBRAIC
    # IDENTITY, not a test:
    #   corrected = recon/coverage = (P/T)/(P/step_serv) = step_serv/T
    #             = (b*tau*1e3/rate_u)/(b*1e3/rate_off) = tau/S_ss = meas
    # Equivalently recon/meas == coverage identically (verified to 4 dp).
    # So P-W8c cannot be tested this way: the profiled step total P
    # cancels, and nothing about the cost MODEL is being checked. The
    # only non-vacuous content is the coverage number itself.
    print("\nIdentity check (RETRACTED as a validation, kept as evidence):")
    for r in out["rows"]:
        r["recon_over_meas"] = round(r["recon_denom"] / r["meas_denom"], 4)
        print(f"  {r['arch']} b{r['b']:<3} recon/meas="
              f"{r['recon_over_meas']:.4f}  coverage="
              f"{r['instrument_coverage']:.4f}  (identical by algebra)")

    print("\nStage-0 (K=4, S_max = 5/denom):")
    for r in out["rows"]:
        a, b = r["arch"], r["b"]
        T = r["T_ms"]
        d_th = wbytes(a, b, ARCH[a]["qb"]) / PEAK_BW * 1e3 / T
        s_th = 5.0 / (4 * d_th + 1.0)
        denom_emp = 4 * r["D_over_T"] + r["V"] + r["ovh_over_T"]
        s_emp = 5.0 / denom_emp
        r["stage0_theory_S_max"] = round(s_th, 3)
        r["stage0_emp_S_max"] = round(s_emp, 3)
        r["stage0_theory"] = "OFF-proven" if s_th <= 1 else "undecided"
        r["stage0_empirical"] = "OFF-proven" if s_emp <= 1 else "survives"
        print(f"  {a} b{b:<3} theory S_max={s_th:5.2f} [{r['stage0_theory']:>10}]"
              f"   empirical S_max={s_emp:5.2f} [{r['stage0_empirical']:>10}]"
              f"   actual S_ss={r['S_ss']:.3f}")

    (W8 / "w8_final.json").write_text(json.dumps(out, indent=1))
    print("\n[W8] saved ->", W8 / "w8_final.json")


if __name__ == "__main__":
    main()
