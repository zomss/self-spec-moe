"""Aggregate W7-Qwen30B headline JSONs into the EP-scaling + EP=8 sweep tables.

Reads research/34_worldA_system/data/w7q_*.json and prints:
  1) EP-scaling: DP=2/4/8 -> spec(FULL_CG) and spec(PIECEWISE) speedup vs no-spec
     at (batch=64, K=2).
  2) EP=8 sweep: (batch,K) -> accept_len, spec tok/s, no-spec tok/s, speedup, for
     both the FULL_CG and PIECEWISE draft variants.
Speedup = spec tok/s / no-spec tok/s at the same (dp, batch).
"""
import glob
import json
import os

DATA = os.environ.get(
    "W7_OUT", "/data/smcho/self-spec-moe/research/34_worldA_system/data"
)


def load(path):
    with open(path) as f:
        return json.load(f)


def rows_of(path):
    d = load(path)
    out = {}
    for r in d.get("results", []):
        if isinstance(r, dict) and "batch" in r and "error" not in r:
            out[r["batch"]] = r
    return out


def nospec_map(dp):
    # nospec file: w7q_epscale_dp{dp}_nospec_cg_nospec.json (epscale tag) or
    # w7q_sweep8_dp8_nospec_cg_nospec.json
    res = {}
    for pat in (f"w7q_*_dp{dp}_nospec_cg_nospec.json",):
        for p in glob.glob(os.path.join(DATA, pat)):
            res.update(rows_of(p))
    return res


def spec_files(tag_prefix, dp):
    # e.g. w7q_epscale_fcg_dp2_spec_cg_fullcg_K2.json
    pats = [
        os.path.join(DATA, f"w7q_{tag_prefix}_dp{dp}_spec_*_K*.json"),
    ]
    files = []
    for pat in pats:
        files += glob.glob(pat)
    # dedupe and pull K from filename
    by_k = {}
    for f in files:
        base = os.path.basename(f)
        k = base.split("_K")[-1].split(".")[0]
        try:
            by_k[int(k)] = f
        except ValueError:
            pass
    return by_k


def fmt(x, nd=1):
    return "-" if x is None or x < 0 else f"{x:.{nd}f}"


def ep_scaling():
    print("## EP-scaling (batch=64, K=2): speedup vs no-spec\n")
    print("| DP=EP | no-spec tok/s | spec(FULL_CG) tok/s | sp(FCG) | accept(FCG) "
          "| spec(PIECEWISE) tok/s | sp(PW) | accept(PW) |")
    print("|------:|------------:|-----------------:|-------:|-----------:"
          "|-------------------:|------:|----------:|")
    for dp in (2, 4, 8):
        ns = nospec_map(dp)
        ns64 = ns.get(64, {})
        ns_tps = ns64.get("tok_s_mean")
        fcg = spec_files("epscale_fcg", dp).get(2)
        pw = spec_files("epscale_pw", dp).get(2)
        fcg_r = rows_of(fcg).get(64, {}) if fcg else {}
        pw_r = rows_of(pw).get(64, {}) if pw else {}
        fcg_tps = fcg_r.get("tok_s_mean")
        pw_tps = pw_r.get("tok_s_mean")
        sp_fcg = (fcg_tps / ns_tps) if (fcg_tps and ns_tps) else None
        sp_pw = (pw_tps / ns_tps) if (pw_tps and ns_tps) else None
        print(f"| {dp} | {fmt(ns_tps)} | {fmt(fcg_tps)} | "
              f"{fmt(sp_fcg, 3)} | {fmt(fcg_r.get('accept_len'), 2)} | "
              f"{fmt(pw_tps)} | {fmt(sp_pw, 3)} | {fmt(pw_r.get('accept_len'), 2)} |")
    print()


def sweep8(variant_tag, label):
    print(f"## EP=8 (DP=8) sweep -- spec draft = {label}\n")
    print("| batch | K | accept_len | spec tok/s | no-spec tok/s | **speedup** |")
    print("|------:|--:|----------:|----------:|-------------:|----------:|")
    ns = nospec_map(8)
    by_k = spec_files(variant_tag, 8)
    best = (0.0, None)
    for k in sorted(by_k):
        rs = rows_of(by_k[k])
        for b in sorted(rs):
            r = rs[b]
            sp_tps = r.get("tok_s_mean")
            ns_tps = ns.get(b, {}).get("tok_s_mean")
            sp = (sp_tps / ns_tps) if (sp_tps and ns_tps) else None
            if sp and sp > best[0]:
                best = (sp, (b, k))
            print(f"| {b} | {k} | {fmt(r.get('accept_len'), 2)} | "
                  f"{fmt(sp_tps)} | {fmt(ns_tps)} | {fmt(sp, 3)} |")
    print(f"\nBest {label}: speedup={best[0]:.3f} at (batch,K)={best[1]}\n")


if __name__ == "__main__":
    ep_scaling()
    sweep8("sweep8_fcg", "FULL_CG (named headline stack)")
    sweep8("sweep8_pw", "PIECEWISE (FULL_CG off, correct accept)")
