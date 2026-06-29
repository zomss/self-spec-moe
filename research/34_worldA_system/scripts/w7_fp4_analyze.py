"""Aggregate W7 timing JSONs into a bf16 vs FP8 vs FP4 (batch,K) speedup table.

Reads:
  - new FP4 stage: w7fp4_v2lite_{nospec,bf16,nvfp4}_<suffix>[_K*].json
    (written by w7_fp4_timing.py).
  - prior FP8 stage: w7fp8_v2lite_fp8_<suffix>[_K*].json (the FP8 spec numbers).
Computes speedup vs the FP4-stage's own nospec baseline (self-consistent), and
the FP8 speedup vs the FP8-stage nospec (its own baseline); both baselines should
agree within ~1%, which the table prints.

Usage: python w7_fp4_analyze.py [DATA_DIR]
"""
import glob
import json
import os
import sys

DATA = sys.argv[1] if len(sys.argv) > 1 else \
    "/data/smcho/ssm-w7fp4/research/34_worldA_system/data"
SUFFIX = "cg_a2a100us"


def load(prefix, tag, mode):
    """Union per-K shard files; fall back to the combined file."""
    by_k = {}
    for f in sorted(glob.glob(
            os.path.join(DATA, f"{prefix}_{tag}_{mode}_{SUFFIX}_*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        k = str(d.get("this_k", 0))
        by_k[k] = d.get("results", [])
    if by_k:
        return {"by_k": by_k}
    path = os.path.join(DATA, f"{prefix}_{tag}_{mode}_{SUFFIX}.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        if d.get("by_k"):
            return d
    return None


def rows_by_batch(d):
    out = {}
    if not d:
        return out
    for r in d["by_k"].get("0", []):
        if "batch" in r and "tok_s_mean" in r:
            out[r["batch"]] = r
    return out


def rows_by_k_batch(d):
    out = {}
    if not d:
        return out
    for k, rows in d["by_k"].items():
        for r in rows:
            if "batch" in r and "tok_s_mean" in r:
                out[(int(k), r["batch"])] = r
    return out


def f(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "NA"


def main():
    fp4_nospec = load("w7fp4", "v2lite_nospec", "nospec")
    fp4_bf16 = load("w7fp4", "v2lite_bf16", "spec")
    fp4_nvfp4 = load("w7fp4", "v2lite_nvfp4", "spec")
    fp8_nospec = load("w7fp8", "v2lite_nospec", "nospec")
    fp8_fp8 = load("w7fp8", "v2lite_fp8", "spec")

    ns = rows_by_batch(fp4_nospec)
    bf = rows_by_k_batch(fp4_bf16)
    nv = rows_by_k_batch(fp4_nvfp4)
    ns8 = rows_by_batch(fp8_nospec)
    fq = rows_by_k_batch(fp8_fp8)

    print("## V2-Lite -- bf16 vs FP8 vs NVFP4 draft (Regime A: PCIe + 100us A2A)\n")
    print("| batch | K | nospec | bf16 sp | fp8 sp | fp4 sp | "
          "fp4 tok/s | acc(bf16) | acc(fp4) |")
    print("|------:|--:|-------:|--------:|-------:|-------:|"
          "----------:|----------:|---------:|")
    keys = sorted(set(bf.keys()) | set(nv.keys()))
    for (k, b) in keys:
        nr = ns.get(b)
        ntp = nr.get("tok_s_mean") if nr else None
        br = bf.get((k, b))
        nvr = nv.get((k, b))
        fr = fq.get((k, b))
        n8 = ns8.get(b)
        ntp8 = n8.get("tok_s_mean") if n8 else None
        btp = br.get("tok_s_mean") if br else None
        nvtp = nvr.get("tok_s_mean") if nvr else None
        ftp = fr.get("tok_s_mean") if fr else None
        bsp = (btp / ntp) if (btp and ntp) else None
        nvsp = (nvtp / ntp) if (nvtp and ntp) else None
        fsp = (ftp / ntp8) if (ftp and ntp8) else None
        bsus = "S" if (br and br.get("suspect")) else ""
        nvsus = "S" if (nvr and nvr.get("suspect")) else ""
        print(f"| {b} | {k} | {f(ntp,1)} | {f(bsp,3)}{bsus} | "
              f"{f(fsp,3)} | {f(nvsp,3)}{nvsus} | {f(nvtp,1)} | "
              f"{f(br.get('accept_len') if br else None,3)} | "
              f"{f(nvr.get('accept_len') if nvr else None,3)} |")

    # best fp4 speedup + crossover
    valid = [(k, b, nv[(k, b)]) for (k, b) in nv
             if ns.get(b) and not nv[(k, b)].get("suspect")]
    best = None
    for (k, b, r) in valid:
        sp = r["tok_s_mean"] / ns[b]["tok_s_mean"]
        if best is None or sp > best[0]:
            best = (sp, k, b)
    if best:
        print(f"\nNVFP4 best speedup = {best[0]:.3f} at K={best[1]} batch={best[2]}")
        crossed = [(k, b) for (k, b, r) in valid
                   if r["tok_s_mean"] / ns[b]["tok_s_mean"] >= 1.0]
        print(f"NVFP4 crosses 1.0x: {crossed if crossed else 'nowhere'}")
    # baseline self-consistency
    if ns and ns8:
        bset = sorted(set(ns) & set(ns8))
        print("\nBaseline self-consistency (fp4-stage nospec vs fp8-stage nospec):")
        for b in bset:
            a = ns[b]["tok_s_mean"]
            c = ns8[b]["tok_s_mean"]
            print(f"  batch={b}: {a:.1f} vs {c:.1f} "
                  f"({100 * (a - c) / c:+.1f}%)")


if __name__ == "__main__":
    main()
