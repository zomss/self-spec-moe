"""Aggregate W7-FP8 timing JSONs into side-by-side (batch,K) speedup tables.

Reads the combined w7fp8_<tag>_<mode>_<suffix>.json files written by
w7_fp8_timing.py and prints:
  (1) V2-Lite: (batch,K) -> bf16 speedup vs FP8 speedup, side by side.
  (2) Qwen3-30B: FP8 spec tok/s vs no-spec, per (batch,K).

Usage: python w7_fp8_analyze.py [DATA_DIR]
"""
import glob
import json
import os
import sys

DATA = sys.argv[1] if len(sys.argv) > 1 else \
    "/data/smcho/ssm-w7q/research/34_worldA_system/data"


def load_tag(tag, mode, suffix="cg_a2a100us"):
    """Build by_k by unioning all per-K shard files (robust to a run that
    failed mid-sweep and left an incomplete combined file). Falls back to the
    combined file only if no shards are present."""
    by_k = {}
    for f in sorted(glob.glob(
            os.path.join(DATA, f"w7fp8_{tag}_{mode}_{suffix}_*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        k = str(d.get("this_k", 0))
        by_k[k] = d.get("results", [])
    if by_k:
        return {"by_k": by_k}
    path = os.path.join(DATA, f"w7fp8_{tag}_{mode}_{suffix}.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        if d.get("by_k"):
            return d
    return None


def rows_by_batch(d):
    """nospec: {batch: row}."""
    out = {}
    for r in d["by_k"].get("0", []):
        if "batch" in r and "tok_s_mean" in r:
            out[r["batch"]] = r
    return out


def rows_by_k_batch(d):
    """spec: {(k,batch): row}."""
    out = {}
    for k, rows in d["by_k"].items():
        for r in rows:
            if "batch" in r and "tok_s_mean" in r:
                out[(int(k), r["batch"])] = r
    return out


def f(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "NA"


def v2lite_table():
    nospec = load_tag("v2lite_nospec", "nospec")
    bf16 = load_tag("v2lite_bf16", "spec")
    fp8 = load_tag("v2lite_fp8", "spec")
    if not (nospec and bf16 and fp8):
        print("V2-Lite: missing one of nospec/bf16/fp8 JSONs; skipping.")
        return
    ns = rows_by_batch(nospec)
    bf = rows_by_k_batch(bf16)
    fq = rows_by_k_batch(fp8)
    print("\n## V2-Lite -- controlled bf16-vs-FP8 draft (Regime A: PCIe + 100us A2A)\n")
    print("| batch | K | nospec tok/s | bf16 spec tok/s | bf16 sp | "
          "fp8 spec tok/s | fp8 sp | acc(bf16) | acc(fp8) |")
    print("|------:|--:|------------:|---------------:|-------:|"
          "--------------:|------:|---------:|---------:|")
    keys = sorted(set(bf.keys()) | set(fq.keys()))
    for (k, b) in keys:
        nr = ns.get(b)
        ntp = nr.get("tok_s_mean") if nr else None
        br = bf.get((k, b))
        fr = fq.get((k, b))
        btp = br.get("tok_s_mean") if br else None
        ftp = fr.get("tok_s_mean") if fr else None
        bsp = (btp / ntp) if (btp and ntp) else None
        fsp = (ftp / ntp) if (ftp and ntp) else None
        bsus = " S" if (br and br.get("suspect")) else ""
        fsus = " S" if (fr and fr.get("suspect")) else ""
        print(f"| {b} | {k} | {f(ntp,1)} | {f(btp,1)} | {f(bsp,3)}{bsus} | "
              f"{f(ftp,1)} | {f(fsp,3)}{fsus} | "
              f"{f(br.get('accept_len') if br else None,3)} | "
              f"{f(fr.get('accept_len') if fr else None,3)} |")
    # best fp8 speedup + crossover
    valid = [(k, b, fq[(k, b)]) for (k, b) in fq
             if ns.get(b) and not fq[(k, b)].get("suspect")]
    best = None
    for (k, b, r) in valid:
        sp = r["tok_s_mean"] / ns[b]["tok_s_mean"]
        if best is None or sp > best[0]:
            best = (sp, k, b)
    if best:
        print(f"\nFP8 best speedup = {best[0]:.3f} at K={best[1]} batch={best[2]}")
        crossed = any(
            r["tok_s_mean"] / ns[b]["tok_s_mean"] >= 1.0 for (k, b, r) in valid
        )
        print(f"FP8 crosses 1.0x anywhere: {crossed}")


def qwen_table():
    nospec = load_tag("qwen30b_nospec", "nospec")
    fp8 = load_tag("qwen30b_fp8", "spec")
    if not fp8:
        print("\nQwen3-30B: no FP8 spec JSON found; skipping.")
        return
    ns = rows_by_batch(nospec) if nospec else {}
    fq = rows_by_k_batch(fp8)
    print("\n## Qwen3-30B-A3B -- FP8 full-replica draft vs no-spec "
          "(Regime A: PCIe + 100us A2A)\n")
    print("| batch | K | nospec tok/s | fp8 spec tok/s | fp8 sp | acc(fp8) | note |")
    print("|------:|--:|------------:|--------------:|------:|---------:|------|")
    for (k, b) in sorted(fq.keys()):
        nr = ns.get(b)
        ntp = nr.get("tok_s_mean") if nr else None
        fr = fq.get((k, b))
        ftp = fr.get("tok_s_mean") if fr else None
        fsp = (ftp / ntp) if (ftp and ntp) else None
        note = "SUSPECT" if fr.get("suspect") else ""
        if fr.get("error"):
            note = "ERR:" + fr["error"][:40]
        print(f"| {b} | {k} | {f(ntp,1)} | {f(ftp,1)} | {f(fsp,3)} | "
              f"{f(fr.get('accept_len'),3)} | {note} |")
    # also surface any nospec errors
    for b, r in ns.items():
        if r.get("error"):
            print(f"| {b} | - | ERR nospec: {r['error'][:60]} | | | | |")


def main():
    v2lite_table()
    qwen_table()


if __name__ == "__main__":
    main()
