"""Aggregate W7 timing JSONs into (batch, K) speedup tables.

Pairs each spec result with the matched no-spec result at the same batch in the
same regime (link + execution mode + emulated A2A). Prints Markdown tables and
writes a combined w7_summary.json.

Usage: python w7_analyze.py [DATA_DIR]
"""
import glob
import json
import os
import sys

DATA = sys.argv[1] if len(sys.argv) > 1 else \
    "/data/smcho/self-spec-moe/research/34_worldA_system/data"


def load_combined():
    """Load the per-mode combined files (w7_<mode>_<link>_<suffix>.json)."""
    files = glob.glob(os.path.join(DATA, "w7_*_*.json"))
    runs = {}  # regime -> {nospec: rows, spec: {k: rows}}
    for f in files:
        base = os.path.basename(f)
        # Skip per-K shards; the combined file has by_k.
        with open(f) as fh:
            d = json.load(fh)
        if "by_k" not in d:
            continue
        regime = (d["link"], d.get("suffix", "eager" if d.get("eager") else "cg"))
        mode = d["mode"]
        runs.setdefault(regime, {"meta": d})
        if mode == "nospec":
            rows = d["by_k"].get("0", [])
            runs[regime]["nospec"] = {r["batch"]: r for r in rows if "batch" in r}
        else:
            spec = runs[regime].setdefault("spec", {})
            for k, rows in d["by_k"].items():
                for r in rows:
                    if "batch" in r and "tok_s_mean" in r:
                        spec[(int(k), r["batch"])] = r
    return runs


def fmt(x, nd=1):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def main():
    runs = load_combined()
    summary = {}
    for regime, data in sorted(runs.items()):
        link, suffix = regime
        nospec = data.get("nospec", {})
        spec = data.get("spec", {})
        if not spec:
            continue
        meta = data["meta"]
        title = (f"link={link} mode={suffix} "
                 f"out_len={meta.get('out_len')} short_len={meta.get('short_len')} "
                 f"a2a_us={meta.get('a2a_emulated_us', 0)} eager={meta.get('eager')}")
        print(f"\n### {title}\n")
        print("| batch | K | accept_len | spec tok/s | nospec tok/s | speedup | |")
        print("|------:|--:|-----------:|-----------:|-------------:|--------:|--|")
        rows_out = []
        for (k, batch) in sorted(spec.keys()):
            sr = spec[(k, batch)]
            nr = nospec.get(batch)
            stp = sr.get("tok_s_mean")
            ntp = nr.get("tok_s_mean") if nr else None
            sp = (stp / ntp) if (stp and ntp) else None
            al = sr.get("accept_len")
            susp = sr.get("suspect", False)
            flag = "SUSPECT" if susp else ""
            print(f"| {batch} | {k} | {fmt(al,3) if al else 'NA'} | "
                  f"{fmt(stp)} | {fmt(ntp) if ntp else 'NA'} | "
                  f"{fmt(sp,3) if sp else 'NA'} | {flag} |")
            rows_out.append({
                "batch": batch, "K": k, "accept_len": al,
                "spec_tok_s": stp, "nospec_tok_s": ntp, "speedup": sp,
                "spec_tok_s_std": sr.get("tok_s_std"), "suspect": susp,
            })
        summary[f"{link}__{suffix}"] = {"meta": title, "rows": rows_out}

        # Best speedup + crossover (trustworthy rows only).
        valid = [r for r in rows_out if r["speedup"] and not r["suspect"]]
        if valid:
            best = max(valid, key=lambda r: r["speedup"])
            print(f"\nbest speedup = {best['speedup']:.3f} at "
                  f"batch={best['batch']} K={best['K']} (accept_len="
                  f"{fmt(best['accept_len'],3)})")
            # Crossover: smallest batch where ALL K dip below 1.0 (per K).
            for k in sorted({r["K"] for r in valid}):
                ks = sorted([r for r in valid if r["K"] == k],
                            key=lambda r: r["batch"])
                cross = next((r["batch"] for r in ks if r["speedup"] < 1.0), None)
                print(f"  K={k}: crossover (speedup<1) at batch="
                      f"{cross if cross else 'none in range'}")

    out = os.path.join(DATA, "w7_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
