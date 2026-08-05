#!/usr/bin/env python3
"""C2 -> C3 handoff: emit a scheduler policy table from C2's measured cells.

The deployed scheduler (vllm/v1/core/sched/scheduler.py) reads a table of
per-(batch, ctx) cells, each carrying options [{K, R, S_ref, f_ref}], and per
step computes argmax_K (1 + f_live*K)/(K*R + 1) against OFF scored at 1.0.
That is Stage C of C2's search with the live accept-EMA substituted for the
profiled f -- so the table is just C2's measurements re-expressed.

Inversion (identical to 82/scripts/compile_policy.py's --solve, so tables from
either source are interchangeable):

    tau      = accept                      (measured, = 1 + f*K)
    f_ref    = (tau - 1) / K
    S_ref    = decode_toks / decode_toks_AR
    R        = (tau / S_ref - 1) / K

Verified against the committed 82 table (policy_table_hum.json b1/c2000 K2:
f 0.893, R 0.595 -> S = 2.786/2.19 = 1.272 = its S_ref).

Scoring uses the 2-sample mean over the R1+R2 content draws where both exist
(winner's-curse discipline from 94/scripts/truth_4arch.py); --sample r1
restricts to the first draw for comparability with older artifacts.

usage:
  compile_from_c2.py --arch dense --config q-hum_w-512_s-none \
      --out research/95_c3_deploy/data/policy_dense_w512.json
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path("/data/smcho/self-spec-moe")
C1 = ROOT / "research/93_c1_grid/data"
C2 = ROOT / "research/94_composition/data"

PAIRS = {
    "dense": ("oracle_dense_", "oracleR2_dense_"),
    "llama": ("oracle_llamafix_", "oracle_llamaR2_"),
    "mla": ("oracle_mla_", "oracleR2_mla_"),
    "moe": ("oracle_moe_", "oracleR2_moe_"),
}
ANCHOR = {"llama": "cells_93_llama_off.csv"}
ARMING_RENT = 0.015           # measured, 82-E0; scheduler's switch margin

MODELS = {
    "dense": "Qwen/Qwen3-8B",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mla": "deepseek-ai/DeepSeek-V2-Lite",
    "moe": "Qwen/Qwen3-30B-A3B",
}


def read(prefix, config):
    """(batch, ctx) -> K -> (decode_toks, accept), or {} if unmeasured."""
    f = C2 / f"{prefix}{config}.csv"
    if not f.exists():
        return {}
    out = {}
    for r in csv.DictReader(open(f)):
        out.setdefault((int(r["batch"]), int(r["ctx"])), {})[int(r["K"])] = (
            float(r["decode_toks"]),
            float(r["accept"]),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=sorted(PAIRS))
    ap.add_argument("--config", required=True,
                    help="q-<quant>_w-<window>_s-<skip>, e.g. q-hum_w-512_s-none")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", default="mean", choices=("mean", "r1"))
    ap.add_argument("--draft", default="", help="draft ckpt path, recorded only")
    args = ap.parse_args()

    p1, p2 = PAIRS[args.arch]
    s1 = read(p1, args.config)
    s2 = read(p2, args.config) if args.sample == "mean" else {}
    if not s1:
        raise SystemExit(f"no C2 measurements for {args.arch} {args.config}")

    ar = {}
    anchor = ANCHOR.get(args.arch, f"cells_93_{args.arch}_off.csv")
    for r in csv.DictReader(open(C1 / anchor)):
        ar[(int(r["batch"]), int(r["ctx"]))] = float(r["decode_toks"])

    cells, skipped = [], []
    for cell in sorted(s1, key=lambda c: (c[0], c[1])):
        if cell not in ar:
            skipped.append(f"b{cell[0]}/c{cell[1]}:no-AR-anchor")
            continue
        opts = []
        for K in sorted(s1[cell]):
            toks, tau = s1[cell][K]
            n = 1
            if cell in s2 and K in s2[cell]:
                toks += s2[cell][K][0]
                tau += s2[cell][K][1]
                n = 2
            toks, tau = toks / n, tau / n
            S_ref = toks / ar[cell]
            f_ref = (tau - 1) / K
            R = (tau / S_ref - 1) / K
            if R <= 0:
                # a draft step measured as free is unphysical -- drop the
                # option rather than let the argmax divide by ~0.
                skipped.append(f"b{cell[0]}/c{cell[1]}-K{K}:R={R:.3f}<=0")
                continue
            opts.append({"K": K, "R": round(R, 4),
                         "S_ref": round(S_ref, 4), "f_ref": round(f_ref, 4),
                         "n_samples": n})
        if opts:
            cells.append({"batch": cell[0], "ctx": cell[1], "options": opts})

    table = {
        "model": MODELS.get(args.arch, args.arch),
        "draft": args.draft,
        "arming_rent": ARMING_RENT,
        "provenance": {
            "source": "phase 94 C2 oracle cells",
            "arch": args.arch,
            "config": args.config,
            "sample": args.sample,
            "ar_anchor": anchor,
            "inversion": "f=(tau-1)/K; S=toks/toks_AR; R=(tau/S-1)/K",
        },
        "cells": cells,
    }
    Path(args.out).write_text(json.dumps(table, indent=1))
    print(f"wrote {args.out}: {len(cells)} cells, "
          f"{sum(len(c['options']) for c in cells)} options"
          + (f"  [skipped: {', '.join(skipped)}]" if skipped else ""))
    for c in cells:
        best = max(c["options"], key=lambda o: o["S_ref"])
        print(f"  b{c['batch']:<3d}/c{c['ctx']:<6d} "
              + "  ".join(f"K{o['K']}: R={o['R']:.3f} S={o['S_ref']:.3f}"
                          for o in c["options"])
              + f"   -> best K{best['K']}")


if __name__ == "__main__":
    main()
