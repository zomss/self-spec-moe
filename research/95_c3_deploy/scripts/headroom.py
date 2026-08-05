#!/usr/bin/env python3
"""C3 prediction basis: decompose the map's value into runtime-actionable layers.

Computed from ALREADY-COMMITTED phase-94 compile-cell measurements, so it
carries no new GPU cost. It sets the pre-registered magnitudes in the phase
README; the live-serving numbers those predict are measured in E2-E4.

Ladder (per architecture):

  static  best single (config, K) held fixed for the whole deployment
  konly   config fixed, best K per cell, OFF available   (= today's C3)
  full    best (config, K) per cell, OFF available       (= the new arm)

OFF availability is modelled as max(S, 1.0): the deployed policy scores OFF at
1.0 and disarms whenever no option beats it (scheduler asymmetric hysteresis).
For the GATE arms (MLA, MoE) this is the dominant term, so `static` is also
reported WITHOUT the OFF clamp -- that difference is the gate's value.

Scoring uses the 2-sample truth basis (mean S over the R1+R2 content draws,
the winner's-curse-proof scoring of 94/scripts/truth_4arch.py), not a single
content draw.
"""
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data/smcho/self-spec-moe")
C1 = ROOT / "research/93_c1_grid/data"
C2 = ROOT / "research/94_composition/data"
OUT = ROOT / "research/95_c3_deploy/data/c3_headroom.json"

# same pairing as 94/scripts/truth_4arch.py
PAIRS = {
    "dense": ("oracle_dense_", "oracleR2_dense_", "switching"),
    "llama": ("oracle_llamafix_", "oracle_llamaR2_", "switching"),
    "mla": ("oracle_mla_", "oracleR2_mla_", "gate"),
    "moe": ("oracle_moe_", "oracleR2_moe_", "gate"),
}
ANCHOR = {"llama": "cells_93_llama_off.csv"}


def load(prefix):
    """cell -> (cfg, K) -> decode_toks"""
    out = defaultdict(dict)
    for f in glob.glob(str(C2 / (prefix + "*.csv"))):
        cfg = re.sub(r"\.csv$", "", Path(f).stem.split(prefix)[1])
        for r in csv.DictReader(open(f)):
            out[(int(r["batch"]), int(r["ctx"]))][(cfg, int(r["K"]))] = float(
                r["decode_toks"]
            )
    return out


def mean(v):
    return sum(v) / len(v)


def analyse(arch, p1, p2, mode):
    s1, s2 = load(p1), load(p2)
    if not s1 or not s2:
        return None
    ar = {}
    for r in csv.DictReader(open(C1 / ANCHOR.get(arch, f"cells_93_{arch}_off.csv"))):
        ar[(int(r["batch"]), int(r["ctx"]))] = float(r["decode_toks"])
    cells = sorted(c for c in set(s1) & set(s2) if c in ar)

    # 2-sample mean S per (cfg, K) per cell
    S = {}
    for c in cells:
        keys = set(s1[c]) & set(s2[c])
        S[c] = {k: (s1[c][k] + s2[c][k]) / (2 * ar[c]) for k in keys}
    cfgs = sorted({k[0] for c in cells for k in S[c]})
    Ks = sorted({k[1] for c in cells for k in S[c]})
    complete = [
        cfg
        for cfg in cfgs
        if all((cfg, K) in S[c] for c in cells for K in Ks)
    ]

    # 1. static: one (cfg, K) everywhere. Reported both unclamped (a
    #    deployment that always speculates) and clamped (allowed to ship OFF).
    best_static, bs_raw = None, -1.0
    for cfg in complete:
        for K in Ks:
            v = mean([S[c][(cfg, K)] for c in cells])
            if v > bs_raw:
                best_static, bs_raw = (cfg, K), v
    bs_gated = max(bs_raw, 1.0)

    # 2. konly: fix cfg, best K per cell, OFF available
    best_cfg, bk_val, bk_pick = None, -1.0, None
    for cfg in complete:
        picks = {c: max((S[c][(cfg, K)], K) for K in Ks) for c in cells}
        v = mean([max(picks[c][0], 1.0) for c in cells])
        if v > bk_val:
            best_cfg, bk_val, bk_pick = cfg, v, picks

    # 3. full: best (cfg, K) per cell, OFF available
    fpick = {c: max((S[c][k], k[0], k[1]) for k in S[c]) for c in cells}
    fval = mean([max(fpick[c][0], 1.0) for c in cells])

    per_cell = []
    for c in cells:
        ko, fs = bk_pick[c], fpick[c]
        ko_s, fs_s = max(ko[0], 1.0), max(fs[0], 1.0)
        per_cell.append(
            {
                "cell": f"b{c[0]}/c{c[1]}",
                "konly": {"K": ko[1], "S": round(ko[0], 3)},
                "full": {"config": fs[1], "K": fs[2], "S": round(fs[0], 3)},
                "gain_pct": round((fs_s / ko_s - 1) * 100, 1),
                "beats_ar": bool(fs[0] > 1.0),
            }
        )

    # 4. AXIS ATTRIBUTION: which switchable axis earns the `full` gain?
    #    Each rung frees one more axis; the config is written q_w_s so we
    #    hold the others at their best fixed value. This scopes the engine
    #    work: window and skip are both capture-shape-keyed (window sets
    #    n_kept = the scratchpad gather shape; skip is load-time static),
    #    so each extra free axis costs a capture set, not a metadata flip.
    def parts(cfg):
        d = dict(p.split("-", 1) for p in cfg.split("_"))
        return d["q"], d["w"], d["s"]

    def best_with(free_axes):
        """Best mean S when `free_axes` may vary per cell, others fixed."""
        best, pick = -1.0, None
        fixed_axes = [a for a in ("q", "w", "s") if a not in free_axes]
        combos = {
            tuple(parts(cfg)[i] for i, a in enumerate("qws") if a in fixed_axes)
            for cfg in cfgs
        }
        for combo in combos:
            tot, sel = [], {}
            ok = True
            for c in cells:
                cand = []
                for k in S[c]:
                    p = parts(k[0])
                    held = tuple(
                        p[i] for i, a in enumerate("qws") if a in fixed_axes
                    )
                    if held == combo:
                        cand.append((S[c][k], k[0], k[1]))
                if not cand:
                    ok = False
                    break
                v = max(cand)
                sel[c] = v
                tot.append(max(v[0], 1.0))
            if ok and mean(tot) > best:
                best, pick = mean(tot), sel
        return best, pick

    axis = {}
    for label, free in (
        ("K_only", ""),
        ("K_window", "w"),
        ("K_skip", "s"),
        ("K_window_skip", "ws"),
    ):
        v, _ = best_with(free)
        axis[label] = round(v, 4)
    axis_pct = {
        k: round((v / axis["K_only"] - 1) * 100, 2) for k, v in axis.items()
    }

    return {
        "arch": arch,
        "arm_role": mode,
        "axis_attribution": {
            "S": axis,
            "pct_over_K_only": axis_pct,
            "note": "K_window_skip is `full` restricted to one quant lever; "
            "each freed axis costs a capture set (window sets the scratchpad "
            "gather shape n_kept; skip is load-time static).",
        },
        "n_cells": len(cells),
        "n_configs": len(cfgs),
        "K_arms": Ks,
        "static": {
            "config": best_static[0],
            "K": best_static[1],
            "S_unconditional": round(bs_raw, 4),
            "S_gated": round(bs_gated, 4),
        },
        "konly": {"config": best_cfg, "S": round(bk_val, 4)},
        "full": {"S": round(fval, 4), "n_distinct_configs": len(
            {p["full"]["config"] for p in per_cell})},
        "gate_value_pct": round((bs_gated / bs_raw - 1) * 100, 1),
        "konly_over_static_pct": round((bk_val / bs_gated - 1) * 100, 1),
        "full_over_konly_pct": round((fval / bk_val - 1) * 100, 1),
        "full_over_static_pct": round((fval / bs_gated - 1) * 100, 1),
        "n_cells_beating_ar": sum(p["beats_ar"] for p in per_cell),
        "quant_levers_in_full_picks": sorted(
            {p["full"]["config"].split("_")[0] for p in per_cell}
        ),
        "per_cell": per_cell,
    }


def main():
    out = {
        "description": "C3 prediction basis: runtime-actionable decomposition "
        "of C2's map, from committed phase-94 compile-cell data "
        "(2-sample truth scoring). Predicts live-serving E2-E4.",
        "protocol_caveat": "compile-cell decode measurements, NOT live serving; "
        "the phase-95 pre-registrations predict the live protocol from these.",
        "arches": [],
    }
    for arch, (p1, p2, mode) in PAIRS.items():
        r = analyse(arch, p1, p2, mode)
        if r is None:
            print(f"[{arch}] missing sample -- skipped")
            continue
        out["arches"].append(r)
        print(
            f"\n===== {arch} ({r['arm_role']} arm) — {r['n_cells']} cells, "
            f"{r['n_configs']} configs, K={r['K_arms']} ====="
        )
        print(
            f"  static unconditional  {r['static']['S_unconditional']:.4f}"
            f"   [{r['static']['config']}-K{r['static']['K']}]"
        )
        print(
            f"  static + OFF gate     {r['static']['S_gated']:.4f}"
            f"   (gate worth {r['gate_value_pct']:+.1f}%)"
        )
        print(
            f"  K-only switching      {r['konly']['S']:.4f}"
            f"   ({r['konly_over_static_pct']:+.1f}% over static)"
        )
        print(
            f"  FULL switching        {r['full']['S']:.4f}"
            f"   ({r['full_over_konly_pct']:+.1f}% over K-only,"
            f" {r['full_over_static_pct']:+.1f}% over static)"
        )
        print(
            f"  cells beating AR: {r['n_cells_beating_ar']}/{r['n_cells']}"
            f"   quant levers used: {r['quant_levers_in_full_picks']}"
        )
        a = r["axis_attribution"]["pct_over_K_only"]
        print(
            "  axis attribution over K-only:  "
            f"+window {a['K_window']:+.2f}%   +skip {a['K_skip']:+.2f}%   "
            f"+both {a['K_window_skip']:+.2f}%"
        )
        for p in r["per_cell"]:
            print(
                f"    {p['cell']:12s} konly K{p['konly']['K']} "
                f"{p['konly']['S']:6.3f}  ->  {p['full']['config']}"
                f"-K{p['full']['K']} {p['full']['S']:6.3f}  {p['gain_pct']:+5.1f}%"
                + ("" if p["beats_ar"] else "   [OFF: below AR]")
            )
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
