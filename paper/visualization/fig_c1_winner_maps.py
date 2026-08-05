#!/usr/bin/env python3
"""C1 Stage-A predicted winner maps (multi-column, 3:1 per arch).

Reads research/93_c1_grid/data/cells_93_<arch>_<lever>.csv, computes
per-cell speedup S = decode_toks(lever,K)/decode_toks(off), picks the
winner (incl. OFF at S=1), and renders one batch x ctx heatmap per
architecture annotated with winner lever/K and S. Also emits the
machine-readable winner map to paper/data/c1_grid_winners_<arch>.json.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import style
from style import COLORS  # noqa: F401  (palette; imshow uses cmap)

REPO = Path(__file__).resolve().parents[2]
GRID = REPO / "research" / "93_c1_grid" / "data"
OUT = REPO / "paper" / "data"

ARCHES = sys.argv[1:] or ["dense", "mla", "moe"]


def load_arch(arch):
    cells = defaultdict(dict)   # (batch, ctx) -> {(lever, K): (toks, acc)}
    # v1 first, then cells_93v2_* (post-collision-fix re-measurements,
    # c1_corruption_ledger.md) override matching (cell, lever, K) rows;
    # the "_nc" (new-cache) suffix maps back to the v1 lever name.
    for pref in (f"cells_93_{arch}_", f"cells_93v2_{arch}_"):
        for f in sorted(GRID.glob(pref + "*.csv")):
            lever = f.stem.replace(pref, "").removesuffix("_nc")
            for r in csv.DictReader(f.open()):
                key = (int(r["batch"]), int(r["ctx"]))
                cells[key][(lever, int(r["K"]))] = (
                    float(r["decode_toks"]), float(r["accept"] or 0))
    return cells


for arch in ARCHES:
    cells = load_arch(arch)
    if not cells:
        print(f"[winner-map] no data for {arch}, skipping")
        continue
    winners = {}
    for key, arms in sorted(cells.items()):
        off = arms.get(("off", 0))
        if off is None:
            continue
        best = ("OFF", 0, 1.0, None)
        for (lever, k), (toks, acc) in arms.items():
            if lever == "off":
                continue
            s = toks / off[0]
            if s > best[2]:
                best = (lever, k, s, acc)
        winners[f"b{key[0]}/c{key[1]}"] = {
            "lever": best[0], "K": best[1],
            "S": round(best[2], 3), "accept": best[3],
            "n_arms": len(arms) - 1}
    OUT.mkdir(exist_ok=True)
    out_json = OUT / f"c1_grid_winners_{arch}.json"
    out_json.write_text(json.dumps(
        {"arch": arch,
         "protocol": "Stage-A compiled cells (decode T(1+N)-T(1), "
                     "batch 1-128 x ctx {2k,8k,14k}); S vs same-boot "
                     "AR; winner incl. OFF",
         "winners": winners}, indent=1))
    print(f"[winner-map] wrote {out_json}")

    batches = sorted({k[0] for k in cells})
    ctxs = sorted({k[1] for k in cells})
    S = np.full((len(ctxs), len(batches)), np.nan)
    lab = np.empty((len(ctxs), len(batches)), dtype=object)
    for (b, c), _ in cells.items():
        w = winners.get(f"b{b}/c{c}")
        if not w:
            continue
        i, j = ctxs.index(c), batches.index(b)
        S[i, j] = w["S"]
        lab[i, j] = ("OFF" if w["lever"] == "OFF"
                     else f"{w['lever']}\nK{w['K']}\n{w['S']:.2f}")

    fig, ax = style.multi_fig()
    im = ax.imshow(S, cmap="RdYlGn", vmin=0.9, vmax=max(2.0, np.nanmax(S)),
                   aspect="auto")
    for i in range(len(ctxs)):
        for j in range(len(batches)):
            if lab[i, j]:
                ax.text(j, i, lab[i, j], ha="center", va="center",
                        fontsize=5.2)
            elif np.isnan(S[i, j]):
                ax.text(j, i, "infeas.", ha="center", va="center",
                        fontsize=5.2, color="#a0aec0")
    ax.set_xticks(range(len(batches)), [f"b{b}" for b in batches],
                  fontsize=7)
    ax.set_yticks(range(len(ctxs)), [f"{c//1000}k" for c in ctxs],
                  fontsize=7)
    ax.grid(False)
    ax.set_xlabel("batch")
    ax.set_ylabel("context")
    ax.set_title(f"{arch}: predicted per-cell winner (Stage A; S vs AR)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="S")
    style.save(fig, f"c1_winner_map_{arch}")
