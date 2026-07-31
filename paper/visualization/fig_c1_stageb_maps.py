#!/usr/bin/env python3
"""C1 Stage-B REAL-DATA winner maps (multi-column, 3:1, one per arch).

dataset x batch heatmaps of the best measured arm vs AR on real
datasets with uncapped generation (MLA: ceiling-2048 protocol, model
loops at T=0 — disclosed). Emits paper/data/c1_grid_stageb_<arch>.json.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

import style

REPO = Path(__file__).resolve().parents[2]
GRID = REPO / "research" / "93_c1_grid" / "data"
OUT = REPO / "paper" / "data"

RIDS = ["R1", "R2", "R3", "R4", "R5", "R5cot", "R6", "R7", "R8"]

for arch in (sys.argv[1:] or ["dense", "mla", "moe"]):
    runs = {}
    for f in glob.glob(str(GRID / f"stageb_{arch}_*.json")):
        d = json.load(open(f))
        runs[f.split(f"stageb_{arch}_")[1][:-5]] = {
            (c["rid"], c["batch"]): c for c in d["cells"] if "toks" in c}
    off = runs.pop("off")
    batches = sorted({k[1] for k in off})
    winners = {}
    S = np.full((len(RIDS), len(batches)), np.nan)
    lab = np.empty((len(RIDS), len(batches)), dtype=object)
    for (rid, b), oc in off.items():
        ranked = sorted(
            ((cells[(rid, b)]["toks"] / oc["toks"], n,
              cells[(rid, b)].get("accept"))
             for n, cells in runs.items() if (rid, b) in cells),
            reverse=True)
        if not ranked:
            continue
        s, n, a = ranked[0]
        i, j = RIDS.index(rid), batches.index(b)
        # physical sanity: decode spec speedup cannot exceed ~K+1; S>3
        # means the AR denominator was corrupted (co-tenant contention on
        # the shared box). Mark suspect; exclude from the win count.
        if s > 3.0:
            S[i, j] = np.nan
            lab[i, j] = "AR?\nsuspect"
            winners[f"{rid}/b{b}"] = {
                "winner": "SUSPECT", "S": round(s, 3),
                "reason": "AR baseline anomalously slow", "accept": a}
            continue
        if s > 1.0:
            S[i, j] = s
            lab[i, j] = f"{n.replace('_', chr(10))}\n{s:.2f}"
        else:
            S[i, j] = 1.0
            lab[i, j] = f"OFF\n({s:.2f})"
        winners[f"{rid}/b{b}"] = {
            "winner": n if s > 1.0 else "OFF", "S": round(s, 3),
            "best_spec": n, "best_spec_S": round(s, 3), "accept": a,
            "ar_toks": oc["toks"], "clip": oc.get("clip_ratio")}
    OUT.mkdir(exist_ok=True)
    (OUT / f"c1_grid_stageb_{arch}.json").write_text(json.dumps(
        {"arch": arch,
         "protocol": "Stage-B e2e serving on real datasets, uncapped "
                     "gen (MLA: ceiling 2048, base model loops at T=0 "
                     "-- disclosed); S vs same-protocol AR",
         "winners": winners}, indent=1))
    print(f"[stageb-map] wrote c1_grid_stageb_{arch}.json")

    fig, ax = style.multi_fig()
    im = ax.imshow(S, cmap="RdYlGn", vmin=0.95,
                   vmax=max(1.5, np.nanmax(S)), aspect="auto")
    for i in range(len(RIDS)):
        for j in range(len(batches)):
            if lab[i, j]:
                ax.text(j, i, lab[i, j], ha="center", va="center",
                        fontsize=4.6)
    ax.set_xticks(range(len(batches)), [f"b{b}" for b in batches],
                  fontsize=7)
    ax.set_yticks(range(len(RIDS)), RIDS, fontsize=6.5)
    ax.grid(False)
    ax.set_xlabel("batch")
    ax.set_title(f"{arch}: real-data winner per dataset x batch "
                 f"(Stage B; S vs AR)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="S")
    style.save(fig, f"c1_stageb_map_{arch}")
