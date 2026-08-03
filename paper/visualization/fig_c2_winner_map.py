#!/usr/bin/env python3
"""C2 winner map, drawn to match C1's so the two can be compared.

Same axes as C1 (context x batch), same colour scale, but each cell now
names the winning COMPOSITION and the single it displaced.

Content validation: a b1 cell is measured from ONE document, and
narrow-window (512) acceptance is content-sensitive. Every b1 cell
whose winner carries a 512 window -- plus b1/c8000, whose apparent
anomaly triggered the check -- was re-measured on 3 extra document
draws. Cells whose margin does not survive are drawn hatched and
labelled with the multi-draw mean +- s.e.m. instead of the single-draw
gain. See paper/data/c2_map_b1_8k_validation.json.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

import style

DATA = Path(__file__).resolve().parents[1] / "data"
SHORT = {"q-none": "", "w-none": "", "s-none": "",
         "q-hum": "Hum", "q-w4a16": "W4", "q-w8int8": "W8",
         "w-512": "w512", "w-2048": "w2k", "s-b2": "sk2"}


def fmt(name):
    parts = [SHORT.get(p, p) for p in name.split("_")]
    return "×".join(p for p in parts if p) or "bf16"


def load_validation():
    """{(arch, cell): {mean, sem, robust}} from the content re-draws."""
    out = {}
    for fn in ("c2_map_b1_8k_validation.json", "c2_map_exposed_validation.json"):
        p = DATA / fn
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for arch, v in d["per_arch"].items():
            out[(arch, v.get("cell", d.get("cell")))] = v
    return out


VALID = load_validation()

for arch in ("dense", "llama"):
    f = DATA / f"c2_oracle_{arch}.json"
    if not f.exists():
        continue
    rows = json.loads(f.read_text())["cells"]
    cells = {r["cell"]: r for r in rows}
    batches = sorted({int(c.split("/")[0][1:]) for c in cells})
    ctxs = sorted({int(c.split("/c")[1]) for c in cells})

    S = np.full((len(ctxs), len(batches)), np.nan)
    lab = np.empty((len(ctxs), len(batches)), dtype=object)
    for key, r in cells.items():
        b = int(key.split("/")[0][1:])
        c = int(key.split("/c")[1])
        i, j = ctxs.index(c), batches.index(b)
        # the search selects over the UNION of singles and compositions;
        # a single legitimately wins some cells.
        comp_wins = r["S_comp"] >= r["S_single"]
        win = r["best_comp"] if comp_wins else r["best_single"]
        S[i, j] = max(r["S_comp"], r["S_single"])
        other = r["best_single"] if comp_wins else r["best_comp"]
        oth_s = r["S_single"] if comp_wins else r["S_comp"]
        kind = "composition" if comp_wins else "SINGLE"
        v = VALID.get((arch, key))
        if v:
            # re-measured on extra document draws: report the mean margin
            gain_txt = (f"{v['mean_gain_pct']:+.1f}±{v['sem']:.1f}%"
                        f" n={v['n_draws']}")
            if not v["robust"]:
                kind = "TIE (content)"
        else:
            gain_txt = f"{r['gain_pct']:+.0f}%"
        lab[i, j] = (f"{fmt(win.rsplit('-K', 1)[0])} K{win.rsplit('-K', 1)[1]}"
                     f"  {S[i, j]:.2f}\n[{kind}]\n"
                     f"(next: {fmt(other.rsplit('-K', 1)[0])} {oth_s:.2f},"
                     f" {gain_txt})")

    fig, ax = style.multi_fig()
    im = ax.imshow(S, cmap="RdYlGn", vmin=0.95, vmax=1.7, aspect="auto")
    for i in range(len(ctxs)):
        for j in range(len(batches)):
            if lab[i, j]:
                key = f"b{batches[j]}/c{ctxs[i]}"
                v = VALID.get((arch, key))
                g = v["mean_gain_pct"] if v else cells[key]["gain_pct"]
                if v and not v["robust"]:
                    # margin did not survive re-drawing the documents
                    ax.add_patch(matplotlib.patches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///",
                        edgecolor="#4a5568", linewidth=0, zorder=1))
                ax.text(j, i, lab[i, j], ha="center", va="center",
                        fontsize=5.0, zorder=2,
                        fontweight="bold" if g >= 5 else "normal")
            else:
                ax.text(j, i, "infeas.", ha="center", va="center",
                        fontsize=5.2, color="#a0aec0")
    ax.set_xticks(range(len(batches)), [f"b{b}" for b in batches], fontsize=7)
    ax.set_yticks(range(len(ctxs)), [f"{c//1000}k" for c in ctxs], fontsize=7)
    ax.grid(False)
    ax.set_xlabel("batch")
    ax.set_ylabel("context")
    nc = sum(1 for r in rows if r["S_comp"] >= r["S_single"])
    nv = sum(1 for k in VALID if k[0] == arch and not VALID[k]["robust"])
    ax.set_title(f"{arch}: winner per cell, singles + compositions  "
                 f"({nc}/{len(rows)} composition; hatched = within "
                 f"content noise, n={nv})", fontsize=7.5)
    fig.colorbar(im, ax=ax, shrink=0.85, label="S vs AR")
    style.save(fig, f"c2_winner_map_{arch}")
