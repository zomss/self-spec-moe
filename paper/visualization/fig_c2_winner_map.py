#!/usr/bin/env python3
"""C2 winner map, drawn to match C1's so the two can be compared.

Same axes as C1 (context x batch), same colour scale, but each cell now
names the winning COMPOSITION and the single it displaced.
"""
import json
from pathlib import Path

import numpy as np

import style

DATA = Path(__file__).resolve().parents[1] / "data"
SHORT = {"q-none": "", "w-none": "", "s-none": "",
         "q-hum": "Hum", "q-w4a16": "W4", "q-w8int8": "W8",
         "w-512": "w512", "w-2048": "w2k", "s-b2": "sk2"}


def fmt(name):
    parts = [SHORT.get(p, p) for p in name.split("_")]
    return "×".join(p for p in parts if p) or "bf16"


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
        S[i, j] = r["S_comp"]
        single = fmt(r["best_single"].rsplit("-K", 1)[0])
        kc = r["best_comp"].rsplit("-K", 1)[1]
        lab[i, j] = (f"{fmt(r['best_comp'].rsplit('-K', 1)[0])}\n"
                     f"K{kc}  {r['S_comp']:.2f}\n"
                     f"({single} {r['S_single']:.2f}, {r['gain_pct']:+.0f}%)")

    fig, ax = style.multi_fig()
    im = ax.imshow(S, cmap="RdYlGn", vmin=0.95, vmax=1.7, aspect="auto")
    for i in range(len(ctxs)):
        for j in range(len(batches)):
            if lab[i, j]:
                g = cells[f"b{batches[j]}/c{ctxs[i]}"]["gain_pct"]
                ax.text(j, i, lab[i, j], ha="center", va="center",
                        fontsize=5.0,
                        fontweight="bold" if g >= 5 else "normal")
            else:
                ax.text(j, i, "infeas.", ha="center", va="center",
                        fontsize=5.2, color="#a0aec0")
    ax.set_xticks(range(len(batches)), [f"b{b}" for b in batches], fontsize=7)
    ax.set_yticks(range(len(ctxs)), [f"{c//1000}k" for c in ctxs], fontsize=7)
    ax.grid(False)
    ax.set_xlabel("batch")
    ax.set_ylabel("context")
    ax.set_title(f"{arch}: C2 winning COMPOSITION per cell "
                 f"(displaced single in parentheses)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="S vs AR")
    style.save(fig, f"c2_winner_map_{arch}")
