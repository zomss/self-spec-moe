#!/usr/bin/env python3
"""C1 acceptance-surface diversity (single-column, 1.5:1): beta of each
single lever across the five model columns — no lever ranks the same
everywhere. Data: paper/data/c1_beta_singles.json."""
import json
from pathlib import Path

import numpy as np

import style

d = json.loads((Path(__file__).resolve().parents[1] / "data" /
                "c1_beta_singles.json").read_text())["beta_singles_16k"]

models = ["Q2.5-7B", "Q3-8B", "Q3-32B", "Q3-30B\nMoE", "V2-Lite\nMLA"]
show = ["win512", "win128", "kvq_fp8", "q_fp8_Wonly", "q_int4_RTN",
        "skip125_contig"]
labels = ["win512", "win128", "kvq fp8", "W fp8", "W int4", "skip 12.5%"]
M = np.array([[v if v is not None else np.nan
               for v in d["levers"][k]] for k in show])

fig, ax = style.single_fig()
im = ax.imshow(M, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if not np.isnan(M[i, j]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=6,
                    color="black" if M[i, j] > 0.55 else "white")
ax.set_xticks(range(len(models)), models, fontsize=6.5)
ax.set_yticks(range(len(labels)), labels, fontsize=6.5)
ax.grid(False)
ax.set_title("Acceptance factor β per lever × model:\nno lever ranks the same everywhere")
fig.colorbar(im, ax=ax, shrink=0.85, label="β")
style.save(fig, "c1_beta_heatmap")
