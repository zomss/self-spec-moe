#!/usr/bin/env python3
"""C1 regime axis (multi-column, 3:1): best AR-relative speedup per
canonical regime at 8B and 32B, winning lever/K annotated, naive-grid
losses marked. Data: paper/data/c1_regime_winners.json."""
import json
from pathlib import Path

import numpy as np

import style
from style import COLORS

data = json.loads(
    (Path(__file__).resolve().parents[1] / "data" /
     "c1_regime_winners.json").read_text())
regs = data["regimes"]

labels = [f"{r['regime']}\n{r['shape']}" for r in regs]
s8 = [r["best_8b"]["speedup"] for r in regs]
s32 = [r["best_32b"]["speedup"] for r in regs]
k8 = [f"K{r['best_8b']['K']}" for r in regs]
l32 = [("OFF" if r["best_32b"]["lever"] == "OFF" else
        f"{r['best_32b']['lever']} K{r['best_32b']['K']}") for r in regs]

x = np.arange(len(regs))
w = 0.38
fig, ax = style.multi_fig()
b1 = ax.bar(x - w / 2, s8, w, color=COLORS["primary"], label="Qwen3-8B")
b2 = ax.bar(x + w / 2, s32, w, color=COLORS["secondary"],
            label="Qwen3-32B (TP2)")
ax.axhline(1.0, color=COLORS["bad"], ls="--", lw=1)
ax.text(len(regs) - 0.45, 1.015, "AR", color=COLORS["bad"], fontsize=7)

for xi, (v, k) in enumerate(zip(s8, k8)):
    ax.text(xi - w / 2, v + 0.02, k, ha="center", fontsize=6,
            color=COLORS["primary"])
for xi, (v, lv) in enumerate(zip(s32, l32)):
    ax.text(xi + w / 2, v + 0.02, lv.replace(" ", "\n"), ha="center",
            fontsize=5.2, color=COLORS["secondary"])
# naive-grid losses the framework converted
for xi, r in enumerate(regs):
    if "naive_grid_K4" in r["best_8b"]:
        ax.scatter([xi - w / 2], [r["best_8b"]["naive_grid_K4"]],
                   marker="x", s=18, color=COLORS["bad"], zorder=3)
ax.scatter([], [], marker="x", s=18, color=COLORS["bad"],
           label="naive K4 grid (loss)")

ax.set_xticks(x, labels, fontsize=6.2)
ax.set_ylabel("speedup vs AR")
ax.set_ylim(0.75, 1.95)
ax.legend(loc="upper right", ncols=3)
ax.set_title("No universal draft: 9 regimes need K2–K6 + OFF; "
             "winners split across kernel × depth × composition at 32B")
style.save(fig, "c1_regimes")
