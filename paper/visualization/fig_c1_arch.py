#!/usr/bin/env python3
"""C1 architecture axis (single-column, 1.5:1): best-lever vs
wrong-lever e2e speedup per architecture — the 30-60% wrong-lever
cost. Data: paper/data/c1_arch_ledger.json."""
import json
from pathlib import Path

import numpy as np

import style
from style import COLORS

d = json.loads((Path(__file__).resolve().parents[1] / "data" /
                "c1_arch_ledger.json").read_text())

# (arch, selected config/speedup, wrong config/speedup) — all measured:
# dense = Q3-8B summarization cell (T8: K2 wins where naive K4 loses);
# MoE/MLA = T2 ledger rows.
cases = [
    ("dense Q3-8B\n(summ. b8)", ("Hum K2", 1.08), ("naive K4", 0.84)),
    ("MoE\nQ3-30B-A3B", ("win512 K3", 1.15), ("dense-chain flr50", 0.63)),
    ("MLA\nV2-Lite", ("OFF", 1.00), ("self-draft K5", 0.56)),
]

x = np.arange(len(cases))
w = 0.38
fig, ax = style.single_fig()
ax.bar(x - w / 2, [c[1][1] for c in cases], w, color=COLORS["good"],
       label="selected lever")
ax.bar(x + w / 2, [c[2][1] for c in cases], w, color=COLORS["bad"],
       label="wrong lever")
ax.axhline(1.0, color=COLORS["neutral"], ls="--", lw=1)
for xi, c in enumerate(cases):
    ax.text(xi - w / 2, c[1][1] + 0.03, c[1][0], ha="center", fontsize=5.5,
            color=COLORS["good"])
    ax.text(xi + w / 2, c[2][1] + 0.03, c[2][0], ha="center", fontsize=5.5,
            color=COLORS["bad"])
ax.set_xticks(x, [c[0] for c in cases], fontsize=7)
ax.set_ylabel("speedup vs AR")
ax.set_ylim(0, 2.15)
ax.legend(loc="upper right")
ax.set_title("Wrong lever costs 30–60%\n(same lever family across architectures)")
style.save(fig, "c1_arch")
