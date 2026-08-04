#!/usr/bin/env python3
"""No universal search heuristic: ranker regret across four architectures.

Left: per-architecture regret for the three rules fed identical measured
inputs -- each wins somewhere, none wins more than two.
Right: worst case across architectures, which is what a deployed default
has to survive. Plain `ours` wins it, so neither union is justified.
"""
import json
from pathlib import Path

import numpy as np

import style

DATA = Path(__file__).resolve().parents[1] / "data"
RULES = [("ours", style.COLORS["primary"]),
         ("knapspec", style.COLORS["secondary"]),
         ("product_rank", style.COLORS["accent"])]
ARCHES = ["dense", "llama", "mla", "moe"]

m = json.loads((DATA / "c2_mechanism_4arch.json").read_text())
key = {a: next(k for k in m if k.lower() == a or k.lower().startswith(a[:3]))
       for a in ARCHES}

fig, (ax, ax2) = style.multi_fig(ncols=2, gridspec_kw={"width_ratios": [2.2, 1]})

x = np.arange(len(ARCHES))
w = 0.26
for i, (rule, c) in enumerate(RULES):
    vals = [m[key[a]]["regret"][rule] for a in ARCHES]
    b = ax.bar(x + (i - 1) * w, vals, w, color=c,
               label=rule.replace("_rank", ""))
    # mark the winner per architecture
    for j, a in enumerate(ARCHES):
        if m[key[a]]["best"] == rule:
            ax.text(j + (i - 1) * w, vals[j] + 0.15, "*", ha="center",
                    fontsize=9, color=c, fontweight="bold")
ax.set_xticks(x, [a.upper() if a in ("mla", "moe") else a for a in ARCHES])
ax.set_ylabel("regret vs oracle (%)")
ax.set_title("each rule wins somewhere (* = best); none wins more than two")
ax.legend(ncol=3, loc="upper left")
ax.set_ylim(0, 11)

# worst case across architectures -- what a default must survive
u = json.loads((DATA / "c2_union3.json").read_text())
cand = {"ours": None, "knapspec": None, "product_rank": None,
        "union2": None, "union3": None}
for k in cand:
    cand[k] = max(v[k] for v in u.values())
names = sorted(cand, key=cand.get)
cols = [style.COLORS["good"] if n == names[0] else style.COLORS["light"]
        for n in names]
ax2.barh(range(len(names)), [cand[n] for n in names], color=cols)
ax2.set_yticks(range(len(names)),
               [n.replace("_rank", "") for n in names], fontsize=6.5)
ax2.invert_yaxis()
ax2.set_xlabel("worst-case regret (%)")
ax2.set_title("worst case over 4 arches")
for i, n in enumerate(names):
    ax2.text(cand[n] + 0.15, i, f"{cand[n]:.2f}", va="center", fontsize=6)
ax2.set_xlim(0, max(cand.values()) * 1.35)

style.save(fig, "c2_ranker_4arch")
print("wrote paper/figures/c2_ranker_4arch.{png,pdf}")
