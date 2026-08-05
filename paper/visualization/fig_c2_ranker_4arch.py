#!/usr/bin/env python3
"""No universal search heuristic: ranker regret across four architectures,
judged against the 2-SAMPLE truth (c2_truth_4arch.json).

Left: per-architecture mean regret for the three rules fed identical
measured inputs; whiskers span the two pick-directions (search derived
from sample R1 vs from R2) — the content-luck band. Each rule wins
somewhere; no rule wins more than two; the dense winner is a 0.01pp tie.
Right: per-architecture rule separation (max-min of the rule means)
against the content spread (mean |from_R1 - from_R2| over rules) — rule
choice is second-order wherever the first bar is below the second.
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

t = json.loads((DATA / "c2_truth_4arch.json").read_text())

fig, (ax, ax2) = style.multi_fig(ncols=2, gridspec_kw={"width_ratios": [2.2, 1]})

x = np.arange(len(ARCHES))
w = 0.26
for i, (rule, c) in enumerate(RULES):
    means = [t[a][rule]["mean"] for a in ARCHES]
    lo = [t[a][rule]["mean"] - min(t[a][rule]["from_R1"], t[a][rule]["from_R2"])
          for a in ARCHES]
    hi = [max(t[a][rule]["from_R1"], t[a][rule]["from_R2"]) - t[a][rule]["mean"]
          for a in ARCHES]
    ax.bar(x + (i - 1) * w, means, w, color=c, yerr=[lo, hi],
           error_kw={"lw": 0.7, "capsize": 1.5, "ecolor": "#666"},
           label=rule.replace("_rank", ""))
    for j, a in enumerate(ARCHES):
        if t[a]["best_rule_mean"] == rule:
            ax.text(j + (i - 1) * w, means[j] + hi[j] + 0.25, "*", ha="center",
                    fontsize=9, color=c, fontweight="bold")
ax.set_xticks(x, [a.upper() if a in ("mla", "moe") else a for a in ARCHES])
ax.set_ylabel("regret vs 2-sample truth (%)")
ax.set_title("each rule wins somewhere (* = best mean);\n"
             "whiskers: pick-from-R1 vs pick-from-R2")
ax.legend(ncol=3, loc="upper left")
ax.set_ylim(0, 8.5)

rule_sep = [max(t[a][r]["mean"] for r, _ in RULES)
            - min(t[a][r]["mean"] for r, _ in RULES) for a in ARCHES]
content = [np.mean([abs(t[a][r]["from_R1"] - t[a][r]["from_R2"])
                    for r, _ in RULES]) for a in ARCHES]
xb = np.arange(len(ARCHES))
ax2.bar(xb - 0.18, rule_sep, 0.36, color=style.COLORS["primary"],
        label="rule separation")
ax2.bar(xb + 0.18, content, 0.36, color=style.COLORS["light"],
        label="content spread")
ax2.set_xticks(xb, [a.upper() if a in ("mla", "moe") else a for a in ARCHES],
               fontsize=6.5)
ax2.set_ylabel("percentage points")
ax2.set_title("rule choice vs content noise")
ax2.legend(fontsize=5.5, loc="upper left")

style.save(fig, "c2_ranker_4arch")
print("wrote paper/figures/c2_ranker_4arch.{png,pdf}")
