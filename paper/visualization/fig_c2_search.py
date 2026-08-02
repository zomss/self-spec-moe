#!/usr/bin/env python3
"""C2 figures: (a) regret vs budget for the search against the oracle
(single column, 1.5:1); (b) composition gain per cell (multi, 3:1)."""
import json
from pathlib import Path

import numpy as np

import style
from style import COLORS

DATA = Path(__file__).resolve().parents[1] / "data"

# ---------------- (a) regret vs budget ----------------
fig, ax = style.single_fig()
marks = {"dense": "o", "llama": "s"}
for arch, col in (("dense", COLORS["primary"]), ("llama", COLORS["secondary"])):
    f = DATA / f"c2_search_{arch}.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text())
    x = [r["budget_frac_pct"] for r in d["curve"]]
    y = [r["mean_regret_pct"] for r in d["curve"]]
    ax.plot(x, y, marks[arch] + "-", color=col, lw=1.8, ms=4,
            label=f"{arch}: search")
    ax.axhline(d["best_single_regret_pct"], color=col, ls=":", lw=1.2)
    ax.text(62, d["best_single_regret_pct"] + 0.3,
            f"{arch} best-single-only", color=col, fontsize=5.5, ha="right")
    xr = [r["budget_frac_pct"] for r in d["curve"]]
    yr = [r["random_mean_regret_pct"] for r in d["curve"]]
    ax.plot(xr, yr, marks[arch] + "--", color=COLORS["neutral"], lw=1,
            ms=3, alpha=0.7,
            label="random @ equal budget" if arch == "dense" else None)
ax.set_xlabel("measurement budget (% of exhaustive boots)")
ax.set_ylabel("mean regret vs oracle (%)")
ax.set_title("Search reaches the optimum at a fraction of\nexhaustive cost")
ax.legend(loc="upper right", fontsize=5.5)
ax.set_ylim(bottom=-0.5)
style.save(fig, "c2_regret_vs_budget")

# ---------------- (b) composition gain per cell ----------------
fig, ax = style.multi_fig()
archs = [a for a in ("dense", "llama") if (DATA / f"c2_oracle_{a}.json").exists()]
allcells = []
for a in archs:
    allcells += [r["cell"] for r in json.loads(
        (DATA / f"c2_oracle_{a}.json").read_text())["cells"]]
cells = sorted(set(allcells), key=lambda c: (int(c.split("/c")[1]),
                                             int(c.split("/")[0][1:])))
x = np.arange(len(cells))
w = 0.38
for i, (a, col) in enumerate(zip(archs, (COLORS["primary"], COLORS["secondary"]))):
    d = {r["cell"]: r for r in json.loads(
        (DATA / f"c2_oracle_{a}.json").read_text())["cells"]}
    vals = [d[c]["gain_pct"] if c in d else np.nan for c in cells]
    ax.bar(x + (i - 0.5) * w, vals, w, color=col, label=a)
ax.axhline(0, color=COLORS["neutral"], lw=1)
ax.axhline(5, color=COLORS["bad"], ls="--", lw=1)
ax.text(len(cells) - 0.4, 5.4, "+5%", color=COLORS["bad"], fontsize=6, ha="right")
ax.set_xticks(x, cells, fontsize=6)
ax.set_ylabel("best composition vs\nbest single (%)")
ax.set_title("Composition pays where two cost terms bind "
             "(mid-batch x long context), not elsewhere")
ax.legend(loc="upper left")
style.save(fig, "c2_composition_gain")
