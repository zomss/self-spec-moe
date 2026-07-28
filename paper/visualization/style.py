"""Unified figure style for all paper visualizations.

Every figure script imports this module and uses:
  - single_fig() / multi_fig(): canvas at the mandated aspect ratios
      single column -> width:height = 1.5:1  (3.5 x 2.333 in)
      multi  column -> width:height = 3:1    (7.0 x 2.333 in)
  - COLORS: the shared palette (colorblind-safe anchors)
  - save(fig, name): writes PNG(300dpi)+PDF to paper/figures/
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = Path(__file__).resolve().parents[1] / "figures"

SINGLE = (3.5, 3.5 / 1.5)   # 1.5 : 1
MULTI = (7.0, 7.0 / 3.0)    # 3 : 1

COLORS = {
    "primary": "#2b6cb0",    # blue   — our system / 8B / main series
    "secondary": "#c05621",  # orange — comparison / 32B / second series
    "accent": "#805ad5",     # purple — third series / annotations
    "good": "#2f855a",       # green  — wins
    "bad": "#b91c1c",        # red    — losses / thresholds
    "neutral": "#4a5568",    # gray   — baselines / context
    "light": "#cbd5e0",      # light gray — fills / grids
}

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e2e8f0",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
})


def single_fig(**kw):
    return plt.subplots(figsize=SINGLE, **kw)


def multi_fig(**kw):
    return plt.subplots(figsize=MULTI, **kw)


def save(fig, name):
    """Save to paper/figures/<name>.{png,pdf}; name without extension."""
    FIGDIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=300)
    print(f"wrote {FIGDIR}/{name}.png/.pdf")
