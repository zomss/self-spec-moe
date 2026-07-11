#!/usr/bin/env python3
"""E4: figures for the Phase 76 lever x regime map.

fig1_cost_map        R heatmaps (lever x cell), diverging around R=1
fig2_crossovers      R vs context lines, faceted by batch (dense / MoE)
fig3_strategy_map    winner-per-cell map with composed speedup + gamma*
fig4_validation      predicted vs measured (5 e2e ground-truth cells)
fig5_composition     (R, beta) plane + iso-speedup contours: why skip dies

Colors: dataviz reference palette, validated (CVD worst-adjacent dE 47.2).
Lever hue is FIXED across all figures (color follows the entity).
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

PHASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PHASE / "scripts"))
from e2_strategy_map import BETA, best_speedup  # noqa: E402

FIG = PHASE / "figures"
FIG.mkdir(exist_ok=True)

# palette roles (validated set)
C = {
    "window": "#2a78d6",     # blue
    "weight-quant": "#1baf7a",  # aqua
    "kv-quant": "#eda100",   # yellow
    "local-route": "#4a3aa7",  # violet
    "act-quant": "#e87ba4",  # magenta
    "layer-skip": "#eb6834",  # orange
    "none": "#9b9a94",
}
INK, INK2, MUT = "#0b0b0b", "#52514e", "#8a8983"
SURF, GRID = "#fcfcfb", "#e8e7e3"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": False,
    "font.size": 9.5, "axes.titlesize": 10.5, "figure.dpi": 180,
})

ARM_LABEL = {
    "d_w4marlin": ("W4 Marlin", "weight-quant"), "d_w4machete": ("W4 Machete", "weight-quant"),
    "d_fp8w8a8": ("fp8 W8A8", "act-quant"), "d_kvq": ("KV fp8", "kv-quant"),
    "d_win": ("window 512", "window"), "d_skip50": ("skip50", "layer-skip"),
    "d_skip25": ("skip25", "layer-skip"),
    "m_fp8marlin": ("fp8 Marlin", "weight-quant"), "m_fp8block": ("fp8 native", "weight-quant"),
    "m_kvq": ("KV fp8", "kv-quant"), "m_win": ("window 512", "window"),
    "m_localroute": ("local route", "local-route"), "m_skipa2a": ("skip-a2a", "local-route"),
    "m_skip50": ("skip50", "layer-skip"),
}
CELLS = [(b, c) for b in (0, 1, 2) for c in (2, 16, 32)]  # placeholder; real per group


def load():
    R = defaultdict(dict)  # group -> arm -> {(b,c): (ratio, overpool)}
    for row in csv.DictReader((PHASE / "data/e1/summary.csv").open()):
        R[row["group"]].setdefault(row["arm"], {})[(int(row["batch"]), int(row["ctx"]))] = (
            float(row["ratio"]), int(row.get("overpool", 0)))
    return R


R = load()
DENSE_B, MOE_B, CTXS = [1, 8, 32], [4, 8, 32], [2, 16, 32]
DENSE_ARMS = ["d_win", "d_w4marlin", "d_w4machete", "d_fp8w8a8", "d_kvq", "d_skip50", "d_skip25"]
MOE_ARMS = ["m_win", "m_fp8marlin", "m_fp8block", "m_kvq", "m_localroute", "m_skipa2a", "m_skip50"]


def cell_labels(bs):
    return [f"b{b}\n{c}k" for b in bs for c in CTXS]


# ---------------------------------------------------------------- fig 1: cost map
def fig1():
    cmap = LinearSegmentedColormap.from_list("rdiv", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8c1d1d"])
    norm = TwoSlopeNorm(vmin=0.3, vcenter=1.0, vmax=1.4)
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 6.4), constrained_layout=True)
    for ax, (group, arms, bs, title) in zip(axes, [
        ("dense", DENSE_ARMS, DENSE_B, "dense Qwen2.5-7B (TP1)"),
        ("moe", MOE_ARMS, MOE_B, "MoE Qwen3-30B-A3B (attention-DP4+EP4, global batch)"),
    ]):
        cells = [(b, c) for b in bs for c in CTXS]
        M = np.full((len(arms), len(cells)), np.nan)
        OP = np.zeros_like(M, dtype=bool)
        for i, arm in enumerate(arms):
            for j, cell in enumerate(cells):
                if cell in R[group].get(arm, {}):
                    M[i, j], op = R[group][arm][cell]
                    OP[i, j] = bool(op)
        ax.imshow(np.where(OP, np.nan, M), cmap=cmap, norm=norm, aspect="auto")
        for i in range(len(arms)):
            for j in range(len(cells)):
                if np.isnan(M[i, j]):
                    ax.text(j, i, "–", ha="center", va="center", color=MUT)
                elif OP[i, j]:
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=True, fc="#f0efec",
                                           hatch="///", ec=GRID, lw=0.5))
                    ax.text(j, i, "n/a", ha="center", va="center", color=MUT, fontsize=8)
                else:
                    v = M[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                            color="#ffffff" if (v < 0.55 or v > 1.3) else INK)
        ax.set_xticks(range(len(cells)), cell_labels(bs), fontsize=8)
        ax.set_yticks(range(len(arms)), [ARM_LABEL[a][0] for a in arms], fontsize=9)
        for i, arm in enumerate(arms):
            ax.get_yticklabels()[i].set_color(C[ARM_LABEL[arm][1]])
            ax.get_yticklabels()[i].set_fontweight("bold")
        ax.set_title(title, loc="left", color=INK)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.7, pad=0.01)
    cb.set_label("R = decode-step time vs bf16  (blue = cheaper draft)", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle("E1 cost map: draft-step latency ratio R per lever and regime",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.savefig(FIG / "fig1_cost_map.png")
    plt.close(fig)


# ------------------------------------------------------------- fig 2: crossovers
def fig2():
    panels = [
        ("dense", DENSE_B, ["d_win", "d_w4marlin", "d_fp8w8a8", "d_kvq", "d_skip50"]),
        ("moe", MOE_B, ["m_win", "m_fp8marlin", "m_localroute", "m_kvq", "m_skip50"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.6), sharey=True, constrained_layout=True)
    xs = range(len(CTXS))
    handles = {}
    for r, (group, bs, arms) in enumerate(panels):
        for cidx, b in enumerate(bs):
            ax = axes[r][cidx]
            ax.axhline(1.0, color=MUT, lw=1, ls=(0, (4, 3)), zorder=1)
            for arm in arms:
                lab, fam = ARM_LABEL[arm]
                vals = [R[group].get(arm, {}).get((b, c), (np.nan, 0)) for c in CTXS]
                y = [v if not op else np.nan for v, op in vals]
                (ln,) = ax.plot(xs, y, color=C[fam], lw=2, marker="o", ms=4.5,
                                zorder=3, solid_capstyle="round")
                handles.setdefault(lab, ln)
                if cidx == len(bs) - 1:
                    valid = [i for i, v in enumerate(y) if not np.isnan(v)]
                    if valid:
                        i = valid[-1]
                        ax.annotate(lab, (xs[i], y[i]), xytext=(6, 0),
                                    textcoords="offset points", va="center",
                                    color=C[fam], fontsize=8.5, fontweight="bold")
            ax.set_xticks(list(xs), [f"{c}k" for c in CTXS])
            ax.set_title(f"{'dense' if r == 0 else 'MoE'}  b={b}", loc="left", fontsize=9.5)
            ax.grid(axis="y", color=GRID, lw=0.7)
            ax.set_ylim(0.28, 1.40)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[r][0].set_ylabel("R (lower = cheaper draft)")
    axes[0][2].annotate("bf16 over-capacity\nat 32k", (2, 1.06), ha="center",
                        fontsize=7.5, color=MUT)
    for ax in axes[1]:
        ax.set_xlabel("context")
    order = ["window 512", "W4 Marlin", "fp8 W8A8", "fp8 Marlin", "local route", "KV fp8", "skip50"]
    fig.legend([handles[k] for k in order if k in handles],
               [k for k in order if k in handles],
               loc="lower center", ncol=7, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.045))
    fig.suptitle("Crossovers: which term binds depends on (batch × context)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.get_layout_engine().set(rect=(0, 0, 0.9, 1))
    fig.savefig(FIG / "fig2_crossovers.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------- fig 3: strategy map
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), constrained_layout=True)
    for ax, (group, bs, sel, title) in zip(axes, [
        ("dense", DENSE_B, ["d_win", "d_w4marlin", "d_w4machete", "d_fp8w8a8"],
         "dense Qwen2.5-7B"),
        ("moe", MOE_B, ["m_win", "m_fp8marlin", "m_fp8block", "m_localroute"],
         "MoE Qwen3-30B-A3B"),
    ]):
        for yi, b in enumerate(bs):
            for xi, c in enumerate(CTXS):
                best = (1.0, None, 0)
                overcap = False
                for arm in sel:
                    entry = R[group].get(arm, {}).get((b, c))
                    if entry is None:
                        continue
                    ratio, op = entry
                    if op:
                        overcap = True
                        continue
                    s, g = best_speedup(BETA[arm][0], ratio)
                    if s > best[0]:
                        best = (s, arm, g)
                if best[1] is None and overcap:
                    ax.add_patch(Rectangle((xi, yi), 1, 1, fc="#f0efec", ec=SURF, lw=3, hatch="///"))
                    ax.text(xi + .5, yi + .5, "bf16\nover-capacity", ha="center", va="center",
                            fontsize=8, color=MUT)
                    continue
                if best[1] is None:
                    fam, label, txt = "none", "OFF", "no lever wins\n(≤1.0×)"
                else:
                    label, fam = ARM_LABEL[best[1]]
                    txt = f"{label}\n{best[0]:.2f}×  (γ{best[2]})"
                col = C[fam]
                ax.add_patch(Rectangle((xi, yi), 1, 1, fc=col, alpha=0.22, ec=SURF, lw=3))
                ax.text(xi + .5, yi + .58, txt.split("\n")[0], ha="center", va="center",
                        fontsize=9.5, fontweight="bold", color=col if fam != "none" else INK2)
                ax.text(xi + .5, yi + .36, txt.split("\n")[1], ha="center", va="center",
                        fontsize=8.5, color=INK2)
                # weak-win cells: best barely above 1 -> annotate as toggle-off zone
                if best[1] is not None and best[0] < 1.05:
                    ax.text(xi + .5, yi + .16, "≈ parity: leave OFF", ha="center",
                            va="center", fontsize=7.5, color=MUT)
                # S4 placement-noise artifact (contradicts P74 measured fp8 parity)
                if (group, b, c) == ("moe", 4, 32):
                    ax.text(xi + .5, yi + .16, "noisy cell — not robust (S4)",
                            ha="center", va="center", fontsize=7.5, color=MUT)
        ax.set_xlim(0, 3); ax.set_ylim(0, 3)
        ax.set_xticks([i + .5 for i in range(3)], [f"{c}k" for c in CTXS])
        ax.set_yticks([i + .5 for i in range(3)], [f"b{b}" for b in bs])
        ax.set_xlabel("context"); ax.set_ylabel("batch (global)")
        ax.set_title(title, loc="left")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Strategy map: best lever per regime  (speedup* = max_γ τ/(γ·R+1), P75-validated)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.savefig(FIG / "fig3_strategy_map.png")
    plt.close(fig)


# ------------------------------------------------------------ fig 4: validation
def fig4():
    pts = [  # (label, predicted, measured, family)
        ("dense b1/2k\nW4 γ3 (P75)", 1.24, 1.21, "weight-quant"),
        ("MoE b8/16k\nwin (P74)", 1.21, 1.16, "window"),
        ("MoE b8/32k\nwin (P74)", 1.24, 1.34, "window"),
        ("dense b32/16k\nwin K4 (E3)", 1.40, 1.54, "window"),
        ("MoE b32/32k\nwin K6 (E3)", 1.90, 1.64, "window"),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 5.2), constrained_layout=True)
    lo, hi = 1.0, 2.05
    xs = np.linspace(lo, hi, 10)
    ax.fill_between(xs, xs * 0.85, xs * 1.15, color="#f0efec", zorder=1)
    ax.plot(xs, xs, color=MUT, lw=1.2, zorder=2)
    ax.text(1.97, 1.97, "y = x", color=MUT, fontsize=8.5, va="bottom", ha="right", rotation=38)
    ax.text(1.99, 1.72, "±15%", color=MUT, fontsize=8)
    off = {0: (8, -11), 1: (8, -13), 2: (-66, 6), 3: (-72, 4), 4: (8, -4)}
    for i, (lab, p, m, fam) in enumerate(pts):
        filled = "E3" in lab
        ax.scatter(p, m, s=90, color=C[fam], zorder=4,
                   edgecolors=SURF, linewidths=1.5, marker="D" if filled else "o")
        ax.annotate(lab, (p, m), xytext=off[i], textcoords="offset points",
                    fontsize=8, color=INK2)
    ax.scatter([], [], marker="o", color=INK2, label="back-prediction (P74/P75)")
    ax.scatter([], [], marker="D", color=INK2, label="forward prediction (E3)")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("strategy-map predicted speedup")
    ax.set_ylabel("measured end-to-end speedup (real tok/s)")
    ax.grid(color=GRID, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Map vs ground truth: five e2e-measured cells",
                 loc="left", fontweight="bold", fontsize=11.5)
    fig.savefig(FIG / "fig4_validation.png")
    plt.close(fig)


# ---------------------------------------------- fig 5: (R, beta) composition plane
def fig5():
    rs = np.linspace(0.2, 1.25, 160)
    bt = np.linspace(0.40, 0.995, 160)
    RR, BB = np.meshgrid(rs, bt)
    S = np.zeros_like(RR)
    for g in range(1, 9):
        tau = (1 - BB ** (g + 1)) / (1 - BB)
        S = np.maximum(S, tau / (g * RR + 1))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharey=True, constrained_layout=True)
    panels = [
        ("dense b32/16k", "dense", (32, 16),
         [("d_win", 0.94), ("d_w4marlin", 0.91), ("d_fp8w8a8", 0.95), ("d_kvq", 0.91)],
         "d_skip50"),
        ("MoE b32/32k", "moe", (32, 32),
         [("m_win", 0.94), ("m_fp8marlin", 0.95), ("m_localroute", 0.85), ("m_kvq", 0.91)],
         "m_skip50"),
    ]
    for ax, (title, group, cell, arms, skiparm) in zip(axes, panels):
        cs = ax.contour(RR, BB, S, levels=[1.0, 1.2, 1.5, 2.0],
                        colors=[INK2, MUT, MUT, MUT], linewidths=[1.8, 1, 1, 1])
        ax.clabel(cs, fmt=lambda v: f"{v:g}×", fontsize=8, colors=INK2)
        ax.contourf(RR, BB, S, levels=[0, 1.0], colors=["#f0efec"])
        ax.text(1.1, 0.44, "speculation\nloses", color=MUT, fontsize=8.5, ha="center")
        LOFF = {"d_w4marlin": (-14, -18), "d_kvq": (9, -4), "d_fp8w8a8": (7, 7),
                "d_win": (7, 7), "m_win": (7, 7), "m_fp8marlin": (-52, 9),
                "m_kvq": (9, 5), "m_localroute": (9, 2)}
        for arm, beta in arms:
            entry = R[group].get(arm, {}).get(cell)
            if entry is None or entry[1]:
                continue
            lab, fam = ARM_LABEL[arm]
            ax.scatter(entry[0], beta, s=95, color=C[fam], zorder=5,
                       edgecolors=SURF, linewidths=1.5)
            ax.annotate(lab, (entry[0], beta), xytext=LOFF.get(arm, (7, 5)),
                        textcoords="offset points",
                        fontsize=8.5, fontweight="bold", color=C[fam])
        sk = R[group].get(skiparm, {}).get(cell)
        if sk and not sk[1]:
            lab, fam = ARM_LABEL[skiparm]
            ax.annotate("", xy=(sk[0], 0.42), xytext=(sk[0], 0.68),
                        arrowprops=dict(arrowstyle="-|>", color=C[fam], lw=2))
            ax.scatter([sk[0]], [0.68], s=70, facecolors="none", edgecolors=C[fam], linewidths=2, zorder=5)
            ax.annotate(f"{lab}: cheap but\nβ collapses (P17)", (sk[0], 0.55),
                        xytext=(8, 0), textcoords="offset points", fontsize=8, color=C[fam])
        ax.set_title(title, loc="left")
        ax.set_xlabel("R (draft-step cost ratio, measured)")
        ax.set_xlim(0.2, 1.25); ax.set_ylim(0.40, 1.0)
        ax.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("β (per-token acceptance, measured)")
    fig.suptitle("Why composition decides: iso-speedup contours in the (cost, accept) plane",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.savefig(FIG / "fig5_composition_plane.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("wrote", *sorted(p.name for p in FIG.glob("fig*.png")))
