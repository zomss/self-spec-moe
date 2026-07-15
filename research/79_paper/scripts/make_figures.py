#!/usr/bin/env python3
"""Paper figures (fig1-fig8) -> research/79_paper/figures/.

Reuses 76/e4_figures.py style (validated palette, fixed lever hues).
Data: 76 summary.csv (R, incl. MLA tier), 77 beta.csv, 80 beta_combo.csv,
79 beta_menu_ext.csv + recorded tables (v4 winners, e2e cells, 81 anatomy)
hardcoded from the committed results docs where a single number is cleaner
than a loader.
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

PHASE = Path(__file__).resolve().parents[1]
P76 = PHASE.parent / "76_lever_latency_sweep"
P77 = PHASE.parent / "77_acceptance_map"
P80 = PHASE.parent / "80_lever_composition"
sys.path.insert(0, str(P76 / "scripts"))
import e4_figures as E4  # noqa: E402  (style + loaders; module-level rcParams)
from e2_strategy_map import best_speedup, tau  # noqa: E402

FIG = PHASE / "figures"
FIG.mkdir(exist_ok=True)
C, INK, INK2, MUT, GRID = E4.C, E4.INK, E4.INK2, E4.MUT, E4.GRID
R = E4.R
CTXS = [2, 16, 32]

BETA_S = {}
for r in csv.DictReader((P77 / "data/beta.csv").open()):
    BETA_S[(r["model"], r["arm"], int(r["ctx"]))] = float(r["beta_greedy"])
BETA_C = {}
for r in csv.DictReader((P80 / "data/beta_combo.csv").open()):
    key = (r["model"], "+".join(sorted(r["combo"].split("+"))), int(r["ctx"]))
    BETA_C[key] = float(r["beta_greedy"])
MENU = {}
for r in csv.DictReader((PHASE / "data/beta_menu_ext.csv").open()):
    MENU[r["arm"]] = float(r["beta_greedy"])


# ---------------------------------------------------------------- fig1: 3-arch cost map
def fig1():
    cmap = LinearSegmentedColormap.from_list(
        "rdiv", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8c1d1d"])
    norm = TwoSlopeNorm(vmin=0.3, vcenter=1.0, vmax=1.4)
    MLA_ARMS = ["ds_win", "ds_fp8marlin", "ds_fp8block", "ds_kvq", "ds_skip50"]
    MLA_LBL = {"ds_win": ("window 512", "window"),
               "ds_fp8marlin": ("fp8 Marlin", "weight-quant"),
               "ds_fp8block": ("fp8 native", "weight-quant"),
               "ds_kvq": ("KV fp8", "kv-quant"),
               "ds_skip50": ("skip50", "layer-skip")}
    panels = [("dense", E4.DENSE_ARMS, E4.DENSE_B, E4.ARM_LABEL,
               "dense Qwen2.5-7B (TP1)"),
              ("moe", E4.MOE_ARMS, E4.MOE_B, E4.ARM_LABEL,
               "MoE Qwen3-30B-A3B (attention-DP4+EP4)"),
              ("mla", MLA_ARMS, [4, 8, 32], MLA_LBL,
               "MLA DeepSeek-V2-Lite (one-anchor tier)")]
    hr = [len(p[1]) for p in panels]
    fig, axes = plt.subplots(3, 1, figsize=(9.6, 9.2), constrained_layout=True,
                             gridspec_kw=dict(height_ratios=hr))
    for ax, (group, arms, bs, lbl, title) in zip(axes, panels):
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
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=True,
                                           fc="#f0efec", hatch="///", ec=GRID, lw=0.5))
                    ax.text(j, i, "n/a", ha="center", va="center", color=MUT, fontsize=8)
                else:
                    v = M[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="#ffffff" if (v < 0.55 or v > 1.3) else INK)
        ax.set_xticks(range(len(cells)),
                      [f"b{b}\n{c}k" for b in bs for c in CTXS], fontsize=8)
        ax.set_yticks(range(len(arms)), [lbl[a][0] for a in arms], fontsize=9)
        for i, arm in enumerate(arms):
            ax.get_yticklabels()[i].set_color(C[lbl[arm][1]])
            ax.get_yticklabels()[i].set_fontweight("bold")
        ax.set_title(title, loc="left", color=INK)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                      shrink=0.5, pad=0.01)
    cb.set_label("R = draft decode-step time / bf16 (blue = cheaper)", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle("Fig 1 — the cost map: R per lever, regime, architecture",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.savefig(FIG / "fig1_cost_map.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig3: strategy map v4
V4 = {  # (group,(b,ck)): (winner, speedup, gamma)  -- strategy_map_v4.md
    ("dense", (1, 2)): ("q_int4", 1.31, 3), ("dense", (1, 16)): ("q_int4", 1.32, 3),
    ("dense", (1, 32)): ("q_int4+win512", 1.31, 3),
    ("dense", (8, 2)): ("q_int4", 1.32, 3), ("dense", (8, 16)): ("q_int4+win512", 1.55, 4),
    ("dense", (8, 32)): ("q_int4+win512", 1.61, 5),
    ("dense", (32, 2)): ("q_int4+win128", 1.29, 3),
    ("dense", (32, 16)): ("q_int4+win512", 1.91, 6), ("dense", (32, 32)): None,
    # v5 (Phase 83): the 2k band + b4/16k flip to profiled expert selection
    ("moe", (4, 2)): ("flr50+q_fp8", 1.12, 2), ("moe", (4, 16)): ("flr50+q_fp8+win512", 1.11, 2),
    ("moe", (4, 32)): ("win128", 1.58, 8),
    ("moe", (8, 2)): ("flr50+q_fp8", 1.13, 2), ("moe", (8, 16)): ("win512", 1.27, 5),
    ("moe", (8, 32)): ("win512", 1.37, 7),
    ("moe", (32, 2)): ("flr50+q_fp8", 1.15, 2), ("moe", (32, 16)): ("win128", 1.44, 6),
    ("moe", (32, 32)): ("win512", 2.22, 8),
    ("mla", (4, 2)): ("OFF", 1.08, 6), ("mla", (4, 16)): ("OFF", 1.02, 1),
    ("mla", (4, 32)): ("OFF", 1.07, 5),
    ("mla", (8, 2)): ("OFF", 1.02, 1), ("mla", (8, 16)): ("OFF", 1.03, 1),
    ("mla", (8, 32)): ("OFF", 1.05, 1),
    ("mla", (32, 2)): ("OFF", 1.07, 5), ("mla", (32, 16)): ("OFF", 1.13, 7),
    ("mla", (32, 32)): ("OFF", 1.04, 4),
}
FAM = {"q_int4": "weight-quant", "q_fp8": "weight-quant", "win512": "window",
       "win128": "window", "flr50": "local-route", "OFF": "none"}


def fig3():
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4), constrained_layout=True)
    for ax, (group, bs, title) in zip(axes, [
            ("dense", [1, 8, 32], "dense"), ("moe", [4, 8, 32], "MoE"),
            ("mla", [4, 8, 32], "MLA")]):
        for i, b in enumerate(bs):
            for j, ck in enumerate(CTXS):
                ent = V4.get((group, (b, ck)))
                if ent is None:
                    ax.add_patch(Rectangle((j, i), 1, 1, fc="#f0efec",
                                           hatch="///", ec="white", lw=2))
                    ax.text(j + .5, i + .5, "over-\ncapacity", ha="center",
                            va="center", fontsize=7, color=MUT)
                    continue
                win, sp, g = ent
                fam = FAM.get(win.split("+")[0], "none")
                shade = min(1.0, 0.25 + (sp - 1.0))
                base = C[fam]
                ax.add_patch(Rectangle((j, i), 1, 1, fc=base, alpha=shade,
                                       ec="white", lw=2))
                lbl = (win.replace("q_int4", "W4").replace("flr50", "freq-lr")
                       .replace("+", "\n+"))
                dark = sp > 1.5
                ax.text(j + .5, i + .58, lbl, ha="center", va="center",
                        fontsize=7.5, fontweight="bold",
                        color="#ffffff" if dark else INK)
                ax.text(j + .5, i + .24, f"{sp:.2f}×  γ{g}", ha="center",
                        va="center", fontsize=7,
                        color="#ffffff" if dark else INK2)
        ax.set_xlim(0, 3)
        ax.set_ylim(3, 0)
        ax.set_xticks([j + .5 for j in range(3)], [f"{c}k" for c in CTXS])
        ax.set_yticks([i + .5 for i in range(3)], [f"b{b}" for b in bs])
        ax.set_title(title, loc="left", color=INK)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Fig 3 — strategy map v5 (argmax over the full space, profiled lever forms)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.savefig(FIG / "fig3_strategy_map.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig4: validation scatter
def fig4():
    # (predicted, measured, family) from the committed records
    pts = {
        "V1/V2 cost model (R)": ([0.95, 0.75, 0.63, 0.54, 0.97, 0.68, 0.65],
                                 [0.949, 0.749, 0.626, 0.541, 0.965, 0.677, 0.647],
                                 C["window"], "o"),
        "combo R (term edits)": ([0.799, 0.521, 0.505, 0.353],
                                 [0.772, 0.548, 0.435, 0.311],
                                 C["weight-quant"], "s"),
        "e2e speedup (5 + 81-E3 cells)": (
            [1.55, 1.91, 1.55, 1.54, 1.25, 1.91, 1.85, 1.64, 1.52],
            [1.41, 1.48, 1.64, 1.54, 1.21, 1.91, 1.85, 1.64, 1.52],
            C["layer-skip"], "D"),
    }
    fig, ax = plt.subplots(figsize=(4.9, 4.6), constrained_layout=True)
    lim = [0.25, 2.35]
    ax.plot(lim, lim, color=GRID, lw=1, zorder=0)
    ax.fill_between(lim, [x * 0.9 for x in lim], [x * 1.1 for x in lim],
                    color=GRID, alpha=0.4, zorder=0, label="±10%")
    for name, (px, mx, col, mk) in pts.items():
        ax.scatter(px, mx, s=42, color=col, marker=mk, label=name,
                   ec="white", lw=0.7, zorder=3)
    ax.annotate("broken-chain cells\n(delivery, fixed in §8)",
                xy=(1.91, 1.48), xytext=(1.32, 2.1), fontsize=7.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUT, lw=0.8))
    ax.set_xlabel("predicted (registered before measurement)")
    ax.set_ylabel("measured")
    ax.set_title("Fig 4 — the validation ladder, predicted vs measured",
                 loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(FIG / "fig4_validation.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig6: portability slopegraph
def fig6():
    levers = [("q_fp8", "fp8 weight", "weight-quant"),
              ("q_int4", "int4 weight", "weight-quant"),
              ("win512", "window 512", "window"),
              ("kvq_fp8", "KV fp8", "kv-quant"),
              ("skip125", "skip 12.5%", "layer-skip"),
              ("skip50", "skip 50%", "layer-skip")]
    xs = [0, 1, 2]
    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    ends = []
    for arm, lbl, fam in levers:
        ys = [BETA_S.get((m, arm, 16384)) for m in ("dense", "moe", "mla")]
        ls = "-" if arm.startswith("q") else "--"
        ax.plot(xs, ys, ls, color=C[fam], lw=2, marker="o", ms=5,
                mec="white", mew=0.8)
        ends.append((ys[2], lbl, C[fam]))
    # stagger right-edge labels (min gap 0.05, top-down)
    ends.sort(reverse=True)
    ly = None
    for yv, lbl, col in ends:
        ly = yv if ly is None else min(yv, ly - 0.05)
        ax.annotate(lbl, (2.07, ly), fontsize=8, color=col, va="center")
    ax.set_xticks(xs, ["dense\n(no KV norm)", "MoE\n(QK-norm)", "MLA\n(kv_a_layernorm)"])
    ax.set_ylim(-0.03, 1.06)
    ax.set_xlim(-0.15, 2.75)
    ax.set_ylabel("β (per-token acceptance, 16k)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Fig 6 — β portability: only fp8 weight-quant is flat",
                 loc="left", color=INK)
    fig.savefig(FIG / "fig6_portability.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig7: composition law
def fig7():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3), constrained_layout=True)
    # (a) across levers: measured combo beta vs product of singles
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, zorder=0)
    seen_lbl = set()
    for (model, combo, ctx), meas in BETA_C.items():
        parts = combo.split("+")
        prod = 1.0
        ok = True
        for p in parts:
            v = BETA_S.get((model, p, ctx))
            if v is None:
                ok = False
                break
            prod *= v
        if not ok:
            continue
        rescue = model == "dense" and "kvq_fp8" in parts and any(
            p.startswith("win") for p in parts)
        destr = model == "mla" and ctx == 32768 and any(
            p.startswith("skip") for p in parts) and len(parts) > 1
        col = (C["kv-quant"] if rescue else
               C["layer-skip"] if destr else MUT)
        lbl = ("window rescues kvq (dense)" if rescue else
               "MLA skip×ctx destructive" if destr else "tracks product")
        ax.scatter(prod, meas, s=30, color=col, ec="white", lw=0.6, zorder=3,
                   label=lbl if lbl not in seen_lbl else None)
        seen_lbl.add(lbl)
    ax.set_xlabel("Π βᵢ (product of measured singles)")
    ax.set_ylabel("measured combo β")
    ax.set_title("(a) across levers: β_combo ≈ Πβᵢ (48 cells)", loc="left",
                 color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    # (b) within the skip lever: depth breaks the product
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, zorder=0)
    singles = {int(k[3:]): v for k, v in MENU.items()
               if k.startswith("ls_") and k.count("_") == 1}
    for arm, meas in MENU.items():
        if not arm.startswith(("ls_g_", "ls_na_", "ls_p_")):
            continue
        ids = [int(x) for x in arm.split("_")[-1].split("-")]
        prod = float(np.prod([singles[i] for i in ids]))
        depth = len(ids)
        col = {3: C["window"], 5: C["weight-quant"], 7: C["layer-skip"]}[depth]
        ax.scatter(prod, meas, s=34, color=col, ec="white", lw=0.6, zorder=3)
    for d, col in [(3, C["window"]), (5, C["weight-quant"]), (7, C["layer-skip"])]:
        ax.scatter([], [], s=34, color=col, label=f"budget {d}")
    ax.set_xlabel("Π βᵢ (leave-one-out singles)")
    ax.set_ylabel("measured set β")
    ax.set_title("(b) within layer-skip: DEPTH breaks the product", loc="left",
                 color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Fig 7 — the composition law and its measured limits",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.savefig(FIG / "fig7_composition_law.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig8: floor-free chain
def fig8():
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), constrained_layout=True)
    # (a) chain-step anatomy
    ax = axes[0]
    labels = ["eager chain\n(piecewise)", "fixed chain\n(scratchpad FA3)"]
    active, idle = [3.94, 3.49], [2.53, 0.21]
    ax.bar(labels, active, 0.55, color=C["window"], label="GPU-active")
    ax.bar(labels, idle, 0.55, bottom=active, color="#c9c7c2", label="launch idle")
    for i, (a, s) in enumerate(zip(active, [6.47, 3.70])):
        ax.text(i, s + 0.12, f"{s:.2f} ms", ha="center", fontsize=9,
                fontweight="bold", color=INK)
    ax.set_ylabel("draft chain step (ms), dense b32/16k")
    ax.set_ylim(0, 7.4)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("(a) F11: constant geometry\ncaptures the whole step",
                 loc="left", fontsize=9.5, color=INK)
    # (b) delivered speedup
    ax = axes[1]
    cells = ["b8/16k\nK4", "b32/16k\nK6", "b16/32k\nK4"]
    broken = [1.41, 1.48, np.nan]
    fixed = [1.64, 1.91, 2.77]
    roof = [1.55, 1.91, np.nan]
    x = np.arange(3)
    ax.bar(x - 0.19, broken, 0.34, color="#c9c7c2", label="broken chain")
    ax.bar(x + 0.19, fixed, 0.34, color=C["weight-quant"], label="fixed chain")
    for i, rv in enumerate(roof):
        if not np.isnan(rv):
            ax.plot([i - 0.42, i + 0.42], [rv, rv], color=INK, lw=1.2, ls=":")
    ax.text(0.02, 2.35, "dotted = registered roofline", fontsize=7.2, color=INK2)
    for i, v in enumerate(fixed):
        ax.text(i + 0.19, v + 0.03, f"{v:.2f}×", ha="center", fontsize=8.5,
                fontweight="bold", color=INK)
    ax.set_xticks(x, cells)
    ax.set_ylim(0, 3.1)
    ax.set_ylabel("e2e speedup vs bf16 (composed W4+window)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("(b) F13: delivery ≈ 100%,\ngrowing with context", loc="left",
                 fontsize=9.5, color=INK)
    # (c) the capture law across backends (accept)
    ax = axes[2]
    rows = [("FA3 GQA\nnaive CG", 1.930, 5.685),
            ("TRITON\nnaive CG", 1.954, 5.685),
            ("scratchpad\nFA3 CG", 5.655, 5.685),
            ("FLASH_ATTN_MLA\ncaptured (fg2)", 2.00, 5.916),
            ("FLASH_ATTN_MLA\nfixed (eager)", 5.916, 5.916)]
    y = np.arange(len(rows))[::-1]
    for yi, (lbl, acc, ref) in zip(y, rows):
        collapsed = acc < 0.8 * ref
        ax.barh(yi, acc, 0.55,
                color=C["layer-skip"] if collapsed else C["window"])
        ax.plot([ref, ref], [yi - 0.38, yi + 0.38], color=INK, lw=1.2, ls=":")
        ax.text(acc + 0.08, yi, f"{acc:.2f}", va="center", fontsize=8.2,
                color=INK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7.6)
    ax.set_xlabel("accept length (dotted = reference)")
    ax.set_xlim(0, 7.0)
    ax.set_title("(c) F11 out of sample:\ncollapse iff geometry varies",
                 loc="left", fontsize=9.5, color=INK)
    for a in axes:
        a.tick_params(length=0)
        for s in a.spines.values():
            s.set_visible(False)
        a.grid(axis="y" if a is not axes[2] else "x", color=GRID, lw=0.5)
        a.set_axisbelow(True)
    fig.suptitle("Fig 8 — the floor-free chain: two execution laws cash the composed frontier",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.savefig(FIG / "fig8_floor_free.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    E4.FIG = FIG          # redirect fig2/fig5 outputs into this phase
    E4.fig2()
    fig3()
    fig4()
    E4.fig5()
    fig6()
    fig7()
    fig8()
    print("figures ->", FIG)
