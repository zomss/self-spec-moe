#!/usr/bin/env python3
"""Draft figures for paper/: (1) head-to-head with per-arm detail,
(2) the full measured e2e record across regimes. All values are measured
(committed JSONs / results docs); DRAFT -- not final."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)
C = {"window": "#2a78d6", "quant": "#1baf7a", "composed": "#0e7a54",
     "skip": "#eb6834", "knapspec": "#9b9a94", "fp8": "#eda100",
     "base": "#c9c7c2", "ink": "#0b0b0b", "ink2": "#52514e"}
plt.rcParams.update({"font.size": 9.5, "figure.dpi": 180,
                     "axes.edgecolor": "#e8e7e3"})

# ---------------- fig A: head-to-head, per-arm detail ----------------
def fig_h2h():
    settings = [
        ("Qwen3-8B\nb1 greedy", [("W4 alone", 1.08, "quant"),
                                 ("W4+win (ours)", 1.42, "composed"),
                                 ("KnapSpec", 1.28, "knapspec")]),
        ("Qwen3-8B\nb1 T=0.7", [("W4+win (ours)", 1.32, "composed"),
                                ("KnapSpec*", 1.28, "knapspec")]),
        ("Qwen3-32B\nb1 prose", [("fp8-Marlin+win", 1.19, "fp8"),
                                 ("fp8-native+win", 1.26, "fp8"),
                                 ("W4gptq+win (ours)", 1.41, "composed"),
                                 ("KnapSpec", 1.43, "knapspec")]),
        ("Qwen3-32B\nb1 MATH (their task)", [("W4gptq+win K4", 1.54, "composed"),
                                             ("W4gptq+win K5 (ours)", 1.63, "composed"),
                                             ("KnapSpec", 1.43, "knapspec")]),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.2), constrained_layout=True)
    x = 0
    ticks, ticklabels = [], []
    for name, arms in settings:
        xs = []
        for lbl, v, fam in arms:
            b = ax.bar(x, v, 0.8, color=C[fam],
                       edgecolor="white", linewidth=0.5)
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=8,
                    fontweight="bold" if fam == "composed" else "normal")
            ax.text(x, 0.05, lbl, ha="center", va="bottom", fontsize=7,
                    rotation=90, color="white" if fam != "knapspec" else C["ink"])
            xs.append(x)
            x += 1
        ticks.append(sum(xs) / len(xs))
        ticklabels.append(name)
        x += 1.2
    ax.set_xticks(ticks, ticklabels, fontsize=9)
    ax.set_ylabel("e2e speedup vs own AR baseline")
    ax.set_ylim(0, 1.85)
    ax.axhline(1.0, color="#e8e7e3", lw=1)
    ax.set_title("Head-to-head vs KnapSpec (published numbers; DRAFT, "
                 "ours measured 2026-07, 8-iter runs)", loc="left",
                 color=C["ink"], fontsize=11, fontweight="bold")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(OUT / "figA_headtohead.png")
    plt.close(fig)


# ---------------- fig B: the full measured e2e record by regime ----------------
E2E = [
    # (family, model, ctx_k, batch, config, speedup, accept)
    ("dense", "Q2.5-7B", 16, 8, "W4+win K4", 1.64, 4.38),
    ("dense", "Q2.5-7B", 16, 8, "W4+win K6", 1.52, 5.64),
    ("dense", "Q2.5-7B", 16, 32, "W4+win K6", 1.91, 5.65),
    ("dense", "Q2.5-7B", 16, 32, "W4+win K4", 1.85, 4.29),
    ("dense", "Q2.5-7B", 32, 16, "W4+win K4", 2.77, 4.33),
    ("dense", "Q2.5-7B", 32, 16, "W4+win K6", 2.33, 5.62),
    ("dense", "Q3-8B", 16, 1, "W4+win K4", 1.42, 4.49),
    ("dense", "Q3-8B", 2, 1, "W4 K4", 1.23, 4.48),
    ("dense", "Q3-8B", 16, 8, "W4+win K4", 1.81, 4.51),
    ("dense", "Q3-8B", 16, 16, "W4+win K6", 1.79, 5.69),
    ("dense", "Q3-8B", 16, 8, "W4A8+win K4 Humming", 1.90, 4.58),
    ("dense", "Q3-8B", 16, 16, "W4A8+win K6 Humming", 2.19, 6.02),
    ("dense", "Q3-8B", 16, 16, "W4A8+win K6 Cutlass", 1.78, 5.94),
    ("dense", "Q3-32B", 16, 1, "W4gptq+win K4", 1.41, 4.54),
    ("dense", "Q3-32B", 16, 1, "W4gptq+win K5 math", 1.63, 5.36),
    ("dense", "Q3-32B", 16, 8, "W4+win K5", 1.28, 5.20),
    ("moe", "Q3-30B-A3B", 16, 8, "win K3 fixed-stack", 1.15, 3.77),
    ("moe", "Q3-30B-A3B", 2, 4, "flr50 bf16-partial K2", 1.03, 2.88),
    ("moe", "Q3-30B-A3B", 2, 8, "flr50 (chain-blocked)", 0.63, 2.84),
    ("mla", "DS-V2-Lite", 16, 32, "beta~1 self-draft K5", 0.56, 5.92),
    ("mla", "DS-V2-Lite", 16, 32, "ngram K4 (CPU-lookup)", 0.78, 2.85),
]
FAMC = {"dense": "#1baf7a", "moe": "#2a78d6", "mla": "#eb6834"}


def fig_regimes():
    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    ys, labels = [], []
    order = sorted(E2E, key=lambda r: (r[0], r[1], r[2], r[3]))
    for i, (fam, model, ck, b, cfg, sp, acc) in enumerate(order):
        ax.barh(i, sp, 0.62, color=FAMC[fam], alpha=0.9 if sp >= 1 else 0.45)
        ax.text(sp + 0.02, i, f"{sp:.2f}  (acc {acc:.2f})", va="center",
                fontsize=7.5)
        labels.append(f"{model} b{b}/{ck}k  {cfg}")
        ys.append(i)
    ax.set_yticks(ys, labels, fontsize=7.8)
    ax.axvline(1.0, color=C["ink2"], lw=1, ls=":")
    ax.set_xlabel("measured e2e speedup vs own AR baseline (accept length in parens)")
    ax.set_xlim(0, 3.0)
    ax.invert_yaxis()
    ax.set_title("The measured e2e record across regimes (every committed cell; "
                 "DRAFT)", loc="left", fontsize=11, fontweight="bold",
                 color=C["ink"])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(OUT / "figB_regimes.png")
    plt.close(fig)


# ---------------- fig C: frontiers (skip-set beta vs budget, 3 models) ----------------
def fig_frontiers():
    F = {"Qwen2.5-7B (28L)": [0.954, 0.908, 0.869, 0.817, 0.761, 0.659, 0.557],
         "Qwen3-8B (36L)": [0.951, 0.923, 0.885, 0.849, 0.813, 0.773, 0.694],
         "Qwen3-32B (64L)": [0.976, 0.954, 0.912, 0.892, 0.833, 0.800, 0.770]}
    CONTIG = {"Qwen2.5-7B (28L)": (3.5, 0.448), "Qwen3-8B (36L)": (4.5, 0.754),
              "Qwen3-32B (64L)": (8, 0.659)}
    cols = {"Qwen2.5-7B (28L)": "#eb6834", "Qwen3-8B (36L)": "#eda100",
            "Qwen3-32B (64L)": "#2a78d6"}
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    for name, vals in F.items():
        ax.plot(range(1, 8), vals, "-o", color=cols[name], label=name, ms=4)
        bx, by = CONTIG[name]
        ax.scatter([bx], [by], marker="x", s=60, color=cols[name])
    ax.scatter([], [], marker="x", s=60, color=C["ink2"],
               label="contiguous middle-block (naive)")
    ax.set_xlabel("layers dropped (iterative-greedy set)")
    ax.set_ylabel("β (per-token acceptance, 16k refs)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color="#e8e7e3", lw=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Skip-set frontiers: placement law at three scales; "
                 "scale is NOT monotone (DRAFT)", loc="left", fontsize=10.5,
                 fontweight="bold", color=C["ink"])
    fig.savefig(OUT / "figC_frontiers.png")
    plt.close(fig)


# ---------------- fig D: serving wall-clock e2e (T6, the deployment headline) ----------------
def fig_serving():
    phases = ["b8 x 14k RAG\n(3k-tok CoT)", "b16 x 14k RAG\n(3k-tok CoT)",
              "aggregate"]
    arms = [
        ("W4+win K4", [1.53, 1.72, 1.64], "#7cc7f0"),
        ("W4+win K6", [1.51, 1.76, 1.65], "#2a78d6"),
        ("W4A8 K6 Cutlass", [1.40, 1.61, 1.52], "#c9a3e8"),
        ("W4A8 K6 Humming", [1.67, 1.90, 1.80], "#0e7a54"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    n = len(arms)
    w = 0.8 / n
    for j, (name, vals, col) in enumerate(arms):
        xs = [i + (j - n / 2 + 0.5) * w for i in range(len(phases))]
        ax.bar(xs, vals, w * 0.92, color=col, edgecolor="white", lw=0.5,
               label=name)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5,
                    fontweight="bold" if "Humming" in name else "normal")
    ax.axhline(1.0, color=C["ink2"], lw=1, ls=":")
    ax.text(2.42, 1.015, "AR", fontsize=8, color=C["ink2"])
    ax.set_xticks(range(len(phases)), phases, fontsize=9)
    ax.set_ylabel("serving WALL speedup vs AR (prefill included)")
    ax.set_ylim(0, 2.15)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    ax.set_title("Serving wall-clock e2e (DRAFT) -- Qwen3-8B, real data "
                 "(C4 14k-doc RAG + AIME CoT);\nb16 decode 2.13x reproduces "
                 "the harness record; same-ckpt kernel span 1.52-1.80x",
                 loc="left", fontsize=9.5, fontweight="bold", color=C["ink"])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.savefig(OUT / "figD_serving.png")
    plt.close(fig)


# ---------------- fig E: runtime policy vs statics, three traces (T7) ----------------
def fig_policy():
    # aggregate tok/s, fixed-skip stack (82/results_e2.md, 2026-07-19)
    traces = [
        ("mixed\n(AIME+docs, b1/8/32)", {"OFF": 955.7, "K4": 976.6,
                                         "K6": 914.0, "policy": 979.2}),
        ("long-math RAG\n(rag14k b1/8/16 + aime b32)",
         {"OFF": 584.4, "K4": 769.8, "K6": 753.5, "policy": 752.2}),
        ("long-ctx docs\n(doc16k b1/8/16 + aime b32)",
         {"OFF": 768.1, "K4": 799.5, "K6": 704.9, "policy": 817.8}),
    ]
    arms = ["OFF", "K4", "K6", "policy"]
    cols = {"OFF": C["base"], "K4": "#7cc7f0", "K6": "#2a78d6",
            "policy": "#0e7a54"}
    fig, ax = plt.subplots(figsize=(8.2, 4.2), constrained_layout=True)
    n = len(arms)
    w = 0.8 / n
    for j, arm in enumerate(arms):
        xs = [i + (j - n / 2 + 0.5) * w for i in range(len(traces))]
        vals = [t[1][arm] / t[1]["OFF"] for t in traces]
        ax.bar(xs, vals, w * 0.9, color=cols[arm], edgecolor="white",
               lw=0.5, label=arm if arm != "policy" else "policy (ours)")
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5,
                    fontweight="bold" if arm == "policy" else "normal")
    for i, (_, d) in enumerate(traces):
        winner = max(d, key=d.get)
        j = arms.index(winner)
        ax.text(i + (j - n / 2 + 0.5) * w, d[winner] / d["OFF"] + 0.06,
                "*", ha="center", fontsize=13, color=C["ink"])
    ax.axhline(1.0, color=C["ink2"], lw=1, ls=":")
    ax.set_xticks(range(len(traces)), [t[0] for t in traces], fontsize=8.5)
    ax.set_ylabel("serving throughput vs OFF (AR) on trace")
    ax.set_ylim(0, 1.55)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    ax.set_title("Runtime selection vs every static (DRAFT, fixed-skip "
                 "stack) -- one policy config,\nthree different per-trace "
                 "static winners; policy wins 2/3 outright, -2.3% on the "
                 "third (* = winner)", loc="left", fontsize=9.5,
                 fontweight="bold", color=C["ink"])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.savefig(OUT / "figE_policy.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_h2h()
    fig_regimes()
    fig_frontiers()
    fig_serving()
    fig_policy()
    print("figures ->", OUT)
