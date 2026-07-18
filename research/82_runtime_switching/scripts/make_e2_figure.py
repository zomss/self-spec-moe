#!/usr/bin/env python3
"""Figure: which lever the policy selects per regime + measured results.

Panel A: the policy decision surface in (mean context, offered batch)
space -- K4 / K6 / OFF(schedule) / OFF(ctx-cell veto) -- with the E2
and E2b trace regimes placed on it.
Panel B: measured per-regime tok/s, all arms; the policy bar is
annotated with the lever it selected. E2 numbers are the committed
finals; E2b bars appear automatically when data/e2_*.json carry Q*
phases (the long-ctx run).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PHASE = Path(__file__).resolve().parents[1]
OUT = PHASE / "figures"
OUT.mkdir(exist_ok=True)

C = {"K4": "#1baf7a", "K6": "#0e7a54", "OFF": "#c9c7c2",
     "veto": "#e8e0d8", "ink": "#0b0b0b", "ink2": "#52514e",
     "off_arm": "#9b9a94", "k4_arm": "#7cc7f0", "k6_arm": "#2a78d6",
     "policy": "#eb6834"}
plt.rcParams.update({"font.size": 9.5, "figure.dpi": 180,
                     "axes.edgecolor": "#e8e7e3"})

fig, (ax, bx) = plt.subplots(
    1, 2, figsize=(12.6, 4.6), constrained_layout=True,
    gridspec_kw={"width_ratios": [1.0, 1.35]})

# ---------------- Panel A: decision surface ----------------
CTX_MAX, B_MAX = 16000, 33
# regions (x0, x1, y0, y1, color, label)
regions = [
    (0, CTX_MAX, 1, 8, "K4", None),            # b1-7: K4 (map prior)
    (8000, CTX_MAX, 8, 25, "K4", None),        # b8-24 long ctx: K4
    (0, 8000, 8, 25, "veto", None),            # ctx-cell veto -> OFF
    (0, CTX_MAX, 25, B_MAX, "OFF", None),      # schedule b>=25: OFF
]
for x0, x1, y0, y1, key, _ in regions:
    ax.add_patch(mpatches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        facecolor=C[key], edgecolor="white", linewidth=1.2,
        alpha=0.75 if key in ("K4", "K6") else 1.0))
ax.text(4500, 4.6, "SPEC K4  (b1-7: map prior, accept gate silent)",
        ha="center", fontsize=8.2, color="white", fontweight="bold")
ax.text(12000, 16, "SPEC K4\n(K6 prior FALSIFIED by\nmeasurement: "
        "K4 ≥ K6 at\nevery batch on this column)", ha="center",
        fontsize=8, color="white", fontweight="bold")
ax.text(3800, 15.5, "OFF\n(ctx-cell veto:\n$\\tau^*>K{+}1$,\n"
        "any-accept loses)", ha="center", fontsize=8, color=C["ink2"])
ax.text(8000, 28.6, "OFF (schedule: verify-width regime)",
        ha="center", fontsize=8, color=C["ink2"])
ax.text(15700, 2.4, "accept-EMA content gate armed in spec regions (b≥4)",
        ha="right", fontsize=6.8, style="italic", color="#f0efe9")

TRACES = [  # (name, ctx, batch, selected, dy)
    ("E2:P1 b1 math", 1400, 1, "K4", 0),
    ("E2:P2 b8 docs", 6100, 8, "OFF", 0),
    ("E2:P3 b32 math", 900, 32, "OFF", 0),
    ("E2c:S1 b1 rag14k", 14100, 1, "K4", 0),
    ("E2c:S2 b8 rag14k", 14200, 8, "K4", 0),
    ("E2c:S3 b16 rag14k", 14200, 16, "K4", 0),
    ("E2c:S4 b32 math", 900, 30, "OFF", -2),
]
for name, x, b, sel, dy in TRACES:
    ax.scatter([x], [b], s=52, color=C["ink"], zorder=5,
               edgecolor="white", linewidth=1)
    ax.annotate(f"{name} → {sel}", (x, b),
                textcoords="offset points", xytext=(6, 5 + dy),
                fontsize=7.6, fontweight="bold", color=C["ink"])
ax.set_xlim(0, CTX_MAX)
ax.set_ylim(1, B_MAX)
ax.set_xlabel("mean effective context (tokens)")
ax.set_ylabel("offered batch (running + waiting)")
ax.set_title("A. The policy's decision surface — which lever, where",
             loc="left", fontsize=10.5, fontweight="bold")
for s in ax.spines.values():
    s.set_visible(False)

# ---------------- Panel B: measured per-regime results ----------------
E2_FINAL = {  # committed finals (results_e2.md)
    "P1 b1\nmath": {"off": 149.9, "k4": 173.1, "k6": 168.1,
                    "policy": (162.5, "K4")},
    "P2 b8\ndocs 6k": {"off": 689.9, "k4": 516.6, "k6": 475.6,
                       "policy": (670.5, "OFF")},
    "P3 b32\nmath": {"off": 4180.1, "k4": 3728.8, "k6": 3539.5,
                     "policy": (4052.2, "OFF")},
    "aggregate": {"off": 957.9, "k4": 932.8, "k6": 886.1,
                  "policy": (990.1, "")},
}
E2C_FINAL = {
    "S1 b1\nrag14k": {"off": 128.3, "k4": 145.7, "k6": 140.9,
                      "policy": (145.9, "K4")},
    "S2 b8\nrag14k": {"off": 514.5, "k4": 557.9, "k6": 549.2,
                      "policy": (557.8, "K4")},
    "S3 b16\nrag14k": {"off": 658.4, "k4": 712.9, "k6": 700.9,
                       "policy": (713.7, "K4")},
    "S4 b32\nmath": {"off": 4155.5, "k4": 3755.9, "k6": 3576.0,
                     "policy": (4109.2, "OFF")},
    "aggregate ": {"off": 587.6, "k4": 642.3, "k6": 626.3,
                   "policy": (646.5, "")},
}
E2_FINAL.update(E2C_FINAL)
SELMAP = {"Q1_b1_doc16k": "K4", "Q2_b8_doc16k": "K4",
          "Q3_b16_doc16k": "K6", "Q4_b32_aime2k": "OFF"}
e2b = {}
for arm in ("off", "k4", "k6", "policy"):
    f = PHASE / f"data/e2_{arm}.json"
    if f.exists():
        d = json.loads(f.read_text())
        if any(k.startswith("Q") for k in d):
            for k, v in d.items():
                if k.startswith("Q") or k == "aggregate_toks":
                    key = (k.replace("_", " ")[:9] if k != "aggregate_toks"
                           else "aggregate")
                    lab = (k.split("_")[0] + " " + k.split("_")[1]
                           + "\n" + k.split("_")[2]) \
                        if k != "aggregate_toks" else "aggregate"
                    e2b.setdefault(lab, {})[arm] = (
                        v["toks"] if isinstance(v, dict) else v)

def draw_group(bx, table, title_y, is_e2b):
    x = 0
    ticks, labels = [], []
    for phase, arms in table.items():
        vals = []
        base = arms.get("off", 1)
        for arm in ("off", "k4", "k6", "policy"):
            if arm not in arms:
                continue
            v = arms[arm]
            sel = ""
            if isinstance(v, tuple):
                v, sel = v
            if is_e2b and arm == "policy":
                sel = SELMAP.get(
                    [k for k in SELMAP
                     if k.startswith(phase.split("\n")[0].replace(" ", "_"))
                     ][0], "") if "\n" in phase else ""
            col = C[arm + "_arm"] if arm != "policy" else C["policy"]
            norm = v / base
            bx.bar(x, norm, 0.8, color=col, edgecolor="white", lw=0.5)
            txt = f"{norm:.2f}"
            if arm == "policy" and sel:
                txt += f"\n[{sel}]"
            bx.text(x, norm + 0.02, txt, ha="center", fontsize=6.8,
                    fontweight="bold" if arm == "policy" else "normal")
            vals.append(v)
            x += 1
        ticks.append(x - len(vals) / 2 - 0.5)
        labels.append(phase)
        x += 1.1
    return ticks, labels

ticks, labels = draw_group(bx, E2_FINAL, None, False)
if e2b:
    bx2 = bx  # same axis, offset second block
t2, l2 = [], []
bx.set_xticks(ticks, labels, fontsize=8)
bx.axhline(1.0, color=C["ink2"], lw=0.8, ls=":")
bx.set_ylabel("tok/s normalized to OFF (per regime)")
bx.set_title("B. Per-regime tok/s vs OFF — E2 (left 4) + E2c RAG trace (right 5)",
             loc="left", fontsize=10.5, fontweight="bold")
leg = [mpatches.Patch(color=C["off_arm"], label="OFF static"),
       mpatches.Patch(color=C["k4_arm"], label="K4 static"),
       mpatches.Patch(color=C["k6_arm"], label="K6 static"),
       mpatches.Patch(color=C["policy"], label="policy (selected lever)")]
bx.set_ylim(0, 1.38)
bx.legend(handles=leg, frameon=False, fontsize=7.5, ncol=4,
          loc="upper center", bbox_to_anchor=(0.5, 1.0))
for s in bx.spines.values():
    s.set_visible(False)

fig.suptitle("Runtime lever selection: strategy map compiled into "
             "per-step scheduler switches (Qwen3-8B + W4win draft; "
             "DRAFT)", fontsize=11, fontweight="bold", x=0.01,
             ha="left")
fig.savefig(OUT / "figE2_policy.png")
print("saved ->", OUT / "figE2_policy.png")
if e2b:
    print("E2b data found:", json.dumps(e2b, indent=1)[:400])
