#!/usr/bin/env python3
"""Narrow-window acceptance is content-dependent; full-window is not.

Three panels on the multi-column canvas:
  (a) tau across nine context lengths -- narrow windows are jagged
      because each length truncates the same document stream at a
      different point, so each is a different content draw.
  (b) the same cell (b1/c8000) re-measured on fresh documents: the
      window single recovers, the quant single never moved.
  (c) batching averages the lottery away (b1 sees one document).
"""
import csv
import json
import statistics as st
from pathlib import Path

import style

REPO = Path(__file__).resolve().parents[2]
D = REPO / "research/82_runtime_switching/data"
C1 = REPO / "research/93_c1_grid/data"
DATA = REPO / "paper/data"


def sweep(name):
    f = D / f"anom_{name}_sweep.csv"
    if not f.exists():
        return {}
    return {int(r["ctx"]): float(r["accept"] or 0)
            for r in csv.DictReader(open(f)) if int(r["K"]) == 2}


def c1_batch(arch, arm, ctx=8000):
    f = C1 / f"cells_93_{arch}_{arm}.csv"
    if not f.exists():
        return {}
    return {int(r["batch"]): float(r["accept"] or 0)
            for r in csv.DictReader(open(f))
            if int(r["K"]) == 2 and int(r["ctx"]) == ctx}


ARMS = [("win128", style.COLORS["bad"]),
        ("win512", style.COLORS["secondary"]),
        ("win8192", style.COLORS["primary"])]

fig, axes = style.multi_fig(ncols=3)

# (a) context sweep
ax = axes[0]
for arm, c in ARMS:
    d = sweep(arm)
    if not d:
        continue
    xs = sorted(d)
    ax.plot([x / 1000 for x in xs], [d[x] for x in xs], "o-", ms=3, lw=1.4,
            color=c, label=arm.replace("win", "w="))
ax.axvline(8, color=style.COLORS["light"], lw=4, zorder=0)
ax.set_xlabel("context (k tokens)")
ax.set_ylabel(r"$\tau$  (tokens / draft)")
ax.set_title("(a) same stack, 9 content draws")
ax.legend(loc="lower right")
ax.set_ylim(1.9, 3.15)

# (b) same cell, fresh documents
ax = axes[1]
res = json.loads((DATA / "c2_b1_8k_anomaly.json").read_text())
w = res["window_single_resample_ctx8000"]
comp = res["composition_resample"]["dense"]["samples"]
labels = ["original", "docs +16", "docs +32"]
xs = range(3)
ax.plot(xs, [w["off0"], w["off16"], w["off32"]], "o-", ms=4, lw=1.6,
        color=style.COLORS["secondary"], label="window single (w=512)")
ax.plot(xs, [s["tau_single"] for s in comp], "s--", ms=4, lw=1.6,
        color=style.COLORS["primary"], label="quant single")
ax.set_xticks(list(xs), labels)
ax.set_ylabel(r"$\tau$  at b1 / 8k")
ax.set_title("(b) same cell, fresh documents")
ax.legend(loc="lower right")
ax.set_ylim(1.9, 3.15)

# (c) batching averages it away
ax = axes[2]
for arm, c in ARMS:
    d = c1_batch("dense", arm)
    bs = [b for b in (1, 8, 32) if b in d]
    if bs:
        ax.plot(range(len(bs)), [d[b] for b in bs], "o-", ms=4, lw=1.4,
                color=c, label=arm.replace("win", "w="))
        ax.set_xticks(range(len(bs)), [f"b{b}" for b in bs])
ax.set_ylabel(r"$\tau$  at 8k")
ax.set_xlabel("batch (docs averaged: 1, 8, 16)")
ax.set_title("(c) batching averages it out")
ax.set_ylim(1.9, 3.15)

style.save(fig, "content_variance")
print("wrote paper/figures/content_variance.{png,pdf}")
