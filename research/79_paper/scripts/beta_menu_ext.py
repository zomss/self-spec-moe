#!/usr/bin/env python3
"""Menu-extension beta gate: levers the map never measured.

The winner-hardening search is exhaustive over the MEASURED lever menu; the
design space has unmeasured axes. Gate them by beta first (the map's own
protocol -- cheap filter before any cost work), on 77's dense ondist refs:

  1. Weight pruning: 2:4 structured magnitude (pr24) and 50% unstructured
     magnitude (pr50u). Uncalibrated one-shot = a LOWER bound on beta;
     a SparseGPT-style calibrated prune would sit above it. If even the
     bound's ceiling analysis kills the lever, it's dead; if borderline,
     escalate to a calibrated prune.
  2. Non-contiguous layer selection (KnapSpec's lever): per-layer
     leave-one-out profile (ls_<i>), then greedy lowest-damage sets of the
     same budget as skip125/skip25 (ls_g_<ids>). Contiguous middle-block
     is our measured arm; this checks whether set-selection changes the
     verdict.

Reuses 77/score_accept.score_arm verbatim: unknown arm names fall through
to a plain forward, so pruning is applied destructively BEFORE scoring and
layer sets wrap the call in a context manager. Output:
data/beta_menu_ext.csv (model=dense, ctx=16384, ondist refs).

Usage:
  beta_menu_ext.py --stage prune          # pr24, pr50u (2 model loads)
  beta_menu_ext.py --stage profile        # ls_<i> leave-one-out, i in 2..25
  beta_menu_ext.py --stage greedy         # greedy sets from the profile
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn

PHASE = Path(__file__).resolve().parents[1]
P77 = PHASE.parent / "77_acceptance_map"
sys.path.insert(0, str(P77 / "scripts"))

import score_accept as SA  # noqa: E402

CFG = SA.MODELS["dense"]
REFDIR = P77 / "data/refs/dense_c16384"
OUT = PHASE / "data/beta_menu_ext.csv"
N_LAYERS = 28
KEEP_HEAD, KEEP_TAIL = 2, 2      # early-exit lesson + LayerSkip's tail rule


class Args:
    model, ctx, prompts, gen = "dense", 16384, 12, 96


def prune24_(model):
    """2:4 structured magnitude along the input dim (skip lm_head/embed)."""
    n = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or "lm_head" in name:
            continue
        w = mod.weight.data
        out_f, in_f = w.shape
        if in_f % 4:
            continue
        g = w.view(out_f, in_f // 4, 4)
        idx = g.abs().argsort(-1)                     # ascending
        mask = torch.ones_like(g, dtype=torch.bool)
        mask.scatter_(-1, idx[..., :2], False)        # zero the 2 smallest
        g *= mask
        n += 1
    print(f"[prune24] {n} linears pruned 2:4", flush=True)


def prune_unstr_(model, frac: float):
    """Unstructured magnitude pruning, per-matrix threshold."""
    n = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or "lm_head" in name:
            continue
        w = mod.weight.data
        k = int(w.numel() * frac)
        thr = w.abs().flatten().kthvalue(k).values
        w[w.abs() <= thr] = 0
        n += 1
    print(f"[prune_unstr {frac}] {n} linears pruned", flush=True)


class LayerSet:
    """Drop an arbitrary set of layers; kept layers retain layer_idx
    (mirrors 77's LayerSkip splice)."""

    def __init__(self, model, drop: list):
        self.m, self.cfg = model.model, model.config
        self.kept = [i for i in range(N_LAYERS) if i not in set(drop)]

    def __enter__(self):
        self.layers, self.n = self.m.layers, self.cfg.num_hidden_layers
        self.m.layers = nn.ModuleList([self.layers[i] for i in self.kept])
        self.cfg.num_hidden_layers = len(self.kept)

    def __exit__(self, *a):
        self.m.layers, self.cfg.num_hidden_layers = self.layers, self.n


def append_row(row):
    exists = OUT.exists()
    with OUT.open("a") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(f"[menu-ext] {row['arm']}: beta={row['beta_greedy']}", flush=True)


def done_arms():
    if not OUT.exists():
        return set()
    return {r["arm"] for r in csv.DictReader(OUT.open())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["prune", "profile", "greedy"])
    ap.add_argument("--budget", type=int, default=None,
                    help="greedy: layers to drop (default: both 3 and 7)")
    a = ap.parse_args()
    args = Args()
    done = done_arms()

    if a.stage == "prune":
        for arm, fn in (("pr24", prune24_),
                        ("pr50u", lambda m: prune_unstr_(m, 0.5))):
            if arm in done:
                print(f"skip {arm}")
                continue
            model = SA.load_model(CFG)          # fresh weights per arm
            fn(model)
            append_row(SA.score_arm(model, CFG, arm, REFDIR, args))
            del model
            torch.cuda.empty_cache()

    elif a.stage == "profile":
        model = SA.load_model(CFG)
        for i in range(KEEP_HEAD, N_LAYERS - KEEP_TAIL):
            arm = f"ls_{i}"
            if arm in done:
                continue
            with LayerSet(model, [i]):
                append_row(SA.score_arm(model, CFG, arm, REFDIR, args))

    elif a.stage == "greedy":
        prof = {int(r["arm"][3:]): float(r["beta_greedy"])
                for r in csv.DictReader(OUT.open())
                if r["arm"].startswith("ls_") and "_g_" not in r["arm"]
                and r["arm"].count("_") == 1}
        order = sorted(prof, key=lambda i: -prof[i])   # least damaging first
        model = SA.load_model(CFG)
        for budget in ([a.budget] if a.budget else [3, 7]):
            drop = sorted(order[:budget])
            arm = "ls_g_" + "-".join(map(str, drop))
            if arm in done:
                continue
            with LayerSet(model, drop):
                append_row(SA.score_arm(model, CFG, arm, REFDIR, args))
    return 0


if __name__ == "__main__":
    return_code = main()
    sys.exit(return_code)
