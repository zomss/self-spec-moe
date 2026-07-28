#!/usr/bin/env python3
"""Phase 93 Step 2: search-derived skip sets for MoE / MLA.

The phase-86 measured protocol (the search mechanism that survived
phase-90's proxy falsification): on-policy 16k refs -> per-layer LOO
profile (conditional beta) -> iterative greedy set growth, pool-8,
budget 4. Grid arms use the budget-2 and budget-4 sets.

Usage: skip_search_93.py --model {moe,mla} --stage {profile,greedy}
Appends data/skip_search_<model>.csv.
"""
import argparse
import csv
import sys
from pathlib import Path

import torch.nn as nn

PHASE = Path(__file__).resolve().parents[1]
P77 = PHASE.parent / "77_acceptance_map"
sys.path.insert(0, str(P77 / "scripts"))
import score_accept as SA  # noqa: E402

KEEP_HEAD, KEEP_TAIL = 2, 2
POOL, BUDGET = 8, 4


class Args:
    ctx, prompts, gen, chunk = 16384, 12, 96, 4096

    def __init__(self, model):
        self.model = model


class LayerSet:
    def __init__(self, model, n_layers, drop):
        self.m, self.cfg = model.model, model.config
        self.kept = [i for i in range(n_layers) if i not in set(drop)]

    def __enter__(self):
        self.layers, self.n = self.m.layers, self.cfg.num_hidden_layers
        self.m.layers = nn.ModuleList([self.layers[i] for i in self.kept])
        self.cfg.num_hidden_layers = len(self.kept)

    def __exit__(self, *a):
        self.m.layers, self.cfg.num_hidden_layers = self.layers, self.n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["moe", "mla"])
    ap.add_argument("--stage", required=True, choices=["profile", "greedy"])
    a = ap.parse_args()

    cfg = SA.MODELS[a.model]
    n_layers = cfg["layers"]
    refdir = PHASE / "data" / "refs" / f"{a.model}_c16384"
    out = PHASE / "data" / f"skip_search_{a.model}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    args = Args(a.model)

    def done():
        if not out.exists():
            return {}
        return {r["arm"]: float(r["beta_greedy"])
                for r in csv.DictReader(out.open())}

    def append_row(row):
        row = dict(row)
        exists = out.exists()
        with out.open("a") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if not exists:
                w.writeheader()
            w.writerow(row)
        print(f"[93-skip:{a.model}] {row['arm']}: "
              f"beta={row['beta_greedy']}", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        cfg["hf"], trust_remote_code=cfg["trust"])
    d = done()
    model = SA.load_model(cfg)
    SA.stage_a(model, tok, cfg, args, refdir)

    if a.stage == "profile":
        for i in range(KEEP_HEAD, n_layers - KEEP_TAIL):
            arm = f"ig_{i}"
            if arm in d:
                continue
            with LayerSet(model, n_layers, [i]):
                append_row(SA.score_arm(model, cfg, arm, refdir, args))

    elif a.stage == "greedy":
        singles = {int(k[3:]): v for k, v in d.items()
                   if k.startswith("ig_") and k.count("_") == 1}
        assert singles, "run --stage profile first"
        chosen, col = [], dict(singles)
        for rnd in range(1, BUDGET + 1):
            best_c = max(col, key=col.get)
            chosen.append(best_c)
            print(f"[93-skip:{a.model}] greedy round {rnd}: "
                  f"set {sorted(chosen)} beta={col[best_c]:.4f}",
                  flush=True)
            if rnd == BUDGET:
                break
            cands = [c for c in sorted(col, key=col.get, reverse=True)
                     if c not in chosen][:POOL]
            col = {}
            for c in cands:
                drop = sorted(chosen + [c])
                arm = "igs_" + "-".join(map(str, drop))
                if arm in d:
                    col[c] = d[arm]
                    continue
                with LayerSet(model, n_layers, drop):
                    row = SA.score_arm(model, cfg, arm, refdir, args)
                append_row(row)
                col[c] = float(row["beta_greedy"])
        print(f"[93-skip:{a.model}] FINAL budget-{BUDGET} set: "
              f"{sorted(chosen)}", flush=True)


if __name__ == "__main__":
    main()
