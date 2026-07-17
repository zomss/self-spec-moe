#!/usr/bin/env python3
"""E1: Qwen3-8B beta column (77 harness, new MODELS entry via driver).

Stages: refs (stage A, ondist 16k) then singles + layer profile +
iterative-greedy sets. Appends data/beta_q3_8b.csv.
Usage: --stage {singles,profile,greedy}
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

SA.MODELS["q3_8b"] = dict(hf="Qwen/Qwen3-8B", trust=False, moe=False,
                          layers=36)
CFG = SA.MODELS["q3_8b"]
REFDIR = PHASE / "data/refs/q3_8b_c16384"
OUT = PHASE / "data/beta_q3_8b.csv"
(PHASE / "data").mkdir(parents=True, exist_ok=True)
N_LAYERS = 36
KEEP_HEAD, KEEP_TAIL = 2, 2
POOL, BUDGET = 8, 7


class Args:
    model, ctx, prompts, gen = "q3_8b", 16384, 12, 96


class LayerSet:
    def __init__(self, model, drop):
        self.m, self.cfg = model.model, model.config
        self.kept = [i for i in range(N_LAYERS) if i not in set(drop)]

    def __enter__(self):
        self.layers, self.n = self.m.layers, self.cfg.num_hidden_layers
        self.m.layers = nn.ModuleList([self.layers[i] for i in self.kept])
        self.cfg.num_hidden_layers = len(self.kept)

    def __exit__(self, *a):
        self.m.layers, self.cfg.num_hidden_layers = self.layers, self.n


def append_row(row, extra=None):
    row = dict(row)
    if extra:
        row.update(extra)
    exists = OUT.exists()
    with OUT.open("a") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(f"[q3-8b] {row['arm']}: beta={row['beta_greedy']}", flush=True)


def done():
    if not OUT.exists():
        return {}
    return {r["arm"]: float(r["beta_greedy"]) for r in csv.DictReader(OUT.open())}


def ensure_refs(model, tok, args):
    SA.stage_a(model, tok, CFG, args, REFDIR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["singles", "profile", "greedy"])
    a = ap.parse_args()
    args = Args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CFG["hf"])
    d = done()
    model = SA.load_model(CFG)
    ensure_refs(model, tok, args)

    if a.stage == "singles":
        # order: non-destructive first, quant (destructive) LAST
        for arm in ("win512", "win128", "kvq_fp8", "skip125", "skip25",
                    "q_fp8", "q_int4"):
            if arm in d:
                continue
            if arm == "q_fp8":
                SA.fake_quant_(model, "fp8")
            if arm == "q_int4":
                # fresh weights: reload after fp8 destruction
                del model
                torch.cuda.empty_cache()
                model = SA.load_model(CFG)
                SA.fake_quant_(model, "int4")
            append_row(SA.score_arm(model, CFG, arm, REFDIR, args))

    elif a.stage == "profile":
        for i in range(KEEP_HEAD, N_LAYERS - KEEP_TAIL):
            arm = f"ig_{i}"
            if arm in d:
                continue
            with LayerSet(model, [i]):
                append_row(SA.score_arm(model, CFG, arm, REFDIR, args))

    elif a.stage == "greedy":
        singles = {int(k[3:]): v for k, v in d.items()
                   if k.startswith("ig_") and k.count("_") == 1}
        assert singles, "run --stage profile first"
        chosen, col = [], dict(singles)
        for rnd in range(1, BUDGET + 1):
            best_c = max(col, key=col.get)
            chosen.append(best_c)
            print(f"[q3-8b] greedy round {rnd}: set {sorted(chosen)} "
                  f"beta={col[best_c]:.4f}", flush=True)
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
                with LayerSet(model, drop):
                    row = SA.score_arm(model, CFG, arm, REFDIR, args)
                append_row(row)
                col[c] = float(row["beta_greedy"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
