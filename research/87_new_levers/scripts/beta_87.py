#!/usr/bin/env python3
"""E1 beta gates on Qwen3-8B paired refs (86's refdir).
Arms: actfp8 (A-quant factor isolated), w4a8 (int4 W + fp8 A),
sparse24 (SparseGPT ckpt decompressed), s24w4 (2:4 + RTN int4 = the
marlin_24 combo beta). Usage: --stage {act,sparse}"""
import argparse, csv, sys
from pathlib import Path
import torch
import torch.nn as nn

PHASE = Path(__file__).resolve().parents[1]
P77 = PHASE.parent / "77_acceptance_map"
P86 = PHASE.parent / "86_scale_headtohead"
sys.path.insert(0, str(P77 / "scripts"))
import score_accept as SA  # noqa: E402

SA.MODELS["q3_8b"] = dict(hf="Qwen/Qwen3-8B", trust=False, moe=False, layers=36)
CFG = SA.MODELS["q3_8b"]
REFDIR = P86 / "data/refs/q3_8b_c16384"
OUT = PHASE / "data/beta_87.csv"


class Args:
    model, ctx, prompts, gen, chunk = "q3_8b", 16384, 12, 96, 4096


class ActFP8:
    """Dynamic per-token e4m3 fake-quant on every Linear INPUT (skip lm_head)."""
    def __init__(self, model):
        self.mods = [m for n, m in model.named_modules()
                     if isinstance(m, nn.Linear) and "lm_head" not in n]
        self.hooks = []

    def __enter__(self):
        def pre(mod, inputs):
            x = inputs[0]
            s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 448.0
            xq = (x / s).to(torch.float8_e4m3fn).to(x.dtype) * s
            return (xq,) + inputs[1:]
        for m in self.mods:
            self.hooks.append(m.register_forward_pre_hook(pre))
        return self

    def __exit__(self, *a):
        for h in self.hooks:
            h.remove()


def append_row(row):
    exists = OUT.exists()
    with OUT.open("a") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(f"[87] {row['arm']}: beta={row['beta_greedy']}", flush=True)


def done():
    if not OUT.exists():
        return set()
    return {r["arm"] for r in csv.DictReader(OUT.open())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["act", "sparse"])
    a = ap.parse_args()
    d = done()
    args = Args()
    if a.stage == "act":
        model = SA.load_model(CFG)
        if "actfp8" not in d:
            with ActFP8(model):
                append_row(SA.score_arm(model, CFG, "actfp8", REFDIR, args))
        if "w4a8" not in d:
            SA.fake_quant_(model, "int4")
            with ActFP8(model):
                append_row(SA.score_arm(model, CFG, "w4a8", REFDIR, args))
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            Path.home() / "ckpts/Qwen3-8B-sparse24-sgpt",
            dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="sdpa").eval()
        if "sparse24" not in d:
            append_row(SA.score_arm(model, CFG, "sparse24", REFDIR, args))
        if "s24w4" not in d:
            SA.fake_quant_(model, "int4")
            append_row(SA.score_arm(model, CFG, "s24w4", REFDIR, args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
