#!/usr/bin/env python3
"""Pool-completeness gates on Qwen3-8B paired refs (86's refdir).

Arms:
  kvq_int4               prefix KV int4 (per-token per-head symmetric,
                         group = head_dim) -- the full-potential form of
                         the kvq lever (4x cut vs fp8's 2x)
  ffn{125,25,375,50}     FFN-channel prune, C4-profiled Wanda-style
                         importance (mean|act| x ||down col||), per-layer
                         uniform fraction; realization-free (smaller GEMM)
  head{25,50}            Q-head prune (o_proj column zeroing), same
                         importance form; KV heads/cache untouched
  ffn25head25            the width combo

Importance profiled on C4 (disjoint from eval refs per the
disjoint-artifact rule). Usage: --stage {profile,score}
"""
import argparse
import csv
import glob
import gzip
import json
import sys
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
IMP = PHASE / "data/width_importance.pt"


class Args:
    model, ctx, prompts, gen, chunk = "q3_8b", 16384, 12, 96, 4096


def kv_quant_int4(kv_gpu):
    """Prefix KV int4: symmetric per (head, token), group = head_dim."""
    out = {}
    for li, kv in kv_gpu.items():
        qkv = []
        for t in kv:  # (1, n_kv_heads, seq, head_dim)
            s = t.float().abs().amax(-1, keepdim=True).clamp(min=1e-8) / 7.0
            qkv.append(((t.float() / s).round().clamp(-8, 7) * s).to(t.dtype))
        out[li] = tuple(qkv)
    return out


def c4_texts(n):
    files = glob.glob(
        "/data/smcho/huggingface/hub/datasets--allenai--c4/**/*.json.gz",
        recursive=True)
    with gzip.open(files[0], "rt") as f:
        rows = []
        for line in f:
            rows.append(json.loads(line)["text"])
            if len(rows) >= n:
                return rows
    return rows


@torch.no_grad()
def profile(model, tok, n_seq=32, seq_len=2048):
    """Wanda-style importance: mean|act| per FFN channel / per Q head."""
    layers = model.model.layers
    nL = len(layers)
    ffn_act = [None] * nL
    head_act = [None] * nL
    counts = [0] * nL
    hooks = []

    def mk_ffn(li):
        def pre(mod, inputs):
            a = inputs[0].detach().float().abs().sum(dim=(0, 1))
            ffn_act[li] = a if ffn_act[li] is None else ffn_act[li] + a
        return pre

    def mk_head(li):
        def pre(mod, inputs):
            a = inputs[0].detach().float().abs().sum(dim=(0, 1))
            head_act[li] = a if head_act[li] is None else head_act[li] + a
            counts[li] += inputs[0].shape[0] * inputs[0].shape[1]
        return pre

    for li, lyr in enumerate(layers):
        hooks.append(lyr.mlp.down_proj.register_forward_pre_hook(mk_ffn(li)))
        hooks.append(lyr.self_attn.o_proj.register_forward_pre_hook(mk_head(li)))

    for i, text in enumerate(c4_texts(n_seq)):
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=seq_len).input_ids.cuda()
        if ids.shape[1] < 256:
            continue
        model(input_ids=ids)
        print(f"[prof] seq {i} len {ids.shape[1]}", flush=True)
    for h in hooks:
        h.remove()

    head_dim = model.config.head_dim
    imp = {"ffn": [], "head": []}
    for li, lyr in enumerate(layers):
        wd = lyr.mlp.down_proj.weight.float()          # (hidden, inter)
        imp["ffn"].append((ffn_act[li] / counts[li]) * wd.norm(dim=0))
        wo = lyr.self_attn.o_proj.weight.float()       # (hidden, nH*dh)
        nH = wo.shape[1] // head_dim
        a = (head_act[li] / counts[li]).view(nH, head_dim).mean(-1)
        wn = wo.view(wo.shape[0], nH, head_dim).norm(dim=(0, 2))
        imp["head"].append(a * wn)
    torch.save({k: [t.cpu() for t in v] for k, v in imp.items()}, IMP)
    print(f"[prof] saved -> {IMP}", flush=True)


class WidthMask:
    """Zero pruned FFN-channel / Q-head columns (exact removal equivalent)."""
    def __init__(self, model, ffn_frac=0.0, head_frac=0.0):
        self.model, self.ffn_frac, self.head_frac = model, ffn_frac, head_frac
        self.saved = []

    def __enter__(self):
        imp = torch.load(IMP, weights_only=False)
        head_dim = self.model.config.head_dim
        for li, lyr in enumerate(self.model.model.layers):
            if self.ffn_frac:
                w = lyr.mlp.down_proj.weight
                k = int(w.shape[1] * self.ffn_frac)
                drop = imp["ffn"][li].topk(k, largest=False).indices.to(w.device)
                self.saved.append((w, drop, w.data[:, drop].clone()))
                w.data[:, drop] = 0
            if self.head_frac:
                w = lyr.self_attn.o_proj.weight
                nH = w.shape[1] // head_dim
                kh = int(nH * self.head_frac)
                hd = imp["head"][li].topk(kh, largest=False).indices
                cols = (hd[:, None] * head_dim
                        + torch.arange(head_dim)).flatten().to(w.device)
                self.saved.append((w, cols, w.data[:, cols].clone()))
                w.data[:, cols] = 0
        return self

    def __exit__(self, *a):
        for w, idx, orig in self.saved:
            w.data[:, idx] = orig
        self.saved = []


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


ARMS = [("ffn125", 0.125, 0.0), ("ffn25", 0.25, 0.0), ("ffn375", 0.375, 0.0),
        ("ffn50", 0.50, 0.0), ("head25", 0.0, 0.25), ("head50", 0.0, 0.50),
        ("ffn25head25", 0.25, 0.25)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["profile", "score"])
    a = ap.parse_args()
    args = Args()
    model = SA.load_model(CFG)
    if a.stage == "profile":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(CFG["hf"])
        profile(model, tok)
        return 0
    d = done()
    if "kvq_int4" not in d:
        orig = SA.kv_quant
        SA.kv_quant = kv_quant_int4
        row = SA.score_arm(model, CFG, "kvq_fp8", REFDIR, args)
        SA.kv_quant = orig
        row["arm"] = "kvq_int4"
        append_row(row)
    for arm, ff, hf in ARMS:
        if arm in d:
            continue
        with WidthMask(model, ff, hf):
            append_row(SA.score_arm(model, CFG, arm, REFDIR, args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
