#!/usr/bin/env python3
"""Phase 90 E3: heterogeneous per-layer draft windows for R4
(summarization), searched by DIRECT conditional on-policy measurement.

Protocol (per the E1 verdict -- no proxies):
  1. R4 on-policy refs: CNN/DM ~8k articles -> the target's own
     greedy 96-token summaries.
  2. Per-layer window masks (eager-attention patch): each layer runs
     either FULL causal attention or win512+sink16.
  3. Anchors: all-full (ceiling) and all-win512 (the accept-2.70
     regime). LOO relief singles (one layer full, rest windowed) ->
     measured ranking -> nested top-k sets (k=2,4,6,9,12) + controls
     (random-k, bottom-k).
  4. Gate P3-analogue: some k<=9 set recovers >=70% of the
     win512->full agreement gap.
env: E3H_PROMPTS (6), E3H_TOK (8000), plus HF_HOME etc.
"""
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, "research/88_regime_eval/scripts")

MODEL = "Qwen/Qwen3-8B"
DEV = "cuda:0"
N_PROMPTS = int(os.environ.get("E3H_PROMPTS", "6"))
REF_TOK = int(os.environ.get("E3H_TOK", "8000"))
GEN = 96
WIN, SINKS = 512, 16

FULL_SET = set()          # layers with full ctx (mutated per config)
MASKS = {}                # seq_len -> (mask_full, mask_win) additive


def build_masks(q, dtype, dev):
    if q in MASKS:
        return MASKS[q]
    neg = torch.finfo(dtype).min
    qi = torch.arange(q, device=dev).view(-1, 1)
    ki = torch.arange(q, device=dev).view(1, -1)
    causal = ki <= qi
    winok = causal & ((ki < SINKS) | ((qi - ki) < WIN))
    mf = torch.where(causal, 0.0, neg).to(dtype)[None, None]
    mw = torch.where(winok, 0.0, neg).to(dtype)[None, None]
    MASKS[q] = (mf, mw)
    return MASKS[q]


def patch_eager(modeling):
    orig = modeling.eager_attention_forward

    def wrapped(module, query, key, value, attention_mask, *args, **kw):
        q = query.shape[-2]
        if q > 1 and key.shape[-2] == q:
            mf, mw = build_masks(q, query.dtype, query.device)
            attention_mask = mf if module.layer_idx in FULL_SET else mw
        return orig(module, query, key, value, attention_mask,
                    *args, **kw)

    modeling.eager_attention_forward = wrapped


def main():
    import importlib

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from regime_datasets import load_regime

    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts, _ = load_regime("R4", tok, n=N_PROMPTS, ctx_target=REF_TOK)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEV,
        attn_implementation="eager")
    model.eval()
    L = len(model.model.layers)

    # on-policy refs: full-attention greedy summaries (pre-patch)
    seqs = []
    for i, p in enumerate(prompts):
        ids = tok(p, return_tensors="pt").input_ids.to(DEV)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=GEN,
                                 do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        seqs.append(out)
        print(f"[e3] ref {i+1}/{N_PROMPTS} generated "
              f"({out.shape[-1]} tok)", flush=True)

    patch_eager(importlib.import_module(type(model).__module__))

    def agreement(full_set):
        global FULL_SET
        FULL_SET = set(full_set)
        agree = tot = 0
        with torch.no_grad():
            for s in seqs:
                z = model(s, use_cache=False).logits[0, -GEN - 1:-1]
                am = z.argmax(-1)
                agree += int((am == s[0, -GEN:]).sum())
                tot += GEN
        return agree / tot

    res = {}
    res["all_full"] = agreement(set(range(L)))
    res["all_win"] = agreement(set())
    gap = res["all_full"] - res["all_win"]
    print(f"[e3] anchors: all_full={res['all_full']:.4f} "
          f"all_win512={res['all_win']:.4f} gap={gap:.4f}", flush=True)

    relief = {}
    for li in range(L):
        relief[li] = agreement({li})
        if (li + 1) % 6 == 0:
            print(f"[e3] relief {li+1}/{L}", flush=True)
    res["relief_singles"] = relief

    order = sorted(relief, key=relief.get, reverse=True)
    print(f"[e3] top relief layers: {order[:12]}", flush=True)
    res["sets"] = {}
    for k in (2, 4, 6, 9, 12):
        topk = order[:k]
        a = agreement(set(topk))
        res["sets"][f"top{k}"] = {"layers": topk, "agree": a,
                                  "gap_recovered":
                                  (a - res["all_win"]) / max(gap, 1e-9)}
        print(f"[e3] top{k} {topk}: agree={a:.4f} "
              f"({res['sets'][f'top{k}']['gap_recovered']*100:.0f}% of gap)",
              flush=True)
    # controls
    import random
    random.seed(7)
    rnd = random.sample(range(L), 9)
    res["sets"]["rand9"] = {"layers": rnd, "agree": agreement(set(rnd))}
    bot = order[-9:]
    res["sets"]["bottom9"] = {"layers": bot, "agree": agreement(set(bot))}
    print(f"[e3] controls: rand9={res['sets']['rand9']['agree']:.4f} "
          f"bottom9={res['sets']['bottom9']['agree']:.4f}", flush=True)

    out = Path("research/90_hier_search/data/e3_hetero.json")
    out.write_text(json.dumps(res, indent=1))
    print("[e3] saved ->", out, flush=True)


if __name__ == "__main__":
    main()
