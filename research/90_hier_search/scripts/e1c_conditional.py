#!/usr/bin/env python3
"""Phase 90 E1c: why do objective-exact singles misrank set growth?

H-cond: damage is non-additive -> screening must be CONDITIONAL on
        the current set. Test: micro-LOO of PAIRS {5, x} at 2k ctx
        vs the committed round-2 arms (measured at 16k).
H-ctx:  layer roles shift with context length. Test: micro-LOO
        singles at 8k ctx vs 2k ctx ranking.
"""
import glob
import gzip
import json
import os
import types

import torch

MODEL = "Qwen/Qwen3-8B"
DEV = "cuda:0"


def refs_c4(tok, n, ref_tok):
    texts, buf = [], ""
    for path in sorted(glob.glob(
            "/data/smcho/huggingface/hub/datasets--allenai--c4/**/"
            "*.json.gz", recursive=True)):
        with gzip.open(path, "rt") as f:
            for line in f:
                buf += json.loads(line)["text"] + "\n\n"
                ids = tok.encode(buf)
                if len(ids) >= ref_tok:
                    texts.append(tok.decode(ids[:ref_tok]))
                    buf = ""
                    if len(texts) >= n:
                        return texts
    return texts


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEV,
        attn_implementation="sdpa")
    model.eval()
    layers = model.model.layers

    def identity(layer):
        def fwd(self, hidden_states, *args, **kwargs):
            return hidden_states
        return types.MethodType(fwd, layer)

    def agreement(texts, base_argmax, skip_set):
        origs = {}
        for i in skip_set:
            origs[i] = layers[i].forward
            layers[i].forward = identity(layers[i])
        agree = tot = 0
        with torch.no_grad():
            for text, basea in zip(texts, base_argmax):
                ids = tok(text, return_tensors="pt").input_ids.to(DEV)
                am = model(ids, use_cache=False).logits[0].argmax(-1)
                agree += int((am == basea).sum())
                tot += am.numel()
        for i, f in origs.items():
            layers[i].forward = f
        return agree / tot

    out = {}

    # ---- H-cond: pairs {5, x} at 2k ----
    texts = refs_c4(tok, 6, 2000)
    with torch.no_grad():
        base = [model(tok(t, return_tensors="pt").input_ids.to(DEV),
                      use_cache=False).logits[0].argmax(-1)
                for t in texts]
    cands = [2, 3, 4, 6, 7, 8, 11, 12, 15, 16, 18, 20, 21]
    pair = {}
    for x in cands:
        pair[x] = agreement(texts, base, {5, x})
        print(f"[e1c] pair {{5,{x}}}: agreement={pair[x]:.4f}", flush=True)
    out["pairs_2k"] = pair

    # ---- H-ctx: singles at 8k vs 2k ----
    probe = [3, 4, 5, 6, 12, 15, 16, 18, 20, 21, 26, 32]
    for ref_tok, tag in [(2000, "2k"), (8000, "8k")]:
        texts = refs_c4(tok, 4 if ref_tok > 4000 else 6, ref_tok)
        with torch.no_grad():
            base = [model(tok(t, return_tensors="pt").input_ids.to(DEV),
                          use_cache=False).logits[0].argmax(-1)
                    for t in texts]
        singles = {}
        for x in probe:
            singles[x] = agreement(texts, base, {x})
        out[f"singles_{tag}"] = singles
        print(f"[e1c] singles@{tag}: " + ", ".join(
            f"L{x}={v:.4f}" for x, v in sorted(
                singles.items(), key=lambda kv: -kv[1])), flush=True)

    p = os.path.dirname(os.path.abspath(__file__)) + "/../data/e1c.json"
    json.dump(out, open(p, "w"), indent=1)
    print("[e1c] saved ->", p, flush=True)


if __name__ == "__main__":
    main()
