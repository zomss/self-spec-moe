#!/usr/bin/env python3
"""Phase 90 E1d: micro-LOO singles ON-POLICY (the committed harness's
own refs/positions) — isolates the ref-distribution factor.

For each committed 86 ref prompt (16k target-generated context), skip
one layer at a time and measure argmax agreement with the full model
at the generated-tail positions (the same paired positions the
committed beta used). If this matches committed LOO@16k, the earlier
anti-correlation was REF DISTRIBUTION (off-policy raw text misranks).
"""
import json
import os
import types
from pathlib import Path

import torch

MODEL = "Qwen/Qwen3-8B"
DEV = "cuda:0"
REFDIR = Path("research/86_scale_headtohead/data/refs/q3_8b_c16384")
N_PROMPTS = int(os.environ.get("E1D_PROMPTS", "6"))
TAIL = 96


def main():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEV,
        attn_implementation="sdpa")
    model.eval()
    layers = model.model.layers
    L = len(layers)

    metas = []
    for i in range(N_PROMPTS):
        m = torch.load(REFDIR / f"p{i}_meta.pt", map_location="cpu",
                       weights_only=False)
        metas.append(m["prompt_ids"].to(DEV))

    def tail_argmax(ids):
        with torch.no_grad():
            z = model(ids, use_cache=False).logits[0, -TAIL - 1:-1]
            return z.argmax(-1)

    base = [tail_argmax(ids) for ids in metas]
    print(f"[e1d] baselines done ({N_PROMPTS} prompts x {TAIL} pos)",
          flush=True)

    def identity(layer):
        def fwd(self, hidden_states, *args, **kwargs):
            return hidden_states
        return types.MethodType(fwd, layer)

    res = {}
    for li in range(L):
        orig = layers[li].forward
        layers[li].forward = identity(layers[li])
        agree = tot = 0
        for ids, b in zip(metas, base):
            am = tail_argmax(ids)
            agree += int((am == b).sum())
            tot += am.numel()
        layers[li].forward = orig
        res[li] = agree / tot
        if (li + 1) % 6 == 0:
            print(f"[e1d] {li+1}/{L} (last agree={res[li]:.3f})",
                  flush=True)

    out = Path("research/90_hier_search/data/e1d_onpolicy.json")
    out.write_text(json.dumps(res, indent=1))
    print("[e1d] saved ->", out, flush=True)


if __name__ == "__main__":
    main()
