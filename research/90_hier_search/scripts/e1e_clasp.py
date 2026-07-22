#!/usr/bin/env python3
"""Phase 90 E1e: CLaSp-class objective (on-policy, context-local,
CONTINUOUS hidden-state cosine) validated against the committed
Qwen3-8B measured record.

Objective per skip-config: mean cosine similarity between the skip
model's final hidden states and the FULL model's, at the on-policy
tail positions of the 86 refs (same positions the committed beta
harness scored). Continuous observable -> hypothesis: better sample
efficiency than binary argmax agreement (E1d's +0.34 ceiling).

Configs: all 36 singles + the 33 committed greedy arms.
"""
import json
import os
import re
import types
from collections import defaultdict
from pathlib import Path

import torch

MODEL = "Qwen/Qwen3-8B"
DEV = "cuda:0"
REFDIR = Path("research/86_scale_headtohead/data/refs/q3_8b_c16384")
N_PROMPTS = int(os.environ.get("E1E_PROMPTS", "6"))
TAIL = 96


def committed_arms():
    per = defaultdict(list)
    for m in re.finditer(
            r"\[B\] igs_([0-9-]+) p\d+: beta=(0\.\d+)",
            open("research/86_scale_headtohead/logs/beta_greedy.log",
                 errors="ignore").read()):
        per[tuple(sorted(int(x) for x in m.group(1).split("-")))].append(
            float(m.group(2)))
    return {k: sum(v) / len(v) for k, v in per.items()}


def main():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEV,
        attn_implementation="sdpa")
    model.eval()
    layers = model.model.layers
    L = len(layers)

    ids_list = [torch.load(REFDIR / f"p{i}_meta.pt", map_location="cpu",
                           weights_only=False)["prompt_ids"].to(DEV)
                for i in range(N_PROMPTS)]

    def tail_hidden(ids):
        with torch.no_grad():
            hs = model(ids, use_cache=False,
                       output_hidden_states=True).hidden_states[-1]
            return hs[0, -TAIL - 1:-1].float()

    base_h = [tail_hidden(ids) for ids in ids_list]
    print(f"[e1e] baselines done ({N_PROMPTS} prompts)", flush=True)

    def identity(layer):
        def fwd(self, hidden_states, *args, **kwargs):
            return hidden_states
        return types.MethodType(fwd, layer)

    def objective(skip_set):
        origs = {}
        for i in skip_set:
            origs[i] = layers[i].forward
            layers[i].forward = identity(layers[i])
        sims = []
        for ids, bh in zip(ids_list, base_h):
            h = tail_hidden(ids)
            sims.append(float(torch.nn.functional.cosine_similarity(
                h, bh, dim=-1).mean()))
        for i, f in origs.items():
            layers[i].forward = f
        return sum(sims) / len(sims)

    res = {"singles": {}, "arms": {}}
    for li in range(L):
        res["singles"][li] = objective({li})
        if (li + 1) % 9 == 0:
            print(f"[e1e] singles {li+1}/{L}", flush=True)
    arms = committed_arms()
    for j, (aset, beta) in enumerate(sorted(arms.items())):
        key = "-".join(map(str, aset))
        res["arms"][key] = {"cos": objective(set(aset)), "beta": beta}
        if (j + 1) % 8 == 0:
            print(f"[e1e] arms {j+1}/{len(arms)}", flush=True)

    out = Path("research/90_hier_search/data/e1e_clasp.json")
    out.write_text(json.dumps(res, indent=1))
    print("[e1e] saved ->", out, flush=True)


if __name__ == "__main__":
    main()
