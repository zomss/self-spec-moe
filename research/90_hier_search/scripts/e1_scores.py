#!/usr/bin/env python3
"""Phase 90 E1: importance-proxy scores from ONE forward pass.

Per model x ref-set, computes:
  (a) per-layer IMPORTANCE (angular distance): 1 - cos(h_l, h_{l+1})
      averaged over positions -- the skip-lever proxy.
  (b) per-(layer, head) ATTENTION LOCALITY: attention mass on keys
      OLDER than window W (excluding the first SINKS sink positions),
      averaged over eligible queries -- the window-lever proxy
      ("what a sinks+W draft cannot see").

env: MODEL, REFS (c4|cnndm), N_REFS, REF_TOK, DEV, OUT_PREFIX
"""
import glob
import gzip
import json
import os

import torch

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
REFS = os.environ.get("REFS", "c4")
N_REFS = int(os.environ.get("N_REFS", "6"))
REF_TOK = int(os.environ.get("REF_TOK", "3000"))
DEV = os.environ.get("DEV", "cuda:0")
OUT = os.environ.get("OUT_PREFIX", "scores")
SINKS = 16
WINDOWS = [128, 512, 2048, 4096]

_acc = {}   # (layer, head, W) -> [sum_mass, n_queries]
_imp = {}   # layer -> [sum_angular, n]


def make_ref_texts(tok):
    texts = []
    if REFS == "c4":
        buf = ""
        for path in sorted(glob.glob(
                "/data/smcho/huggingface/hub/datasets--allenai--c4/**/"
                "*.json.gz", recursive=True)):
            with gzip.open(path, "rt") as f:
                for line in f:
                    buf += json.loads(line)["text"] + "\n\n"
                    ids = tok.encode(buf)
                    if len(ids) >= REF_TOK:
                        texts.append(tok.decode(ids[:REF_TOK]))
                        buf = ""
                        if len(texts) >= N_REFS:
                            return texts
    elif REFS == "cnndm":
        from datasets import load_dataset
        ds = load_dataset("abisee/cnn_dailymail", "3.0.0",
                          split="test[:2000]")
        parts, texts_out = [], []
        for ex in ds:
            parts.append(ex["article"])
            body = "\n\n---\n\n".join(parts)
            if len(tok.encode(body)) >= REF_TOK:
                texts_out.append(tok.decode(tok.encode(body)[:REF_TOK]))
                parts = []
                if len(texts_out) >= N_REFS:
                    return texts_out
        return texts_out
    return texts


def record_attn(layer_idx, attn):
    # attn: [1, heads, q, k] post-softmax
    q = attn.shape[-2]
    dev = attn.device
    qi = torch.arange(q, device=dev).unsqueeze(1)
    ki = torch.arange(q, device=dev).unsqueeze(0)
    for W in WINDOWS:
        if q <= W + SINKS + 1:
            continue
        mask = (ki >= SINKS) & (ki <= qi - W - 1)          # [q, k]
        elig = qi.squeeze(1) >= (W + SINKS + 1)            # [q]
        n_elig = int(elig.sum())
        # per head: sum over masked keys, mean over eligible queries
        m = (attn[0] * mask.unsqueeze(0)).sum(-1)          # [heads, q]
        m = m[:, elig].sum(-1)                             # [heads]
        for h in range(attn.shape[1]):
            key = (layer_idx, h, W)
            s = _acc.setdefault(key, [0.0, 0])
            s[0] += float(m[h])
            s[1] += n_elig


def patch_eager(modeling):
    orig = modeling.eager_attention_forward

    def wrapped(module, query, key, value, attention_mask, *args, **kw):
        out, attn = orig(module, query, key, value, attention_mask,
                         *args, **kw)
        with torch.no_grad():
            record_attn(module.layer_idx, attn.float())
        return out, attn

    modeling.eager_attention_forward = wrapped


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    texts = make_ref_texts(tok)
    print(f"[e1] {MODEL} refs={REFS} n={len(texts)} x ~{REF_TOK} tok",
          flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map=DEV,
        attn_implementation="eager")
    model.eval()
    mod_name = type(model).__module__
    import importlib
    patch_eager(importlib.import_module(mod_name))

    for ti, text in enumerate(texts):
        ids = tok(text, return_tensors="pt").input_ids.to(DEV)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # (L+1) x [1, q, d]
        for li in range(len(hs) - 1):
            a = hs[li][0].float()
            b = hs[li + 1][0].float()
            cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
            s = _imp.setdefault(li, [0.0, 0])
            s[0] += float((1 - cos).sum())
            s[1] += cos.numel()
        del out, hs
        torch.cuda.empty_cache()
        print(f"[e1] ref {ti+1}/{len(texts)} done", flush=True)

    base = os.path.dirname(os.path.abspath(__file__)) + "/../data"
    with open(f"{base}/{OUT}_locality.csv", "w") as f:
        f.write("model,refs,layer,head,window,beyond_mass\n")
        for (li, h, W), (s, n) in sorted(_acc.items()):
            f.write(f"{MODEL},{REFS},{li},{h},{W},{s/max(n,1):.6f}\n")
    with open(f"{base}/{OUT}_importance.csv", "w") as f:
        f.write("model,refs,layer,importance\n")
        for li, (s, n) in sorted(_imp.items()):
            f.write(f"{MODEL},{REFS},{li},{s/max(n,1):.6f}\n")
    print(f"[e1] saved -> {base}/{OUT}_*.csv", flush=True)


if __name__ == "__main__":
    main()
