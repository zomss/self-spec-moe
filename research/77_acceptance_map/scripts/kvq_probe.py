#!/usr/bin/env python3
"""Probe the kvq_fp8 instability (dense beta 0.75/0.84/0.60 across ctx).

Variants of the draft-only fp8 KV pool, scored against the SAME stage-A refs
as the sweep (paired):

  base        K+V cast e4m3 scale 1.0 (the sweep arm / vLLM's uncalibrated default)
  k_only      only K quantized -> which side carries the error
  v_only      only V quantized
  keep_sinks  K+V quantized EXCEPT the first 16 tokens (sink-outlier hypothesis:
              sink K norms are outliers; quantizing them warps attention globally)
  scaled      per-token-per-head amax scaling (the calibrated-scales ceiling;
              what a real draft-only pool with stored scales would do)

Prints per-prompt beta for `base` (bimodality check) and the aggregate table.
Usage: CUDA_VISIBLE_DEVICES=1 python kvq_probe.py [--ctx 2048 16384 32768]
"""

import argparse
import sys
from pathlib import Path

import torch

PHASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PHASE / "scripts"))
from score_accept import MODELS, load_model, view_cache  # noqa: E402


def q_e4m3(t):
    return t.clamp(-448, 448).to(torch.float8_e4m3fn).to(t.dtype)


def q_scaled(t):
    # per-token-per-head amax scale (dims: [b, h, s, d] -> scale over d)
    s = t.float().abs().amax(-1, keepdim=True).clamp(min=1e-8) / 448.0
    return ((t.float() / s).to(torch.float8_e4m3fn).float() * s).to(t.dtype)


def q_kchan(t):
    # per-CHANNEL scale for K (KIVI-style: K outliers live in fixed channels;
    # scale over the sequence dim, shared across tokens -> storable in a pool)
    s = t.float().abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 448.0
    return ((t.float() / s).to(torch.float8_e4m3fn).float() * s).to(t.dtype)


VARIANTS = {
    "base": lambda k, v: (q_e4m3(k), q_e4m3(v)),
    "k_only": lambda k, v: (q_e4m3(k), v),
    "v_only": lambda k, v: (k, q_e4m3(v)),
    "keep_sinks": lambda k, v: (
        torch.cat([k[:, :, :16], q_e4m3(k[:, :, 16:])], 2),
        torch.cat([v[:, :, :16], q_e4m3(v[:, :, 16:])], 2)),
    "scaled": lambda k, v: (q_scaled(k), q_scaled(v)),
    "k_chan": lambda k, v: (q_kchan(k), q_e4m3(v)),
}


@torch.no_grad()
def score(model, n_layers, refdir, transform, prompts, per_prompt=False):
    dev = "cuda"
    matches = []
    pp = []
    for pi in range(prompts):
        meta = torch.load(refdir / f"p{pi}_meta.pt", weights_only=False)
        kv_cpu = torch.load(refdir / f"p{pi}_cache.pt", weights_only=False)
        kv = {}
        for li in kv_cpu:
            k, v = (t.to(dev) for t in kv_cpu[li])
            kv[li] = transform(k, v)
        T = meta["prompt_ids"].shape[1]
        m0 = len(matches)
        for step, cur in enumerate(meta["gen_tokens"]):
            seqlen = T + step
            dcache = view_cache(kv, n_layers, seqlen)
            out = model(input_ids=torch.tensor([[cur]], device=dev),
                        past_key_values=dcache, use_cache=True,
                        position_ids=torch.tensor([[seqlen]], device=dev),
                        cache_position=torch.tensor([dcache.get_seq_length()], device=dev))
            dp = out.logits[:, -1].float()
            tp = meta["probs"][step]
            matches.append(float(int(dp.argmax()) == int(tp.argmax())))
            del dcache, out
        pp.append(sum(matches[m0:]) / (len(matches) - m0))
        del kv, kv_cpu
        torch.cuda.empty_cache()
    if per_prompt:
        print(f"    per-prompt: {[round(x, 3) for x in pp]}")
    return sum(matches) / len(matches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, nargs="+", default=[2048, 16384, 32768])
    ap.add_argument("--prompts", type=int, default=12)
    a = ap.parse_args()
    cfg = MODELS["dense"]
    model = load_model(cfg)
    n_layers = model.config.num_hidden_layers

    print(f"{'variant':12s} " + "  ".join(f"c{c//1024}k" for c in a.ctx))
    for name, tf in VARIANTS.items():
        vals = []
        for ctx in a.ctx:
            refdir = PHASE / f"data/refs/dense_c{ctx}"
            if name == "base":
                print(f"  base c{ctx//1024}k:")
            vals.append(score(model, n_layers, refdir, tf, a.prompts,
                              per_prompt=(name == "base")))
        print(f"{name:12s} " + "  ".join(f"{v:.4f}" for v in vals), flush=True)


if __name__ == "__main__":
    main()
