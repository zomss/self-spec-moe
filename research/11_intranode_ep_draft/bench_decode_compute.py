#!/usr/bin/env python3
"""Per-token decode compute latency (single GPU), as the compute floor of a step.

Decode is memory-bound on the active params (attention + top-k expert weights),
so single-GPU per-token latency is a reasonable proxy for the per-step compute
that both draft and verify steps must pay regardless of communication.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=8)
    a = ap.parse_args()

    config = AutoConfig.from_pretrained(a.model)
    model_type = getattr(config, "model_type", "")
    load_kwargs = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if model_type == "gpt_oss":
        from transformers import Mxfp4Config

        load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, **load_kwargs).to(a.device).eval()

    ids = tok("The future of large-scale model serving is", return_tensors="pt").input_ids.to(
        a.device
    )
    with torch.inference_mode():
        out = model(ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)

    times = []
    with torch.inference_mode():
        for i in range(a.warmup + a.decode_steps):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = model(nxt, past_key_values=past, use_cache=True)
            e.record()
            torch.cuda.synchronize()
            past = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
            if i >= a.warmup:
                times.append(s.elapsed_time(e))

    res = {
        "model": a.model,
        "model_type": model_type,
        "decode_steps": a.decode_steps,
        "median_ms_per_token": round(statistics.median(times), 4),
        "mean_ms_per_token": round(statistics.mean(times), 4),
        "p10_ms": round(sorted(times)[len(times) // 10], 4),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
