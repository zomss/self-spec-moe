#!/usr/bin/env python3
"""Real vLLM decode latency vs weight quantization (single GPU).

MoE decode is memory-bound on expert weights (phi_moe~0.7, Phase 16). Quantizing
weights (FP8 ~0.5x bytes, FP4 ~0.25x) cuts that read directly. Decode-latency ratio
bf16/quant is the throughput gain -- and it composes with everything else.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def parse_int_list(raw):
    return [int(v) for v in raw.split(",") if v.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--quantization", default="none")  # none | fp8 | ...
    ap.add_argument("--num-experts", type=int, default=0)  # 0 = model default; else hf_override
    ap.add_argument("--batches", type=parse_int_list, default=[64, 256])
    ap.add_argument("--input-len", type=int, default=8)
    ap.add_argument("--output-len", type=int, default=48)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--output-json", type=Path, required=True)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=a.model,
        tensor_parallel_size=1,
        load_format="dummy",
        enforce_eager=False,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )
    if a.quantization != "none":
        kwargs["quantization"] = a.quantization
    if a.num_experts > 0:
        kwargs["hf_overrides"] = {"num_experts": a.num_experts}
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=a.output_len, detokenize=False)
    rng = np.random.default_rng(0)
    rows = []
    for B in a.batches:
        prompts = [{"prompt_token_ids": rng.integers(10000, size=a.input_len).tolist()} for _ in range(B)]

        def once():
            t = time.perf_counter()
            llm.generate(prompts, sampling_params=sp, use_tqdm=False)
            return time.perf_counter() - t

        for _ in range(a.warmup):
            once()
        lat = [once() for _ in range(a.iters)]
        ms_tok = float(np.median(lat)) * 1000.0 / a.output_len
        rows.append({"batch": B, "ms_per_token": round(ms_tok, 4)})
        print(f"[{a.quantization}] B={B}: {ms_tok:.4f} ms/token")

    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps({"model": a.model, "quantization": a.quantization, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
