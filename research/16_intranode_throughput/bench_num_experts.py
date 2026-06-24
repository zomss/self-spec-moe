#!/usr/bin/env python3
"""Real end-to-end vLLM decode latency vs num_experts (single GPU, no EP).

Overriding num_experts (via hf_overrides, dummy weights) keeps attention identical
and changes only how many expert weights the MoE layer reads. So:
  T(E) = T_attn_and_other + T_moe(E),  T_moe(E) ~ proportional to active experts.
  T_draft/T_full = T(E/2)/T(E)   (the phi=0.5 draft vs full routing, end-to-end)
A linear fit T(E) = a + b*E backs out phi_moe = b*E / (a + b*E).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def parse_int_list(raw: str) -> list[int]:
    return [int(v) for v in raw.split(",") if v.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--num-experts", type=int, required=True)
    ap.add_argument("--batches", type=parse_int_list, default=[64, 256])
    ap.add_argument("--input-len", type=int, default=8)
    ap.add_argument("--output-len", type=int, default=48)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--output-json", type=Path, required=True)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=a.model,
        tensor_parallel_size=1,
        load_format="dummy",
        hf_overrides={"num_experts": a.num_experts},
        enforce_eager=False,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )
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
        print(f"[E={a.num_experts}] B={B}: {ms_tok:.4f} ms/token")

    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps({"num_experts": a.num_experts, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
