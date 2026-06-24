#!/usr/bin/env python3
"""Run a vLLM latency sweep for Phase 02 timing-envelope inputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.sampling_params import SamplingParams


def parse_int_list(raw_value: str) -> list[int]:
    values = [int(value) for value in raw_value.split(",") if value]
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def percentile_summary(latencies: list[float]) -> dict[str, float]:
    percentages = [10, 25, 50, 75, 90, 99]
    values = np.percentile(np.array(latencies), percentages)
    return {str(key): float(value) for key, value in zip(percentages, values)}


def build_prompts(batch_size: int, input_len: int) -> list[PromptType]:
    prompt_token_ids = np.random.randint(10000, size=(batch_size, input_len))
    return [
        {"prompt_token_ids": token_ids}
        for token_ids in prompt_token_ids.tolist()
    ]


def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--batch-sizes",
        type=parse_int_list,
        required=True,
        help="Comma-separated batch sizes to sweep, e.g. 1,2,4,8.",
    )
    parser.add_argument("--input-len", type=int, default=16)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--num-iters-warmup", type=int, default=3)
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help="Exclude detokenization from measured latency.",
    )
    parser = EngineArgs.add_cli_args(parser)
    parser.set_defaults(enable_prefix_caching=False)
    return parser


def main() -> int:
    parser = add_cli_args(argparse.ArgumentParser())
    args = parser.parse_args()

    from vllm import LLM

    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM.from_engine_args(engine_args)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
        detokenize=not args.disable_detokenize,
    )

    results: dict[str, Any] = {
        "model": args.model,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_iters_warmup": args.num_iters_warmup,
        "num_iters": args.num_iters,
        "batch_results": {},
    }

    for batch_size in args.batch_sizes:
        prompts = build_prompts(batch_size, args.input_len)

        def run_once() -> float:
            start = time.perf_counter()
            llm.generate(
                prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            return time.perf_counter() - start

        for _ in tqdm(
            range(args.num_iters_warmup),
            desc=f"warmup batch={batch_size}",
        ):
            run_once()

        latencies = [
            run_once()
            for _ in tqdm(
                range(args.num_iters),
                desc=f"bench batch={batch_size}",
            )
        ]
        results["batch_results"][str(batch_size)] = {
            "avg_latency": float(np.mean(latencies)),
            "latencies": latencies,
            "percentiles": percentile_summary(latencies),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
