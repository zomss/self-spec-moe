#!/usr/bin/env python3
"""Run a multi-process vLLM DP+EP latency sweep for Phase 02."""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Barrier, Process, Queue
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np

from vllm import EngineArgs
from vllm.inputs import PromptType
from vllm.sampling_params import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import get_open_port


def parse_int_list(raw_value: str) -> list[int]:
    values = [int(value) for value in raw_value.split(",") if value]
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def local_batch_size(global_batch_size: int, dp_size: int, rank: int) -> int:
    floor = global_batch_size // dp_size
    remainder = global_batch_size % dp_size
    return floor + int(rank < remainder)


def build_prompts(batch_size: int, input_len: int) -> list[PromptType]:
    token_ids = np.random.randint(10000, size=(batch_size, input_len))
    return [{"prompt_token_ids": row} for row in token_ids.tolist()]


def percentile_summary(latencies: list[float]) -> dict[str, float]:
    percentages = [10, 25, 50, 75, 90, 99]
    values = np.percentile(np.array(latencies), percentages)
    return {str(key): float(value) for key, value in zip(percentages, values)}


def create_parser() -> FlexibleArgumentParser:
    parser = FlexibleArgumentParser(description="DP+EP latency sweep")
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(enable_prefix_caching=False)
    parser.add_argument("--batch-sizes", type=parse_int_list, required=True)
    parser.add_argument("--input-len", type=int, default=1)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--num-iters-warmup", type=int, default=3)
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    return parser


def run_rank(
    *,
    dp_size: int,
    local_rank: int,
    master_ip: str,
    master_port: int,
    engine_args: dict[str, Any],
    batch_sizes: list[int],
    input_len: int,
    output_len: int,
    num_iters_warmup: int,
    num_iters: int,
    barrier: Barrier,
    queue: Queue,
) -> None:
    os.environ["VLLM_DP_RANK"] = str(local_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    from vllm import LLM

    llm = LLM(**engine_args)
    count_active_file = os.getenv("VLLM_SELF_SPEC_A2A_COUNT_ACTIVE_FILE")
    count_active_file_started = False
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=output_len,
        detokenize=False,
    )

    rank_results: dict[str, Any] = {}
    for global_batch_size in batch_sizes:
        rank_batch_size = local_batch_size(global_batch_size, dp_size, local_rank)
        prompts = build_prompts(max(rank_batch_size, 1), input_len)
        if rank_batch_size == 0:
            prompts = []

        def run_once() -> float:
            barrier.wait()
            start = time.perf_counter()
            if prompts:
                llm.generate(
                    prompts,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
            else:
                # Ranks with no local requests still need to stay in lockstep.
                # vLLM's DP coordinator should issue dummy forwards if needed.
                time.sleep(0.0)
            latency = time.perf_counter() - start
            barrier.wait()
            return latency

        for _ in range(num_iters_warmup):
            run_once()

        if count_active_file and not count_active_file_started:
            Path(count_active_file).parent.mkdir(parents=True, exist_ok=True)
            Path(count_active_file).touch()
            count_active_file_started = True

        latencies = [run_once() for _ in range(num_iters)]
        rank_results[str(global_batch_size)] = {
            "local_batch_size": rank_batch_size,
            "avg_latency": float(np.mean(latencies)),
            "latencies": latencies,
            "percentiles": percentile_summary(latencies),
        }

    queue.put((local_rank, rank_results))


def main() -> int:
    parser = create_parser()
    args = vars(parser.parse_args())

    dp_size = args.pop("data_parallel_size")
    timeout = args.pop("timeout")
    batch_sizes = args.pop("batch_sizes")
    input_len = args.pop("input_len")
    output_len = args.pop("output_len")
    num_iters_warmup = args.pop("num_iters_warmup")
    num_iters = args.pop("num_iters")
    output_json = args.pop("output_json")

    if dp_size <= 1:
        parser.error("--data-parallel-size must be > 1")

    master_ip = "127.0.0.1"
    master_port = get_open_port()
    barrier = Barrier(dp_size)
    queue: Queue = Queue()

    processes = []
    for local_rank in range(dp_size):
        proc = Process(
            target=run_rank,
            kwargs={
                "dp_size": dp_size,
                "local_rank": local_rank,
                "master_ip": master_ip,
                "master_port": master_port,
                "engine_args": args,
                "batch_sizes": batch_sizes,
                "input_len": input_len,
                "output_len": output_len,
                "num_iters_warmup": num_iters_warmup,
                "num_iters": num_iters,
                "barrier": barrier,
                "queue": queue,
            },
        )
        proc.start()
        processes.append(proc)

    exit_code = 0
    for proc in processes:
        proc.join(timeout=timeout)
        if proc.exitcode is None:
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    if exit_code:
        return exit_code

    per_rank = {}
    try:
        while len(per_rank) < dp_size:
            rank, rank_results = queue.get(timeout=5)
            per_rank[str(rank)] = rank_results
    except Empty:
        return 1

    batch_results = {}
    for batch_size in batch_sizes:
        key = str(batch_size)
        max_latencies = []
        for iteration in range(num_iters):
            max_latencies.append(
                max(
                    per_rank[str(rank)][key]["latencies"][iteration]
                    for rank in range(dp_size)
                )
            )
        batch_results[key] = {
            "avg_latency": float(np.mean(max_latencies)),
            "latencies": max_latencies,
            "percentiles": percentile_summary(max_latencies),
        }

    output = {
        "input_len": input_len,
        "output_len": output_len,
        "num_iters_warmup": num_iters_warmup,
        "num_iters": num_iters,
        "dp_size": dp_size,
        "batch_results": batch_results,
        "per_rank": per_rank,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(output, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
