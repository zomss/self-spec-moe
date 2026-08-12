# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X8 -- where does piecewise paged FA3 win back against the whole-chain graph?

DIAGNOSTIC, NOT SCORED.

X6 showed whole-chain (FULLCG + WHOLECHAIN) beats piecewise by 10.1 ms/step at
R1 and 7.4 ms at R5. X7 showed the kernels whole-chain ADDS grew +74% from R1 to
R5 while those it REMOVES grew +13%, which reads as an approaching crossover.

That reading was confounded and this probe exists to fix it: **R1 is batch 1 and
R5 is batch 8**, so the R1->R5 comparison moved context and batch together. The
distinction matters because the window caps the draft's attention at
window+sinks (272 keys) regardless of context, so context should NOT drive the
draft's cost at all -- only the target's verify, which is identical in both
modes. If that is right, batch is the real axis and the apparent context effect
was batch all along.

Design: batch x context, fully crossed, both modes.

  short context (R1, ~82 prompt tokens):  batch 1, 4, 16, 32
  long context  (R5, ~14k prompt tokens): batch 1, 4, 8, 16

Long-context batches stop at 16 deliberately. The KV cache holds ~353k tokens;
32 x 14k = 449k would exceed it and the scheduler would preempt, silently
changing the effective batch and making the cell a measurement of preemption
rather than of the crossover.

Batch stays <= 32 throughout because the dynamic-K schedule is [[1, 32, 4]] --
above 32 the chain length itself changes and the arms stop being comparable.

Each boot reports wall (no profiler) AND the kernel mix (profiled separately),
so a crossover can be attributed rather than just observed. Whole-chain capture
is VERIFIED per boot from the log: it keys on batch and maintains a failure set,
so a silent non-capture at large batch would look exactly like a crossover.
That is the specific way this measurement could lie, so it is checked.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import regex as re

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402

matrix = r1.matrix

QUANT = "w4a16-quantized"
SKIP = "2,4,7,11,16,20,25,30"
WINDOW = 256

# (regime, batch) -- batch is applied by requesting that many prompts.
CELLS = [
    ("R1", 1),
    ("R1", 4),
    ("R1", 16),
    ("R1", 32),
    ("R5", 1),
    ("R5", 4),
    ("R5", 8),
    ("R5", 16),
]
MODES = {
    "piecewise": {
        "VLLM_SELF_SPEC_DRAFT_WHOLECHAIN": "0",
        "VLLM_SELF_SPEC_DRAFT_FULLCG": "0",
    },
    "wholechain": {
        "VLLM_SELF_SPEC_DRAFT_WHOLECHAIN": "1",
        "VLLM_SELF_SPEC_DRAFT_FULLCG": "1",
    },
}
ORDER = [(regime, batch, mode) for regime, batch in CELLS for mode in MODES]

WARMUP_STEPS = 30
TIMED_STEPS = 50
PROFILED_STEPS = 12

ARMED_MARKER = "Whole-chain graph captured"

FAMILIES = [
    ("fa3_paged", re.compile(r"_vllm_fa3_C|prepare_varlen", re.I)),
    ("flash_splitkv", re.compile(r"splitkv", re.I)),
    ("flash_other", re.compile(r"flash|FlashAttn", re.I)),
    ("w4a16_gemm", re.compile(r"GemmUniversal|machete", re.I)),
    ("bf16_gemm", re.compile(r"nvjet|cublas|aten::mm", re.I)),
    ("kv_write", re.compile(r"reshape_and_cache", re.I)),
    ("scratchpad", re.compile(r"gather|index_select|elementwise|copy|memcpy", re.I)),
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _family(name: str) -> str:
    for label, pattern in FAMILIES:
        if pattern.search(name):
            return label
    return "other"


def measure(regime: str, batch: int, trace: Path, out_json: Path) -> None:
    """Time steps with no profiler, then profile a few for the kernel mix."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    from vllm import LLMEngine, SamplingParams

    cfg = {
        "quant": QUANT,
        "window": WINDOW,
        "skip_count": len([t for t in SKIP.split(",") if t.strip()]),
    }
    prompts = r1._prompts_for(regime, batch)
    _require(
        len(prompts) == batch,
        f"regime {regime} yielded {len(prompts)} prompts, need {batch}",
    )
    engine = LLMEngine.from_engine_args(r1._engine_args(cfg))
    per_step: list[float] = []
    kernels: dict[str, dict[str, Any]] = {}
    try:
        for index, tokens in enumerate(prompts):
            engine.add_request(
                f"{regime}-{index}",
                {"prompt_token_ids": tokens},
                SamplingParams(
                    temperature=0.0, max_tokens=r1.MEASURE_TOKENS, ignore_eos=True
                ),
            )
        for _ in range(WARMUP_STEPS):
            _require(engine.has_unfinished_requests(), "ran out of work in warmup")
            engine.step()
        for _ in range(TIMED_STEPS):
            _require(engine.has_unfinished_requests(), "ran out of work timing")
            t0 = time.perf_counter()
            engine.step()
            per_step.append(time.perf_counter() - t0)
        torch.accelerator.synchronize()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
        ) as prof:
            for _ in range(PROFILED_STEPS):
                _require(engine.has_unfinished_requests(), "ran out of work profiling")
                engine.step()
            torch.accelerator.synchronize()
        for event in prof.key_averages():
            total_us = 0.0
            for attr in ("self_device_time_total", "self_cuda_time_total"):
                value = getattr(event, attr, None)
                if value:
                    total_us = float(value)
                    break
            if total_us <= 0:
                continue
            kernels[event.key] = {
                "family": _family(event.key),
                "calls": int(event.count),
                "device_us_total": total_us,
            }
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()

    _require(bool(kernels), "no device kernels recorded")
    by_family: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "device_us_total": 0.0}
    )
    for info in kernels.values():
        agg = by_family[info["family"]]
        agg["calls"] += info["calls"]
        agg["device_us_total"] += info["device_us_total"]

    out_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_crossover_sweep",
                "regime": regime,
                "batch": batch,
                "window": WINDOW,
                "prompt_tokens_mean": int(statistics.mean(len(t) for t in prompts)),
                "timed_steps": len(per_step),
                "wall_mean_ms": statistics.mean(per_step) * 1000,
                "wall_median_ms": statistics.median(per_step) * 1000,
                "wall_min_ms": min(per_step) * 1000,
                "wall_stdev_ms": statistics.stdev(per_step) * 1000,
                "profiled_steps": PROFILED_STEPS,
                "device_us_per_step": sum(
                    k["device_us_total"] for k in kernels.values()
                )
                / PROFILED_STEPS,
                "by_family": {
                    k: {
                        "calls_per_step": v["calls"] / PROFILED_STEPS,
                        "us_per_step": v["device_us_total"] / PROFILED_STEPS,
                    }
                    for k, v in by_family.items()
                },
                "loadavg_1min": os.getloadavg()[0],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_all(output_dir: Path) -> None:
    lane = matrix.lane_for_block(1)
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = sorted(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    traces = output_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    for regime, batch, mode in ORDER:
        name = f"{regime}_b{batch:02d}_{mode}"
        target = output_dir / f"{name}.json"
        if target.exists():
            continue
        trace = traces / f"{name}.jsonl"
        _require(not trace.exists(), f"trace {trace} already exists")
        cfg = {
            "quant": QUANT,
            "window": WINDOW,
            "skip_count": len([t for t in SKIP.split(",") if t.strip()]),
        }
        env = r1.boot_environment(cfg, trace)
        env["VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS"] = SKIP
        env["VLLM_SELF_SPEC_PROFILE"] = "0"
        env.update(MODES[mode])
        env = matrix._boot_child_environment(env)
        log = output_dir / f"{name}.log"
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output-dir",
                    str(output_dir),
                    "--arm",
                    json.dumps({"name": name, "regime": regime, "batch": batch}),
                    "--trace",
                    str(trace),
                ],
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                preexec_fn=lambda: os.sched_setaffinity(0, affinity),
            )
        if completed.returncode != 0 or not target.exists():
            # A cell may legitimately fail (KV capacity at long context x large
            # batch). Record it and continue rather than losing the whole sweep.
            (output_dir / f"{name}.FAILED").write_text(
                f"returncode={completed.returncode}\n", encoding="utf-8"
            )
            continue
        text = log.read_text(encoding="utf-8", errors="replace")
        record = r1._load_json(target)
        record["wholechain_requested"] = mode == "wholechain"
        record["wholechain_captured"] = ARMED_MARKER in text
        record["preempted"] = "preempt" in text.lower()
        target.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", help=argparse.SUPPRESS)
    parser.add_argument("--trace", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.arm:
        arm = json.loads(args.arm)
        measure(
            arm["regime"],
            arm["batch"],
            Path(args.trace).resolve(),
            output_dir / f"{arm['name']}.json",
        )
        return 0
    base_env = matrix._boot_child_environment({})
    matrix._preflight_native_sampler(base_env)
    matrix._preflight_inprocess_engine_core(base_env)
    lane = matrix.lane_for_block(1)
    matrix._preflight_gpu_identity_and_idle(
        lane["physical_gpu_index"], lane["physical_gpu_uuid"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_all(output_dir)
    print(json.dumps({"cells": len(ORDER)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
