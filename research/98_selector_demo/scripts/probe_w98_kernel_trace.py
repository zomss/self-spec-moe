# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X2 -- per-kernel profile of the fast and slow quantized draft forwards.

DIAGNOSTIC, NOT SCORED.

Splits the one question every surviving hypothesis depends on:

    Is DIFFERENT code running, or is the SAME code running slower?

Different kernels  -> the W4A16 path is not being taken (D1), or inductor emitted
                      different code for the larger compiled pieces (C1).
Same kernels, slower -> the right code is running against worse memory (E1) or a
                      worse schedule (D2).

Method: torch.profiler over a fixed number of armed steps in each arm, CUDA
activities only, then aggregate device kernels by name. Both arms run the same
prompts for the same number of steps, so the kernel MIX (names, call counts) and
the per-call durations are directly comparable.

The arms are the two that bracket the threshold at a single window, plus the
bf16 reference whose line the slow arm has crossed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import regex as re

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402

matrix = r1.matrix

ARMS = [
    ("quant_skip4", "w4a16-quantized", "2,4,7,16"),
    ("quant_skip8", "w4a16-quantized", "2,4,7,11,16,20,25,30"),
    ("bf16_skip8", "target-matching", "2,4,7,11,16,20,25,30"),
    ("quant_skip0", "w4a16-quantized", ""),
]

WINDOW = 256
REGIME = "R1"
# Enough steps to average out launch jitter, few enough that the trace stays
# small and the profiler's own overhead does not dominate.
WARMUP_STEPS = 40
PROFILED_STEPS = 25

# Kernel families we expect to see; anything unmatched is reported verbatim so
# an unanticipated kernel cannot hide inside an "other" bucket.
FAMILIES = [
    ("machete", re.compile(r"machete", re.I)),
    ("marlin", re.compile(r"marlin", re.I)),
    ("cutlass_gemm", re.compile(r"cutlass.*(gemm|mma)", re.I)),
    ("cublas_gemm", re.compile(r"(cublas|gemm|sm\d+_xmma)", re.I)),
    ("dequant", re.compile(r"dequant|unpack|awq|gptq", re.I)),
    ("attention", re.compile(r"flash|attn|paged", re.I)),
    ("norm_act", re.compile(r"rms_?norm|silu|swiglu|rotary|layernorm", re.I)),
    ("elementwise", re.compile(r"elementwise|vectorized|copy|cat_", re.I)),
    ("reduce", re.compile(r"reduce|softmax|topk|argmax|sort", re.I)),
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _family(name: str) -> str:
    for label, pattern in FAMILIES:
        if pattern.search(name):
            return label
    return "unclassified"


def profile_arm(quant: str, skip: str, trace: Path, out_json: Path) -> None:
    """Boot one arm, profile a fixed number of armed steps, aggregate kernels."""
    import time

    import torch
    from torch.profiler import ProfilerActivity, profile

    from vllm import LLMEngine, SamplingParams

    cfg = {
        "quant": quant,
        "window": WINDOW,
        "skip_count": len([t for t in skip.split(",") if t.strip()]),
    }
    manifest = r1._load_json(matrix._repository_path(r1.PROMPT_MANIFEST))
    spec = {r["regime_id"]: r for r in manifest["prompt_plan"]["regimes"]}[REGIME]
    prompts = r1._prompts_for(REGIME, spec["batch"])
    engine = LLMEngine.from_engine_args(r1._engine_args(cfg))
    try:
        for index, tokens in enumerate(prompts):
            engine.add_request(
                f"{REGIME}-{index}",
                {"prompt_token_ids": tokens},
                SamplingParams(
                    temperature=0.0,
                    max_tokens=r1.MEASURE_TOKENS,
                    ignore_eos=True,
                ),
            )
        for _ in range(WARMUP_STEPS):
            _require(engine.has_unfinished_requests(), "ran out of work in warmup")
            engine.step()
        torch.accelerator.synchronize()
        wall_t0 = time.perf_counter()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
        ) as prof:
            for _ in range(PROFILED_STEPS):
                _require(engine.has_unfinished_requests(), "ran out of work profiling")
                engine.step()
            torch.accelerator.synchronize()
        wall_s = time.perf_counter() - wall_t0
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()

    kernels: dict[str, dict[str, Any]] = {}
    averages = prof.key_averages()
    _require(bool(len(averages)), "profiler returned no events")
    for event in averages:
        # Device-side work only; host events are covered by the fine-region pass.
        # torch renamed this across versions; try both rather than silently
        # aggregating zeros into an empty result.
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
            "device_us_total": float(total_us),
            "device_us_per_call": float(total_us) / max(int(event.count), 1),
        }
    by_family: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "device_us_total": 0.0, "distinct_kernels": 0}
    )
    for info in kernels.values():
        agg = by_family[info["family"]]
        agg["calls"] += info["calls"]
        agg["device_us_total"] += info["device_us_total"]
        agg["distinct_kernels"] += 1
    _require(
        bool(kernels),
        "no device kernels recorded; the profiler attribute names changed",
    )
    # Host-side CUDA API calls. The device totals show the GPU doing LESS work
    # at skip8 while wall time rises, so the cost is idle time; these counters
    # say whether the host is launching more, or blocking.
    host_api: dict[str, dict[str, Any]] = {}
    for event in averages:
        key = event.key
        if not (
            key.startswith(("cuda", "Memcpy", "Memset"))
            or "Launch" in key
            or "Graph" in key
            or "ynchronize" in key
        ):
            continue
        cpu_us = float(getattr(event, "self_cpu_time_total", 0) or 0)
        host_api[key] = {
            "calls": int(event.count),
            "cpu_us_total": cpu_us,
            "cpu_us_per_call": cpu_us / max(int(event.count), 1),
        }
    out_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_kernel_trace",
                "quant": quant,
                "skip_layers": skip,
                "skip_count": cfg["skip_count"],
                "window": WINDOW,
                "regime": REGIME,
                "profiled_steps": PROFILED_STEPS,
                # Measured in the SAME run as the device totals, region
                # profiler OFF, so busy/idle is internally consistent. A
                # region-profiled run would be invalid here: it syncs ~10x per
                # step and drains the very overlap being measured.
                "wall_s_total": wall_s,
                "wall_ms_per_step": wall_s * 1000 / PROFILED_STEPS,
                "device_us_total": sum(k["device_us_total"] for k in kernels.values()),
                "by_family": dict(by_family),
                "host_api": host_api,
                "kernels": kernels,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_arms(output_dir: Path) -> None:
    lane = matrix.lane_for_block(1)
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = sorted(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    traces = output_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    for name, quant, skip in ARMS:
        target = output_dir / f"{name}.json"
        if target.exists():
            continue
        trace = traces / f"{name}.jsonl"
        _require(not trace.exists(), f"trace {trace} already exists")
        cfg = {
            "quant": quant,
            "window": WINDOW,
            "skip_count": len([t for t in skip.split(",") if t.strip()]),
        }
        env = r1.boot_environment(cfg, trace)
        env["VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS"] = skip
        # The profiler's own syncs would perturb the region timers; X2 measures
        # kernels, not regions, so the region profiler stays off.
        env["VLLM_SELF_SPEC_PROFILE"] = "0"
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
                    json.dumps({"name": name, "quant": quant, "skip": skip}),
                    "--trace",
                    str(trace),
                ],
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                preexec_fn=lambda: os.sched_setaffinity(0, affinity),
            )
        _require(
            completed.returncode == 0 and target.exists(),
            f"arm {name} failed; output preserved at {log}",
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
        profile_arm(
            arm["quant"],
            arm["skip"],
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
    run_arms(output_dir)
    print(json.dumps({"arms": [a[0] for a in ARMS]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
