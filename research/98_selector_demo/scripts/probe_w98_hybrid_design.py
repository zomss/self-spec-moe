# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X14 -- can one boot config serve both small and large batch?

DIAGNOSTIC, NOT SCORED.

X13 showed whole-chain wins 1.4x at batch 1 and is worth ~1% at batch >= 32. A
system that must serve both needs either two deployments or one that switches.

The two flags are NOT the same kind of thing, and that is the opening:

  FULLCG      boot-time. Swaps paged FA3 for scratchpad+splitkv. This is where
              the CONSTANT -4.7 ms/step device gain comes from.
  WHOLECHAIN  per-step. Captures the K-step chain as one graph, keyed on
              (batch, table_width, K). This is where the BUBBLE effect comes
              from: -5.6 ms at batch 1, +4.7 ms at batch 32+.

Every measurement so far has been (0,0) or (1,1). **(1,0) is untested**: the
cheaper kernel with a host-driven chain. If the large-batch penalty comes from
the graph rather than the kernel, (1,0) keeps the device gain without the
bubble cost -- and then ONE boot config (FULLCG on) plus a per-step batch gate
on whole-chain serves both regimes, because whole-chain arming is already
per-step and already batch-keyed.

Three modes x four batches. Reading rule, fixed before the data:

  * (1,0) ~ (1,1) at small batch      -> the graph is not what wins; simplify
  * (1,0) beats (1,1) at large batch  -> ONE boot config + per-step gate works,
                                         and the gate threshold is the crossing
  * (1,0) worse than (0,0) at large batch -> the kernel swap itself does not pay
                                         at scale; two deployments are needed
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

# Part A: batch beyond 32, both modes, window fixed at 256.
CROSSOVER_BATCHES = [1, 8, 32, 64]
# Part B: window scaling, whole-chain only, at two batch sizes.
MATERIALISATION_WINDOWS: list[int] = []

MODES = {
    "piecewise": {  # (0,0) paged FA3, host-driven chain
        "VLLM_SELF_SPEC_DRAFT_WHOLECHAIN": "0",
        "VLLM_SELF_SPEC_DRAFT_FULLCG": "0",
    },
    "fullcg_only": {  # (1,0) THE UNTESTED ONE: splitkv kernel, host-driven chain
        "VLLM_SELF_SPEC_DRAFT_WHOLECHAIN": "0",
        "VLLM_SELF_SPEC_DRAFT_FULLCG": "1",
    },
    "wholechain": {  # (1,1) splitkv kernel, one captured graph
        "VLLM_SELF_SPEC_DRAFT_WHOLECHAIN": "1",
        "VLLM_SELF_SPEC_DRAFT_FULLCG": "1",
    },
}

# K must stay 4 across the whole batch range or the arms are not comparable.
WIDE_K_SCHEDULE = [[1, 128, 4]]

ARMS: list[dict[str, Any]] = []
for _b in CROSSOVER_BATCHES:
    for _m in MODES:
        ARMS.append(
            {"part": "A", "regime": "R1", "batch": _b, "window": 256, "mode": _m}
        )
for _w in MATERIALISATION_WINDOWS:
    for _regime, _batch in (("R1", 1), ("R5", 8)):
        ARMS.append(
            {
                "part": "B",
                "regime": _regime,
                "batch": _batch,
                "window": _w,
                "mode": "wholechain",
            }
        )

WARMUP_STEPS = 30
TIMED_STEPS = 40
PROFILED_STEPS = 12
ARMED_MARKER = "Whole-chain graph captured"

FAMILIES = [
    ("scratchpad", re.compile(r"gather|index_select|elementwise|copy|memcpy", re.I)),
    ("flash_splitkv", re.compile(r"splitkv", re.I)),
    ("fa3_paged", re.compile(r"_vllm_fa3_C|prepare_varlen", re.I)),
    ("flash_other", re.compile(r"flash|FlashAttn", re.I)),
    ("w4a16_gemm", re.compile(r"GemmUniversal|machete|marlin", re.I)),
    ("bf16_gemm", re.compile(r"nvjet|cublas|aten::mm", re.I)),
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _family(name: str) -> str:
    for label, pattern in FAMILIES:
        if pattern.search(name):
            return label
    return "other"


def measure(arm: dict[str, Any], trace: Path, out_json: Path) -> None:
    import torch
    from torch.profiler import ProfilerActivity, profile

    from vllm import LLMEngine, SamplingParams

    # Widen K so the chain length is 4 across the whole batch range.
    r1.DYNAMIC_K_SCHEDULE = WIDE_K_SCHEDULE

    cfg = {
        "quant": QUANT,
        "window": arm["window"],
        "skip_count": len([t for t in SKIP.split(",") if t.strip()]),
    }
    prompts = r1._prompts_for(arm["regime"], arm["batch"])
    _require(
        len(prompts) == arm["batch"],
        f"{arm['regime']} gave {len(prompts)} prompts, need {arm['batch']}",
    )
    engine = LLMEngine.from_engine_args(r1._engine_args(cfg))
    per_step: list[float] = []
    kernels: dict[str, dict[str, Any]] = {}
    try:
        for index, tokens in enumerate(prompts):
            engine.add_request(
                f"{arm['regime']}-{index}",
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
            total = 0.0
            for attr in ("self_device_time_total", "self_cuda_time_total"):
                value = getattr(event, attr, None)
                if value:
                    total = float(value)
                    break
            if total <= 0:
                continue
            kernels[event.key] = {
                "family": _family(event.key),
                "calls": int(event.count),
                "device_us_total": total,
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
                "record_type": "w98_hybrid_design",
                **{k: arm[k] for k in ("part", "regime", "batch", "window", "mode")},
                "k_schedule": WIDE_K_SCHEDULE,
                "timed_steps": len(per_step),
                "wall_mean_ms": statistics.mean(per_step) * 1000,
                "wall_median_ms": statistics.median(per_step) * 1000,
                "wall_min_ms": min(per_step) * 1000,
                "wall_stdev_ms": statistics.stdev(per_step) * 1000,
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
    for arm in ARMS:
        name = (
            f"{arm['part']}_{arm['regime']}_b{arm['batch']:03d}"
            f"_w{arm['window']}_{arm['mode']}"
        )
        target = output_dir / f"{name}.json"
        if target.exists():
            continue
        trace = traces / f"{name}.jsonl"
        _require(not trace.exists(), f"trace {trace} already exists")
        cfg = {
            "quant": QUANT,
            "window": arm["window"],
            "skip_count": len([t for t in SKIP.split(",") if t.strip()]),
        }
        env = r1.boot_environment(cfg, trace)
        env["VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS"] = SKIP
        env["VLLM_SELF_SPEC_PROFILE"] = "0"
        env.update(MODES[arm["mode"]])
        # Marlin is the selected kernel (X12); disable Machete so it resolves.
        env["VLLM_DISABLED_KERNELS"] = "MacheteLinearKernel"
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
                    json.dumps({**arm, "name": name}),
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
            (output_dir / f"{name}.FAILED").write_text(
                f"returncode={completed.returncode}\n", encoding="utf-8"
            )
            continue
        text = log.read_text(encoding="utf-8", errors="replace")
        record = r1._load_json(target)
        record["wholechain_captured"] = ARMED_MARKER in text
        record["kernel_resolved"] = next(
            (
                tok
                for line in text.splitlines()
                if "for CompressedTensorsWNA16" in line
                for tok in line.split()
                if tok.endswith("LinearKernel")
            ),
            None,
        )
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
        measure(arm, Path(args.trace).resolve(), output_dir / f"{arm['name']}.json")
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
    print(json.dumps({"arms": len(ARMS)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
