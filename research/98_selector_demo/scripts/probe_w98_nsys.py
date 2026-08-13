# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X24 -- profile a clamped boot with Nsight Systems.

DIAGNOSTIC, NOT SCORED.

Nine hypotheses have been refuted by comparing aggregate timings. This stops
comparing and looks directly at where a step's time goes.

The clamp adds ~4-5 ms per step, ordered by how much GPU work can hide it
(+5.02 ms at batch 1, +0.24 ms at long context), with bit-identical scheduled
work. If that is host-side, a trace will show it as one of:

  * CUDA API calls taking longer -- `cudaLaunchKernel`, `cudaMemcpyAsync`,
    `cudaStreamSynchronize` -- pointing at driver contention;
  * GAPS between API calls on the launching thread, pointing at the host being
    descheduled or blocked in Python;
  * the same API time but longer KERNELS, which would mean it is not host-side
    at all and the whole characterisation is wrong.

Those three are mutually exclusive and the trace distinguishes them directly,
which no amount of aggregate timing has managed.

Method
------

Only the three CHEAP regimes run (R8, R1, R6, in manifest order), so the
measurement-shape gate can still judge whether this boot was clamped -- a
profile of a clean boot answers nothing.

Profiling is confined to a window of R1 steps via `cudaProfilerStart/Stop` with
`--capture-range=cudaProfilerApi`. R1 is batch 1, where the clamp is most
exposed (+12 to +25%), and a window keeps the trace to a size `nsys stats` can
chew. Startup, compile and capture are excluded by construction.

`osrt` tracing is enabled alongside `cuda`, because thread-state and syscall
data is what distinguishes "descheduled" from "blocked in the driver".

Note the profiler perturbs what it measures -- nsys adds per-API overhead. That
is tolerable here because the question is WHICH component grew, not by how much,
and the gate verdict comes from the unprofiled regimes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402
import w98_host_load as hostload  # noqa: E402

matrix = r1.matrix

CELL = {"quant": "target-matching", "window": "off", "skip_count": 0}
# Manifest order, restricted to the regimes the gate needs.
PROFILE_REGIMES = ("R8", "R1", "R6")
PROFILED = "R1"
# Skip this many engine steps before profiling, then profile this many. The
# skip clears warmup; the window keeps the trace small enough to summarise.
PROFILE_SKIP_STEPS = 60
PROFILE_WINDOW_STEPS = 60
NSYS = "nsys"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def measure(trace: Path, out_json: Path) -> None:
    """Boot, run the cheap regimes, and profile a window of R1 steps."""
    import torch

    from vllm import LLMEngine, SamplingParams
    from vllm.v1.spec_decode.self_spec_profiler import get_profiler

    manifest = r1._load_json(matrix._repository_path(r1.PROMPT_MANIFEST))
    regimes = {r["regime_id"]: r for r in manifest["prompt_plan"]["regimes"]}
    profiler = get_profiler()
    _require(profiler.enabled, "the self-spec profiler is not enabled")
    engine = LLMEngine.from_engine_args(r1._engine_args(CELL))
    observations: dict[str, Any] = {}
    windows: dict[str, Any] = {}
    try:
        for regime_id in PROFILE_REGIMES:
            spec = regimes[regime_id]
            prompts = r1._prompts_for(regime_id, spec["batch"])
            before = r1._trace_len(trace)
            profiler.reset()
            for index, tokens in enumerate(prompts):
                engine.add_request(
                    f"{regime_id}-{index}",
                    {"prompt_token_ids": tokens},
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=r1.MEASURE_TOKENS,
                        ignore_eos=True,
                    ),
                )
            steps = 0
            started = stopped = False
            while engine.has_unfinished_requests():
                if (
                    regime_id == PROFILED
                    and not started
                    and steps == PROFILE_SKIP_STEPS
                ):
                    torch.cuda.profiler.start()
                    started = True
                    windows["start_step"] = steps
                engine.step()
                steps += 1
                if (
                    regime_id == PROFILED
                    and started
                    and not stopped
                    and steps >= PROFILE_SKIP_STEPS + PROFILE_WINDOW_STEPS
                ):
                    torch.cuda.profiler.stop()
                    stopped = True
                    windows["stop_step"] = steps
            if regime_id == PROFILED and started and not stopped:
                torch.cuda.profiler.stop()
                windows["stop_step"] = steps
            armed = r1._armed_steps(trace, before)
            chain = profiler.summary(warmup=r1.PROFILER_WARMUP).get("draft_chain", {})
            observations[regime_id] = {
                "draft_chain_s": (
                    chain["mean_ms"] / 1000.0 if chain.get("mean_ms") else None
                ),
                "armed_step_count": len(armed),
                "mean_armed_step_s": statistics.mean(armed) if armed else None,
                "engine_steps": steps,
                "batch": spec["batch"],
            }
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()
    verdict = hostload.measurement_verdict(observations)
    out_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_nsys_profile",
                "config": CELL,
                "observations": observations,
                "profile_window": windows,
                "measurement_gate": verdict.as_record() if verdict else None,
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
    trace = output_dir / "koff.jsonl"
    target = output_dir / "result.json"
    report = output_dir / "profile"
    if target.exists():
        return
    _require(not trace.exists(), f"trace {trace} already exists")
    env = matrix._boot_child_environment(r1.boot_environment(CELL, trace))
    log = output_dir / "nsys.log"
    argv = [
        NSYS,
        "profile",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--trace=cuda,osrt,nvtx",
        "--sample=cpu",
        "--force-overwrite=true",
        "-o",
        str(report),
        sys.executable,
        str(Path(__file__).resolve()),
        "--output-dir",
        str(output_dir),
        "--child",
        "--trace-path",
        str(trace),
    ]
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            argv,
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            preexec_fn=lambda: os.sched_setaffinity(0, affinity),
        )
    print(f"[x24] nsys rc={completed.returncode}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--trace-path", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.child:
        measure(Path(args.trace_path).resolve(), output_dir / "result.json")
        return 0
    with contextlib.suppress(Exception):
        base_env = matrix._boot_child_environment({})
        matrix._preflight_native_sampler(base_env)
        matrix._preflight_inprocess_engine_core(base_env)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_all(output_dir)
    result = output_dir / "result.json"
    if result.is_file():
        record = r1._load_json(result)
        gate = record.get("measurement_gate") or {}
        print(
            json.dumps(
                {
                    "gate": gate.get("verdict"),
                    "relative_spread": gate.get("relative_spread"),
                    "draft_chain_ms": {
                        r: round(o["draft_chain_s"] * 1000.0, 2)
                        for r, o in record["observations"].items()
                        if o.get("draft_chain_s")
                    },
                    "profile_window": record.get("profile_window"),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
