# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X4 -- find WHICH host operation grows at quantized x skip8.

DIAGNOSTIC, NOT SCORED.

X3 localised the penalty to host-side idle: the whole +3.6 ms/step is GPU idle
(busy -1.72, idle +5.34), 95.3% of it outside any CUDA call, concentrated in
~90 extra 50-200 us gaps per step against ~116 piecewise cudagraph replays. The
host is executing framework code between replays and that per-replay cost
roughly triples. This probe asks which frame.

Method: cProfile over a fixed number of engine steps per arm, then diff the
per-function totals. The arms run the SAME number of steps, so a function whose
total time RISES at skip8 is doing so despite four fewer layers to work on --
which is the signal. Functions that fall are just tracking the layer count.

Reading rule, fixed before the data: the culprit must
  (a) rise in absolute time at skip8 despite fewer layers, and
  (b) not rise by the same amount in the bf16 skip8 control, since the defect
      is quant-specific.
A frame that rises in both is a property of skipping, not of the interaction.

Caveat recorded up front: cProfile sees Python frames only. If the growth is in
C++ dispatch or the CUDA driver, every Python frame will look flat and the
answer is "not in Python" -- itself informative, and the cue to rerun with a
native-stack sampler rather than to keep staring at Python.
"""

from __future__ import annotations

import argparse
import contextlib
import cProfile
import json
import os
import pstats
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402

matrix = r1.matrix

ARMS = [
    ("quant_skip4", "w4a16-quantized", "2,4,7,16"),
    ("quant_skip8", "w4a16-quantized", "2,4,7,11,16,20,25,30"),
    # Control: isolates "grew because layers were skipped" from "grew because
    # of the quant x skip interaction".
    ("bf16_skip8", "target-matching", "2,4,7,11,16,20,25,30"),
]

WINDOW = 256
REGIME = "R1"
WARMUP_STEPS = 40
PROFILED_STEPS = 25


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def profile_arm(quant: str, skip: str, trace: Path, out_json: Path) -> None:
    """Boot one arm and cProfile a fixed number of engine steps."""
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
                    temperature=0.0, max_tokens=r1.MEASURE_TOKENS, ignore_eos=True
                ),
            )
        for _ in range(WARMUP_STEPS):
            _require(engine.has_unfinished_requests(), "ran out of work in warmup")
            engine.step()
        profiler = cProfile.Profile()
        wall_t0 = time.perf_counter()
        profiler.enable()
        for _ in range(PROFILED_STEPS):
            _require(engine.has_unfinished_requests(), "ran out of work profiling")
            engine.step()
        profiler.disable()
        wall_s = time.perf_counter() - wall_t0
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()

    stats = pstats.Stats(profiler)
    functions: dict[str, dict[str, Any]] = {}
    for (filename, line, name), entry in stats.stats.items():  # type: ignore[attr-defined]
        calls, _, tottime, cumtime, _ = entry
        short = f"{Path(filename).name}:{line}({name})"
        functions[short] = {
            "calls": calls,
            "tottime_s": tottime,
            "cumtime_s": cumtime,
            "tottime_us_per_step": tottime * 1e6 / PROFILED_STEPS,
        }
    out_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_cpu_profile",
                "quant": quant,
                "skip_layers": skip,
                "skip_count": cfg["skip_count"],
                "window": WINDOW,
                "regime": REGIME,
                "profiled_steps": PROFILED_STEPS,
                # cProfile inflates wall substantially; recorded so the arms can
                # be checked to still differ in the expected direction, NOT to
                # be compared against any profiler-free number.
                "wall_ms_per_step_under_cprofile": wall_s * 1000 / PROFILED_STEPS,
                "total_tottime_s": sum(f["tottime_s"] for f in functions.values()),
                "functions": functions,
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
