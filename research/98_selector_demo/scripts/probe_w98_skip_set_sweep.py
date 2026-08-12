# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate 'how many layers are skipped' from 'which layers are skipped'.

DIAGNOSTIC, NOT SCORED.

G98-B found the quantized draft's per-layer forward cost reverts to the bf16
cost at skip8 (0.190 ms/layer vs 0.139 at skip4, against a bf16 line of 0.186).
The forward is 89% of the chain and host orchestration is flat at ~2.9 ms, so
the anomaly is inside the model forward. Qwen3-8B is dense with 36 homogeneous
layers and uniform W4A16 (only lm_head ignored), so removing four more identical
layers cannot add work. Something global changes at skip8.

This probe asks the one question that splits the hypothesis space:

    Is the penalty a function of the COUNT (8 skips), or of the SET
    ({2,4,7,11,16,20,25,30} specifically)?

Four different 8-layer sets are booted. If all four are slow, the count is the
trigger and the cause is a threshold -- compile/partition/alignment. If the
declared set is slow and the others are fast, the indices are the trigger and
the cause is positional. Anything in between is itself informative.

skip0 and skip4 reproduce the known-good points in the same session, and a bf16
skip4 boot pins the reference line that quantization is failing to beat.
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

matrix = r1.matrix

# Every arm holds window and quant fixed except the two controls, so the only
# thing moving is the skip set. Arm names are the record keys.
ARMS = [
    # reference points, reproduced in-session rather than cited across runs
    ("bf16_skip4", "target-matching", "2,4,7,16"),
    ("quant_skip0", "w4a16-quantized", ""),
    ("quant_skip4_declared", "w4a16-quantized", "2,4,7,16"),
    # the anomaly
    ("quant_skip8_declared", "w4a16-quantized", "2,4,7,11,16,20,25,30"),
    # same count, different positions
    ("quant_skip8_tail", "w4a16-quantized", "28,29,30,31,32,33,34,35"),
    ("quant_skip8_head", "w4a16-quantized", "1,2,3,4,5,6,7,8"),
    ("quant_skip8_spread", "w4a16-quantized", "4,8,12,16,20,24,28,32"),
]

WINDOW = 256
# The anomaly is flat in context; R1 alone separates the arms.
PROBE_REGIMES = ["R1"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def arm_environment(quant: str, skip: str, trace: Path) -> dict[str, str]:
    """Round 1's boot environment with the skip set overridden per arm."""
    cfg = {"quant": quant, "window": WINDOW, "skip_count": len(_indices(skip))}
    env = r1.boot_environment(cfg, trace)
    env["VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS"] = skip
    return env


def _indices(skip: str) -> list[int]:
    return [int(t) for t in skip.split(",") if t.strip()]


def measure_arm(quant: str, skip: str, trace: Path) -> dict[str, Any]:
    """Boot one arm and record the forward and chain regions."""
    from vllm import LLMEngine, SamplingParams
    from vllm.v1.spec_decode.self_spec_profiler import get_profiler

    cfg = {"quant": quant, "window": WINDOW, "skip_count": len(_indices(skip))}
    manifest = r1._load_json(matrix._repository_path(r1.PROMPT_MANIFEST))
    regimes = {r["regime_id"]: r for r in manifest["prompt_plan"]["regimes"]}
    profiler = get_profiler()
    _require(profiler.enabled, "the self-spec profiler is not enabled")
    engine = LLMEngine.from_engine_args(r1._engine_args(cfg))
    out: dict[str, Any] = {}
    try:
        for regime_id in PROBE_REGIMES:
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
            while engine.has_unfinished_requests():
                engine.step()
            steps = r1._armed_steps(trace, before)
            summary = profiler.summary(warmup=r1.PROFILER_WARMUP)
            out[regime_id] = {
                "regions": {
                    label: {"mean_ms": v.get("mean_ms"), "n": v.get("n")}
                    for label, v in sorted(summary.items())
                },
                "armed_step_count": len(steps),
                "mean_armed_step_s": statistics.mean(steps) if steps else None,
            }
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()
    return out


def run_arms(output_dir: Path) -> None:
    """Boot every arm in its own pinned child."""
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
        env = matrix._boot_child_environment(arm_environment(quant, skip, trace))
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
    if args.arm:  # child
        arm = json.loads(args.arm)
        observations = measure_arm(
            arm["quant"], arm["skip"], Path(args.trace).resolve()
        )
        (output_dir / f"{arm['name']}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "record_type": "w98_skip_set_sweep_measurement",
                    "arm": arm["name"],
                    "quant": arm["quant"],
                    "skip_layers": arm["skip"],
                    "skip_count": len(_indices(arm["skip"])),
                    "window": WINDOW,
                    "observations": observations,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
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
