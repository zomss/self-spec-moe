# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decompose the quantized x skip8 x windowed draft-chain floor.

DIAGNOSTIC, NOT SCORED. This probe carries no authorization, produces no
coverage number, and grants no Round-2 authority. It exists to answer one
question left open by G98-B: Round 1 measured `draft_chain` as a total, and the
two `w4a16 x skip8 x windowed` cells sat at 27.8 ms flat when both the fit and a
linear extrapolation from the quantized points say ~18.8 ms.

Design:

* Two anomalous cells and TWO CONTROLS that behaved in Round 1. A sub-region
  time means nothing in isolation -- only the anomalous-minus-control
  difference does, and the controls must run under identical instrumentation.

* Two passes per cell. The coarse pass records the labels the profiler emits
  without extra syncs (`draft_chain`, `draft_forward`, `wc_replay`, `verify`)
  and is directly comparable to Round 1. The fine pass sets
  VLLM_SELF_SPEC_PROFILE_FINE, which adds per-step CUDA syncs that INFLATE the
  enclosing chain total -- its absolute numbers are not comparable to Round 1
  and are used only to attribute within a pass.

* Two regimes. The anomaly is context-flat, so R1 and R5 bracket it; the other
  four regimes would add cost without adding evidence.

The coarse pass alone may settle it: `draft_forward` is not fine-gated, so
`draft_chain - draft_forward` is an undisturbed measure of host orchestration.
If the anomalous cells carry a normal forward and an inflated remainder, the
floor is host dispatch; if the forward itself is inflated, it is execution.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402

matrix = r1.matrix

# Two anomalous cells and two that behaved. Every cell is windowed and every
# cell was measured in Round 1, so each has a baseline to compare against.
CELLS = [
    {"quant": "w4a16-quantized", "window": 1024, "skip_count": 8},  # anomalous
    {"quant": "w4a16-quantized", "window": 128, "skip_count": 8},  # anomalous
    {"quant": "w4a16-quantized", "window": 256, "skip_count": 4},  # control
    {"quant": "target-matching", "window": 256, "skip_count": 8},  # control
]

# The anomaly is flat in context, so the short and long ends bracket it.
PROBE_REGIMES = ["R1", "R5"]

PASSES = {"coarse": "0", "fine": "1"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def probe_environment(cfg: Mapping[str, Any], trace: Path, fine: str) -> dict[str, str]:
    """Round 1's boot environment plus the fine-region gate."""
    env = r1.boot_environment(cfg, trace)
    env["VLLM_SELF_SPEC_PROFILE_FINE"] = fine
    return env


def measure_regions(cfg: Mapping[str, Any], trace: Path) -> dict[str, Any]:
    """Boot one cell and record EVERY profiler region, per regime."""
    from vllm import LLMEngine, SamplingParams
    from vllm.v1.spec_decode.self_spec_profiler import get_profiler

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
                # Every label the profiler saw, not a chosen few -- the point
                # of the probe is that we do not yet know which one carries it.
                "regions": {
                    label: {"mean_ms": v.get("mean_ms"), "n": v.get("n")}
                    for label, v in sorted(summary.items())
                },
                "armed_step_count": len(steps),
                "mean_armed_step_s": statistics.mean(steps) if steps else None,
                "context_tokens": int(
                    statistics.mean(len(t) for t in prompts) + r1.MEASURE_TOKENS / 2
                ),
            }
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()
    return out


def run_cells(output_dir: Path) -> None:
    """Boot every cell under every pass, each in its own pinned child."""
    lane = matrix.lane_for_block(1)
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = sorted(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    for pass_name, fine in PASSES.items():
        stage = output_dir / pass_name
        stage.mkdir(parents=True, exist_ok=True)
        traces = output_dir / "traces" / pass_name
        traces.mkdir(parents=True, exist_ok=True)
        for cfg in CELLS:
            key = r1._config_key(cfg).replace("/", "_")
            target = stage / f"{key}.json"
            if target.exists():
                continue
            trace = traces / f"{key}.jsonl"
            _require(not trace.exists(), f"trace {trace} already exists")
            env = matrix._boot_child_environment(probe_environment(cfg, trace, fine))
            log = stage / f"{key}.log"
            with log.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--output-dir",
                        str(output_dir),
                        "--measure-cell",
                        json.dumps(cfg),
                        "--stage-dir",
                        str(stage),
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
                f"probe boot {pass_name}/{key} failed; output preserved at {log}",
            )


def summarize(output_dir: Path) -> dict[str, Any]:
    """Table the regions per pass, with the anomalous-minus-control deltas."""
    result: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "w98_skip8_floor_probe",
        "scored": False,
        "note": (
            "diagnostic only; the fine pass adds CUDA syncs that inflate the "
            "chain total, so its absolute times are not comparable to Round 1"
        ),
        "passes": {},
    }
    for pass_name in PASSES:
        stage = output_dir / pass_name
        rows = {}
        for path in sorted(stage.glob("*.json")):
            rows[path.stem] = r1._load_json(path)
        result["passes"][pass_name] = rows
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--measure-cell", help=argparse.SUPPRESS)
    parser.add_argument("--stage-dir", help=argparse.SUPPRESS)
    parser.add_argument("--trace", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.measure_cell:  # child
        cfg = json.loads(args.measure_cell)
        observations = measure_regions(cfg, Path(args.trace).resolve())
        target = Path(args.stage_dir).resolve() / (
            r1._config_key(cfg).replace("/", "_") + ".json"
        )
        target.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "record_type": "w98_skip8_floor_measurement",
                    "config": dict(cfg),
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
    run_cells(output_dir)
    summary = summarize(output_dir)
    (output_dir / "probe_result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"cells": len(CELLS), "passes": list(PASSES)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
