# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X5 -- interleaved A/B control for the quantized x skip8 penalty.

DIAGNOSTIC, NOT SCORED.

Every measurement so far ran the arms in sequence: skip4, then skip8. This box
has co-tenants (`cudafe++`, `cc1plus` at ~100% CPU each) that this project has
already caught perturbing results once -- Phase 97's certification failure was
root-caused to exactly that. Our lane pin (CPUs 0-15) constrains OUR threads; it
does not stop a co-tenant from being scheduled onto the same cores. A sequential
A-then-B design cannot separate "skip8 is slower" from "the second half of the
session was slower".

X4 makes this control mandatory rather than optional: it attributed the growth
to pure-Python frames whose bytecode is trivial (`_resolve_layer_name` is a
one-line isinstance check), which means those frames are STALLING, and external
CPU contention is one of the few things that stalls trivial Python.

Design: alternate A/B/A/B/A/B in one session, no profiler of any kind, plain
wall time per engine step. Contention drifting over the session hits both arms
equally; a real configuration effect survives the interleave. Reported per
repeat so drift is visible rather than averaged away.

Reading rule, fixed before the data:
  * gap stable across all three repeats -> real, and X4's localisation stands
  * gap present in repeat 1 and shrinking -> session drift/contention, and the
    whole quant x skip8 finding needs re-measuring under a quiet box
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
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402

matrix = r1.matrix

CONFIGS = {
    "skip4": ("w4a16-quantized", "2,4,7,16"),
    "skip8": ("w4a16-quantized", "2,4,7,11,16,20,25,30"),
}
REPEATS = 3
ORDER = [(rep, arm) for rep in range(REPEATS) for arm in ("skip4", "skip8")]

WINDOW = 256
REGIME = "R1"
WARMUP_STEPS = 40
MEASURED_STEPS = 60


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def measure(quant: str, skip: str, trace: Path, out_json: Path) -> None:
    """Time engine steps with no profiler of any kind."""
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
    per_step: list[float] = []
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
        for _ in range(MEASURED_STEPS):
            _require(engine.has_unfinished_requests(), "ran out of work measuring")
            t0 = time.perf_counter()
            engine.step()
            per_step.append(time.perf_counter() - t0)
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()

    out_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_interleave_control",
                "quant": quant,
                "skip_layers": skip,
                "skip_count": cfg["skip_count"],
                "window": WINDOW,
                "regime": REGIME,
                "measured_steps": len(per_step),
                "mean_ms": statistics.mean(per_step) * 1000,
                "median_ms": statistics.median(per_step) * 1000,
                "stdev_ms": statistics.stdev(per_step) * 1000,
                # The minimum is the least contended step observed, so it is the
                # measure least sensitive to a co-tenant stealing cycles.
                "min_ms": min(per_step) * 1000,
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
    for rep, arm in ORDER:
        name = f"r{rep}_{arm}"
        target = output_dir / f"{name}.json"
        if target.exists():
            continue
        quant, skip = CONFIGS[arm]
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


def summarize(output_dir: Path) -> dict[str, Any]:
    rows = {}
    for rep, arm in ORDER:
        path = output_dir / f"r{rep}_{arm}.json"
        if path.exists():
            rows[f"r{rep}_{arm}"] = r1._load_json(path)
    return {"schema_version": 1, "record_type": "w98_interleave_summary", "runs": rows}


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
    run_all(output_dir)
    (output_dir / "interleave_summary.json").write_text(
        json.dumps(summarize(output_dir), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"runs": [f"r{r}_{a}" for r, a in ORDER]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
