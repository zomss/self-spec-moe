# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X19 -- is the skip0 discrepancy a mid-measurement starvation clamp?

DIAGNOSTIC, NOT SCORED.

X18 re-measured two v6 reference cells. skip8 reproduced within 1% on every
regime. skip0 did not: +10.15% at R1, +5.57% at R8, ~0% elsewhere. Both boots
passed the host-load gate and both had low dispersion (R1 CV 0.43%), so neither
the gate nor the variance flagged it.

Sorting skip0's regimes by their v6 cost shows the structure:

    regime    v6      now    delta
    R1        30.36   33.44  +3.08
    R8        31.75   33.52  +1.77
    R6        33.92   34.17  +0.25
    R4        43.59   43.73  +0.14
    R5cot     51.84   51.94  +0.10
    R5        51.89   51.82  -0.07

The absolute delta shrinks monotonically to zero as the baseline grows. That is
not an offset, it is a FLOOR at ~33.5 ms: every regime cheaper than the floor
was lifted to it, every regime above it was untouched. It is the same clamp that
made the contaminated skip16 boot read flat across a 33x context range, only
milder.

The session order fits. skip8 ran first, skip16 second (every regime <= 29.3 ms,
no clamp), skip0 last -- by which time new co-tenants had arrived. The gate reads
compile and capture time, which happen at STARTUP, so contention that begins
after the boot is invisible to it.

Hypotheses, and what separates them
-----------------------------------

H1 STARVATION -- the host stalls the draft chain during measurement. The clamp
   is a property of the BOX AT THE TIME, so under load every cell clamps to a
   COMMON floor regardless of its levers.
H2 CELL-SPECIFIC -- something about skip0 (36 layers, draft aliases the target
   exactly, 99.7% acceptance) makes it fragile. Then skip0 shifts and skip8 does
   not, whatever the box is doing.

X18 cannot separate these: skip0 was measured under load and skip8 was not, so
"skip0 shifted" and "the loaded cell shifted" are the same observation.

This probe separates them by measuring BOTH cells back to back under whatever
load exists now, twice, interleaved (skip8, skip0, skip8, skip0). H1 predicts a
common floor across both cells; H2 predicts skip0 alone moves.

The instrument
--------------

Starvation is measured directly rather than inferred. Linux exposes per-task
scheduling statistics in /proc: field 2 of `schedstat` is nanoseconds spent
WAITING on a runqueue while runnable -- exactly the stall that a host-bound
draft chain suffers when co-tenants take its CPUs. Sampling it across every
thread before and after each regime gives the fraction of wall time this process
spent waiting for a CPU, per regime, with no profiler overhead.

GPU clocks are sampled alongside, to eliminate thermal capping: at the time of
writing every busy GPU on the box reported 1980/1980 MHz at 42-69 C, so capping
looks unlikely, but it costs nothing to record.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402
import w98_host_load as hl  # noqa: E402

matrix = r1.matrix

ORDER = [
    ("r0_skip8", {"quant": "target-matching", "window": "off", "skip_count": 8}),
    ("r0_skip0", {"quant": "target-matching", "window": "off", "skip_count": 0}),
    ("r1_skip8", {"quant": "target-matching", "window": "off", "skip_count": 8}),
    ("r1_skip0", {"quant": "target-matching", "window": "off", "skip_count": 0}),
]
V6_QUIET_MS = {
    "skip0": {
        "R1": 30.36,
        "R4": 43.59,
        "R5": 51.89,
        "R5cot": 51.84,
        "R6": 33.92,
        "R8": 31.75,
    },
    "skip8": {
        "R1": 24.43,
        "R4": 33.67,
        "R5": 39.74,
        "R5cot": 40.67,
        "R6": 26.96,
        "R8": 25.48,
    },
}
# A clamp is only visible in regimes whose unclamped cost is BELOW the floor.
# These are the cheap ones; the long-context regimes sit above any floor seen so
# far and stay flat whatever happens.
CLAMP_SENSITIVE = ("R1", "R8", "R6")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def runqueue_wait_ns() -> int:
    """Nanoseconds this process's threads have spent waiting for a CPU.

    Field 2 of ``/proc/<pid>/task/<tid>/schedstat`` is time spent on a runqueue
    while runnable but not running -- host starvation, measured directly by the
    kernel. Summing across threads captures the engine's worker threads as well
    as the main loop.

    Returns:
        Total runqueue wait in nanoseconds, or 0 if unavailable.
    """
    total = 0
    try:
        for task in Path("/proc/self/task").iterdir():
            with contextlib.suppress(OSError, ValueError, IndexError):
                fields = (task / "schedstat").read_text().split()
                total += int(fields[1])
    except OSError:
        return 0
    return total


def gpu_clocks(device: str) -> dict[str, Any]:
    """Sample SM clock, temperature and power for the measured GPU."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device}",
                "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout.strip()
        sm, sm_max, temp, power = (v.strip() for v in out.split(","))
        return {
            "sm_mhz": float(sm),
            "sm_max_mhz": float(sm_max),
            "temp_c": float(temp),
            "power_w": float(power),
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def measure(cfg: dict[str, Any], trace: Path, out_json: Path) -> None:
    """Round-1 measurement with per-regime starvation instrumentation.

    Mirrors ``r1.measure_config`` exactly -- same engine args, same prompts,
    same profiler and warmup, same regime order -- and only adds sampling
    around each regime so a stall can be attributed to the regime it hit.
    """
    import time

    from vllm import LLMEngine, SamplingParams
    from vllm.v1.spec_decode.self_spec_profiler import get_profiler

    manifest = r1._load_json(matrix._repository_path(r1.PROMPT_MANIFEST))
    regimes = {r["regime_id"]: r for r in manifest["prompt_plan"]["regimes"]}
    device = matrix.lane_for_block(1)["physical_gpu_index"]
    profiler = get_profiler()
    _require(profiler.enabled, "the self-spec profiler is not enabled")
    engine = LLMEngine.from_engine_args(r1._engine_args(cfg))
    observations: dict[str, Any] = {}
    try:
        for regime_id, spec in regimes.items():
            prompts = r1._prompts_for(regime_id, spec["batch"])
            before_trace = r1._trace_len(trace)
            profiler.reset()
            wait0, wall0 = runqueue_wait_ns(), time.monotonic()
            clocks_before = gpu_clocks(str(device))
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
            wall = time.monotonic() - wall0
            wait_s = (runqueue_wait_ns() - wait0) / 1e9
            steps = r1._armed_steps(trace, before_trace)
            chain = profiler.summary(warmup=r1.PROFILER_WARMUP).get("draft_chain", {})
            observations[regime_id] = {
                "draft_chain_s": (
                    chain["mean_ms"] / 1000.0 if chain.get("mean_ms") else None
                ),
                "armed_step_count": len(steps),
                "mean_armed_step_s": statistics.mean(steps) if steps else None,
                "stdev_armed_step_s": (
                    statistics.stdev(steps) if len(steps) > 1 else None
                ),
                # The starvation measurement: runqueue wait as a fraction of
                # wall time. A host-bound chain stalls exactly here.
                "runqueue_wait_s": wait_s,
                "wall_s": wall,
                "runqueue_wait_frac": wait_s / wall if wall else None,
                "gpu_before": clocks_before,
                "gpu_after": gpu_clocks(str(device)),
                "batch": spec["batch"],
            }
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()
    out_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_starvation_clamp",
                "config": cfg,
                "observations": observations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_all(output_dir: Path) -> None:
    lane = matrix.lane_for_block(1)
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    traces = output_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    for name, cfg in ORDER:
        target = output_dir / f"{name}.json"
        if target.exists():
            continue
        trace = traces / f"{name}.jsonl"
        _require(not trace.exists(), f"trace {trace} already exists")
        env = matrix._boot_child_environment(r1.boot_environment(cfg, trace))
        log = output_dir / f"{name}.log"
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output-dir",
                    str(output_dir),
                    "--cell",
                    json.dumps({"name": name, "cfg": cfg}),
                    "--trace",
                    str(trace),
                ],
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        if completed.returncode != 0 or not target.exists():
            (output_dir / f"{name}.FAILED").write_text(
                f"returncode={completed.returncode}\n", encoding="utf-8"
            )
            print(f"[x19] FAILED {name}", flush=True)
            continue
        print(f"[x19] ok {name}", flush=True)


def summarise(output_dir: Path) -> dict[str, Any]:
    """Decide between the starvation clamp and a cell-specific effect."""
    cells: dict[str, Any] = {}
    for name, cfg in ORDER:
        path = output_dir / f"{name}.json"
        if not path.is_file():
            cells[name] = {"status": "FAILED"}
            continue
        record = r1._load_json(path)
        key = f"skip{cfg['skip_count']}"
        quiet = V6_QUIET_MS[key]
        now = {
            regime: round(obs["draft_chain_s"] * 1000.0, 2)
            for regime, obs in record["observations"].items()
            if obs.get("draft_chain_s")
        }
        cells[name] = {
            "status": "ok",
            "skip_count": cfg["skip_count"],
            "draft_chain_ms": now,
            "delta_pct_vs_v6": {
                regime: round((now[regime] / quiet[regime] - 1.0) * 100.0, 2)
                for regime in quiet
                if regime in now
            },
            "runqueue_wait_frac": {
                regime: round(obs["runqueue_wait_frac"], 4)
                for regime, obs in record["observations"].items()
                if obs.get("runqueue_wait_frac") is not None
            },
            "sm_mhz": {
                regime: obs["gpu_after"].get("sm_mhz")
                for regime, obs in record["observations"].items()
            },
            "host_load": hl.signature_from_log_path(
                output_dir / f"{name}.log"
            ).as_record(),
            # Under a clamp every cheap regime is lifted to a common value.
            "cheap_regime_spread_ms": (
                round(
                    max(now[r] for r in CLAMP_SENSITIVE if r in now)
                    - min(now[r] for r in CLAMP_SENSITIVE if r in now),
                    2,
                )
                if all(r in now for r in CLAMP_SENSITIVE)
                else None
            ),
        }
    ok = [c for c in cells.values() if c.get("status") == "ok"]
    verdict = "INCONCLUSIVE"
    detail: dict[str, Any] = {}
    if len(ok) >= 2:
        by_skip: dict[int, list[float]] = {}
        for cell in ok:
            by_skip.setdefault(cell["skip_count"], []).append(
                cell["draft_chain_ms"].get("R1", 0.0)
            )
        skip0_r1 = statistics.mean(by_skip.get(0, [0.0]))
        skip8_r1 = statistics.mean(by_skip.get(8, [0.0]))
        worst = max(abs(d) for c in ok for d in c["delta_pct_vs_v6"].values())
        detail = {
            "skip0_r1_ms": round(skip0_r1, 2),
            "skip8_r1_ms": round(skip8_r1, 2),
            "worst_abs_delta_pct_vs_v6": round(worst, 2),
            "max_runqueue_wait_frac": round(
                max(
                    (v for c in ok for v in c["runqueue_wait_frac"].values()),
                    default=0.0,
                ),
                4,
            ),
        }
        # Order matters. If NEITHER cell deviates from its quiet-box reference
        # then no clamp occurred at all, and asking which cell moved is asking
        # the wrong question -- the X18 excursion simply did not reproduce.
        if worst < 3.0:
            verdict = "NOT_REPRODUCED"
        elif abs(skip0_r1 - skip8_r1) < 2.0:
            # Both lifted to a common value: a property of the box, not a lever.
            verdict = "H1_COMMON_FLOOR"
        else:
            verdict = "H2_CELL_SPECIFIC"
    return {
        "cells": cells,
        "verdict": verdict,
        "verdict_detail": detail,
        "v6_quiet_ms": V6_QUIET_MS,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summarise-only", action="store_true")
    parser.add_argument("--cell", help=argparse.SUPPRESS)
    parser.add_argument("--trace", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.cell:
        cell = json.loads(args.cell)
        measure(
            cell["cfg"],
            Path(args.trace).resolve(),
            output_dir / f"{cell['name']}.json",
        )
        return 0
    if not args.summarise_only:
        with contextlib.suppress(Exception):
            base_env = matrix._boot_child_environment({})
            matrix._preflight_native_sampler(base_env)
            matrix._preflight_inprocess_engine_core(base_env)
        lane = matrix.lane_for_block(1)
        matrix._preflight_gpu_identity_and_idle(
            lane["physical_gpu_index"], lane["physical_gpu_uuid"]
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        run_all(output_dir)
    summary = summarise(output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
