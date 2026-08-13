# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X21 -- is the clamp a GPU clock effect driven by chassis power?

DIAGNOSTIC, NOT SCORED.

What is established
-------------------

The clamp inflates the draft chain by 10-60% on a boot that is otherwise
identical. From the traces of a clamped boot and a clean boot of the SAME cell:

  * the work is identical -- KV blocks in use (415/29/818 medians), their
    maxima, and H_target_steps (16/1/32) match exactly, regime for regime;
  * the inflation is FLAT within each regime and switches at regime
    BOUNDARIES, not at arbitrary wall-clock times;
  * it lands on R8 and R1 and spares R6, which sits between them in cost, so
    it is not a simple cost floor;
  * per-step added time is +0.5/+0.3/+0.3/+3.2/+4.6/+0.5 ms across the
    measurement order R4, R5, R5cot, R8, R1, R6.

What has been excluded
----------------------

  * mid-measurement CPU starvation (X19), and runqueue wait never exceeds 0.1%
  * lane CPU pinning (X20): pinned and unpinned agree to a ratio of 1.001
  * differences in scheduled work (identical counters, above)

The remaining hypothesis
------------------------

Neither the runqueue instrument nor the clock sampler has ever observed a
CONTAMINATED boot -- both have only ever run during clean ones, so both are
unvalidated as detectors. That gap is the point of this probe.

Right now GPUs 2 and 3 run a tensor-parallel job at 100% utilisation drawing
512 W and 539 W, over 1 kW from one job, while the machine load average is 5.78
across 192 cores. If the chassis power or thermal budget is shared, GPU 0's
clocks can be capped when the neighbours draw hard. That would explain every
observation above:

  * identical work at different speed;
  * regime-boundary alignment, since each regime has its own power draw and the
    cap responds to it;
  * worst impact on R1 (batch 1) and R8 (batch 16), which are latency-bound and
    depend on clock, while R4/R5/R5cot are long-context and bandwidth-bound, and
    R6 at batch 32 has enough occupancy to be less clock-sensitive;
  * intermittency, tracking whatever the neighbours are doing.

Design
------

Boot the campaign's own anchor cell while sampling GPU 0's SM clock, memory
clock, power, temperature and the throttle reason bitmask at ~10 Hz from a
background thread, and report the distribution PER REGIME alongside the timing.

The decisive quantity is `clocks_throttle_reasons`, which NVML reports
directly: it names SW_POWER_CAP, HW_POWER_BRAKE, SW_THERMAL and HW_THERMAL
rather than leaving them to be inferred from a clock number.

Read as follows:

  * clamped regimes show reduced SM clock and a non-zero throttle reason ->
    the cause is power or thermal, and the fix is scheduling, not code;
  * clamped regimes show full 1980 MHz clocks -> the cause is NOT the GPU, the
    hypothesis dies, and the shape-based gate stays the only defence.

A run in which nothing clamps is INCONCLUSIVE, not a refutation -- the effect is
intermittent and has resisted three attempts to reproduce it on demand.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402

matrix = r1.matrix

CELL = {"quant": "target-matching", "window": "off", "skip_count": 0}
SAMPLE_HZ = 4.0
# Sampler runs here, well clear of lane-a's CPUs 0-15.
SAMPLER_CPUS = frozenset(range(64, 96))
# Clean reference for this exact cell, from v6 and reproduced by X19/X20.
V6_QUIET_MS = {
    "R1": 30.36,
    "R4": 43.59,
    "R5": 51.89,
    "R5cot": 51.84,
    "R6": 33.92,
    "R8": 31.75,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


class ClockSampler:
    """Sample GPU telemetry in a background thread while a boot measures.

    Runs `nvidia-smi` in a loop rather than binding NVML, because the probe must
    not add a dependency to the measured process. The sampler thread is not
    pinned to the lane, so it cannot steal the lane's CPUs.
    """

    FIELDS = (
        "clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,"
        "clocks_throttle_reasons.active"
    )

    def __init__(self, device: int, hz: float = SAMPLE_HZ) -> None:
        self.device = device
        self.interval = 1.0 / hz
        self.samples: list[tuple[float, dict[str, Any]]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _read(self) -> dict[str, Any] | None:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.device}",
                    f"--query-gpu={self.FIELDS}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout.strip()
            sm, mem, power, temp, util, throttle = (v.strip() for v in out.split(","))
            return {
                "sm_mhz": float(sm),
                "mem_mhz": float(mem),
                "power_w": float(power),
                "temp_c": float(temp),
                "util_pct": float(util),
                "throttle": throttle,
            }
        except (OSError, subprocess.SubprocessError, ValueError):
            return None

    def _loop(self) -> None:
        # Keep the sampler off the lane's CPUs. Each sample spawns nvidia-smi,
        # which costs real CPU; left on the lane it would perturb the very
        # measurement it exists to explain.
        with contextlib.suppress(OSError, AttributeError):
            os.sched_setaffinity(threading.get_native_id(), SAMPLER_CPUS)
        while not self._stop.is_set():
            sample = self._read()
            if sample is not None:
                self.samples.append((time.monotonic(), sample))
            self._stop.wait(self.interval)

    def __enter__(self) -> ClockSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def between(self, start: float, end: float) -> list[dict[str, Any]]:
        return [s for t, s in self.samples if start <= t <= end]


def _summarise(window: list[dict[str, Any]]) -> dict[str, Any]:
    if not window:
        return {}
    sm = [s["sm_mhz"] for s in window]
    throttles = sorted(
        {
            s["throttle"]
            for s in window
            if s["throttle"] and s["throttle"].lower() not in ("not active", "0x0000")
        }
    )
    return {
        "n": len(window),
        "sm_mhz_median": statistics.median(sm),
        "sm_mhz_min": min(sm),
        "sm_mhz_p10": sorted(sm)[max(0, len(sm) // 10 - 1)],
        "power_w_median": statistics.median(s["power_w"] for s in window),
        "temp_c_max": max(s["temp_c"] for s in window),
        "util_pct_median": statistics.median(s["util_pct"] for s in window),
        "throttle_reasons_seen": throttles,
    }


def measure(cfg: dict[str, Any], trace: Path, out_json: Path) -> None:
    """Round-1 measurement with GPU telemetry sampled throughout."""
    from vllm import LLMEngine, SamplingParams
    from vllm.v1.spec_decode.self_spec_profiler import get_profiler

    manifest = r1._load_json(matrix._repository_path(r1.PROMPT_MANIFEST))
    regimes = {r["regime_id"]: r for r in manifest["prompt_plan"]["regimes"]}
    device = matrix.lane_for_block(1)["physical_gpu_index"]
    profiler = get_profiler()
    _require(profiler.enabled, "the self-spec profiler is not enabled")
    engine = LLMEngine.from_engine_args(r1._engine_args(cfg))
    observations: dict[str, Any] = {}
    neighbours_before = _neighbour_state(device)
    try:
        with ClockSampler(device) as sampler:
            for regime_id, spec in regimes.items():
                prompts = r1._prompts_for(regime_id, spec["batch"])
                before = r1._trace_len(trace)
                profiler.reset()
                start = time.monotonic()
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
                end = time.monotonic()
                steps = r1._armed_steps(trace, before)
                chain = profiler.summary(warmup=r1.PROFILER_WARMUP).get(
                    "draft_chain", {}
                )
                observations[regime_id] = {
                    "draft_chain_s": (
                        chain["mean_ms"] / 1000.0 if chain.get("mean_ms") else None
                    ),
                    "armed_step_count": len(steps),
                    "mean_armed_step_s": (statistics.mean(steps) if steps else None),
                    "batch": spec["batch"],
                    "wall_s": end - start,
                    "gpu": _summarise(sampler.between(start, end)),
                }
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()
    out_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_clock_clamp",
                "config": cfg,
                "observations": observations,
                "neighbours_before": neighbours_before,
                "neighbours_after": _neighbour_state(device),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _neighbour_state(device: int) -> list[str]:
    """What the other GPUs are drawing -- the suspected driver of the cap."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,power.draw,clocks.sm,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines()]


def run_all(output_dir: Path, repeats: int) -> None:
    lane = matrix.lane_for_block(1)
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    traces = output_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    for repeat in range(repeats):
        name = f"r{repeat}"
        target = output_dir / f"{name}.json"
        if target.exists():
            continue
        trace = traces / f"{name}.jsonl"
        _require(not trace.exists(), f"trace {trace} already exists")
        env = matrix._boot_child_environment(r1.boot_environment(CELL, trace))
        log = output_dir / f"{name}.log"
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output-dir",
                    str(output_dir),
                    "--cell",
                    json.dumps({"name": name}),
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
            print(f"[x21] FAILED {name}", flush=True)
            continue
        print(f"[x21] ok {name}", flush=True)


def summarise(output_dir: Path) -> dict[str, Any]:
    """Pair each regime's inflation with the clocks observed while measuring it."""
    boots: dict[str, Any] = {}
    for path in sorted(output_dir.glob("r*.json")):
        record = r1._load_json(path)
        rows = {}
        for regime, obs in record["observations"].items():
            if not obs.get("draft_chain_s"):
                continue
            now = obs["draft_chain_s"] * 1000.0
            rows[regime] = {
                "ms": round(now, 2),
                "delta_pct": round((now / V6_QUIET_MS[regime] - 1.0) * 100.0, 2),
                "sm_mhz_median": obs["gpu"].get("sm_mhz_median"),
                "sm_mhz_min": obs["gpu"].get("sm_mhz_min"),
                "power_w_median": obs["gpu"].get("power_w_median"),
                "throttle_reasons_seen": obs["gpu"].get("throttle_reasons_seen"),
            }
        clamped = [r for r, v in rows.items() if v["delta_pct"] > 4.0]
        boots[path.stem] = {
            "regimes": rows,
            "clamped_regimes": sorted(clamped),
            "neighbours_before": record.get("neighbours_before"),
        }
    any_clamped = any(b["clamped_regimes"] for b in boots.values())
    verdict = "INCONCLUSIVE_NO_CLAMP_OBSERVED"
    if any_clamped:
        throttled = any(
            v["throttle_reasons_seen"]
            for b in boots.values()
            for r, v in b["regimes"].items()
            if r in b["clamped_regimes"]
        )
        reduced = any(
            (v["sm_mhz_median"] or 0) < 1900
            for b in boots.values()
            for r, v in b["regimes"].items()
            if r in b["clamped_regimes"]
        )
        verdict = (
            "CLOCK_CAP_CONFIRMED" if (throttled or reduced) else "CLOCK_NOT_THE_CAUSE"
        )
    return {"boots": boots, "verdict": verdict, "v6_quiet_ms": V6_QUIET_MS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--summarise-only", action="store_true")
    parser.add_argument("--cell", help=argparse.SUPPRESS)
    parser.add_argument("--trace", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.cell:
        name = json.loads(args.cell)["name"]
        measure(CELL, Path(args.trace).resolve(), output_dir / f"{name}.json")
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
        run_all(output_dir, args.repeats)
    summary = summarise(output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
