# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X20 -- does lane CPU pinning cause the clamp when the box is loud?

DIAGNOSTIC, NOT SCORED.

The G98-C campaign was aborted after its first boot. The anchor cell
`target-matching/woff/skip0`, measured at 30.33-30.36 ms at R1 across every v6
boot with CV 0.027%, came back at 38.18 ms -- +25.8%:

    regime  campaign    v6    delta
    R1        38.18   30.36  +25.8%
    R8        38.26   31.75  +20.5%
    R6        34.54   33.92   +1.8%
    R4        43.70   43.59   +0.3%
    R5cot     52.34   51.84   +1.0%
    R5        52.44   51.89   +1.1%

A floor at ~38.2 ms: R1 and R8 lifted onto it, everything above it untouched.

Two instruments failed to catch it.

The startup host-load gate passed the boot (compile 10.24 s, capture 9.0 s,
both comfortably quiet). The cheap-regime spread read 3.72 ms against v6's
3.56 -- apparently normal. That second failure is a defect in the metric, not
bad luck: `max - min` is DIRECTION-BLIND. On a quiet box the cheap regimes rise
with batch (30.36 < 31.75 < 33.92), and under this clamp R1 and R8 are lifted
ABOVE R6 (38.18 > 34.54), which inverts the ordering while leaving the spread
almost unchanged. Monotonicity in batch was the signal all along.

The hypothesis
--------------

The campaign runner pins each boot to the lane's CPUs (0-15) via
`preexec_fn=os.sched_setaffinity`. None of the X17-X19 probes did. Co-tenants
on this box are unpinned (`Cpus_allowed_list` 0-191), so they compete for those
16 cores, while an unpinned boot can migrate onto the other 176.

That means X19's "the clamp does not reproduce" conclusion tested a
configuration the campaign does not use, and cannot be carried over. It is
withdrawn pending this probe.

If the hypothesis holds, then under identical load:

  * PINNED boots show inflated R1/R8, an inverted cheap-regime ordering, and a
    LARGE runqueue wait;
  * UNPINNED boots reproduce v6 with a near-zero runqueue wait.

This also supplies the positive case the runqueue instrument has never had --
it has only ever read ~0, so it is currently unvalidated as a detector.

Design
------

One cell, the campaign's own anchor, four boots interleaved
(pinned, unpinned, pinned, unpinned) so drift cannot masquerade as the effect.
Runqueue wait is sampled per regime from `/proc/self/task/*/schedstat`, and the
cheap-regime ordering is reported alongside the spread so the two can be
compared as candidate detectors.
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
import w98_host_load as hostload  # noqa: E402

matrix = r1.matrix

CELL = {"quant": "target-matching", "window": "off", "skip_count": 0}
ORDER = [
    ("r0_pinned", True),
    ("r0_unpinned", False),
    ("r1_pinned", True),
    ("r1_unpinned", False),
]
# The campaign anchor's v6 values, quiet box, same prompts.
V6_QUIET_MS = {
    "R1": 30.36,
    "R4": 43.59,
    "R5": 51.89,
    "R5cot": 51.84,
    "R6": 33.92,
    "R8": 31.75,
}
# Ordered by batch: 1, 16, 32. On a quiet box draft chain rises along this list.
CHEAP_BY_BATCH = ("R1", "R8", "R6")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def runqueue_wait_ns() -> int:
    """Nanoseconds this process's threads spent runnable but not running."""
    total = 0
    try:
        for task in Path("/proc/self/task").iterdir():
            with contextlib.suppress(OSError, ValueError, IndexError):
                total += int((task / "schedstat").read_text().split()[1])
    except OSError:
        return 0
    return total


def measure(cfg: dict[str, Any], trace: Path, out_json: Path) -> None:
    """The Round-1 measurement with per-regime runqueue sampling."""
    from vllm import LLMEngine, SamplingParams
    from vllm.v1.spec_decode.self_spec_profiler import get_profiler

    manifest = r1._load_json(matrix._repository_path(r1.PROMPT_MANIFEST))
    regimes = {r["regime_id"]: r for r in manifest["prompt_plan"]["regimes"]}
    profiler = get_profiler()
    _require(profiler.enabled, "the self-spec profiler is not enabled")
    engine = LLMEngine.from_engine_args(r1._engine_args(cfg))
    observations: dict[str, Any] = {}
    try:
        for regime_id, spec in regimes.items():
            prompts = r1._prompts_for(regime_id, spec["batch"])
            before = r1._trace_len(trace)
            profiler.reset()
            wait0, wall0 = runqueue_wait_ns(), time.monotonic()
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
            steps = r1._armed_steps(trace, before)
            chain = profiler.summary(warmup=r1.PROFILER_WARMUP).get("draft_chain", {})
            observations[regime_id] = {
                "draft_chain_s": (
                    chain["mean_ms"] / 1000.0 if chain.get("mean_ms") else None
                ),
                "armed_step_count": len(steps),
                "mean_armed_step_s": statistics.mean(steps) if steps else None,
                "runqueue_wait_s": wait_s,
                "wall_s": wall,
                "runqueue_wait_frac": wait_s / wall if wall else None,
                "batch": spec["batch"],
                "cpu_affinity_size": len(os.sched_getaffinity(0)),
            }
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()
    out_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_affinity_starvation",
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
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = sorted(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    traces = output_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    for name, pinned in ORDER:
        target = output_dir / f"{name}.json"
        if target.exists():
            continue
        trace = traces / f"{name}.jsonl"
        _require(not trace.exists(), f"trace {trace} already exists")
        env = matrix._boot_child_environment(r1.boot_environment(CELL, trace))
        log = output_dir / f"{name}.log"
        # The single manipulated variable: whether the child inherits the
        # lane's 16-CPU affinity or the machine's full set.
        pre = (lambda: os.sched_setaffinity(0, affinity)) if pinned else None
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
                preexec_fn=pre,
            )
        if completed.returncode != 0 or not target.exists():
            (output_dir / f"{name}.FAILED").write_text(
                f"returncode={completed.returncode}\n", encoding="utf-8"
            )
            print(f"[x20] FAILED {name}", flush=True)
            continue
        record = r1._load_json(target)
        record["pinned"] = pinned
        record["host_load"] = hostload.signature_from_log_path(log).as_record()
        target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"[x20] ok {name} pinned={pinned}", flush=True)


def summarise(output_dir: Path) -> dict[str, Any]:
    """Compare pinned against unpinned, and the two candidate detectors."""
    cells: dict[str, Any] = {}
    by_arm: dict[bool, list[float]] = {True: [], False: []}
    for name, pinned in ORDER:
        path = output_dir / f"{name}.json"
        if not path.is_file():
            cells[name] = {"status": "FAILED"}
            continue
        record = r1._load_json(path)
        obs = record["observations"]
        now = {r: round(o["draft_chain_s"] * 1000.0, 2) for r, o in obs.items()}
        cheap = [now[r] for r in CHEAP_BY_BATCH]
        by_arm[pinned].append(now["R1"])
        cells[name] = {
            "status": "ok",
            "pinned": pinned,
            "cpu_affinity_size": obs["R1"]["cpu_affinity_size"],
            "draft_chain_ms": now,
            "delta_pct_vs_v6": {
                r: round((now[r] / quiet - 1.0) * 100.0, 2)
                for r, quiet in V6_QUIET_MS.items()
            },
            "runqueue_wait_frac": {
                r: round(o["runqueue_wait_frac"], 4)
                for r, o in obs.items()
                if o.get("runqueue_wait_frac") is not None
            },
            "host_load_gate": record["host_load"]["verdict"],
            # Candidate detector A, direction-blind and known to have missed
            # the aborted campaign boot.
            "cheap_spread_ms": round(max(cheap) - min(cheap), 2),
            # Candidate detector B: cost must rise with batch on a quiet box.
            "cheap_monotonic_in_batch": all(
                cheap[i] < cheap[i + 1] for i in range(len(cheap) - 1)
            ),
        }
    ok = [c for c in cells.values() if c.get("status") == "ok"]
    verdict = "INCONCLUSIVE"
    if by_arm[True] and by_arm[False]:
        pinned_r1 = statistics.mean(by_arm[True])
        unpinned_r1 = statistics.mean(by_arm[False])
        verdict = (
            "PINNING_CAUSES_CLAMP"
            if pinned_r1 > unpinned_r1 * 1.05
            else "PINNING_NOT_THE_CAUSE"
        )
        cells["_arms"] = {
            "pinned_r1_ms": round(pinned_r1, 2),
            "unpinned_r1_ms": round(unpinned_r1, 2),
            "ratio": round(pinned_r1 / unpinned_r1, 4),
        }
    return {
        "cells": cells,
        "verdict": verdict,
        "detector_comparison": {
            "gate_caught": [
                c.get("host_load_gate") for c in ok if c.get("host_load_gate")
            ],
            "monotonic_flags": {
                c["pinned"]: c["cheap_monotonic_in_batch"] for c in ok if "pinned" in c
            },
        },
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
        run_all(output_dir)
    summary = summarise(output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
