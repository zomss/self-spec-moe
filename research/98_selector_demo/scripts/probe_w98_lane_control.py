# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X22 -- is the clamp specific to lane-a's GPU, or is it the whole box?

DIAGNOSTIC, NOT SCORED.

G98-C run 3 aborted at its first anchor: three attempts, all three rejected by
the measurement gate at relative spreads of 0.0084, 0.0172 and 0.0132 against a
threshold of 0.03. The gate worked -- it refused to emit a contaminated number
and failed loudly -- but the campaign cannot proceed while the clamp fires on
essentially every boot.

The clamp is now frequent enough to be tested against rather than hunted for.
Earlier it resisted reproduction: X21 measured this exact cell three times at
~23:00 and got three clean boots. Half an hour later the same cell clamped three
times running. That flip is the opportunity -- a discriminating test only needs
the effect to be present, not understood.

The question this answers
-------------------------

Lane-a is GPU 0 with CPUs 0-15. Lane-b is GPU 1 with CPUs 96-111, a separate
cache root, and its own place in the chassis. The neighbours currently drawing
576 W and 594 W are GPUs 2 and 3.

Booting the identical cell on both lanes, interleaved, separates:

  * BOTH clamp -> the cause is box-wide (chassis power, a shared rail, the
    host), and no lane choice escapes it. Waiting for the neighbours is the
    only remedy.
  * ONLY lane-a clamps -> the cause is local to GPU 0 or its CPUs, which both
    explains why four probes that varied host-side factors found nothing and
    offers lane-b as a workaround.

Note what a lane change would cost. Round 1 and every v6 reference number were
measured on lane-a, and the preregistration requires Round 2 to be comparable to
Round 1. Moving lanes is therefore NOT a free fix; it would need its own
re-measurement of the reference cell and an argument that the two lanes agree
when both are quiet. This probe measures whether that argument is even
available.

Arms are interleaved (a, b, a, b) so drift cannot masquerade as a lane effect.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
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
ORDER = [("r0_lane_a", 0), ("r0_lane_b", 1), ("r1_lane_a", 0), ("r1_lane_b", 1)]
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


def _lane_for_gpu(index: int) -> dict[str, Any]:
    for lane in matrix.V12_LANE_ASSIGNMENT:
        if lane["physical_gpu_index"] == index:
            return dict(lane)
    raise RuntimeError(f"no lane owns GPU {index}")


def measure(trace: Path, out_json: Path) -> None:
    """The Round-1 measurement, unchanged, on whichever GPU is pinned."""
    observations = r1.measure_config(CELL, trace)
    out_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_lane_control",
                "config": CELL,
                "observations": observations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_all(output_dir: Path) -> None:
    traces = output_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    for name, gpu in ORDER:
        target = output_dir / f"{name}.json"
        if target.exists():
            continue
        lane = _lane_for_gpu(gpu)
        Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
        lo, hi = lane["cpu_affinity"].split("-")
        affinity = sorted(range(int(lo), int(hi) + 1))
        trace = traces / f"{name}.jsonl"
        _require(not trace.exists(), f"trace {trace} already exists")
        env = matrix._boot_child_environment(r1.boot_environment(CELL, trace))
        # r1.boot_environment pins lane-a; override to the lane under test so
        # the GPU, its CPUs and its cache root all move together.
        env[matrix.DEVICE_PIN_ENV] = str(lane["physical_gpu_index"])
        env[matrix.CACHE_ROOT_ENV] = lane["cache_root"]
        log = output_dir / f"{name}.log"
        with (
            log.open("w", encoding="utf-8") as handle,
            hostload.GpuTelemetry(lane["physical_gpu_index"]) as telem,
        ):
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
                preexec_fn=lambda affinity=affinity: os.sched_setaffinity(0, affinity),
            )
        if completed.returncode != 0 or not target.exists():
            (output_dir / f"{name}.FAILED").write_text(
                f"returncode={completed.returncode}\n", encoding="utf-8"
            )
            print(f"[x22] FAILED {name}", flush=True)
            continue
        record = r1._load_json(target)
        record["gpu"] = lane["physical_gpu_index"]
        record["lane"] = lane["lane_id"]
        record["telemetry"] = telem.summary()
        record["host_load"] = hostload.signature_from_log_path(log).as_record()
        verdict = hostload.measurement_verdict(record["observations"])
        record["measurement_gate"] = verdict.as_record() if verdict else None
        target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"[x22] ok {name} gpu={gpu}", flush=True)


def summarise(output_dir: Path) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    by_lane: dict[str, list[str]] = {}
    for name, gpu in ORDER:
        path = output_dir / f"{name}.json"
        if not path.is_file():
            cells[name] = {"status": "FAILED"}
            continue
        record = r1._load_json(path)
        ms = {
            r: round(o["draft_chain_s"] * 1000.0, 2)
            for r, o in record["observations"].items()
            if o.get("draft_chain_s")
        }
        gate = record.get("measurement_gate") or {}
        cells[name] = {
            "status": "ok",
            "gpu": gpu,
            "draft_chain_ms": ms,
            "delta_pct_vs_v6": {
                r: round((ms[r] / q - 1.0) * 100.0, 2)
                for r, q in V6_QUIET_MS.items()
                if r in ms
            },
            "measurement_gate": gate.get("verdict"),
            "relative_spread": gate.get("relative_spread"),
            "telemetry": record.get("telemetry"),
        }
        by_lane.setdefault(record["lane"], []).append(gate.get("verdict"))
    verdict = "INCONCLUSIVE"
    if len(by_lane) == 2:
        a = by_lane.get("lane-a", [])
        b = by_lane.get("lane-b", [])
        a_bad = any(v == "clamped" for v in a)
        b_bad = any(v == "clamped" for v in b)
        if a_bad and b_bad:
            verdict = "BOX_WIDE"
        elif a_bad and not b_bad:
            verdict = "LANE_A_SPECIFIC"
        elif not a_bad and not b_bad:
            verdict = "NO_CLAMP_OBSERVED"
        else:
            verdict = "LANE_B_SPECIFIC"
    return {"cells": cells, "by_lane": by_lane, "verdict": verdict}


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
        measure(Path(args.trace).resolve(), output_dir / f"{name}.json")
        return 0
    if not args.summarise_only:
        with contextlib.suppress(Exception):
            base_env = matrix._boot_child_environment({})
            matrix._preflight_native_sampler(base_env)
            matrix._preflight_inprocess_engine_core(base_env)
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
