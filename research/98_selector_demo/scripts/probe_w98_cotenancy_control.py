# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X18 -- is the skip16 anomaly real, or is the box just loud?

DIAGNOSTIC, NOT SCORED.

X17 measured `target-matching / woff / skip16` at 29.56 ms of draft chain at R1.
Against the Round-1 v6 ladder (skip0 30.36, skip4 27.30, skip8 24.43) that is a
5 ms REGRESSION from skip8 while removing 8 more layers -- keep_frac falls to
0.556 and the chain gets slower. If true it breaks monotonicity in keep_frac and
the R2 model with it.

But the two numbers come from different sessions. The v6 ladder ran at 03:00 on
a quiet box. X17 ran with four co-tenant `VLLM::EngineCore` processes pinned to
GPUs 4-7 at ~103% CPU each, and the draft chain under piecewise is substantially
HOST-bound (X8-X12). Three host-side quantities in the X17 boot corroborate the
load rather than the lever:

    quantity                  v6 skip8 (quiet)   X17 skip16 (loud)
    graph compile             5.34 / 4.49 s      8.69 / 6.94 s
    piecewise capture rate    7.56 it/s          5.01 it/s
    full-decode capture rate  11.58 it/s         6.63 it/s

Capture and compile should get FASTER with 16 layers removed, not ~40% slower.
That is not a lever effect; it is the host.

So this re-measures the two v6 reference points IN THE CURRENT SESSION, under
whatever load exists right now, and repeats skip16 between them. X5's lesson
applies: comparing across sessions understated drift by 50x, and only
interleaved measurement is trustworthy.

Reading:

  * skip0 and skip8 come back near their v6 values (30.36 / 24.43) -> the box is
    comparable and skip16 IS anomalous. A real finding, and the R2 lattice has a
    problem, since skip16 carries 7 of the 23 scored boots.
  * skip0 and skip8 are both inflated ~20% -> co-tenancy, X17's ladder
    comparison is void, and skip16 must be judged against THESE baselines.
    That also gates the campaign itself: sigma_repro was measured quiet
    (CV 0.03-0.26%), so a loud box invalidates the envelope.

Either way the campaign cannot start until this is settled.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402

matrix = r1.matrix

# Interleaved so that a monotone drift over the control session cannot be read
# as a difference between cells: the two v6 reference points bracket the repeat.
ORDER = [
    ("skip8_ctl", {"quant": "target-matching", "window": "off", "skip_count": 8}),
    ("skip16_rpt", {"quant": "target-matching", "window": "off", "skip_count": 16}),
    ("skip0_ctl", {"quant": "target-matching", "window": "off", "skip_count": 0}),
]
# Round-1 v6, measured on a quiet box. Draft chain in ms, per regime.
V6_QUIET_DRAFT_CHAIN_MS = {
    "skip0_ctl": {
        "R1": 30.36,
        "R4": 43.59,
        "R5": 51.89,
        "R5cot": 51.84,
        "R6": 33.92,
        "R8": 31.75,
    },
    "skip8_ctl": {
        "R1": 24.43,
        "R4": 33.67,
        "R5": 39.74,
        "R5cot": 40.67,
        "R6": 26.96,
        "R8": 25.48,
    },
}
# The X17 measurement this control exists to interpret.
X17_SKIP16_DRAFT_CHAIN_MS = {
    "R1": 29.56,
    "R4": 29.65,
    "R5": 32.66,
    "R5cot": 32.90,
    "R6": 29.70,
    "R8": 29.69,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def host_load() -> dict[str, Any]:
    """Record the co-tenancy that this control is measuring against."""
    import os
    import shutil

    load1, load5, load15 = os.getloadavg()
    gpus: list[str] = []
    smi = shutil.which("nvidia-smi")
    if smi:
        with contextlib.suppress(Exception):
            gpus = subprocess.run(
                [
                    smi,
                    "--query-gpu=index,memory.used,utilization.gpu",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            ).stdout.splitlines()
    return {"loadavg": [load1, load5, load15], "gpus": gpus}


def measure(cfg: dict[str, Any], trace: Path, out_json: Path) -> None:
    """Run the scored Round-1 measurement and stamp the host load around it."""
    before = host_load()
    observations = r1.measure_config(cfg, trace)
    out_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_cotenancy_control",
                "config": cfg,
                "observations": observations,
                "host_load_before": before,
                "host_load_after": host_load(),
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
            text = log.read_text(encoding="utf-8", errors="replace")
            (output_dir / f"{name}.FAILED").write_text(
                f"returncode={completed.returncode}\n\n" + text[-8000:],
                encoding="utf-8",
            )
            print(f"[control] FAILED {name}", flush=True)
            continue
        print(f"[control] ok {name}", flush=True)


def summarise(output_dir: Path) -> dict[str, Any]:
    """Compare each control cell against its quiet-box v6 value."""
    cells: dict[str, Any] = {}
    inflations: list[float] = []
    for name, _cfg in ORDER:
        path = output_dir / f"{name}.json"
        if not path.is_file():
            cells[name] = {"status": "FAILED"}
            continue
        record = r1._load_json(path)
        now = {
            regime: (
                round(obs["draft_chain_s"] * 1000.0, 2)
                if obs.get("draft_chain_s")
                else None
            )
            for regime, obs in record["observations"].items()
        }
        quiet = V6_QUIET_DRAFT_CHAIN_MS.get(name)
        delta = None
        if quiet:
            delta = {
                regime: round((now[regime] / quiet[regime] - 1.0) * 100.0, 2)
                for regime in sorted(quiet)
                if now.get(regime)
            }
            inflations.extend(delta.values())
        cells[name] = {
            "status": "ok",
            "draft_chain_ms": now,
            "v6_quiet_ms": quiet,
            "inflation_pct_vs_quiet": delta,
            "loadavg_before": record["host_load_before"]["loadavg"],
        }
    verdict = "INCONCLUSIVE"
    if inflations:
        worst = max(abs(v) for v in inflations)
        # sigma_repro on a quiet box was 0.03-0.26% CV, so anything past a few
        # percent is load, not measurement noise.
        verdict = "BOX_COMPARABLE" if worst < 3.0 else "BOX_LOUD"
    return {
        "cells": cells,
        "x17_skip16_ms": X17_SKIP16_DRAFT_CHAIN_MS,
        "verdict": verdict,
        "max_abs_inflation_pct": (
            round(max(abs(v) for v in inflations), 2) if inflations else None
        ),
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
        # NOT the idle preflight: the whole point is to measure a box that may
        # not be idle. GPU identity is still checked so the lane is right.
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
