# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X23 -- campaign path vs probe path, interleaved.

DIAGNOSTIC, NOT SCORED.

Five G98-C campaign sessions have clamped and six probe sessions have not. That
looks damning, but every one of those comparisons was made ACROSS TIME, and this
phase has already produced three causal claims that dissolved for exactly that
reason -- neighbour GPU activity, and CPU pinning twice. The clamp switches on a
timescale of minutes, so any A-then-B comparison can manufacture a difference
out of nothing.

So this alternates the two paths back to back, A B A B, on the same cell:

  * PROBE path    -- a child that imports the Round-1 runner and calls
    `measure_config` directly, which is what every clean probe does.
  * CAMPAIGN path -- the real `run_w98_g98c_round2.py --measure-config` child,
    which additionally imports the Round-2 model and validates the
    authorization before measuring.

Both get byte-identical environments from `r1.boot_environment`, the same lane
pinning, the same cell, and the same measurement function. The only thing that
varies is which entry point runs it.

Reading:

  * campaign clamps and probe does not, alternating -> something in the
    campaign child is responsible, and the next step is to bisect it;
  * both clamp or both clean -> the path is irrelevant and the earlier
    5-vs-6 tally was an artifact of when each happened to run, which would
    retire the last standing hypothesis.
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
ORDER = [
    ("r0_probe", "probe"),
    ("r0_campaign", "campaign"),
    ("r1_probe", "probe"),
    ("r1_campaign", "campaign"),
]
V6_QUIET_MS = {"R1": 30.36, "R8": 31.75, "R6": 33.92}
CAMPAIGN_RUNNER = "research/98_selector_demo/scripts/run_w98_g98c_round2.py"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def measure(trace: Path, out_json: Path) -> None:
    """The probe path: Round-1 measurement, nothing else loaded."""
    observations = r1.measure_config(CELL, trace)
    out_json.write_text(
        json.dumps(
            {
                "record_type": "w98_path_ab",
                "config": CELL,
                "observations": observations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _child_argv(path: str, output_dir: Path, name: str, trace: Path) -> list[str]:
    if path == "probe":
        return [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output-dir",
            str(output_dir),
            "--cell",
            json.dumps({"name": name}),
            "--trace",
            str(trace),
        ]
    stage = output_dir / f"stage_{name}"
    stage.mkdir(parents=True, exist_ok=True)
    import run_w98_g98c_round2 as g98c

    return [
        sys.executable,
        str(matrix._repository_path(CAMPAIGN_RUNNER)),
        "--authorization",
        str(matrix._repository_path(g98c.AUTHORIZATION_PATH)),
        "--output-dir",
        str(output_dir),
        "--measure-config",
        json.dumps(CELL),
        "--stage-dir",
        str(stage),
        "--trace",
        str(trace),
    ]


def _result_path(path: str, output_dir: Path, name: str) -> Path:
    if path == "probe":
        return output_dir / f"{name}.json"
    return output_dir / f"stage_{name}" / "target-matching_woff_skip0.json"


def run_all(output_dir: Path) -> None:
    lane = matrix.lane_for_block(1)
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = sorted(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    traces = output_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    for name, path in ORDER:
        summary_path = output_dir / f"{name}.summary.json"
        if summary_path.exists():
            continue
        trace = traces / f"{name}.jsonl"
        _require(not trace.exists(), f"trace {trace} already exists")
        env = matrix._boot_child_environment(r1.boot_environment(CELL, trace))
        log = output_dir / f"{name}.log"
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                _child_argv(path, output_dir, name, trace),
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                preexec_fn=lambda: os.sched_setaffinity(0, affinity),
            )
        produced = _result_path(path, output_dir, name)
        if completed.returncode != 0 or not produced.is_file():
            summary_path.write_text(
                json.dumps({"status": "FAILED", "path": path}) + "\n", encoding="utf-8"
            )
            print(f"[x23] FAILED {name} ({path})", flush=True)
            continue
        record = r1._load_json(produced)
        verdict = hostload.measurement_verdict(record["observations"])
        ms = {
            regime: round(record["observations"][regime]["draft_chain_s"] * 1000.0, 2)
            for regime in V6_QUIET_MS
        }
        summary_path.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "path": path,
                    "draft_chain_ms": ms,
                    "delta_pct_vs_v6": {
                        r: round((ms[r] / q - 1.0) * 100.0, 2)
                        for r, q in V6_QUIET_MS.items()
                    },
                    "gate": verdict.as_record() if verdict else None,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[x23] ok {name} ({path}) R1={ms['R1']}", flush=True)


def summarise(output_dir: Path) -> dict[str, Any]:
    cells = {}
    by_path: dict[str, list[str]] = {}
    for name, path in ORDER:
        p = output_dir / f"{name}.summary.json"
        if not p.is_file():
            continue
        record = r1._load_json(p)
        cells[name] = record
        if record.get("status") == "ok":
            by_path.setdefault(path, []).append(
                (record.get("gate") or {}).get("verdict")
            )
    verdict = "INCONCLUSIVE"
    if len(by_path) == 2:
        camp = by_path.get("campaign", [])
        prob = by_path.get("probe", [])
        c_bad = any(v == "clamped" for v in camp)
        p_bad = any(v == "clamped" for v in prob)
        if c_bad and not p_bad:
            verdict = "CAMPAIGN_PATH_IMPLICATED"
        elif c_bad and p_bad:
            verdict = "PATH_IRRELEVANT_BOTH_CLAMP"
        elif not c_bad and not p_bad:
            verdict = "PATH_IRRELEVANT_BOTH_CLEAN"
        else:
            verdict = "PROBE_PATH_IMPLICATED"
    return {"cells": cells, "by_path": by_path, "verdict": verdict}


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
