# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X26 -- do co-tenant GPUs cause the clamp? Controlled, interleaved.

DIAGNOSTIC, NOT SCORED.

All night the clamp tracked a neighbouring tensor-parallel job on GPUs 2 and 3,
but the correlation could never be tested: that job belonged to someone else, so
every comparison was observational and taken across time. Two conclusions drawn
that way later dissolved, and the one refutation of the neighbour hypothesis
(campaign run 4, clamped while the neighbours read 0% at a single spot-check)
rested on far weaker evidence than it was presented with.

The box is now idle, so the load can be MANUFACTURED. That converts the question
into an experiment: alternate boots with and without synthetic load on GPUs 2
and 3, back to back, in one session.

  * A `noload` -- neighbours idle at ~70 W.
  * B `load`   -- `w98_cotenant_load` driving GPUs 2 and 3 to 100% SM and
    ~700 W, exceeding the real neighbour's ~500 W.

Interleaved A B A B, because a several-minute switching timescale makes any
A-then-B design capable of manufacturing a difference from nothing.

Reading:

  * load clamps and noload does not, alternating -> co-tenant GPU activity is
    the cause, the earlier refutation was wrong, and the operational rule is
    simply that the campaign needs idle neighbours;
  * neither clamps -> the synthetic load does not reproduce it, and the
    difference from the real neighbour (memory-bound inference at 91% memory
    utilisation, plus NCCL host spin, versus compute-bound GEMMs at 30%) points
    at which property matters;
  * both clamp -> the box is in a bad state independent of load, and the
    neighbour correlation is finally dead.

The load deliberately omits the neighbour's HOST behaviour -- no NCCL spin, no
busy polling -- so a positive result isolates GPU-side activity as sufficient.
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
REGIMES = ("R8", "R1", "R6")
ORDER = [
    ("r0_noload", False),
    ("r0_load", True),
    ("r1_noload", False),
    ("r1_load", True),
]
LOAD_DEVICES = "2,3"
LOAD_RAMP_S = 60.0  # vLLM co-tenant needs time to load and reach steady decode
LOAD_MAX_S = 900.0
# Load driver runs here: clear of lane-a (0-15) and lane-b (96-111).
LOAD_CPUS = "128-159"
# Faithful co-tenant: a real vLLM TP=2 engine, matching the neighbour on memory
# utilisation, footprint, NCCL host spin and NVLink traffic all at once.
LOAD_MODE = "hold"
V6_QUIET_MS = {"R1": 30.36, "R8": 31.75, "R6": 33.92}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def measure(trace: Path, out_json: Path) -> None:
    """Boot and measure the cheap regimes; the caller controls the load."""
    from vllm import LLMEngine, SamplingParams
    from vllm.v1.spec_decode.self_spec_profiler import get_profiler

    manifest = r1._load_json(matrix._repository_path(r1.PROMPT_MANIFEST))
    regimes = {r["regime_id"]: r for r in manifest["prompt_plan"]["regimes"]}
    profiler = get_profiler()
    _require(profiler.enabled, "the self-spec profiler is not enabled")
    engine = LLMEngine.from_engine_args(r1._engine_args(CELL))
    observations: dict[str, Any] = {}
    try:
        for regime_id in REGIMES:
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
            armed = r1._armed_steps(trace, before)
            chain = profiler.summary(warmup=r1.PROFILER_WARMUP).get("draft_chain", {})
            observations[regime_id] = {
                "draft_chain_s": (
                    chain["mean_ms"] / 1000.0 if chain.get("mean_ms") else None
                ),
                "mean_armed_step_s": statistics.mean(armed) if armed else None,
                "armed_step_count": len(armed),
                "batch": spec["batch"],
            }
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()
    verdict = hostload.measurement_verdict(observations)
    out_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_cotenant_ab",
                "config": CELL,
                "observations": observations,
                "measurement_gate": verdict.as_record() if verdict else None,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _neighbour_state() -> list[str]:
    with contextlib.suppress(Exception):
        return (
            subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={LOAD_DEVICES}",
                    "--query-gpu=index,utilization.gpu,utilization.memory,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            .stdout.strip()
            .splitlines()
        )
    return []


@contextlib.contextmanager
def cotenant_load(enabled: bool):
    """Drive GPUs 2 and 3 hard for the duration, if enabled."""
    if not enabled:
        yield []
        return
    procs = []
    logs = []
    for device in LOAD_DEVICES.split(","):
        log = open(f"/tmp/w98_load_gpu{device}.log", "w")  # noqa: SIM115
        logs.append(log)
        procs.append(
            subprocess.Popen(
                [
                    "taskset",
                    "-c",
                    LOAD_CPUS,
                    sys.executable,
                    str(SCRIPT_DIR / "w98_cotenant_load.py"),
                    "--devices",
                    device,
                    "--seconds",
                    str(LOAD_MAX_S),
                    "--mode",
                    LOAD_MODE,
                ],
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        )
    try:
        # Wait for the load to actually appear. A silent failure here produced
        # an invalid "load does not reproduce" verdict once already: the arms
        # were identical because the load never started.
        deadline = time.monotonic() + LOAD_RAMP_S
        state: list[str] = []
        while time.monotonic() < deadline:
            time.sleep(5.0)
            state = _neighbour_state()
            busy = [
                r
                for r in state
                if len(r.split(",")) > 1 and float(r.split(",")[1]) > 20
            ]
            if len(busy) == len(procs):
                break
        _require(
            bool(state)
            and all(
                float(r.split(",")[2]) >= 0 for r in state if len(r.split(",")) > 1
            ),
            f"co-tenant load did not start; neighbours={state}. "
            "See /tmp/w98_load_gpu*.log",
        )
        yield state
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            with contextlib.suppress(Exception):
                proc.wait(timeout=60)
        for log in logs:
            with contextlib.suppress(Exception):
                log.close()
        time.sleep(5.0)


def run_all(output_dir: Path) -> None:
    lane = matrix.lane_for_block(1)
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = sorted(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    traces = output_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    for name, loaded in ORDER:
        target = output_dir / f"{name}.json"
        if target.exists():
            continue
        trace = traces / f"{name}.jsonl"
        _require(not trace.exists(), f"trace {trace} already exists")
        env = matrix._boot_child_environment(r1.boot_environment(CELL, trace))
        log = output_dir / f"{name}.log"
        with cotenant_load(loaded) as neighbours:
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
                    preexec_fn=lambda: os.sched_setaffinity(0, affinity),
                )
            during = _neighbour_state()
        if completed.returncode != 0 or not target.exists():
            (output_dir / f"{name}.FAILED").write_text(
                f"returncode={completed.returncode}\n", encoding="utf-8"
            )
            print(f"[x26] FAILED {name}", flush=True)
            continue
        record = r1._load_json(target)
        record["cotenant_load"] = loaded
        record["neighbours_at_start"] = neighbours
        record["neighbours_at_end"] = during
        target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        gate = (record.get("measurement_gate") or {}).get("verdict")
        print(f"[x26] {name} load={loaded} gate={gate}", flush=True)


def summarise(output_dir: Path) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    arms: dict[bool, list[str]] = {}
    chains: dict[bool, dict[str, list[float]]] = {True: {}, False: {}}
    for name, loaded in ORDER:
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
        for regime, value in ms.items():
            chains[loaded].setdefault(regime, []).append(value)
        gate = record.get("measurement_gate") or {}
        arms.setdefault(loaded, []).append(gate.get("verdict"))
        cells[name] = {
            "status": "ok",
            "cotenant_load": loaded,
            "draft_chain_ms": ms,
            "delta_pct_vs_v6": {
                r: round((ms[r] / q - 1.0) * 100.0, 2)
                for r, q in V6_QUIET_MS.items()
                if r in ms
            },
            "measurement_gate": gate.get("verdict"),
            "relative_spread": gate.get("relative_spread"),
            "neighbours_at_start": record.get("neighbours_at_start"),
        }
    verdict = "INCONCLUSIVE"
    if len(arms) == 2:
        load_bad = any(v == "clamped" for v in arms.get(True, []))
        idle_bad = any(v == "clamped" for v in arms.get(False, []))
        if load_bad and not idle_bad:
            verdict = "COTENANT_LOAD_CAUSES_CLAMP"
        elif not load_bad and not idle_bad:
            verdict = "SYNTHETIC_LOAD_DOES_NOT_REPRODUCE"
        elif load_bad and idle_bad:
            verdict = "BOTH_CLAMP_LOAD_IRRELEVANT"
        else:
            verdict = "IDLE_CLAMPS_ONLY"
    per_regime = {}
    for regime in V6_QUIET_MS:
        on, off = chains[True].get(regime), chains[False].get(regime)
        if on and off:
            per_regime[regime] = {
                "load_ms": round(statistics.mean(on), 2),
                "idle_ms": round(statistics.mean(off), 2),
                "delta_pct": round(
                    100.0 * (statistics.mean(on) / statistics.mean(off) - 1.0), 2
                ),
            }
    return {"cells": cells, "per_regime": per_regime, "verdict": verdict}


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
        measure(Path(args.trace).resolve(), output_dir / f"{cell['name']}.json")
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
