# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X29 -- does suppressing the mmap churn remove the clamp?

DIAGNOSTIC, NOT SCORED.

The chain of evidence this tests: the box is a KVM guest (clamp doc section
9); vCPU preemption is refuted by steal accounting (entry 16); the anonymous
page-fault path is NOT inflated during the clamped state (~1.5 us/fault,
bare-metal-normal). What remains is munmap TLB-shootdown amplification: the
engine step performs 144 mmap/munmap pairs (one per flash-attention call,
verify forwards included -- it is the shared FA path, not the chain loop),
each munmap IPIs every CPU running one of the process's threads, and under
KVM each IPI is emulated via VM exits. ~2 us of added cost per exit across
~15 target CPUs reproduces the whole 4-5 ms clamp.

The manipulation: `glibc.malloc.mmap_threshold` raised to 1 GiB makes the
per-call host temporaries come from the heap, which glibc reuses -- no
mmap/munmap, no shootdowns. An environment variable, zero code change, same
scheduled GPU work.

Interleaved A/B on the CLAMPED box, which for once makes the bad state an
asset: control boots must clamp for the test to have power, and heap boots
going quiet while controls clamp confirms the mechanism AND hands the
campaign a mitigation (a v7 re-issue decision, since it changes the pinned
boot environment and removes ~1 ms/step of real syscall time from measured
costs).

Manipulation check (the X26 lesson: verify the manipulation is active):
minor faults per engine step, read from getrusage in the measuring process.
Fresh mappings fault on every touch; heap reuse does not. If the heap arms'
fault rate does not collapse, the churn is not glibc's and the verdict is
INVALID, not refuted.
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
import w98_host_load as hostload  # noqa: E402

matrix = r1.matrix

CELL = {"quant": "target-matching", "window": "off", "skip_count": 0}
REGIMES = ("R8", "R1", "R6")
TUNABLE = "glibc.malloc.mmap_threshold=1073741824"
ORDER = [
    ("r0_control", False),
    ("r0_heap", True),
    ("r1_control", False),
    ("r1_heap", True),
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def measure(trace: Path, out_json: Path) -> None:
    import resource

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
            steps = 0
            minflt_before = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
            while engine.has_unfinished_requests():
                engine.step()
                steps += 1
            minflt = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
            armed = r1._armed_steps(trace, before)
            chain = profiler.summary(warmup=r1.PROFILER_WARMUP).get("draft_chain", {})
            observations[regime_id] = {
                "draft_chain_s": (
                    chain["mean_ms"] / 1000.0 if chain.get("mean_ms") else None
                ),
                "mean_armed_step_s": statistics.mean(armed) if armed else None,
                "armed_step_count": len(armed),
                "engine_steps": steps,
                "minflt_per_engine_step": (
                    round((minflt - minflt_before) / steps, 1) if steps else None
                ),
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
                "record_type": "w98_malloc_ab",
                "config": CELL,
                "glibc_tunables": os.environ.get("GLIBC_TUNABLES", ""),
                "observations": observations,
                "measurement_gate": verdict.as_record() if verdict else None,
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
    for name, heap in ORDER:
        target = output_dir / f"{name}.json"
        if target.exists():
            continue
        trace = traces / f"{name}.jsonl"
        _require(not trace.exists(), f"trace {trace} already exists")
        env = matrix._boot_child_environment(r1.boot_environment(CELL, trace))
        if heap:
            # The single manipulated variable.
            prev = env.get("GLIBC_TUNABLES")
            env["GLIBC_TUNABLES"] = f"{prev}:{TUNABLE}" if prev else TUNABLE
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
                preexec_fn=lambda: os.sched_setaffinity(0, affinity),
            )
        if completed.returncode != 0 or not target.exists():
            text = log.read_text(encoding="utf-8", errors="replace")
            (output_dir / f"{name}.FAILED").write_text(
                f"returncode={completed.returncode}\n\n{text[-6000:]}",
                encoding="utf-8",
            )
            print(f"[x29] FAILED {name}", flush=True)
            continue
        record = r1._load_json(target)
        record["arm"] = "heap" if heap else "control"
        target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"[x29] {name} arm={'heap' if heap else 'control'}", flush=True)


def summarise(output_dir: Path) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    verdicts: dict[str, list[str]] = {"heap": [], "control": []}
    minflt_r1: dict[str, list[float]] = {"heap": [], "control": []}
    chain_r1: dict[str, list[float]] = {"heap": [], "control": []}
    for name, heap in ORDER:
        path = output_dir / f"{name}.json"
        if not path.is_file():
            cells[name] = {"status": "FAILED"}
            continue
        record = r1._load_json(path)
        arm = "heap" if heap else "control"
        gate = (record.get("measurement_gate") or {}).get("verdict", "unknown")
        verdicts[arm].append(gate)
        r1_obs = record["observations"].get("R1", {})
        if r1_obs.get("minflt_per_engine_step") is not None:
            minflt_r1[arm].append(r1_obs["minflt_per_engine_step"])
        if r1_obs.get("draft_chain_s"):
            chain_r1[arm].append(round(r1_obs["draft_chain_s"] * 1000.0, 2))
        cells[name] = {
            "status": "ok",
            "arm": arm,
            "gate": gate,
            "draft_chain_ms": {
                r: round(o["draft_chain_s"] * 1000.0, 2)
                for r, o in record["observations"].items()
                if o.get("draft_chain_s")
            },
            "minflt_per_engine_step_R1": r1_obs.get("minflt_per_engine_step"),
        }
    out: dict[str, Any] = {
        "cells": cells,
        "by_arm": verdicts,
        "minflt_R1": minflt_r1,
        "chain_R1_ms": chain_r1,
    }
    manipulation_ok = (
        bool(minflt_r1["heap"])
        and bool(minflt_r1["control"])
        and statistics.mean(minflt_r1["heap"])
        < 0.5 * statistics.mean(minflt_r1["control"])
    )
    out["manipulation_active"] = manipulation_ok
    heap_v, ctl_v = set(verdicts["heap"]), set(verdicts["control"])
    if not manipulation_ok:
        out["verdict"] = "INVALID_MANIPULATION_NOT_ACTIVE"
    elif ctl_v == {"clamped"} and heap_v == {"quiet"}:
        out["verdict"] = "SHOOTDOWN_CONFIRMED_MITIGATION_WORKS"
    elif ctl_v == {"clamped"} and "clamped" in heap_v:
        out["verdict"] = "SHOOTDOWN_REFUTED_HEAP_ARM_CLAMPS"
    elif "clamped" not in ctl_v:
        out["verdict"] = "UNINFORMATIVE_CONTROLS_DID_NOT_CLAMP"
    else:
        out["verdict"] = "MIXED_SEE_CELLS"
    return out


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
        measure(
            Path(args.trace).resolve(),
            output_dir / f"{json.loads(args.cell)['name']}.json",
        )
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
