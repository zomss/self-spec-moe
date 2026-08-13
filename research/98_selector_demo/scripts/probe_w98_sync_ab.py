# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X25 -- what does the profiler's own synchronisation cost?

DIAGNOSTIC, NOT SCORED. Changes nothing on the scored path.

X24's trace found 12 `cudaDeviceSynchronize` calls per engine step, and all of
them come from `SelfSpecProfiler.region`, which syncs at both ends of every
timed region:

    region                       per step   syncs
    draft_chain (outer)                 1       2
    draft_forward_first                 1       2
    draft_forward (in chain loop)       3       6
    verify                              1       2
                                            -----
                                               12

Confirmed by kernel counts: 8640 flash-attention kernels over 60 profiled steps
is 144 per step, i.e. 36 layers x 4 forwards.

`_fine_only()` gates per-step sub-timers behind VLLM_SELF_SPEC_PROFILE_FINE,
but matches only labels starting with `step_`, `step0_` or `chain_setup`.
`draft_forward` fires once per CHAIN STEP and is therefore a per-step timer, yet
it is not on that list -- so eight of the twelve syncs sit INSIDE the
`draft_chain` region being measured.

Why this could matter a great deal
----------------------------------

A sync adds no GPU work; it removes OVERLAP. Normally the host runs ahead
launching while the device executes and a step costs roughly max(host, gpu).
Synchronising on every forward makes them alternate, so the step costs
host + gpu. X24 measured ~6 ms/step of launch API against 5.14 ms/step of GPU
kernels, so the lost overlap could be several milliseconds per step at batch 1.

If so, two things follow. Our `draft_chain` overstates production cost at low
batch, since production does not sync per forward. And the instrument maximises
sensitivity to the clamp, by deleting the GPU slack that would otherwise hide a
host-side delay -- which would explain why the clamp scales exactly with how
much GPU work is available to hide it.

Method
------

Interleaved A B A B on one cell, because this phase has repeatedly been fooled
by comparisons made across time:

  * A `syncs_on`  -- the profiler exactly as the scored campaign runs it.
  * B `syncs_off` -- `_fine_only` monkeypatched in the child to also match
    `draft_forward`, leaving only the outer `draft_chain` and `verify` syncs
    (12 -> 4 per step). Nothing on disk changes; the patch lives in the probe
    process only.

Both arms run the three cheap regimes so the measurement gate can report
whether a boot was clamped. The box is currently clamping every boot, which
does not invalidate the comparison -- the clamp hits both arms and the design
is interleaved -- but it is recorded per boot so the reader can judge.

What this does NOT do
---------------------

It does not change the scored measurement. Round 1's v6 numbers were taken with
these syncs in place and the preregistration requires Round 2 cost to be
comparable to Round 1, so quietly making the profiler cheaper would break the
cross-round comparison the design rests on. If the cost is large, that is a
finding about every draft_chain number in this phase and belongs in the paper,
and any change belongs in a registered decision -- not in a probe.
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
ORDER = [
    ("r0_syncs_on", True),
    ("r0_syncs_off", False),
    ("r1_syncs_on", True),
    ("r1_syncs_off", False),
]
# v6 reference for this cell, quiet box, syncs ON.
V6_QUIET_MS = {"R1": 30.36, "R8": 31.75, "R6": 33.92}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _disable_inner_syncs() -> None:
    """Gate `draft_forward*` behind FINE, so only the outer regions sync.

    Monkeypatched rather than edited: `self_spec_profiler.py` is shared by the
    scored path, and this probe must not alter what a campaign would measure.
    """
    from vllm.v1.spec_decode import self_spec_profiler as ssp

    ssp.SelfSpecProfiler._fine_only = staticmethod(
        lambda label: label.startswith(
            ("step_", "step0_", "chain_setup", "draft_forward")
        )
    )


def measure(trace: Path, out_json: Path, syncs_on: bool) -> None:
    """Boot and measure the cheap regimes under the requested sync regime."""
    if not syncs_on:
        _disable_inner_syncs()
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
            summary = profiler.summary(warmup=r1.PROFILER_WARMUP)
            chain = summary.get("draft_chain", {})
            observations[regime_id] = {
                "draft_chain_s": (
                    chain["mean_ms"] / 1000.0 if chain.get("mean_ms") else None
                ),
                # The whole armed engine step is measured WITHOUT profiler
                # syncs either way, so it is the arm-independent yardstick:
                # if the syncs cost real time, this moves too.
                "mean_armed_step_s": statistics.mean(armed) if armed else None,
                "armed_step_count": len(armed),
                "draft_forward_timed": "draft_forward" in summary,
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
                "record_type": "w98_sync_ab",
                "config": CELL,
                "syncs_on": syncs_on,
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
    for name, syncs_on in ORDER:
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
                    json.dumps({"name": name, "syncs_on": syncs_on}),
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
            (output_dir / f"{name}.FAILED").write_text(
                f"returncode={completed.returncode}\n", encoding="utf-8"
            )
            print(f"[x25] FAILED {name}", flush=True)
            continue
        print(f"[x25] ok {name} syncs_on={syncs_on}", flush=True)


def summarise(output_dir: Path) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    arms: dict[bool, dict[str, list[float]]] = {True: {}, False: {}}
    steps: dict[bool, dict[str, list[float]]] = {True: {}, False: {}}
    for name, syncs_on in ORDER:
        path = output_dir / f"{name}.json"
        if not path.is_file():
            cells[name] = {"status": "FAILED"}
            continue
        record = r1._load_json(path)
        chain = {
            r: round(o["draft_chain_s"] * 1000.0, 2)
            for r, o in record["observations"].items()
            if o.get("draft_chain_s")
        }
        step = {
            r: round(o["mean_armed_step_s"] * 1000.0, 2)
            for r, o in record["observations"].items()
            if o.get("mean_armed_step_s")
        }
        for regime, value in chain.items():
            arms[syncs_on].setdefault(regime, []).append(value)
        for regime, value in step.items():
            steps[syncs_on].setdefault(regime, []).append(value)
        gate = record.get("measurement_gate") or {}
        cells[name] = {
            "status": "ok",
            "syncs_on": syncs_on,
            "draft_chain_ms": chain,
            "mean_armed_step_ms": step,
            "draft_forward_timed": record["observations"]["R1"]["draft_forward_timed"],
            "measurement_gate": gate.get("verdict"),
        }
    delta = {}
    for regime in V6_QUIET_MS:
        on = arms[True].get(regime)
        off = arms[False].get(regime)
        if not on or not off:
            continue
        mon, moff = statistics.mean(on), statistics.mean(off)
        entry = {
            "chain_syncs_on_ms": round(mon, 2),
            "chain_syncs_off_ms": round(moff, 2),
            "chain_delta_ms": round(mon - moff, 2),
            "chain_delta_pct": round(100.0 * (mon / moff - 1.0), 2),
        }
        son, soff = steps[True].get(regime), steps[False].get(regime)
        if son and soff:
            entry["step_syncs_on_ms"] = round(statistics.mean(son), 2)
            entry["step_syncs_off_ms"] = round(statistics.mean(soff), 2)
            entry["step_delta_ms"] = round(
                statistics.mean(son) - statistics.mean(soff), 2
            )
        delta[regime] = entry
    return {"cells": cells, "per_regime": delta, "v6_quiet_ms": V6_QUIET_MS}


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
            Path(args.trace).resolve(),
            output_dir / f"{cell['name']}.json",
            cell["syncs_on"],
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
    print(json.dumps(summary["per_regime"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
