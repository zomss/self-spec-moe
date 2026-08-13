# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X27 -- does whole-chain remove the host-bound cost, correctly?

DIAGNOSTIC, NOT SCORED.

X24 measured the batch-1 engine step at 48 ms wall against 5.14 ms of GPU
kernel time: 10.7% utilisation, 822 kernel launches per step, and no forced
device-to-host syncs anywhere in the chain loop. There is no bad operation to
delete -- the host cost IS the per-layer, per-chain-step dispatch that piecewise
capture leaves on the critical path. Whole-chain capture replaces it with a
single graph launch.

Two fixes were required first, both real bugs for runtime switching:

  * `_wc_key` omitted the window. It keyed on the block-table width, which
    follows the CONTEXT; the window changes the scratchpad geometry while
    leaving that width alone, so a window switch would replay a graph captured
    for a different kept-page count. Now keyed by `_lever_signature()`, which
    also carries skip and quant so making those switchable cannot reintroduce
    the bug silently.
  * `_sp_col_arange` reallocated whenever the window changed the cap, moving a
    tensor whose address captured graphs already hold -- the exact IMA class the
    Phase-93 fix exists to prevent. Now max-allocated once and sliced.

This measures both arms on the same cell, interleaved, and checks the thing
that matters more than speed: ACCEPTANCE. A faster chain that drafts differently
is not the same chain. Acceptance is read from the koff trace, which is written
identically in both arms.

`woff` cannot be used here: whole-chain requires the scratchpad, hence a window.
That is exactly why the Round-2 cost lattice is measured under piecewise (it
must sample the skip and quant axes at `woff`), and why serving is a separate
decision -- the selector never has to reach `woff`, which is 37-63% WORSE than
w128 at long context anyway.
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

# Whole-chain needs a window; w256 is a registered lattice point.
CELL = {"quant": "target-matching", "window": 256, "skip_count": 0}
REGIMES = tuple(os.environ.get("W98_WC_REGIMES", "R8,R1,R6").split(","))
# Fine regions attribute host time per chain-step phase. They add syncs, so
# absolute values shift, but the QUESTION here is which component grew.
FINE = os.environ.get("W98_WC_FINE", "0")
ORDER = [
    ("r0_piecewise", False),
    ("r0_wholechain", True),
    ("r1_piecewise", False),
    ("r1_wholechain", True),
]
V6_QUIET_MS = {"R1": 31.63, "R8": 32.41, "R6": 34.16}  # v6 w256/skip0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def acceptance(trace: Path) -> dict[str, Any]:
    """Accepted draft tokens per armed request -- the correctness check."""
    if not trace.is_file():
        return {}
    a = d = 0
    with trace.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("record_type") != "koff_engine_step":
                continue
            counters = row.get("counters") or {}
            if not counters.get("D_armed"):
                continue
            a += int(counters.get("A_accepted", 0))
            d += int(counters.get("D_armed", 0))
    return {"A": a, "D": d, "per_armed_request": (a / d) if d else None}


def measure(trace: Path, out_json: Path) -> None:
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
                "mean_armed_step_s": statistics.mean(armed) if armed else None,
                "armed_step_count": len(armed),
                "batch": spec["batch"],
                "regions_ms": {
                    k: round(v["mean_ms"], 3)
                    for k, v in summary.items()
                    if isinstance(v, dict) and v.get("mean_ms")
                },
            }
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()
    verdict = hostload.measurement_verdict(observations)
    out_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_wholechain_ab",
                "config": CELL,
                "observations": observations,
                "acceptance": acceptance(trace),
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
    for name, wholechain in ORDER:
        target = output_dir / f"{name}.json"
        if target.exists():
            continue
        trace = traces / f"{name}.jsonl"
        _require(not trace.exists(), f"trace {trace} already exists")
        env = matrix._boot_child_environment(r1.boot_environment(CELL, trace))
        # The single manipulated variable.
        env["VLLM_SELF_SPEC_DRAFT_WHOLECHAIN"] = "1" if wholechain else "0"
        env["VLLM_SELF_SPEC_DRAFT_FULLCG"] = "1" if wholechain else "0"
        env["VLLM_SELF_SPEC_PROFILE_FINE"] = FINE
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
                f"returncode={completed.returncode}\n\n{text[-6000:]}", encoding="utf-8"
            )
            print(f"[x27] FAILED {name}", flush=True)
            continue
        record = r1._load_json(target)
        record["wholechain"] = wholechain
        text = log.read_text(encoding="utf-8", errors="replace")
        # Prove the arm actually took: whole-chain never logs the piecewise mark.
        record["ran_piecewise"] = "Draft chain PIECEWISE:" in text
        record["wc_capture_seen"] = "capture sizes augmented" in text
        target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"[x27] {name} wholechain={wholechain}", flush=True)


def summarise(output_dir: Path) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    arms: dict[bool, dict[str, list[float]]] = {True: {}, False: {}}
    acc: dict[bool, list[float]] = {True: [], False: []}
    for name, wholechain in ORDER:
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
            arms[wholechain].setdefault(regime, []).append(value)
        rate = (record.get("acceptance") or {}).get("per_armed_request")
        if rate:
            acc[wholechain].append(rate)
        cells[name] = {
            "status": "ok",
            "wholechain": wholechain,
            "ran_piecewise": record.get("ran_piecewise"),
            "draft_chain_ms": ms,
            "accepted_per_armed_request": round(rate, 4) if rate else None,
            "measurement_gate": (record.get("measurement_gate") or {}).get("verdict"),
        }
    per_regime = {}
    for regime in V6_QUIET_MS:
        wc, pw = arms[True].get(regime), arms[False].get(regime)
        if wc and pw:
            mwc, mpw = statistics.mean(wc), statistics.mean(pw)
            per_regime[regime] = {
                "piecewise_ms": round(mpw, 2),
                "wholechain_ms": round(mwc, 2),
                "speedup": round(mpw / mwc, 3),
                "saved_ms": round(mpw - mwc, 2),
            }
    out = {"cells": cells, "per_regime": per_regime}
    if acc[True] and acc[False]:
        out["acceptance"] = {
            "piecewise": round(statistics.mean(acc[False]), 4),
            "wholechain": round(statistics.mean(acc[True]), 4),
            "delta": round(statistics.mean(acc[True]) - statistics.mean(acc[False]), 4),
        }
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
