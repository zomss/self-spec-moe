#!/usr/bin/env python3
"""X32 -- does our stack cost anything when it is NOT speculating?

The claim under test is the one a reviewer will make first: "you added a
speculative-decoding runtime to vLLM; what does it cost me when it decides not
to speculate?" If the answer is anything but ~zero, every OFF cell in the phase
-- including the fail-closed rule's whole value proposition -- is measured
against a baseline our own stack degraded.

## Arms

* `stock`    -- vLLM with NO `speculative_config` and none of the
                VLLM_SELF_SPEC_* environment. Plain autoregressive decode.
* `koff_off` -- our engine, `speculative_config` present and loaded, K/OFF
                policy pinned to K=0. The draft shares the target's weights
                (`target-matching`), so no extra weights are resident and the
                comparison is compute, not memory.

Interleaved and repeated, so box drift cannot be read as an effect.

## Why the profiler is OFF in both arms

`SelfSpecProfiler.region()` syncs at both ends and costs +2.4 ms/step when
ungated (X25). Leaving it on would measure the instrument rather than the
stack, and stock vLLM has no trace to read anyway. So both arms run bare and
the currency is wall clock.

## How decode is isolated without a trace

Wall clock includes prefill, which is identical across arms but not zero. So
each arm is timed at TWO token budgets and the per-token decode cost is the
slope:

    time(N) = prefill + N * per_token
    per_token = (time(N_hi) - time(N_lo)) / (N_hi - N_lo)

Prefill cancels exactly, no trace required, and the intercept is reported as a
by-product so a broken run is visible rather than silent.
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
PHASE = SCRIPT_DIR.parent
REPO = PHASE.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402
import run_w98_g98e_d3 as d3  # noqa: E402  (rebinds the lane to this host)

matrix = r1.matrix

REGIMES = ("R1", "R8", "R6")
TOKENS_LO = 64
TOKENS_HI = 576
ORDER = [
    ("r0_stock", "stock"),
    ("r0_koff_off", "koff_off"),
    ("r1_stock", "stock"),
    ("r1_koff_off", "koff_off"),
]
CELL = {"quant": "target-matching", "window": "off", "skip_count": 0}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _engine(arm: str):
    from vllm import EngineArgs, LLMEngine

    if arm == "koff_off":
        return LLMEngine.from_engine_args(
            d3._engine_args({**CELL, "action": "off"})
        )
    # Stock: same engine geometry, no speculative_config at all.
    base = r1._engine_args(CELL)
    args = EngineArgs(
        model=base.model,
        tensor_parallel_size=base.tensor_parallel_size,
        pipeline_parallel_size=base.pipeline_parallel_size,
        max_model_len=base.max_model_len,
        max_num_batched_tokens=base.max_num_batched_tokens,
        max_num_seqs=base.max_num_seqs,
        enable_chunked_prefill=base.enable_chunked_prefill,
        gpu_memory_utilization=base.gpu_memory_utilization,
        enable_prefix_caching=base.enable_prefix_caching,
        async_scheduling=base.async_scheduling,
        enforce_eager=base.enforce_eager,
        enable_flashinfer_autotune=base.enable_flashinfer_autotune,
        seed=base.seed,
        disable_log_stats=base.disable_log_stats,
        generation_config=base.generation_config,
    )
    return LLMEngine.from_engine_args(args)


def _timed_run(engine, prompts, regime_id: str, budget: int) -> float:
    from vllm import SamplingParams

    for index, tokens in enumerate(prompts):
        engine.add_request(
            f"{regime_id}-{budget}-{index}",
            {"prompt_token_ids": tokens},
            SamplingParams(temperature=0.0, max_tokens=budget, ignore_eos=True),
        )
    started = time.perf_counter()
    while engine.has_unfinished_requests():
        engine.step()
    return time.perf_counter() - started


def measure(out_json: Path, arm: str) -> None:
    manifest = r1._load_json(matrix._repository_path(r1.PROMPT_MANIFEST))
    regimes = {r["regime_id"]: r for r in manifest["prompt_plan"]["regimes"]}
    engine = _engine(arm)
    observations: dict[str, Any] = {}
    try:
        for regime_id in REGIMES:
            spec = regimes[regime_id]
            prompts = r1._prompts_for(regime_id, spec["batch"])
            # Warm the shapes so the first timed run is not paying capture.
            _timed_run(engine, prompts, regime_id, 8)
            lo = _timed_run(engine, prompts, regime_id, TOKENS_LO)
            hi = _timed_run(engine, prompts, regime_id, TOKENS_HI)
            batch = spec["batch"]
            per_token = (hi - lo) / (TOKENS_HI - TOKENS_LO)
            observations[regime_id] = {
                "batch": batch,
                "seconds_lo": round(lo, 6),
                "seconds_hi": round(hi, 6),
                "per_decode_step_s": round(per_token, 9),
                "decode_tokens_per_s": (
                    round(batch / per_token, 4) if per_token > 0 else None
                ),
                "implied_prefill_s": round(lo - TOKENS_LO * per_token, 6),
            }
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()
    out_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_offpath_vs_stock",
                "arm": arm,
                "cell": CELL if arm == "koff_off" else None,
                "tokens_lo": TOKENS_LO,
                "tokens_hi": TOKENS_HI,
                "observations": observations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def run_all(output_dir: Path) -> None:
    lane = d3.lane_for_block(1)
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = sorted(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    for name, arm in ORDER:
        target = output_dir / f"{name}.json"
        if target.exists():
            continue
        if arm == "koff_off":
            env = matrix._boot_child_environment(
                d3.boot_environment({**CELL, "action": "off"}, output_dir / "t.jsonl",
                                    "corrected")
            )
            # Measure the STACK, not the instrument.
            env["VLLM_SELF_SPEC_PROFILE"] = "0"
            env.pop("VLLM_SELF_SPEC_KOFF_TRACE", None)
        else:
            env = dict(os.environ)
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(lane["physical_gpu_index"]),
                    "VLLM_CACHE_ROOT": lane["cache_root"],
                    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                    "VLLM_DISABLED_KERNELS": "MacheteLinearKernel",
                    "TOKENIZERS_PARALLELISM": "false",
                }
            )
            # Nothing self-spec: the point of the arm.
            for key in [k for k in env if k.startswith("VLLM_SELF_SPEC_")]:
                env.pop(key)
        print(f"[x32] {name} ({arm}) ...", flush=True)
        log = output_dir / f"{name}.log"
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--measure",
                    "--arm",
                    arm,
                    "--out",
                    str(target),
                ],
                env=env,
                cwd=str(REPO),
                stdout=handle,
                stderr=subprocess.STDOUT,
                preexec_fn=lambda: os.sched_setaffinity(0, affinity),
            )
        if completed.returncode != 0 or not target.exists():
            raise RuntimeError(f"{name} failed ({completed.returncode}); see {log}")
        print(f"[x32] ok {name}", flush=True)


def summarise(output_dir: Path) -> dict[str, Any]:
    arms: dict[str, list[dict[str, Any]]] = {}
    for name, arm in ORDER:
        path = output_dir / f"{name}.json"
        if path.exists():
            arms.setdefault(arm, []).append(json.loads(path.read_text()))

    rows: dict[str, Any] = {}
    for regime in REGIMES:
        entry: dict[str, Any] = {}
        for arm, records in arms.items():
            values = [
                r["observations"][regime]["decode_tokens_per_s"]
                for r in records
                if regime in r["observations"]
                and r["observations"][regime]["decode_tokens_per_s"]
            ]
            if values:
                entry[arm] = round(statistics.mean(values), 4)
                entry[f"{arm}_runs"] = values
        if "stock" in entry and "koff_off" in entry:
            entry["koff_over_stock"] = round(entry["koff_off"] / entry["stock"], 6)
            entry["cost_pct"] = round((entry["stock"] / entry["koff_off"] - 1) * 100, 3)
        rows[regime] = entry
    return {
        "record_type": "w98_offpath_summary",
        "schema_version": 1,
        "note": (
            "decode rate from the two-budget slope, so prefill cancels; "
            "profiler and trace disabled in both arms"
        ),
        "regimes": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--arm")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=PHASE / "data/probe_offpath_stock"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.measure:
        measure(args.out, args.arm)
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_all(args.output_dir)
    summary = summarise(args.output_dir)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(f"\n{'regime':8s} {'stock tok/s':>12s} {'koff-OFF tok/s':>15s} "
          f"{'ratio':>8s} {'our cost':>9s}")
    for regime, row in summary["regimes"].items():
        if "koff_over_stock" not in row:
            continue
        print(
            f"{regime:8s} {row['stock']:12.2f} {row['koff_off']:15.2f} "
            f"{row['koff_over_stock']:8.4f} {row['cost_pct']:+8.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
