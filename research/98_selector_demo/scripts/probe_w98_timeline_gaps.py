# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X3 -- localise the GPU idle gaps in the quantized x skip8 draft chain.

DIAGNOSTIC, NOT SCORED.

X2 established what this is NOT. At skip8 the W4A16 path is taken, the same
kernels run at the same per-call speed, total device work goes DOWN 6.5%, and
the host issues FEWER launches (117 graph launches/step vs 133, 385 kernel
launches vs 417 -- identical to the bf16 arm). Yet the profiler-free step time
rises from 29.03 ms to 33.97-36.58 ms.

Less host work and less device work but more wall time can only be idle. It
cannot be quantified by dividing a device total by a wall time, though:
`self_device_time_total` sums per-kernel time across streams, so it exceeds
wall wherever compute and memcpy overlap (it reports 32.52 ms of "device" for a
31.61 ms bf16 step). Occupancy needs ONE timeline from ONE instrument, which is
what this probe builds.

Method: export a chrome trace, sort device kernels by timestamp, find the idle
intervals between consecutive kernels, and attribute each gap to the host event
that spans it. Reported as a gap histogram plus the top blocking host calls, so
the answer is "N gaps of M us each, sitting under <this call>" rather than an
aggregate.

Kept to a few steps: the question is where the gaps are, not their mean.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402

matrix = r1.matrix

ARMS = [
    ("quant_skip4", "w4a16-quantized", "2,4,7,16"),
    ("quant_skip8", "w4a16-quantized", "2,4,7,11,16,20,25,30"),
]

WINDOW = 256
REGIME = "R1"
WARMUP_STEPS = 40
PROFILED_STEPS = 6
# Below this a gap is ordinary launch latency, not a stall worth attributing.
GAP_US = 20.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def profile_arm(quant: str, skip: str, trace: Path, out_json: Path) -> None:
    """Boot one arm, export a chrome trace, and attribute device idle gaps."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    from vllm import LLMEngine, SamplingParams

    cfg = {
        "quant": quant,
        "window": WINDOW,
        "skip_count": len([t for t in skip.split(",") if t.strip()]),
    }
    manifest = r1._load_json(matrix._repository_path(r1.PROMPT_MANIFEST))
    spec = {r["regime_id"]: r for r in manifest["prompt_plan"]["regimes"]}[REGIME]
    prompts = r1._prompts_for(REGIME, spec["batch"])
    engine = LLMEngine.from_engine_args(r1._engine_args(cfg))
    chrome = out_json.with_suffix(".chrome.json")
    try:
        for index, tokens in enumerate(prompts):
            engine.add_request(
                f"{REGIME}-{index}",
                {"prompt_token_ids": tokens},
                SamplingParams(
                    temperature=0.0, max_tokens=r1.MEASURE_TOKENS, ignore_eos=True
                ),
            )
        for _ in range(WARMUP_STEPS):
            _require(engine.has_unfinished_requests(), "ran out of work in warmup")
            engine.step()
        torch.accelerator.synchronize()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
        ) as prof:
            for _ in range(PROFILED_STEPS):
                _require(engine.has_unfinished_requests(), "ran out of work profiling")
                engine.step()
            torch.accelerator.synchronize()
        prof.export_chrome_trace(str(chrome))
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()

    analysis = analyse_chrome(chrome)
    analysis.update(
        {
            "schema_version": 1,
            "record_type": "w98_timeline_gaps",
            "quant": quant,
            "skip_layers": skip,
            "skip_count": cfg["skip_count"],
            "window": WINDOW,
            "profiled_steps": PROFILED_STEPS,
        }
    )
    out_json.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # The chrome trace is large and reproducible; keep the analysis, drop it.
    with contextlib.suppress(OSError):
        chrome.unlink()


def analyse_chrome(chrome: Path) -> dict[str, Any]:
    """Find device idle gaps and attribute each to the host call spanning it."""
    with chrome.open(encoding="utf-8") as handle:
        doc = json.load(handle)
    events = doc["traceEvents"] if isinstance(doc, dict) else doc

    kernels, host = [], []
    for e in events:
        if e.get("ph") != "X" or "ts" not in e or "dur" not in e:
            continue
        cat = (e.get("cat") or "").lower()
        if cat in {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"}:
            kernels.append((float(e["ts"]), float(e["dur"]), e.get("name", "")))
        elif cat in {"cuda_runtime", "cuda_driver", "runtime"}:
            host.append((float(e["ts"]), float(e["dur"]), e.get("name", "")))
    _require(bool(kernels), "no device events in the chrome trace")
    kernels.sort()
    host.sort()

    span = kernels[-1][0] + kernels[-1][1] - kernels[0][0]
    busy = sum(d for _, d, _ in kernels)

    gaps = []
    prev_end, prev_name = kernels[0][0] + kernels[0][1], kernels[0][2]
    for ts, dur, name in kernels[1:]:
        if ts - prev_end >= GAP_US:
            gaps.append((prev_end, ts - prev_end, prev_name, name))
        prev_end = max(prev_end, ts + dur)
        prev_name = name

    # Attribute each gap to the innermost host call covering its midpoint.
    by_host: dict[str, dict[str, float]] = defaultdict(
        lambda: {"gaps": 0, "idle_us": 0.0}
    )
    for start, dur, _, _ in gaps:
        mid = start + dur / 2
        best = None
        for hts, hdur, hname in host:
            if hts > mid:
                break
            if hts <= mid <= hts + hdur and (best is None or hdur < best[0]):
                best = (hdur, hname)
        label = best[1] if best else "<no host call spans the gap>"
        by_host[label]["gaps"] += 1
        by_host[label]["idle_us"] += dur

    buckets = {"20-50us": 0, "50-200us": 0, "200us-1ms": 0, ">1ms": 0}
    for _, dur, _, _ in gaps:
        key = (
            "20-50us"
            if dur < 50
            else "50-200us"
            if dur < 200
            else "200us-1ms"
            if dur < 1000
            else ">1ms"
        )
        buckets[key] += 1

    top_pairs: dict[str, dict[str, float]] = defaultdict(
        lambda: {"gaps": 0, "idle_us": 0.0}
    )
    for _, dur, before, after in gaps:
        key = f"{before[:48]} -> {after[:48]}"
        top_pairs[key]["gaps"] += 1
        top_pairs[key]["idle_us"] += dur

    return {
        "device_span_us": span,
        "device_busy_us": busy,
        "device_idle_us": span - busy,
        "occupancy": busy / span if span else None,
        "gap_threshold_us": GAP_US,
        "gap_count": len(gaps),
        "gap_idle_us": sum(d for _, d, _, _ in gaps),
        "gap_size_histogram": buckets,
        "idle_by_host_call": dict(
            sorted(by_host.items(), key=lambda kv: -kv[1]["idle_us"])[:12]
        ),
        "idle_by_kernel_boundary": dict(
            sorted(top_pairs.items(), key=lambda kv: -kv[1]["idle_us"])[:12]
        ),
    }


def run_arms(output_dir: Path) -> None:
    lane = matrix.lane_for_block(1)
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = sorted(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    traces = output_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    for name, quant, skip in ARMS:
        target = output_dir / f"{name}.json"
        if target.exists():
            continue
        trace = traces / f"{name}.jsonl"
        _require(not trace.exists(), f"trace {trace} already exists")
        cfg = {
            "quant": quant,
            "window": WINDOW,
            "skip_count": len([t for t in skip.split(",") if t.strip()]),
        }
        env = r1.boot_environment(cfg, trace)
        env["VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS"] = skip
        env["VLLM_SELF_SPEC_PROFILE"] = "0"
        env = matrix._boot_child_environment(env)
        log = output_dir / f"{name}.log"
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output-dir",
                    str(output_dir),
                    "--arm",
                    json.dumps({"name": name, "quant": quant, "skip": skip}),
                    "--trace",
                    str(trace),
                ],
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                preexec_fn=lambda: os.sched_setaffinity(0, affinity),
            )
        _require(
            completed.returncode == 0 and target.exists(),
            f"arm {name} failed; output preserved at {log}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", help=argparse.SUPPRESS)
    parser.add_argument("--trace", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.arm:
        arm = json.loads(args.arm)
        profile_arm(
            arm["quant"],
            arm["skip"],
            Path(args.trace).resolve(),
            output_dir / f"{arm['name']}.json",
        )
        return 0
    base_env = matrix._boot_child_environment({})
    matrix._preflight_native_sampler(base_env)
    matrix._preflight_inprocess_engine_core(base_env)
    lane = matrix.lane_for_block(1)
    matrix._preflight_gpu_identity_and_idle(
        lane["physical_gpu_index"], lane["physical_gpu_uuid"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_arms(output_dir)
    print(json.dumps({"arms": [a[0] for a in ARMS]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
