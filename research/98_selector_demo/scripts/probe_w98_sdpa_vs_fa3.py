# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X7 -- what does FULLCG's masked SDPA cost, and where does paged FA3 win back?

DIAGNOSTIC, NOT SCORED.

X6 showed FULLCG + WHOLECHAIN takes quant x skip8 from 38.4 to 25.2 ms/step.
That is a claim about WALL time, and it does not imply the SDPA attention kernel
is faster than the paged FA3 decode it replaces. The opposite is expected: FA3
decode is heavily tuned, while FULLCG materialises a sinks+window scratchpad and
runs a masked SDPA over it. The plausible reading is that FULLCG buys
CAPTURABILITY -- one graph replay instead of ~116 host round-trips -- and PAYS
for it in device work.

If that reading is right, two things must be true, and neither is established:

  1. Device time should go UP under FULLCG even as wall time goes down.
  2. There must be a CROSSOVER. The host-round-trip saving is a fixed ~13 ms
     per step; the extra device work grows with the attention workload. Once
     the chain is GPU-bound rather than host-bound, the faster kernel wins and
     FULLCG should LOSE.

Crossover matters more than the headline number: it decides whether FULLCG is
a global default or a regime-dependent lever the selector must choose, which
in turn decides whether the cost model needs a term for it.

Method, per boot: time 60 steps with NO profiler (the honest wall), then profile
15 steps for the kernel mix (device time is trustworthy under the profiler even
though wall is not). Both halves from one boot, so they describe the same
configuration.

Regimes R1 (431 tokens) and R5 (14,357 tokens) bracket the attention workload by
33x, which is where a crossover would show if the window did not already cap the
key set. Both are windowed at 256, so FULLCG's scratchpad is ~272 keys in both
-- meaning any crossover comes from the batch/prefill side, not the window.
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
from collections import defaultdict
from pathlib import Path
from typing import Any

import regex as re

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402

matrix = r1.matrix

QUANT = "w4a16-quantized"
SKIP = "2,4,7,11,16,20,25,30"
WINDOW = 256
REGIMES = ["R1", "R5"]
MODES = {
    "piecewise": {
        "VLLM_SELF_SPEC_DRAFT_WHOLECHAIN": "0",
        "VLLM_SELF_SPEC_DRAFT_FULLCG": "0",
    },
    "wholechain": {
        "VLLM_SELF_SPEC_DRAFT_WHOLECHAIN": "1",
        "VLLM_SELF_SPEC_DRAFT_FULLCG": "1",
    },
}
ORDER = [(regime, mode) for regime in REGIMES for mode in MODES]

WARMUP_STEPS = 40
TIMED_STEPS = 60
PROFILED_STEPS = 15

ARMED_MARKER = "Whole-chain graph captured"

FAMILIES = [
    ("flash_attn", re.compile(r"flash|fa3|fmha", re.I)),
    ("sdpa_attn", re.compile(r"sdpa|scaled_dot|attn_fwd|masked", re.I)),
    ("paged", re.compile(r"paged|block_table", re.I)),
    ("machete", re.compile(r"machete", re.I)),
    ("gemm", re.compile(r"cutlass|cublas|gemm", re.I)),
    ("scratchpad_copy", re.compile(r"gather|index_select|copy|cat_|memcpy", re.I)),
    ("norm_act", re.compile(r"rms_?norm|silu|swiglu|rotary", re.I)),
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _family(name: str) -> str:
    for label, pattern in FAMILIES:
        if pattern.search(name):
            return label
    return "other"


def measure(regime: str, trace: Path, out_json: Path) -> None:
    """Time steps without a profiler, then profile a few for the kernel mix."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    from vllm import LLMEngine, SamplingParams

    cfg = {
        "quant": QUANT,
        "window": WINDOW,
        "skip_count": len([t for t in SKIP.split(",") if t.strip()]),
    }
    manifest = r1._load_json(matrix._repository_path(r1.PROMPT_MANIFEST))
    spec = {r["regime_id"]: r for r in manifest["prompt_plan"]["regimes"]}[regime]
    prompts = r1._prompts_for(regime, spec["batch"])
    engine = LLMEngine.from_engine_args(r1._engine_args(cfg))
    per_step: list[float] = []
    kernels: dict[str, dict[str, Any]] = {}
    try:
        for index, tokens in enumerate(prompts):
            engine.add_request(
                f"{regime}-{index}",
                {"prompt_token_ids": tokens},
                SamplingParams(
                    temperature=0.0, max_tokens=r1.MEASURE_TOKENS, ignore_eos=True
                ),
            )
        for _ in range(WARMUP_STEPS):
            _require(engine.has_unfinished_requests(), "ran out of work in warmup")
            engine.step()
        # Half 1: the honest wall, no profiler attached.
        for _ in range(TIMED_STEPS):
            _require(engine.has_unfinished_requests(), "ran out of work timing")
            t0 = time.perf_counter()
            engine.step()
            per_step.append(time.perf_counter() - t0)
        # Half 2: kernel mix. Device time is trustworthy under the profiler even
        # though wall is not, which is why the two halves are kept separate.
        torch.accelerator.synchronize()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
        ) as prof:
            for _ in range(PROFILED_STEPS):
                _require(engine.has_unfinished_requests(), "ran out of work profiling")
                engine.step()
            torch.accelerator.synchronize()
        for event in prof.key_averages():
            total_us = 0.0
            for attr in ("self_device_time_total", "self_cuda_time_total"):
                value = getattr(event, attr, None)
                if value:
                    total_us = float(value)
                    break
            if total_us <= 0:
                continue
            kernels[event.key] = {
                "family": _family(event.key),
                "calls": int(event.count),
                "device_us_total": total_us,
                "device_us_per_call": total_us / max(int(event.count), 1),
            }
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()

    _require(bool(kernels), "no device kernels recorded")
    by_family: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "device_us_total": 0.0}
    )
    for info in kernels.values():
        agg = by_family[info["family"]]
        agg["calls"] += info["calls"]
        agg["device_us_total"] += info["device_us_total"]

    out_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_sdpa_vs_fa3",
                "regime": regime,
                "window": WINDOW,
                "skip_layers": SKIP,
                "timed_steps": len(per_step),
                "wall_mean_ms": statistics.mean(per_step) * 1000,
                "wall_median_ms": statistics.median(per_step) * 1000,
                "wall_min_ms": min(per_step) * 1000,
                "wall_stdev_ms": statistics.stdev(per_step) * 1000,
                "profiled_steps": PROFILED_STEPS,
                "device_us_per_step": sum(
                    k["device_us_total"] for k in kernels.values()
                )
                / PROFILED_STEPS,
                "by_family": dict(by_family),
                "kernels": kernels,
                "loadavg_1min": os.getloadavg()[0],
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
    for regime, mode in ORDER:
        name = f"{regime}_{mode}"
        target = output_dir / f"{name}.json"
        if target.exists():
            continue
        trace = traces / f"{name}.jsonl"
        _require(not trace.exists(), f"trace {trace} already exists")
        cfg = {
            "quant": QUANT,
            "window": WINDOW,
            "skip_count": len([t for t in SKIP.split(",") if t.strip()]),
        }
        env = r1.boot_environment(cfg, trace)
        env["VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS"] = SKIP
        env["VLLM_SELF_SPEC_PROFILE"] = "0"
        env.update(MODES[mode])
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
                    json.dumps({"name": name, "regime": regime}),
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
        armed = ARMED_MARKER in log.read_text(encoding="utf-8", errors="replace")
        record = r1._load_json(target)
        record["wholechain_requested"] = mode == "wholechain"
        record["wholechain_captured"] = armed
        target.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
        measure(
            arm["regime"],
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
    run_all(output_dir)
    print(json.dumps({"arms": [f"{r}_{m}" for r, m in ORDER]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
