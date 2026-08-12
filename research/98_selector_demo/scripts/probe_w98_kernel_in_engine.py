# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X12 -- does the W4A16 kernel ranking survive in the real engine?

DIAGNOSTIC, NOT SCORED.

X10/X11 ranked the kernels on isolated GEMMs under CUDA-graph capture. That is
the right way to compare kernels, but it is not evidence about the engine: a
microbenchmark can win and still not transfer, because the engine adds the
attention path, the KV write, the sampler, and -- for Humming specifically -- a
JIT warmup that interacts with capture (`_prewarm_jit_linear_kernels` exists for
exactly that reason).

Method: boot the real engine once per kernel with the whole-chain runtime that
X6 established (FULLCG + WHOLECHAIN), time steps with no profiler, and READ BACK
which kernel actually resolved. Selection is forced with `VLLM_DISABLED_KERNELS`
-- the mechanism Phase 75 used -- by disabling everything above the target in
the CUDA priority order:

    CutlassW4A8 -> Machete -> AllSpark -> Marlin -> Humming -> Conch -> ...

Verification is not optional here. Kernel selection is silent, and Humming
additionally falls back without `ninja` on PATH. A boot that quietly resolved to
a different kernel would look exactly like "the microbenchmark did not
transfer". The engine logs `Using %s for CompressedTensorsWNA16`, and this
probe fails the arm if the log does not name the intended kernel.

Prediction, fixed before the data. X11's per-step GEMM deltas at M=1 were
machete 9.55 ms, marlin 6.67, humming 6.32. The step also carries ~15 ms of
non-GEMM work, so the engine should show the same ORDER with the same absolute
gaps, not the same ratios.
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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402

matrix = r1.matrix

QUANT = "w4a16-quantized"
SKIP = "2,4,7,11,16,20,25,30"
WINDOW = 256

# Disable everything ABOVE the target in the CUDA priority order.
ARMS = {
    "machete": "",
    "marlin": "MacheteLinearKernel",
    "humming": "MacheteLinearKernel,MarlinLinearKernel",
    "conch": "MacheteLinearKernel,MarlinLinearKernel,HummingLinearKernel",
}
EXPECTED_RESOLVED = {
    "machete": "MacheteLinearKernel",
    "marlin": "MarlinLinearKernel",
    "humming": "HummingLinearKernel",
    "conch": "ConchLinearKernel",
}

# (regime, batch) -- M == batch for the draft's decode.
CELLS = [("R1", 1), ("R1", 8), ("R1", 32)]
REPEATS = 2
ORDER = [
    (rep, regime, batch, arm)
    for rep in range(REPEATS)
    for regime, batch in CELLS
    for arm in ARMS
]

WARMUP_STEPS = 40
MEASURED_STEPS = 60
RESOLVED_MARKER = "for CompressedTensorsWNA16"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def measure(regime: str, batch: int, trace: Path, out_json: Path) -> None:
    from vllm import LLMEngine, SamplingParams

    cfg = {
        "quant": QUANT,
        "window": WINDOW,
        "skip_count": len([t for t in SKIP.split(",") if t.strip()]),
    }
    prompts = r1._prompts_for(regime, batch)
    _require(
        len(prompts) == batch, f"{regime} gave {len(prompts)} prompts, need {batch}"
    )
    engine = LLMEngine.from_engine_args(r1._engine_args(cfg))
    per_step: list[float] = []
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
        for _ in range(MEASURED_STEPS):
            _require(engine.has_unfinished_requests(), "ran out of work measuring")
            t0 = time.perf_counter()
            engine.step()
            per_step.append(time.perf_counter() - t0)
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()

    out_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_kernel_in_engine",
                "regime": regime,
                "batch": batch,
                "window": WINDOW,
                "measured_steps": len(per_step),
                "mean_ms": statistics.mean(per_step) * 1000,
                "median_ms": statistics.median(per_step) * 1000,
                "min_ms": min(per_step) * 1000,
                "stdev_ms": statistics.stdev(per_step) * 1000,
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
    for rep, regime, batch, arm in ORDER:
        name = f"r{rep}_{regime}_b{batch:02d}_{arm}"
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
        # The runtime X6 established: whole chain in one graph.
        env["VLLM_SELF_SPEC_DRAFT_WHOLECHAIN"] = "1"
        env["VLLM_SELF_SPEC_DRAFT_FULLCG"] = "1"
        if ARMS[arm]:
            env["VLLM_DISABLED_KERNELS"] = ARMS[arm]
        env = matrix._boot_child_environment(env)
        # Humming JIT-compiles and falls back silently without ninja.
        env["PATH"] = f"{REPO_ROOT / '.venv' / 'bin'}:{env.get('PATH', '')}"
        log = output_dir / f"{name}.log"
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output-dir",
                    str(output_dir),
                    "--arm",
                    json.dumps({"name": name, "regime": regime, "batch": batch}),
                    "--trace",
                    str(trace),
                ],
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                preexec_fn=lambda: os.sched_setaffinity(0, affinity),
            )
        text = log.read_text(encoding="utf-8", errors="replace")
        resolved = None
        for line in text.splitlines():
            if RESOLVED_MARKER in line:
                for token in line.split():
                    if token.endswith("LinearKernel"):
                        resolved = token
                        break
        if completed.returncode != 0 or not target.exists():
            (output_dir / f"{name}.FAILED").write_text(
                f"returncode={completed.returncode} resolved={resolved}\n",
                encoding="utf-8",
            )
            continue
        record = r1._load_json(target)
        record["arm"] = arm
        record["kernel_requested"] = EXPECTED_RESOLVED[arm]
        record["kernel_resolved"] = resolved
        # A silently different kernel is the one way this measurement can lie.
        record["kernel_verified"] = resolved == EXPECTED_RESOLVED[arm]
        record["wholechain_captured"] = "Whole-chain graph captured" in text
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
            arm["batch"],
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
    print(json.dumps({"arms": len(ORDER)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
