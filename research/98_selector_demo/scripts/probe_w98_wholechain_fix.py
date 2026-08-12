# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X6 -- does whole-chain CUDA graph capture recover the quant x skip8 penalty?

DIAGNOSTIC, NOT SCORED. This is a candidate FIX test, not a new measurement of
the defect.

The chain of evidence points at one remedy. X2: nothing is slower -- same
kernels, same per-call speed, less device work. X3: the whole penalty is GPU
idle (+5.34 ms), 95% of it outside any CUDA call, concentrated in ~90 extra
50-200 us host gaps against ~116 piecewise graph replays per step. X4: the
growth sits in the eager attention / KV-cache-update frames that run BETWEEN
those replays, whose Python is too trivial to be doing real work -- they stall.

`unified_attention_with_output` is the piecewise split op, so every attention
layer forces a return to Python between graph replays. The draft chain therefore
pays ~116 host round-trips per step. Phase 82 already built the alternative --
`_wholechain` captures the entire K-step chain as ONE graph -- but
`VLLM_SELF_SPEC_DRAFT_WHOLECHAIN` defaults to 0 and no Round-1 or probe boot set
it. `wc_replay` never firing in any arm is the direct confirmation it was off.

Prediction, fixed before the data: if the mechanism is host round-trips, whole
chain removes most of the skip8 penalty and helps skip4 much less, because
skip4 was not starving (its bubble was 1/3 the size).

Design: 2 skip settings x whole-chain off/on, interleaved, no profiler, plain
wall time. Arming is VERIFIED per boot from the log rather than assumed -- the
arm conditions include all_greedy and a batch-size allowlist, so a silent
non-arm would look exactly like "the fix did not work".
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

matrix = r1.matrix

SKIPS = {
    "skip4": "2,4,7,16",
    "skip8": "2,4,7,11,16,20,25,30",
}
# _wholechain requires BOTH flags: WHOLECHAIN alone is inert. FULLCG swaps the
# paged FA3 decode for a masked SDPA the graph can capture -- without it the
# attention split points remain and no whole-chain graph is possible. FULLCG
# itself requires KV_WINDOW > 0, which every cell here has (window 256).
WHOLECHAIN = {
    "wcoff": {
        "VLLM_SELF_SPEC_DRAFT_WHOLECHAIN": "0",
        "VLLM_SELF_SPEC_DRAFT_FULLCG": "0",
    },
    "wcon": {
        "VLLM_SELF_SPEC_DRAFT_WHOLECHAIN": "1",
        "VLLM_SELF_SPEC_DRAFT_FULLCG": "1",
    },
}
REPEATS = 2
ORDER = [
    (rep, skip, wc) for rep in range(REPEATS) for skip in SKIPS for wc in WHOLECHAIN
]

QUANT = "w4a16-quantized"
WINDOW = 256
REGIME = "R1"
WARMUP_STEPS = 40
MEASURED_STEPS = 60

# Emitted by llm_base_proposer when a whole-chain graph is captured.
ARMED_MARKER = "Whole-chain graph captured"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def measure(quant: str, skip: str, trace: Path, out_json: Path) -> None:
    """Time engine steps with no profiler of any kind."""
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
    per_step: list[float] = []
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
                "record_type": "w98_wholechain_fix",
                "quant": quant,
                "skip_layers": skip,
                "skip_count": cfg["skip_count"],
                "window": WINDOW,
                "regime": REGIME,
                "measured_steps": len(per_step),
                "mean_ms": statistics.mean(per_step) * 1000,
                "median_ms": statistics.median(per_step) * 1000,
                "stdev_ms": statistics.stdev(per_step) * 1000,
                "min_ms": min(per_step) * 1000,
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
    for rep, skip_name, wc_name in ORDER:
        name = f"r{rep}_{skip_name}_{wc_name}"
        target = output_dir / f"{name}.json"
        if target.exists():
            continue
        skip = SKIPS[skip_name]
        trace = traces / f"{name}.jsonl"
        _require(not trace.exists(), f"trace {trace} already exists")
        cfg = {
            "quant": QUANT,
            "window": WINDOW,
            "skip_count": len([t for t in skip.split(",") if t.strip()]),
        }
        env = r1.boot_environment(cfg, trace)
        env["VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS"] = skip
        env["VLLM_SELF_SPEC_PROFILE"] = "0"
        env.update(WHOLECHAIN[wc_name])
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
                    json.dumps({"name": name, "quant": QUANT, "skip": skip}),
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
        # Verify arming from the log rather than trusting the env var: a silent
        # non-arm is indistinguishable from a fix that did not help.
        armed = ARMED_MARKER in log.read_text(encoding="utf-8", errors="replace")
        record = r1._load_json(target)
        record["wholechain_requested"] = wc_name == "wcon"
        record["wholechain_captured"] = armed
        target.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def summarize(output_dir: Path) -> dict[str, Any]:
    rows = {}
    for rep, skip_name, wc_name in ORDER:
        path = output_dir / f"r{rep}_{skip_name}_{wc_name}.json"
        if path.exists():
            rows[f"r{rep}_{skip_name}_{wc_name}"] = r1._load_json(path)
    return {"schema_version": 1, "record_type": "w98_wholechain_summary", "runs": rows}


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
    run_all(output_dir)
    (output_dir / "wholechain_summary.json").write_text(
        json.dumps(summarize(output_dir), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"runs": [f"r{r}_{s}_{w}" for r, s, w in ORDER]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
