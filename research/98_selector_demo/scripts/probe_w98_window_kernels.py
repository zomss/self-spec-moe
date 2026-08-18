# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Why the window saves less under quantization -- at the kernel level.

DIAGNOSTIC, NOT SCORED.

Section 24 measured the window's saving per weight version and found it
family-dependent on IDENTICAL KV traffic: the same window over the same
shared target KV saves **4.544 +/- 0.151 ms** in the bf16 draft and
**2.823 +/- 0.165 ms** in the quantized one, a 1.61x difference at 7.7
sigma. Nothing about quantizing the draft's WEIGHTS touches the KV cache, so
no KV-bytes account can produce that.

The bandwidth arithmetic says the KV read is only partly exposed in both.
Per engine step the chain reads the KV `batch * K = 32` times (verified in
`llm_base_proposer.py:1631`: step 0 plus `num_speculative_tokens - 1`
further forwards, each at query width 1). At w256 that removes 18.78 GB per
step, which at this box's 3.35 TB/s peak would take 5.607 ms -- against
measured savings of 4.544 (81% exposed) and 2.823 ms (50% exposed).

The two chains also sit at very different distances from their own
bandwidth roofline -- 61.3% for bf16, 38.5% for w4a16 -- which is the
leading hypothesis: a step that is not bandwidth-bound gains less from
freeing bandwidth.

This splits the question the same way X2 did:

    Does the ATTENTION kernel itself cost the same in both families?

Bytes say it must -- same KV, same window, same context, same dtype.

    same attention time, different step saving -> the window's value is
        being set by what SURROUNDS attention (idle, launch, overlap), and
        `kappa_kv` is an exposure coefficient, not a bandwidth one;
    different attention time                   -> the attention kernel
        itself runs differently depending on the draft's weight version,
        which points at occupancy or scheduling rather than traffic.

Four arms: {bf16, w4a16} x {woff, w256} at keep = 1, on the LO content and
batch of the equal-work sweep, warmed to a comparable context so the window
is doing the same job it does there.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import regex as re

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402
import run_w98_g98e_d3 as d3  # noqa: E402
import run_w98_g98g_e2e as g98g  # noqa: E402

matrix = r1.matrix

PROMPTS = (
    SCRIPT_DIR.parent.parent / "100_baselines/data/registration/w100_prompts.jsonl.gz"
)
CELL = "LO"
MAX_MODEL_LEN = 40_960
BATCH = 8
# The equal-work sweep integrates a mean of 4252 KV positions; warming to a
# comparable context makes the window's job here the same as the job whose
# cost section 24 fitted.
WARMUP_TOKENS = int(os.environ.get("W98_WK_WARMUP", "4000"))
PROFILED_STEPS = int(os.environ.get("W98_WK_STEPS", "25"))

ARMS = [
    ("bf16_woff", "target-matching", "off"),
    ("bf16_w256", "target-matching", 256),
    ("q4_woff", "w4a16-quantized", "off"),
    ("q4_w256", "w4a16-quantized", 256),
]

FAMILIES = [
    ("marlin", re.compile(r"marlin", re.I)),
    ("machete", re.compile(r"machete", re.I)),
    ("attention", re.compile(r"flash|attn|paged", re.I)),
    ("cutlass_gemm", re.compile(r"cutlass.*(gemm|mma)", re.I)),
    ("cublas_gemm", re.compile(r"(cublas|gemm|sm\d+_xmma)", re.I)),
    ("dequant", re.compile(r"dequant|unpack|awq|gptq", re.I)),
    ("norm_act", re.compile(r"rms_?norm|silu|swiglu|rotary|layernorm", re.I)),
    ("elementwise", re.compile(r"elementwise|vectorized|copy|cat_", re.I)),
    ("reduce", re.compile(r"reduce|softmax|topk|argmax|sort", re.I)),
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _family(name: str) -> str:
    for label, pattern in FAMILIES:
        if pattern.search(name):
            return label
    return "unclassified"


def prompts(limit: int) -> list[list[int]]:
    out = []
    with gzip.open(PROMPTS, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("cell") == CELL:
                out.append(row["token_ids"])
            if len(out) >= limit:
                break
    _require(len(out) >= limit, f"not enough {CELL} prompts")
    return out


def profile_arm(cfg: dict[str, Any], out_json: Path) -> None:
    """Warm to a target context, then profile a fixed number of steps."""
    import time

    import torch
    from torch.profiler import ProfilerActivity, profile

    from vllm import LLMEngine, SamplingParams

    args = d3._engine_args(dict(cfg))
    args.max_model_len = MAX_MODEL_LEN
    engine = LLMEngine.from_engine_args(args)
    try:
        loaded = prompts(BATCH)
        for index, tokens in enumerate(loaded):
            engine.add_request(
                f"{CELL}-{index}",
                {"prompt_token_ids": tokens},
                SamplingParams(
                    temperature=0.0, max_tokens=WARMUP_TOKENS + 4096, ignore_eos=True
                ),
            )
        generated: dict[str, int] = {}
        warmup_steps = 0
        while engine.has_unfinished_requests():
            for request_output in engine.step():
                generated[request_output.request_id] = len(
                    request_output.outputs[0].token_ids
                )
            warmup_steps += 1
            if generated and min(generated.values()) >= WARMUP_TOKENS:
                break
        _require(bool(generated), "no tokens generated during warmup")
        reached = min(generated.values())
        context = len(loaded[0]) + reached
        torch.accelerator.synchronize()
        wall_t0 = time.perf_counter()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
        ) as prof:
            for _ in range(PROFILED_STEPS):
                _require(engine.has_unfinished_requests(), "ran out of work profiling")
                engine.step()
            torch.accelerator.synchronize()
        wall_s = time.perf_counter() - wall_t0
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()

    kernels: dict[str, dict[str, Any]] = {}
    averages = prof.key_averages()
    _require(bool(len(averages)), "profiler returned no events")
    for event in averages:
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
            "device_us_total": float(total_us),
            "device_us_per_call": float(total_us) / max(int(event.count), 1),
        }
    _require(bool(kernels), "no device kernels recorded")
    by_family: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "device_us_total": 0.0, "distinct_kernels": 0}
    )
    for info in kernels.values():
        agg = by_family[info["family"]]
        agg["calls"] += info["calls"]
        agg["device_us_total"] += info["device_us_total"]
        agg["distinct_kernels"] += 1
    device_us = sum(k["device_us_total"] for k in kernels.values())
    out_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_window_kernels",
                "config": cfg,
                "content_cell": CELL,
                "batch": BATCH,
                "warmup_tokens_target": WARMUP_TOKENS,
                "warmup_tokens_reached": reached,
                "warmup_steps": warmup_steps,
                "context_positions": context,
                "profiled_steps": PROFILED_STEPS,
                "wall_s_total": wall_s,
                "wall_ms_per_step": wall_s * 1000 / PROFILED_STEPS,
                "device_us_total": device_us,
                "device_ms_per_step": device_us / 1000.0 / PROFILED_STEPS,
                # What the step spends NOT running a kernel: the quantity the
                # roofline gap in section 24 is a proxy for.
                "idle_ms_per_step": (wall_s * 1000 - device_us / 1000.0)
                / PROFILED_STEPS,
                "by_family": dict(by_family),
                "kernels": kernels,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_arms(output_dir: Path, gpu: int) -> None:
    lane = g98g.bind_box(gpu)
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = sorted(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    traces = output_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    for name, quant, window in ARMS:
        target = output_dir / f"{name}.json"
        if target.exists():
            continue
        cfg = {"quant": quant, "window": window, "skip_count": 0}
        trace = traces / f"{name}.jsonl"
        if trace.exists():
            trace.unlink()
        env = d3.boot_environment({**cfg, "action": "armed"}, trace, "corrected")
        # Kernels, not regions: the region profiler syncs ~10x per step and
        # would drain the very overlap this measures.
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
                    "--gpu",
                    str(gpu),
                    "--arm",
                    json.dumps({"name": name, **cfg}),
                ],
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                preexec_fn=lambda: os.sched_setaffinity(0, affinity),
            )
        if completed.returncode != 0 or not target.exists():
            text = log.read_text(encoding="utf-8", errors="replace")
            (output_dir / f"{name}.FAILED").write_text(text[-4000:], encoding="utf-8")
            print(f"[wk] FAILED {name}", flush=True)
            continue
        record = json.loads(target.read_text())
        print(
            f"[wk] {name:10s} ctx {record['context_positions']:6d}  "
            f"wall {record['wall_ms_per_step']:7.3f} ms/step  "
            f"device {record['device_ms_per_step']:7.3f}  "
            f"idle {record['idle_ms_per_step']:7.3f}",
            flush=True,
        )


def summarise(output_dir: Path) -> dict[str, Any]:
    rows = {}
    for name, _, _ in ARMS:
        path = output_dir / f"{name}.json"
        if not path.is_file():
            continue
        record = json.loads(path.read_text())
        attention = record["by_family"].get("attention", {})
        rows[name] = {
            "context_positions": record["context_positions"],
            "wall_ms_per_step": round(record["wall_ms_per_step"], 4),
            "device_ms_per_step": round(record["device_ms_per_step"], 4),
            "idle_ms_per_step": round(record["idle_ms_per_step"], 4),
            "attention_ms_per_step": round(
                attention.get("device_us_total", 0.0)
                / 1000.0
                / record["profiled_steps"],
                4,
            ),
            "attention_calls_per_step": round(
                attention.get("calls", 0) / record["profiled_steps"], 2
            ),
            "by_family_ms_per_step": {
                k: round(v["device_us_total"] / 1000.0 / record["profiled_steps"], 4)
                for k, v in sorted(record["by_family"].items())
            },
        }
    deltas = {}
    for family, off_arm, win_arm in (
        ("target-matching", "bf16_woff", "bf16_w256"),
        ("w4a16-quantized", "q4_woff", "q4_w256"),
    ):
        if off_arm in rows and win_arm in rows:
            deltas[family] = {
                key: round(rows[off_arm][key] - rows[win_arm][key], 4)
                for key in (
                    "wall_ms_per_step",
                    "device_ms_per_step",
                    "idle_ms_per_step",
                    "attention_ms_per_step",
                )
            }
    return {
        "schema_version": 1,
        "record_type": "w98_window_kernels_summary",
        "arms": rows,
        "window_saving_off_minus_w256": deltas,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--summarise-only", action="store_true")
    parser.add_argument("--arm", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.arm:
        g98g.bind_box(args.gpu)
        arm = json.loads(args.arm)
        name = arm.pop("name")
        output_dir.mkdir(parents=True, exist_ok=True)
        profile_arm(arm, output_dir / f"{name}.json")
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarise_only:
        run_arms(output_dir, args.gpu)
    summary = summarise(output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
