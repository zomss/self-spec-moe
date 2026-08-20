# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where the engine tax goes: our runtime parked against plain vLLM.

DIAGNOSTIC, NOT SCORED.

`off` is our runtime with the K schedule pinned to 0 -- the draft is resident
and the self-spec path is live, but no token is ever drafted. It should
therefore do exactly the target's work, and it does not: it measures 15% below
stock at LO and **37% at SS b32**, cell-dependent and worst where steps are
cheapest.

Two candidate causes are already eliminated by measurements this phase owns:

* **HBM residency.** `off` is within 0.20-0.54% of itself whether the draft is
  a separate 6.1 GB resident (`w4a16-quantized`) or shares the target's own
  tensors (`target-matching`, `SHARE_WEIGHTS=1`). A cost that does not move
  when 6.1 GB appears is not a memory cost.
* **The instrument.** Section 32 removed the profiler and koff trace and the
  tax fell from 17-22% to 5.6-9.4% at LI/LIO. What remains reproduces on
  independent hardware -- Campaign 1 measured 0.933/0.925 on h103 against our
  0.915/0.930 -- so the residual is our implementation, not this box.

So the residual is per-step HOST work, and it is structured: about **1.2 ms
fixed plus ~0.25 ms per request** per step, which is why it dominates at SS
b32 (cheap steps, 32 requests) and barely registers at LO b1.

This measures where it goes. Both arms run the same prompts for the same
number of steps under `torch.profiler` with CPU and CUDA activities, and the
comparison is:

    device time    must be IDENTICAL -- K=0 means the same forward passes
    host API       launches, syncs, memcpies: what our path adds
    idle           wall minus device: the gap the tax actually lives in

A difference in device time would mean `off` is not doing the target's work,
which is a correctness question rather than an overhead one.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

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
MAX_MODEL_LEN = 40_960
CELL = os.environ.get("W98_TAX_CELL", "SS")
BATCH = int(os.environ.get("W98_TAX_BATCH", "32"))
WARMUP_STEPS = int(os.environ.get("W98_TAX_WARMUP", "30"))
PROFILED_STEPS = int(os.environ.get("W98_TAX_STEPS", "40"))

ARMS = [
    ("stock", {"action": "stock"}),
    (
        "off",
        {"action": "off", "quant": "w4a16-quantized", "window": "off", "skip_count": 0},
    ),
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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


def profile_arm(name: str, cfg: dict[str, Any], out_json: Path) -> None:
    import time

    import torch
    from torch.profiler import ProfilerActivity, profile

    from vllm import LLM, LLMEngine, SamplingParams  # noqa: F401

    if cfg.get("action") == "stock":
        from vllm import EngineArgs

        args = EngineArgs(
            model=r1._engine_args({"quant": "target-matching"}).model,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=BATCH,
            max_num_batched_tokens=8192,
            enable_chunked_prefill=True,
            gpu_memory_utilization=0.90,
            enable_prefix_caching=False,
            enforce_eager=False,
            seed=0,
            disable_log_stats=True,
        )
    else:
        args = d3._engine_args(cfg)
        args.max_model_len = MAX_MODEL_LEN
    args.max_num_seqs = max(args.max_num_seqs or 0, BATCH)
    engine = LLMEngine.from_engine_args(args)
    try:
        for index, tokens in enumerate(prompts(BATCH)):
            engine.add_request(
                f"{CELL}-{index}",
                {"prompt_token_ids": tokens},
                SamplingParams(temperature=0.0, max_tokens=4096, ignore_eos=True),
            )
        for _ in range(WARMUP_STEPS):
            _require(engine.has_unfinished_requests(), "ran out of work in warmup")
            engine.step()
        torch.accelerator.synchronize()
        t0 = time.perf_counter()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
        ) as prof:
            for _ in range(PROFILED_STEPS):
                _require(engine.has_unfinished_requests(), "ran out of work")
                engine.step()
            torch.accelerator.synchronize()
        wall = time.perf_counter() - t0
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()

    averages = prof.key_averages()
    device: dict[str, dict[str, Any]] = {}
    host: dict[str, dict[str, Any]] = {}
    for event in averages:
        dev = 0.0
        for attr in ("self_device_time_total", "self_cuda_time_total"):
            value = getattr(event, attr, None)
            if value:
                dev = float(value)
                break
        cpu = float(getattr(event, "self_cpu_time_total", 0) or 0)
        if dev > 0:
            device[event.key] = {
                "calls": int(event.count),
                "us_total": dev,
                "us_per_step": dev / PROFILED_STEPS,
            }
        if cpu > 0:
            host[event.key] = {
                "calls": int(event.count),
                "us_total": cpu,
                "us_per_step": cpu / PROFILED_STEPS,
                "calls_per_step": event.count / PROFILED_STEPS,
            }
    device_us = sum(v["us_total"] for v in device.values())
    out_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_engine_tax",
                "arm": name,
                "config": cfg,
                "content_cell": CELL,
                "batch": BATCH,
                "profiled_steps": PROFILED_STEPS,
                "wall_ms_per_step": wall * 1000 / PROFILED_STEPS,
                "device_ms_per_step": device_us / 1000.0 / PROFILED_STEPS,
                "idle_ms_per_step": (wall * 1000 - device_us / 1000.0) / PROFILED_STEPS,
                "host_cpu_ms_per_step": sum(v["us_total"] for v in host.values())
                / 1000.0
                / PROFILED_STEPS,
                "device_kernels": device,
                "host_events": host,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_all(output_dir: Path, gpu: int) -> None:
    lane = g98g.bind_box(gpu)
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = sorted(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    traces = output_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    for name, cfg in ARMS:
        target = output_dir / f"{name}.json"
        if target.exists():
            continue
        trace = traces / f"{name}.jsonl"
        base = d3.boot_environment(
            {
                **cfg,
                "quant": cfg.get("quant", "target-matching"),
                "window": cfg.get("window", "off"),
                "skip_count": cfg.get("skip_count", 0),
            },
            trace,
            "corrected",
        )
        if cfg.get("action") == "stock":
            env = {k: v for k, v in base.items() if not k.startswith("VLLM_SELF_SPEC")}
        else:
            env = dict(base)
            env["VLLM_SELF_SPEC_PROFILE"] = "0"
            env.pop("VLLM_SELF_SPEC_KOFF_TRACE", None)
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
                    json.dumps({"name": name, "cfg": cfg}),
                ],
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                preexec_fn=lambda: os.sched_setaffinity(0, affinity),
            )
        if completed.returncode != 0 or not target.exists():
            (output_dir / f"{name}.FAILED").write_text(
                log.read_text(encoding="utf-8", errors="replace")[-4000:],
                encoding="utf-8",
            )
            print(f"[tax] FAILED {name}", flush=True)
            continue
        rec = json.loads(target.read_text())
        print(
            f"[tax] {name:6s} wall {rec['wall_ms_per_step']:7.3f}  "
            f"device {rec['device_ms_per_step']:7.3f}  "
            f"idle {rec['idle_ms_per_step']:7.3f}  "
            f"hostcpu {rec['host_cpu_ms_per_step']:7.3f} ms/step",
            flush=True,
        )


def summarise(output_dir: Path) -> dict[str, Any]:
    recs = {}
    for name, _ in ARMS:
        p = output_dir / f"{name}.json"
        if p.is_file():
            recs[name] = json.loads(p.read_text())
    if len(recs) < 2:
        return {"record_type": "w98_engine_tax_summary", "arms": list(recs)}
    s, o = recs["stock"], recs["off"]
    deltas = []
    keys = set(s["host_events"]) | set(o["host_events"])
    for k in keys:
        a = s["host_events"].get(k, {}).get("us_per_step", 0.0)
        b = o["host_events"].get(k, {}).get("us_per_step", 0.0)
        if abs(b - a) > 20.0:
            deltas.append(
                {
                    "event": k,
                    "stock_us_per_step": round(a, 1),
                    "off_us_per_step": round(b, 1),
                    "delta_us_per_step": round(b - a, 1),
                    "stock_calls_per_step": round(
                        s["host_events"].get(k, {}).get("calls_per_step", 0.0), 2
                    ),
                    "off_calls_per_step": round(
                        o["host_events"].get(k, {}).get("calls_per_step", 0.0), 2
                    ),
                }
            )
    deltas.sort(key=lambda d: -abs(d["delta_us_per_step"]))
    return {
        "schema_version": 1,
        "record_type": "w98_engine_tax_summary",
        "content_cell": CELL,
        "batch": BATCH,
        "per_step_ms": {
            n: {
                "wall": round(r["wall_ms_per_step"], 4),
                "device": round(r["device_ms_per_step"], 4),
                "idle": round(r["idle_ms_per_step"], 4),
                "host_cpu": round(r["host_cpu_ms_per_step"], 4),
            }
            for n, r in recs.items()
        },
        "tax_ms_per_step": round(o["wall_ms_per_step"] - s["wall_ms_per_step"], 4),
        "device_identical": abs(
            o["device_ms_per_step"] / max(s["device_ms_per_step"], 1e-9) - 1.0
        )
        < 0.05,
        "top_host_deltas": deltas[:25],
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
        spec = json.loads(args.arm)
        output_dir.mkdir(parents=True, exist_ok=True)
        profile_arm(spec["name"], spec["cfg"], output_dir / f"{spec['name']}.json")
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarise_only:
        run_all(output_dir, args.gpu)
    summary = summarise(output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
