#!/usr/bin/env python3
"""X31 -- per-lever Nsight breakdown: what is expected, and what is overhead.

DIAGNOSTIC, NOT SCORED.

Every cost number in this phase is an aggregate: `d_hat`, the fitted
draft-chain mean. Aggregates tell you a lever is slower than it should be but
never where the time went, which is why the window's short-context overhead
took a fitted model, a wall-clock A/B and a no-window control to pin down.
This measures it directly instead.

Two instruments make that possible:

* **NVTX ranges** (`VLLM_SELF_SPEC_PROFILE_NVTX=1`) on every profiler region.
  Unlike the wall-clock timers those need no `cuda.synchronize()`, so the
  fine-grained sub-regions -- `chain_setup`, `step0_*`, `kv_window_rewrite` --
  become measurable without the +2.4 ms/step of lost overlap that forced them
  to be gated (X25).
* **`nsys stats`** kernel summaries, which split the GPU time inside those
  ranges into attention / GEMM / elementwise, so "expected" stops being an
  assumption and becomes a measured quantity.

## The accounting

For each lever the analysis asks what the modification SHOULD cost:

* `skip N`  -- the draft runs (36-N)/36 of the layers, so its model time
  should scale by that ratio and nothing else should move.
* `window W` at context L -- only ATTENTION reads less KV, by roughly
  min(L, W + sinks)/L. GEMM and elementwise time should not move at all, and
  at short context where L <= W + sinks nothing should move.
* `w4a16`   -- the linear layers load ~4x fewer weight bytes; in a
  memory-bound decode that is a GEMM-time effect and nothing else.

Overhead is then measured-minus-expected, per range and per kernel class, and
is attributable rather than inferred.

## Reading the numbers

nsys inflates absolute time through per-API overhead, so this answers "where
does the time go" and never "how much time is there". Ratios within one trace
are the currency; cross-trace absolutes are not.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE = SCRIPT_DIR.parent
REPO = PHASE.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402
import run_w98_g98e_d3 as d3  # noqa: E402  (rebinds the lane to this host)

matrix = r1.matrix
NSYS = "nsys"

# Single levers first -- a composition cannot be decomposed until each lever's
# own cost is attributed -- then one real winner to check the parts add up.
COMBINATIONS: dict[str, dict[str, Any]] = {
    "off": {"quant": "target-matching", "window": "off", "skip_count": 0,
            "armed": False},
    "base": {"quant": "target-matching", "window": "off", "skip_count": 0,
             "armed": True},
    "quant": {"quant": "w4a16-quantized", "window": "off", "skip_count": 0,
              "armed": True},
    "w1024": {"quant": "target-matching", "window": 1024, "skip_count": 0,
              "armed": True},
    "w128": {"quant": "target-matching", "window": 128, "skip_count": 0,
             "armed": True},
    "skip4": {"quant": "target-matching", "window": "off", "skip_count": 4,
              "armed": True},
    "skip8": {"quant": "target-matching", "window": "off", "skip_count": 8,
              "armed": True},
    "composed": {"quant": "w4a16-quantized", "window": 512, "skip_count": 4,
                 "armed": True},
}

PROFILE_SKIP_STEPS = 40
PROFILE_WINDOW_STEPS = 40
LAYERS = 36

# nsys report names differ across versions; try in order and keep what works.
KERNEL_REPORTS = ("cuda_gpu_kern_sum", "cuda_kern_exec_sum")
NVTX_REPORTS = ("nvtx_gpu_proj_sum", "nvtx_sum")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def classify_kernel(name: str) -> str:
    """Bucket a kernel by what the lever should or should not move."""
    lowered = name.lower()
    if any(k in lowered for k in ("flash", "attention", "attn", "paged")):
        return "attention"
    # GEMM epilogues before the generic reduce/elementwise patterns:
    # `cublasLt::splitKreduce_kernel` is the split-K reduction of a GEMM, not
    # elementwise work, and it is ~0.5 ms/step of the bf16 draft. Marlin does
    # not use split-K, so mis-bucketing it made the quantized draft look as
    # though its elementwise work had vanished.
    if any(k in lowered for k in ("splitkreduce", "cublaslt", "cublas")):
        return "gemm"
    if any(
        k in lowered
        for k in ("gemm", "cutlass", "marlin", "gemv", "matmul", "sm90", "s16816")
    ):
        return "gemm"
    if any(k in lowered for k in ("memcpy", "memset")):
        return "memory"
    if any(k in lowered for k in ("elementwise", "vectorized", "reduce", "norm",
                                  "rope", "rotary", "silu", "cat", "copy")):
        return "elementwise"
    return "other"


def measure(trace: Path, out_json: Path, combo: str, regime_id: str) -> None:
    import torch
    from vllm import LLMEngine, SamplingParams
    from vllm.v1.spec_decode.self_spec_profiler import get_profiler

    cell = COMBINATIONS[combo]
    cfg = {k: cell[k] for k in ("quant", "window", "skip_count")}
    engine_cfg = {**cfg, "action": "armed" if cell["armed"] else "off"}
    manifest = r1._load_json(matrix._repository_path(r1.PROMPT_MANIFEST))
    regimes = {r["regime_id"]: r for r in manifest["prompt_plan"]["regimes"]}
    spec = regimes[regime_id]
    profiler = get_profiler()
    engine = LLMEngine.from_engine_args(d3._engine_args(engine_cfg))
    window: dict[str, Any] = {}
    try:
        prompts = r1._prompts_for(regime_id, spec["batch"])
        profiler.reset()
        for index, tokens in enumerate(prompts):
            engine.add_request(
                f"{regime_id}-{index}",
                {"prompt_token_ids": tokens},
                SamplingParams(
                    temperature=0.0, max_tokens=r1.MEASURE_TOKENS, ignore_eos=True
                ),
            )
        steps = 0
        started = stopped = False
        while engine.has_unfinished_requests():
            if not started and steps == PROFILE_SKIP_STEPS:
                torch.cuda.profiler.start()
                started = True
                window["start_step"] = steps
            engine.step()
            steps += 1
            if (
                started
                and not stopped
                and steps >= PROFILE_SKIP_STEPS + PROFILE_WINDOW_STEPS
            ):
                torch.cuda.profiler.stop()
                stopped = True
                window["stop_step"] = steps
        if started and not stopped:
            torch.cuda.profiler.stop()
            window["stop_step"] = steps
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()
    out_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_nsys_lever",
                "combination": combo,
                "config": cfg,
                "armed": cell["armed"],
                "regime": regime_id,
                "batch": spec["batch"],
                "capture_window": window,
                "wall_summary": profiler.summary(warmup=r1.PROFILER_WARMUP),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _nsys_stats(report: Path, name: str) -> list[dict[str, str]] | None:
    completed = subprocess.run(
        [NSYS, "stats", "--report", name, "--format", "csv", "--force-export=true",
         str(report)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0:
        return None
    # nsys prints a banner before the CSV; the header row starts the table.
    lines = completed.stdout.splitlines()
    for index, line in enumerate(lines):
        if line.count(",") >= 3 and not line.startswith("**"):
            body = "\n".join(lines[index:])
            try:
                return list(csv.DictReader(io.StringIO(body)))
            except Exception:
                return None
    return None


def extract(report: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"nvtx": None, "kernels": None, "kernel_classes": None}
    for name in NVTX_REPORTS:
        rows = _nsys_stats(report, name)
        if rows:
            out["nvtx"] = {"report": name, "rows": rows[:60]}
            break
    for name in KERNEL_REPORTS:
        rows = _nsys_stats(report, name)
        if not rows:
            continue
        out["kernels"] = {"report": name, "rows": rows[:80]}
        classes: dict[str, float] = {}
        key_name = next(
            (k for k in rows[0] if "Name" in k or "name" in k), None
        )
        key_time = next(
            (k for k in rows[0] if "Total Time" in k or "total" in k.lower()), None
        )
        if key_name and key_time:
            for row in rows:
                with contextlib.suppress(ValueError, TypeError):
                    value = float(str(row[key_time]).replace(",", ""))
                    bucket = classify_kernel(row[key_name])
                    classes[bucket] = classes.get(bucket, 0.0) + value
        out["kernel_classes"] = classes or None
        break
    return out


def run_all(output_dir: Path, regime_id: str, only: list[str] | None) -> None:
    lane = d3.lane_for_block(1)
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = sorted(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    traces = output_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    for combo, cell in COMBINATIONS.items():
        if only and combo not in only:
            continue
        stem = f"{combo}__{regime_id}"
        target = output_dir / f"{stem}.json"
        if target.exists():
            continue
        trace = traces / f"{stem}.jsonl"
        report = output_dir / stem
        cfg = {k: cell[k] for k in ("quant", "window", "skip_count")}
        env = matrix._boot_child_environment(
            d3.boot_environment({**cfg, "action": "armed"}, trace, "corrected")
        )
        env["VLLM_SELF_SPEC_PROFILE_NVTX"] = "1"
        print(f"[x31] {stem} ...", flush=True)
        log = output_dir / f"{stem}.nsys.log"
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [
                    NSYS, "profile",
                    "--capture-range=cudaProfilerApi",
                    "--capture-range-end=stop",
                    "--trace=cuda,nvtx",
                    # WITHOUT this nsys defaults to graph granularity and
                    # reports each CUDA-graph replay as ONE entry, so a
                    # 36-layer forward collapses to a single row: the first
                    # run showed 200 GEMM instances for 200 forwards, which is
                    # impossible, and made the quantized draft look as though
                    # it ran no quantized kernel at all. Per-node is required
                    # for any per-kernel attribution under piecewise CUDA
                    # graphs.
                    "--cuda-graph-trace=node",
                    "--sample=none",
                    "--cuda-memory-usage=false",
                    "--force-overwrite=true",
                    "-o", str(report),
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--measure",
                    "--combo", combo,
                    "--regime", regime_id,
                    "--trace", str(trace),
                    "--out", str(target),
                ],
                env=env,
                cwd=str(REPO),
                stdout=handle,
                stderr=subprocess.STDOUT,
                preexec_fn=lambda: os.sched_setaffinity(0, affinity),
            )
        if completed.returncode != 0 or not target.exists():
            raise RuntimeError(f"{stem} failed ({completed.returncode}); see {log}")
        stats = extract(Path(f"{report}.nsys-rep"))
        record = json.loads(target.read_text())
        record["nsys"] = stats
        target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"[x31] ok {stem}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--combo")
    parser.add_argument("--regime", default="R1")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--only", nargs="*")
    parser.add_argument(
        "--output-dir", type=Path, default=PHASE / "data/probe_nsys_levers"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.measure:
        measure(args.trace, args.out, args.combo, args.regime)
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_all(args.output_dir, args.regime, args.only)
    print(f"\nwrote traces + stats under {args.output_dir}")
    print("run analyze_w98_nsys_accounting.py to build the expected/overhead table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
