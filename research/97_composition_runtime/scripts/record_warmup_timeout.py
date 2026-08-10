#!/usr/bin/env python3
"""Record a Phase 97 diagnostic that ended before the worker wrote JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--mbt", type=int, required=True)
    parser.add_argument(
        "--draft-execution", choices=("compiled", "eager"), required=True
    )
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--runner-status", type=int, required=True)
    parser.add_argument("--boot-class", default="weight_q_kv_q")
    parser.add_argument(
        "--draft-kv-dtype",
        choices=("fp8", "bfloat16"),
        default="fp8",
    )
    parser.add_argument(
        "--draft-trace",
        choices=("", "layer", "operator"),
        default="",
    )
    parser.add_argument(
        "--linear-backend",
        choices=("auto", "cutlass"),
        default="auto",
    )
    return parser.parse_args()


def gpu_snapshot(gpu: int) -> dict[str, int] | None:
    fields = "memory.total,memory.used,memory.free,utilization.gpu"
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode or not result.stdout.strip():
        return None
    values = [int(part.strip()) for part in result.stdout.splitlines()[0].split(",")]
    names = ("total_mib", "used_mib", "free_mib", "util_pct")
    return dict(zip(names, values, strict=True))


def failure_stage(log_text: str, draft_execution: str) -> str:
    warmups = log_text.count("Initial profiling/warmup run took")
    if "GPU KV cache size:" in log_text:
        return "after KV sizing"
    if warmups >= 2:
        return "after target and draft initial profiling/warmup"
    if warmups == 1 and draft_execution == "compiled":
        return "initial compiled draft profiling/warmup"
    if warmups == 1:
        return "initial scoped-eager draft profiling/warmup"
    if "pre-warmed JIT linear kernels" in log_text:
        return "before initial target profiling/warmup"
    return "engine initialization"


def last_trace_marker(log_text: str) -> str | None:
    for line in reversed(log_text.splitlines()):
        if "[draft-trace]" in line:
            return line.split("[draft-trace]", maxsplit=1)[1].strip()
    return None


def main() -> int:
    args = parse_args()
    log_bytes = args.log.read_bytes() if args.log.is_file() else b""
    log_text = log_bytes.decode(errors="replace")
    timed_out = args.runner_status in (124, 137)
    stem = args.out.stem
    draft_eager = "1" if args.draft_execution == "eager" else "0"
    kv_q = args.draft_kv_dtype == "fp8"
    diagnostic_only = args.boot_class.startswith("diagnostic_")
    if args.draft_trace:
        proxy_for = (
            f"diagnostic only: localize first non-completing draft {args.draft_trace}"
        )
    elif args.linear_backend == "cutlass":
        proxy_for = "diagnostic only: isolate W4A8 linear kernel selection"
    elif diagnostic_only:
        proxy_for = "diagnostic only: isolate private KV from fp8 dtype"
    else:
        proxy_for = "baseline + both optional paths"
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "timed_out" if timed_out else "failed_without_record",
        "non_scored": True,
        "static_proxy": True,
        "operator_record": True,
        "boot_class": args.boot_class,
        "proxy_for": proxy_for,
        "gpu_physical": args.gpu,
        "model": "Qwen/Qwen3-8B",
        "draft": "/data/smcho/ckpts/Qwen3-8B-W4A8-gptq",
        "weight_q": True,
        "kv_q": kv_q,
        "draft_kv_mode": f"private-{args.draft_kv_dtype}",
        "draft_kv_dtype": args.draft_kv_dtype,
        "private_draft_kv": True,
        "diagnostic_only": diagnostic_only,
        "draft_execution": args.draft_execution,
        "draft_trace": args.draft_trace or None,
        "target_execution": "compiled",
        "compile_cache_root": f"/data/smcho/.cache/vllm-p97-warmup/{stem}",
        "engine_args": {
            "k": 4,
            "max_model_len": 20480,
            "max_num_seqs": 32,
            "max_num_batched_tokens": args.mbt,
            "gpu_memory_utilization": 0.9,
            "enforce_eager": False,
            "flashinfer_autotune": False,
            "linear_backend": args.linear_backend,
        },
        "environment": {
            "VLLM_DISABLED_KERNELS": (
                "MacheteLinearKernel,AllSparkLinearKernel"
                if args.linear_backend == "cutlass"
                else "MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel"
            ),
            "VLLM_SELF_SPEC_CPU_ORCH": "1",
            "VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD": "1",
            "VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE": "1",
            "VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU": "1",
            "VLLM_SELF_SPEC_DRAFT_EAGER": draft_eager,
            "VLLM_SELF_SPEC_DRAFT_TRACE": args.draft_trace,
            "VLLM_SELF_SPEC_DRAFT_FULL_CG": "1",
            "VLLM_SELF_SPEC_DRAFT_KV_DTYPE": args.draft_kv_dtype,
            "VLLM_SELF_SPEC_DRAFT_KV_WINDOW": "0",
            "VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS": "",
            "VLLM_SELF_SPEC_SHARED_KV": "0",
            "VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE": "0",
            "VLLM_SELF_SPEC_SHARE_WEIGHTS": "0",
            "VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT": "0",
        },
        "failure_stage": failure_stage(log_text, args.draft_execution),
        "last_trace_marker": last_trace_marker(log_text),
        "observed_elapsed_before_stop_seconds_gte": args.timeout_seconds,
        "runner_status": args.runner_status,
        "termination": "runner stopped only the owned run at the threshold",
        "gpu_after_shutdown": gpu_snapshot(args.gpu),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "source_log": str(args.log),
        "source_log_sha256": hashlib.sha256(log_bytes).hexdigest(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"[P97] wrote terminal diagnostic record -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
