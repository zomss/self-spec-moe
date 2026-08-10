#!/usr/bin/env python3
"""Summarize the registered Phase 97 MBT/compile-mode diagnostic matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED = (
    (2048, "compiled", 4),
    (2048, "eager", 5),
    (4096, "compiled", 5),
    (4096, "eager", 4),
)

TERMINAL_STATUSES = {"booted", "failed", "timed_out", "failed_without_record"}

CONTROL_CASES = (
    {
        "stem": "weight_q_private_bf16_mbt2048_compiled",
        "draft_execution": "compiled",
        "draft_trace": None,
        "timeout_seconds": 900,
        "conclusion": (
            "FP8 KV dtype is not required; the large "
            "W4A8/private-draft-KV path remains implicated"
        ),
    },
    {
        "stem": "weight_q_private_bf16_mbt2048_operator_trace_eager",
        "draft_execution": "eager",
        "draft_trace": "operator",
        "timeout_seconds": 360,
        "conclusion": (
            "the first non-completing operation is the layer-0 Humming QKV "
            "projection; rotary embedding and private-KV attention are not "
            "reached"
        ),
    },
)

CUTLASS_CONTROL_CASES = (
    {
        "stem": "weight_q_kv_q_mbt2048_eager_cutlass",
        "draft_execution": "eager",
    },
    {
        "stem": "weight_q_kv_q_mbt2048_compiled_cutlass",
        "draft_execution": "compiled",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing"}
    return json.loads(path.read_text())


def log_evidence(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="replace") if path.is_file() else ""
    compile_s = [
        float(value) for value in re.findall(r"torch\.compile took ([0-9.]+) s", text)
    ]
    warmup_s = [
        float(value)
        for value in re.findall(r"Initial profiling/warmup run took ([0-9.]+) s", text)
    ]
    return {
        "torch_compile_totals_s": compile_s,
        "initial_warmups_s": warmup_s,
        "jit_linear_prewarm_reached": "pre-warmed JIT linear kernels" in text,
        "kv_sizing_reached": "GPU KV cache size:" in text,
        "graph_capture_reached": "Capturing CUDA graphs" in text,
        "engine_ready": "init engine (profile, create kv cache" in text,
    }


def gpu_evidence(path: Path) -> dict[str, Any] | None:
    samples: list[tuple[int, int, int]] = []
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 4:
                continue
            try:
                samples.append(tuple(int(float(value)) for value in fields[1:4]))
            except ValueError:
                continue
    if not samples:
        return None

    peak_used_mib = max(used for used, _, _ in samples)
    return {
        "used_mib": peak_used_mib,
        "sm_util_pct": max(sm for _, sm, _ in samples),
        "memory_util_pct": min(memory for _, _, memory in samples),
        "telemetry_samples": len(samples),
        "samples_at_peak_used_mib": sum(
            used == peak_used_mib for used, _, _ in samples
        ),
        "samples_at_100_sm_util": sum(sm == 100 for _, sm, _ in samples),
        "samples_at_0_memory_util": sum(memory == 0 for _, _, memory in samples),
    }


def compact(
    record: dict[str, Any], mbt: int, mode: str, gpu: int, log: Path
) -> dict[str, Any]:
    cache = record.get("cache") or {}
    smoke = record.get("smoke") or {}
    after_shutdown = record.get("gpu_after_shutdown") or {}
    record_path = (
        log.parent.parent.parent
        / "data"
        / "diagnostics"
        / f"weight_q_kv_q_mbt{mbt}_{mode}.json"
    )
    return {
        "max_num_batched_tokens": mbt,
        "draft_execution": mode,
        "target_execution": record.get("target_execution", "compiled"),
        "gpu_physical": record.get("gpu_physical", gpu),
        "status": record.get("status"),
        "boot_seconds": record.get("boot_seconds"),
        "kv_cache_size_tokens": cache.get("kv_cache_size_tokens"),
        "kv_cache_max_concurrency": cache.get("kv_cache_max_concurrency"),
        "smoke_output_tokens": smoke.get("output_tokens"),
        "accepted_tokens": smoke.get("counters", {}).get("accepted_tokens"),
        "draft_tokens": smoke.get("counters", {}).get("draft_tokens"),
        "failure_stage": record.get("failure_stage"),
        "timeout_seconds": record.get("observed_elapsed_before_stop_seconds_gte"),
        "gpu_after_shutdown_mib": after_shutdown.get("used_mib"),
        "observed_gpu_during_stall": record.get("observed_gpu_during_stall"),
        "log_evidence": log_evidence(log),
        "source_record": str(record_path),
        "source_log": str(log),
    }


def compact_control(phase: Path, case: dict[str, Any]) -> dict[str, Any]:
    stem = case["stem"]
    record_path = phase / "data" / "diagnostics" / f"{stem}.json"
    log_path = phase / "logs" / "diagnostics" / f"{stem}.log"
    gpu_path = phase / "logs" / "diagnostics" / f"{stem}.gpu.csv"
    record = read_json(record_path)
    after_shutdown = record.get("gpu_after_shutdown") or {}
    control = {
        "max_num_batched_tokens": 2048,
        "boot_class": "diagnostic_weight_q_private_bf16",
        "weight_q": True,
        "draft_kv_mode": "private-bfloat16",
        "draft_kv_dtype": "bfloat16",
        "private_draft_kv": True,
        "diagnostic_only": True,
        "draft_execution": case["draft_execution"],
        "target_execution": "compiled",
        "gpu_physical": record.get("gpu_physical", 4),
        "status": record.get("status"),
        "kv_cache_size_tokens": None,
        "kv_cache_max_concurrency": None,
        "smoke_output_tokens": None,
        "accepted_tokens": None,
        "draft_tokens": None,
        "failure_stage": record.get("failure_stage"),
        "timeout_seconds": record.get(
            "observed_elapsed_before_stop_seconds_gte",
            case["timeout_seconds"],
        ),
        "gpu_after_shutdown_mib": after_shutdown.get("used_mib"),
        "observed_gpu_during_stall": gpu_evidence(gpu_path),
        "log_evidence": log_evidence(log_path),
        "conclusion": case["conclusion"],
        "source_record": str(record_path),
        "source_log": str(log_path),
        "source_gpu_telemetry": str(gpu_path),
    }
    if case["draft_trace"]:
        control["draft_trace"] = case["draft_trace"]
        control["last_trace_marker"] = record.get("last_trace_marker")
        control["operator_localization"] = {
            "first_non_completing_module": ("model.layers.0.self_attn.qkv_proj"),
            "module_type": "QKVParallelLinear",
            "w4a8_kernel": "Humming",
            "matching_exit_observed": False,
            "rotary_embedding_reached": False,
            "private_kv_attention_reached": False,
        }
    return control


def compact_cutlass_control(phase: Path, case: dict[str, Any]) -> dict[str, Any]:
    stem = case["stem"]
    record_path = phase / "data" / "diagnostics" / f"{stem}.json"
    log_path = phase / "logs" / "diagnostics" / f"{stem}.log"
    gpu_path = phase / "logs" / "diagnostics" / f"{stem}.gpu.csv"
    record = read_json(record_path)
    log_text = log_path.read_text(errors="replace") if log_path.is_file() else ""
    cache = record.get("cache") or {}
    smoke = record.get("smoke") or {}
    counters = smoke.get("counters") or {}
    after_shutdown = record.get("gpu_after_shutdown") or {}
    selection_line = "Using CutlassW4A8LinearKernel for CompressedTensorsW4A8Fp8"
    selection_verified = (
        record.get("engine_args", {}).get("linear_backend") == "cutlass"
        and selection_line in log_text
    )
    return {
        "max_num_batched_tokens": 2048,
        "boot_class": "diagnostic_weight_q_kv_q_cutlass",
        "weight_q": True,
        "draft_kv_mode": "private-fp8",
        "draft_kv_dtype": "fp8",
        "private_draft_kv": True,
        "diagnostic_only": True,
        "draft_execution": case["draft_execution"],
        "target_execution": "compiled",
        "linear_backend": record.get("engine_args", {}).get("linear_backend"),
        "cutlass_selection_verified": selection_verified,
        "gpu_physical": record.get("gpu_physical", 5),
        "status": record.get("status"),
        "boot_seconds": record.get("boot_seconds"),
        "kv_cache_size_tokens": cache.get("kv_cache_size_tokens"),
        "kv_cache_max_concurrency": cache.get("kv_cache_max_concurrency"),
        "smoke_seconds": smoke.get("seconds"),
        "smoke_output_tokens": smoke.get("output_tokens"),
        "accepted_tokens": counters.get("accepted_tokens"),
        "draft_counter": counters.get("draft_tokens"),
        "preemptions": counters.get("preemptions"),
        "gpu_after_shutdown_mib": after_shutdown.get("used_mib"),
        "peak_used_mib": record.get("memory_sampler", {}).get("peak_used_mib"),
        "gpu_telemetry": gpu_evidence(gpu_path),
        "log_evidence": log_evidence(log_path),
        "conclusion": (
            "Cutlass completes the MBT2048 combined realization; the prior "
            "stall requires the Humming W4A8 realization"
        ),
        "source_record": str(record_path),
        "source_log": str(log_path),
        "source_gpu_telemetry": str(gpu_path),
    }


def main() -> int:
    args = parse_args()
    rows = []
    for mbt, mode, gpu in EXPECTED:
        stem = f"weight_q_kv_q_mbt{mbt}_{mode}"
        record_path = args.phase / "data" / "diagnostics" / f"{stem}.json"
        log_path = args.phase / "logs" / "diagnostics" / f"{stem}.log"
        rows.append(compact(read_json(record_path), mbt, mode, gpu, log_path))

    controls = [compact_control(args.phase, case) for case in CONTROL_CASES]
    kernel_controls = [
        compact_cutlass_control(args.phase, case) for case in CUTLASS_CONTROL_CASES
    ]
    protocol = {
        "boot_class": "weight_q_kv_q",
        "model": "Qwen/Qwen3-8B",
        "k": 4,
        "max_model_len": 20480,
        "max_num_seqs": 32,
        "max_num_batched_tokens": [2048, 4096],
        "draft_execution": ["compiled", "eager"],
        "target_execution": "compiled",
        "draft_kv_dtype": "fp8",
        "gpu_memory_utilization": 0.9,
        "gpu_assignment": {
            "mbt2048_compiled": 4,
            "mbt2048_eager": 5,
            "mbt4096_compiled": 5,
            "mbt4096_eager": 4,
        },
        "compile_cache": "separate cold root per case",
        "timeout_seconds": 900,
        "note": "non-scored warmup localization with private fp8 draft KV",
    }
    kernel_control_protocol = {
        "boot_class": "diagnostic_weight_q_kv_q_cutlass",
        "model": "Qwen/Qwen3-8B",
        "k": 4,
        "max_model_len": 20480,
        "max_num_seqs": 32,
        "max_num_batched_tokens": 2048,
        "draft_execution": ["eager", "compiled"],
        "target_execution": "compiled",
        "draft_kv_dtype": "fp8",
        "linear_backend": "cutlass",
        "gpu_memory_utilization": 0.9,
        "gpu_physical": 5,
        "compile_cache": "separate cold root per case",
        "timeout_seconds": 900,
        "note": "non-scored W4A8 kernel-isolation control",
    }
    ledger = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_complete": all(row["status"] in TERMINAL_STATUSES for row in rows),
        "all_booted": all(row["status"] == "booted" for row in rows),
        "additional_control_complete": (controls[0]["status"] in TERMINAL_STATUSES),
        "operator_trace_complete": (
            controls[1]["status"] in TERMINAL_STATUSES
            and controls[1].get("last_trace_marker") is not None
        ),
        "kernel_controls_complete": all(
            control["status"] in TERMINAL_STATUSES
            and control["cutlass_selection_verified"]
            for control in kernel_controls
        ),
        "kernel_controls_all_booted": all(
            control["status"] == "booted" for control in kernel_controls
        ),
        "non_scored": True,
        "static_proxy": True,
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True).encode()
        ).hexdigest(),
        "kernel_control_protocol": kernel_control_protocol,
        "kernel_control_protocol_sha256": hashlib.sha256(
            json.dumps(kernel_control_protocol, sort_keys=True).encode()
        ).hexdigest(),
        "rows": rows,
        "controls": controls,
        "kernel_controls": kernel_controls,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(
        f"[P97] warmup matrix complete={ledger['matrix_complete']} "
        f"all_booted={ledger['all_booted']} -> {args.out}"
    )
    return 0 if ledger["matrix_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
