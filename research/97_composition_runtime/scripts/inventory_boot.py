#!/usr/bin/env python3
"""Inventory one Phase 97 static proxy boot.

This is a non-scored resource run. It measures a boot-static realization and
does not claim that quantized and baseline paths can switch at runtime.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL = "Qwen/Qwen3-8B"
WEIGHT_Q_DRAFT = "/data/smcho/ckpts/Qwen3-8B-W4A8-gptq"
DISABLED_KERNELS = "MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel"
CUTLASS_CONTROL_DISABLED_KERNELS = "MacheteLinearKernel,AllSparkLinearKernel"

CLASS_CONFIGS = {
    "baseline": {
        "weight_q": False,
        "kv_q": False,
        "draft_kv_mode": "shared-target",
        "diagnostic_only": False,
        "proxy_for": "baseline",
    },
    "weight_q": {
        "weight_q": True,
        "kv_q": False,
        "draft_kv_mode": "shared-target",
        "diagnostic_only": False,
        "proxy_for": "baseline + optional weight-q",
    },
    "kv_q": {
        "weight_q": False,
        "kv_q": True,
        "draft_kv_mode": "private-fp8",
        "diagnostic_only": False,
        "proxy_for": "baseline + optional KV-q",
    },
    "weight_q_kv_q": {
        "weight_q": True,
        "kv_q": True,
        "draft_kv_mode": "private-fp8",
        "diagnostic_only": False,
        "proxy_for": "baseline + both optional paths",
    },
    "diagnostic_weight_q_private_bf16": {
        "weight_q": True,
        "kv_q": False,
        "draft_kv_mode": "private-bfloat16",
        "diagnostic_only": True,
        "proxy_for": "diagnostic only: isolate private KV from fp8 dtype",
    },
    "diagnostic_weight_q_kv_q_cutlass": {
        "weight_q": True,
        "kv_q": True,
        "draft_kv_mode": "private-fp8",
        "diagnostic_only": True,
        "linear_backend": "cutlass",
        "proxy_for": "diagnostic only: isolate W4A8 linear kernel selection",
    },
}

COMMON_ENV = {
    "VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU": "1",
    "VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD": "1",
    "VLLM_SELF_SPEC_CPU_ORCH": "1",
    "VLLM_SELF_SPEC_DRAFT_FULL_CG": "1",
    "VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE": "1",
    "VLLM_SELF_SPEC_DRAFT_KV_WINDOW": "0",
    "VLLM_DISABLED_KERNELS": DISABLED_KERNELS,
}

MANAGED_ENV = (
    *COMMON_ENV,
    "VLLM_SELF_SPEC_DRAFT_EAGER",
    "VLLM_SELF_SPEC_DRAFT_TRACE",
    "VLLM_SELF_SPEC_SHARED_KV",
    "VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE",
    "VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT",
    "VLLM_SELF_SPEC_SHARE_WEIGHTS",
    "VLLM_SELF_SPEC_DRAFT_KV_DTYPE",
    "VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boot-class", choices=CLASS_CONFIGS, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=20480)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument(
        "--draft-execution",
        choices=("compiled", "eager"),
        default="compiled",
        help="Use the normal compiled draft or the scoped draft-eager control.",
    )
    parser.add_argument(
        "--draft-trace",
        choices=("", "layer", "operator"),
        default="",
        help="Trace the first max-token eager draft profiling forward.",
    )
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--smoke-tokens", type=int, default=16)
    return parser.parse_args()


def run_text(command: list[str], timeout: float = 20.0) -> str:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode:
        return ""
    return result.stdout.strip()


def gpu_snapshot(gpu: int) -> dict[str, int] | None:
    fields = "memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu"
    output = run_text(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return None
    values = [int(part.strip()) for part in output.splitlines()[0].split(",")]
    names = ("total_mib", "used_mib", "free_mib", "util_pct", "temp_c")
    return dict(zip(names, values, strict=True))


def gpu_processes(gpu: int) -> list[dict[str, Any]]:
    uuid = run_text(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    output = run_text(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4 or parts[0] != uuid:
            continue
        rows.append(
            {
                "gpu_uuid": parts[0],
                "pid": int(parts[1]),
                "process_name": parts[2],
                "used_mib": int(parts[3]),
            }
        )
    return rows


class GpuMemorySampler:
    """Sample driver-visible memory while the blocking boot runs."""

    def __init__(self, gpu: int, interval_s: float = 0.5) -> None:
        self.gpu = gpu
        self.interval_s = interval_s
        self.samples: list[dict[str, int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            snapshot = gpu_snapshot(self.gpu)
            if snapshot is not None:
                self.samples.append(snapshot)
            self._stop.wait(self.interval_s)

    def summary(self) -> dict[str, Any]:
        used = [sample["used_mib"] for sample in self.samples]
        return {
            "interval_s": self.interval_s,
            "sample_count": len(self.samples),
            "peak_used_mib": max(used) if used else None,
        }


def configure_environment(
    boot_class: str,
    draft_execution: str,
    draft_trace: str = "",
) -> tuple[dict[str, Any], str]:
    for key in MANAGED_ENV:
        os.environ.pop(key, None)
    os.environ.update(COMMON_ENV)
    os.environ["VLLM_SELF_SPEC_DRAFT_EAGER"] = (
        "1" if draft_execution == "eager" else "0"
    )
    if draft_trace and draft_execution != "eager":
        raise ValueError("draft tracing requires --draft-execution eager")
    os.environ["VLLM_SELF_SPEC_DRAFT_TRACE"] = draft_trace

    config = CLASS_CONFIGS[boot_class]
    linear_backend = str(config.get("linear_backend", "auto"))
    if linear_backend == "cutlass":
        os.environ["VLLM_DISABLED_KERNELS"] = CUTLASS_CONTROL_DISABLED_KERNELS
    weight_q = bool(config["weight_q"])
    draft_kv_mode = str(config["draft_kv_mode"])
    draft = WEIGHT_Q_DRAFT if weight_q else MODEL

    os.environ["VLLM_SELF_SPEC_SHARE_WEIGHTS"] = "0" if weight_q else "1"
    if draft_kv_mode != "shared-target":
        os.environ["VLLM_SELF_SPEC_SHARED_KV"] = "0"
        os.environ["VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE"] = "0"
        os.environ["VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT"] = "0"
        os.environ["VLLM_SELF_SPEC_DRAFT_KV_DTYPE"] = (
            "fp8" if draft_kv_mode == "private-fp8" else "bfloat16"
        )
    else:
        os.environ["VLLM_SELF_SPEC_SHARED_KV"] = "1"
        os.environ["VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE"] = "1"
        os.environ["VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT"] = "1"
        os.environ["VLLM_SELF_SPEC_DRAFT_KV_DTYPE"] = ""
    return config, draft


def cache_record(llm: Any) -> dict[str, Any]:
    cache = llm.llm_engine.vllm_config.cache_config
    return {
        "num_gpu_blocks": cache.num_gpu_blocks,
        "block_size": cache.block_size,
        "kv_cache_size_tokens": getattr(cache, "kv_cache_size_tokens", None),
        "kv_cache_max_concurrency": getattr(cache, "kv_cache_max_concurrency", None),
        "target_cache_dtype": str(cache.cache_dtype),
        "gpu_memory_utilization": cache.gpu_memory_utilization,
    }


def speculative_counters(llm: Any) -> dict[str, float]:
    wanted = {
        "vllm:spec_decode_num_accepted_tokens": "accepted_tokens",
        "vllm:spec_decode_num_drafts": "draft_tokens",
        "vllm:num_preemptions": "preemptions",
    }
    counters = {name: 0.0 for name in wanted.values()}
    for metric in llm.get_metrics():
        output_name = wanted.get(metric.name)
        if output_name is not None:
            counters[output_name] = float(metric.value)
    return counters


def git_record() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    commit = run_text(["git", "-C", str(root), "rev-parse", "HEAD"])
    status = subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet"],
        check=False,
    )
    return {"commit": commit, "tracked_worktree_dirty": status.returncode != 0}


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    config, draft = configure_environment(
        args.boot_class,
        args.draft_execution,
        args.draft_trace,
    )
    proxy_for = config["proxy_for"]
    if args.draft_trace:
        proxy_for = (
            f"diagnostic only: localize first non-completing draft {args.draft_trace}"
        )
    if config["weight_q"] and not Path(WEIGHT_Q_DRAFT).is_dir():
        raise FileNotFoundError(f"missing weight-q checkpoint: {WEIGHT_Q_DRAFT}")

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "starting",
        "non_scored": True,
        "static_proxy": True,
        "boot_class": args.boot_class,
        "proxy_for": proxy_for,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_physical": args.gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git": git_record(),
        "model": MODEL,
        "draft": draft,
        "draft_execution": args.draft_execution,
        "draft_trace": args.draft_trace or None,
        "target_execution": "compiled",
        "weight_q": config["weight_q"],
        "kv_q": config["kv_q"],
        "draft_kv_mode": config["draft_kv_mode"],
        "draft_kv_dtype": (
            config["draft_kv_mode"].removeprefix("private-")
            if config["draft_kv_mode"] != "shared-target"
            else "shared-target"
        ),
        "private_draft_kv": config["draft_kv_mode"] != "shared-target",
        "diagnostic_only": config["diagnostic_only"],
        "engine_args": {
            "k": args.k,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": 0.90,
            "enable_prefix_caching": False,
            "async_scheduling": True,
            "flashinfer_autotune": False,
            "linear_backend": str(config.get("linear_backend", "auto")),
            "enforce_eager": False,
        },
        "environment": {key: os.environ.get(key, "") for key in MANAGED_ENV},
        "gpu_before": gpu_snapshot(args.gpu),
        "gpu_processes_before": gpu_processes(args.gpu),
    }

    llm = None
    sampler = GpuMemorySampler(args.gpu)
    sampler.start()
    started = time.perf_counter()
    exit_code = 0
    try:
        from vllm import LLM, SamplingParams

        speculative_config = {
            "method": "draft_model",
            "model": draft,
            "num_speculative_tokens": args.k,
            "draft_tensor_parallel_size": 1,
        }
        boot_started = time.perf_counter()
        llm = LLM(
            model=MODEL,
            speculative_config=speculative_config,
            tensor_parallel_size=1,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=0.90,
            max_num_seqs=args.max_num_seqs,
            enable_prefix_caching=False,
            disable_log_stats=False,
            async_scheduling=True,
            max_num_batched_tokens=args.max_num_batched_tokens,
            kernel_config={
                "enable_flashinfer_autotune": False,
                "linear_backend": str(config.get("linear_backend", "auto")),
            },
            trust_remote_code=True,
        )
        result["boot_seconds"] = round(time.perf_counter() - boot_started, 6)
        result["cache"] = cache_record(llm)
        result["gpu_after_boot"] = gpu_snapshot(args.gpu)
        result["gpu_processes_after_boot"] = gpu_processes(args.gpu)

        prompt = "State one benefit of speculative decoding in one sentence."
        smoke_started = time.perf_counter()
        outputs = llm.generate(
            [prompt],
            SamplingParams(
                max_tokens=args.smoke_tokens,
                min_tokens=args.smoke_tokens,
                temperature=0.0,
                ignore_eos=True,
            ),
            use_tqdm=False,
        )
        smoke_output = outputs[0].outputs[0]
        result["smoke"] = {
            "seconds": round(time.perf_counter() - smoke_started, 6),
            "requested_tokens": args.smoke_tokens,
            "output_tokens": len(smoke_output.token_ids),
            "finish_reason": smoke_output.finish_reason,
            "counters": speculative_counters(llm),
        }
        result["gpu_after_smoke"] = gpu_snapshot(args.gpu)
        result["status"] = "booted"
    except BaseException as error:
        exit_code = 1
        result["status"] = "failed"
        result["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        traceback.print_exc()
    finally:
        sampler.stop()
        result["memory_sampler"] = sampler.summary()
        result["elapsed_seconds_before_shutdown"] = round(
            time.perf_counter() - started, 6
        )
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown()
            except BaseException as error:
                result["shutdown_error"] = f"{type(error).__name__}: {error}"
        llm = None
        gc.collect()
        time.sleep(3)
        result["gpu_after_shutdown"] = gpu_snapshot(args.gpu)
        result["gpu_processes_after_shutdown"] = gpu_processes(args.gpu)
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_result(args.out, result)
        print(
            f"[P97] {args.boot_class}/{args.draft_execution}: "
            f"{result['status']} -> {args.out}",
            flush=True,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
