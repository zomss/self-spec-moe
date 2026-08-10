#!/usr/bin/env python3
"""Run the separately authorized non-scored P4 chunked-prefill probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
AUTHORIZATION_PATH = (
    PHASE_DIR / "data" / "p4" / "p4_b0_chunked_prefill_probe_authorization_v5.json"
)
OUTPUT_DIR = PHASE_DIR / "data" / "p4" / "run_b0_chunked_prefill_probe_v5"
GPU4_UUID = "GPU-c9d19019-5065-2353-80a9-f1797eb19d51"
MINIMUM_SHARED_KV_BLOCKS = 21682
CONFIGURED_MAX_NUM_BATCHED_TOKENS = 8192
EFFECTIVE_MAX_NUM_SCHEDULED_TOKENS = 8160
PROBE_REGIMES = ("R4", "R5", "R5cot")
PROBE_ACTION_ID = "target-matching-k4"
PROBE_CONTENT_SEED = 0
PROBE_BATCH = 8
MEASURED_TOKENS = 1


class P4ChunkedPrefillProbeError(RuntimeError):
    """Raised when the one-shot probe cannot preserve its authorization."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise P4ChunkedPrefillProbeError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise P4ChunkedPrefillProbeError(
            f"cannot load JSON artifact {path}: {exc}"
        ) from exc
    _require(isinstance(value, dict), f"JSON artifact must be an object: {path}")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise P4ChunkedPrefillProbeError(
            f"refusing to overwrite probe artifact: {path}"
        ) from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _NonScoredRecorder:
    capture_id = "p4-b0-chunked-prefill-gpu4-probe"

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.closed = False

    def record(self, event: Mapping[str, Any], _evidence: Any) -> None:
        self.events.append(dict(event))

    def close(self) -> None:
        self.closed = True


def _probe_cells(
    manifest: Mapping[str, Any],
) -> list[tuple[dict[str, Any], tuple[str, ...]]]:
    prompts = manifest.get("prompts")
    _require(isinstance(prompts, list), "prompt manifest has no prompt records")
    cells = []
    for regime_id in PROBE_REGIMES:
        request_ids = tuple(
            prompt["record_id"]
            for prompt in prompts
            if isinstance(prompt, Mapping)
            and prompt.get("regime_id") == regime_id
            and prompt.get("content_seed") == PROBE_CONTENT_SEED
        )[:PROBE_BATCH]
        _require(
            len(request_ids) == PROBE_BATCH and len(set(request_ids)) == PROBE_BATCH,
            f"probe regime {regime_id} does not have eight frozen prompts",
        )
        cells.append(
            (
                {
                    "capture_id": f"p4-b0-chunked-prefill-probe-{regime_id}",
                    "matrix": {"action_id": PROBE_ACTION_ID},
                    "generation": {
                        "temperature": 0.0,
                        "max_output_tokens": MEASURED_TOKENS,
                        "generation_seed": 0,
                    },
                },
                request_ids,
            )
        )
    return cells


def _engine_spec(authorization: Mapping[str, Any]) -> dict[str, Any]:
    contract = authorization["run_contract"]
    return {
        "model": dict(contract["model"]),
        "engine": dict(contract["engine"]),
        "dynamic_k_schedule": [[1, 32, 4]],
    }


def _runtime_handles(engine: Any) -> tuple[Any, Any]:
    from vllm.v1.engine.core_client import InprocClient

    client = engine.engine_core
    _require(isinstance(client, InprocClient), "probe did not use InprocClient")
    core = client.engine_core
    executor = core.model_executor
    worker_wrapper = getattr(executor, "driver_worker", None)
    worker = None if worker_wrapper is None else getattr(worker_wrapper, "worker", None)
    _require(worker is not None, "probe cannot inspect the in-process GPU worker")
    return core.scheduler, worker


def _resource_evidence(scheduler: Any, worker: Any) -> dict[str, Any]:
    actual_graph_memory = getattr(worker, "cuda_graph_memory_bytes", None)
    estimated_graph_memory = getattr(worker, "cudagraph_memory_estimate", None)
    capacity = scheduler.kv_cache_manager.block_pool.num_gpu_blocks
    configured_budget = scheduler.scheduler_config.max_num_batched_tokens
    effective_budget = scheduler.max_num_scheduled_tokens
    _require(
        configured_budget == CONFIGURED_MAX_NUM_BATCHED_TOKENS,
        "probe configured chunked-prefill budget drifted",
    )
    _require(
        effective_budget == EFFECTIVE_MAX_NUM_SCHEDULED_TOKENS,
        "probe effective scheduler budget drifted",
    )
    _require(
        type(actual_graph_memory) is int and actual_graph_memory > 0,
        "probe did not observe positive actual CUDA graph memory",
    )
    _require(
        type(estimated_graph_memory) is int and estimated_graph_memory >= 0,
        "probe did not observe the CUDA graph estimate",
    )
    _require(
        type(capacity) is int and capacity >= MINIMUM_SHARED_KV_BLOCKS,
        "probe shared target-KV capacity is below the registered floor",
    )
    return {
        "configured_max_num_batched_tokens": configured_budget,
        "effective_max_num_scheduled_tokens": effective_budget,
        "actual_cuda_graph_memory_bytes": actual_graph_memory,
        "estimated_cuda_graph_memory_bytes": estimated_graph_memory,
        "shared_target_kv_block_capacity": capacity,
        "minimum_shared_target_kv_blocks": MINIMUM_SHARED_KV_BLOCKS,
    }


def _validate_probe_evidence(
    histories: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    expected_request_ids: Sequence[Sequence[str]],
) -> list[dict[str, Any]]:
    from vllm.v1.spec_decode.koff_runtime import (
        KOffRuntimeError,
        canonicalize_p4_request_ids,
    )

    _require(len(histories) == len(PROBE_REGIMES), "probe lost cohort history")
    _require(len(events) == len(PROBE_REGIMES), "probe emitted extra target events")
    normalized_events = []
    for regime_id, history, event, request_ids in zip(
        PROBE_REGIMES, histories, events, expected_request_ids, strict=True
    ):
        _require(
            history.get("state") == "complete"
            and history.get("action_id") == PROBE_ACTION_ID
            and history.get("max_num_batched_tokens")
            == CONFIGURED_MAX_NUM_BATCHED_TOKENS
            and history.get("max_num_scheduled_tokens")
            == EFFECTIVE_MAX_NUM_SCHEDULED_TOKENS
            and history.get("pure_first_measured_decode") is True
            and history.get("unmeasured_prefill_tokens_per_request") == 1
            and history.get("measured_decode_tokens_per_request") == MEASURED_TOKENS
            and history.get("abort_reason") is None,
            f"probe cohort {regime_id} did not close the barrier contract",
        )
        _require(
            history.get("committed_by_request")
            == dict.fromkeys(request_ids, MEASURED_TOKENS),
            f"probe cohort {regime_id} changed exact measured work",
        )
        quality = event.get("quality")
        rows = event.get("request_steps")
        _require(
            event.get("pure_decode") is True
            and event.get("score_eligible") is True
            and isinstance(quality, Mapping)
            and quality.get("preemptions") == 0
            and quality.get("recomputed_tokens") == 0
            and quality.get("invalid_spec_tokens") == 0,
            f"probe cohort {regime_id} has invalid execution quality",
        )
        _require(
            isinstance(rows, list)
            and rows
            and all(isinstance(row, Mapping) for row in rows),
            f"probe cohort {regime_id} target rows changed identity",
        )
        try:
            canonical_ids = canonicalize_p4_request_ids(
                [row.get("request_id") for row in rows],
                request_ids,
            )
        except KOffRuntimeError as exc:
            raise P4ChunkedPrefillProbeError(
                f"probe cohort {regime_id} target rows changed identity: {exc}"
            ) from exc
        _require(
            set(canonical_ids) == set(request_ids),
            f"probe cohort {regime_id} target rows changed identity",
        )
        normalized_event = dict(event)
        normalized_rows = [dict(row) for row in rows]
        for row, canonical_id in zip(normalized_rows, canonical_ids, strict=True):
            row["request_id"] = canonical_id
        normalized_event["request_steps"] = normalized_rows
        normalized_events.append(normalized_event)
    return normalized_events


def run_child(authorization: Mapping[str, Any], output_dir: Path) -> None:
    """Execute three ingress cohorts in one non-scored physical boot."""
    from run_p4_b0_value_screen import (
        _engine_args,
        _load_prompt_rows,
        _run_prompt_chunk,
    )

    from vllm import LLMEngine

    trace_path = output_dir / "koff_trace.jsonl"
    _require(
        os.environ.get("VLLM_SELF_SPEC_KOFF_TRACE") == str(trace_path),
        "child K/OFF trace path differs from the authorized output",
    )
    manifest, prompt_rows = _load_prompt_rows()
    cells = _probe_cells(manifest)
    engine = LLMEngine.from_engine_args(_engine_args(_engine_spec(authorization)))
    scheduler, worker = _runtime_handles(engine)
    recorder = _NonScoredRecorder()
    scheduler._p4_capture_recorder = recorder
    resource_evidence = _resource_evidence(scheduler, worker)
    expected_request_ids = []
    try:
        for cell, request_ids in cells:
            expected_request_ids.append(request_ids)
            _run_prompt_chunk(engine, cell, request_ids, prompt_rows)
        histories = scheduler.get_p4_capture_cohort_history()
        normalized_events = _validate_probe_evidence(
            histories,
            recorder.events,
            expected_request_ids,
        )
    finally:
        engine.engine_core.shutdown()

    _require(recorder.closed, "probe recorder did not close")
    _require(trace_path.is_file(), "probe did not emit its non-scored K/OFF trace")
    result = {
        "schema_version": 1,
        "artifact_id": "p4-b0-chunked-prefill-gpu4-probe-result-v5",
        "status": "pass",
        "scored": False,
        "authorization_consumed": True,
        "gpu": {"physical_index": 4, "uuid": GPU4_UUID},
        "action_id": PROBE_ACTION_ID,
        "regimes": list(PROBE_REGIMES),
        "resources": resource_evidence,
        "cohorts": list(histories),
        "target_events": normalized_events,
        "claims": {
            "chunked_prefill_barrier_gpu_proven": True,
            "v10_authorized": False,
            "performance_claim_allowed": False,
        },
        "authorizations": {
            "v10_value_screen": False,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
    }
    _write_json_exclusive(output_dir / "probe_result.json", result)


def _child_environment(
    authorization: Mapping[str, Any], output_dir: Path
) -> dict[str, str]:
    from run_p4_b0_value_screen import _boot_child_environment

    environment = _boot_child_environment(authorization["run_contract"]["environment"])
    environment["VLLM_SELF_SPEC_KOFF_TRACE"] = str(output_dir / "koff_trace.jsonl")
    return environment


def execute_parent(
    authorization: Mapping[str, Any],
    authorization_path: Path,
    output_dir: Path,
) -> None:
    """Consume the authorization with exactly one GPU-4 child launch."""
    from run_p4_b0_value_screen import (
        _preflight_gpu4_identity_and_idle,
        _preflight_inprocess_engine_core,
        _preflight_native_sampler,
    )

    child_env = _child_environment(authorization, output_dir)
    _preflight_native_sampler(child_env)
    _preflight_inprocess_engine_core(child_env)
    identity = _preflight_gpu4_identity_and_idle()
    _require(identity["gpu_uuid"] == GPU4_UUID, "GPU-4 preflight identity drifted")
    output_dir.mkdir(parents=False, exist_ok=False)
    _write_json_exclusive(
        output_dir / "preparation.json",
        {
            "authorization_path": str(authorization_path.relative_to(REPO_ROOT)),
            "authorization_sha256": _sha256(authorization_path),
            "output_dir": str(output_dir.relative_to(REPO_ROOT)),
            "physical_boot_count": 1,
            "scored": False,
        },
    )
    log_path = output_dir / "child.log"
    script_path = Path(__file__).resolve()
    with log_path.open("x", encoding="utf-8") as stream:
        subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--authorization",
                str(authorization_path),
                "--output-dir",
                str(output_dir),
                "--child",
            ],
            cwd=REPO_ROOT,
            env=child_env,
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    _require(
        (output_dir / "probe_result.json").is_file(),
        "authorized child exited without a probe result",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from validate_p4_b0_chunked_prefill_probe_authorization_v5 import (
        validate_authorization,
    )

    authorization_path = args.authorization.resolve()
    output_dir = args.output_dir.resolve()
    authorization = _load_json(authorization_path)
    validate_authorization(
        authorization,
        authorization_path=authorization_path,
        output_dir=output_dir,
        require_output_absent=not args.child,
    )
    if args.prepare_only:
        _require(not args.child, "child mode cannot be prepare-only")
        print(
            json.dumps(
                {
                    "status": "pass",
                    "gpu_executed": False,
                    "authorization_consumed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.child:
        _require(output_dir.is_dir(), "child output directory is missing")
        preparation = _load_json(output_dir / "preparation.json")
        _require(
            preparation.get("authorization_sha256") == _sha256(authorization_path),
            "child preparation does not bind the authorization",
        )
        run_child(authorization, output_dir)
    else:
        execute_parent(authorization, authorization_path, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
