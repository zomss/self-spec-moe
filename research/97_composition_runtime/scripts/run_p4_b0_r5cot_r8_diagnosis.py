#!/usr/bin/env python3
"""Diagnose isolated R8 versus the R5cot-to-R8 K4 transition."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import traceback
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
AUTHORIZATION_PATH = (
    PHASE_DIR / "data" / "p4" / "p4_b0_r5cot_r8_diagnosis_authorization.json"
)
OUTPUT_DIR = PHASE_DIR / "data" / "p4" / "run_b0_r5cot_r8_diagnosis_v1"
PLAN_PATH = (
    PHASE_DIR
    / "data"
    / "p4"
    / "run_b0_value_screen_v9"
    / "plans"
    / "p4-b0-b1-p2-k4.json"
)
EXPECTED_PACKAGE_ID = "p4-b0-r5cot-r8-diagnosis-authorization-v1"
EXPECTED_FAILURE_MESSAGE = (
    "target-matching-k4 non-decode dispatch requires a positive draft step-0 "
    "query width and pure-prefill cohort arming"
)
MINIMUM_SHARED_KV_BLOCKS = 21682
CONFIGURED_MAX_NUM_BATCHED_TOKENS = 8192
EFFECTIVE_MAX_NUM_SCHEDULED_TOKENS = 8160
ACTION_ID = "target-matching-k4"
JOBS = {
    "isolated-r8": {
        "gpu": {
            "physical_index": 0,
            "uuid": "GPU-4938442e-5508-9249-0fa6-37baa1985703",
        },
        "cohorts": [
            {
                "label": "r8-first",
                "capture_id": "capture-b1-p2-k4-r8-s0-r1",
                "slice_start": 0,
                "slice_stop": 16,
            }
        ],
    },
    "r5cot-to-r8": {
        "gpu": {
            "physical_index": 1,
            "uuid": "GPU-ba39f4f0-61fe-34ca-c1af-ffe565b70923",
        },
        "cohorts": [
            {
                "label": "r5cot-last",
                "capture_id": "capture-b1-p2-k4-r5cot-s1-r4",
                "slice_start": 24,
                "slice_stop": 32,
            },
            {
                "label": "r8-first",
                "capture_id": "capture-b1-p2-k4-r8-s0-r1",
                "slice_start": 0,
                "slice_stop": 16,
            },
        ],
    },
}
REQUIRED_SOURCE_PATHS = {
    "capture_manifest": (
        "research/97_composition_runtime/data/p4/run_b0_value_screen_v9/"
        "capture_manifest.json"
    ),
    "capture_plan": (
        "research/97_composition_runtime/data/p4/run_b0_value_screen_v9/"
        "plans/p4-b0-b1-p2-k4.json"
    ),
    "capture_runner": (
        "research/97_composition_runtime/scripts/run_p4_b0_value_screen.py"
    ),
    "diagnosis_runner": (
        "research/97_composition_runtime/scripts/"
        "run_p4_b0_r5cot_r8_diagnosis.py"
    ),
    "diagnosis_tests": (
        "research/97_composition_runtime/tests/"
        "test_p4_b0_r5cot_r8_diagnosis.py"
    ),
    "draft_proposer": "vllm/v1/spec_decode/llm_base_proposer.py",
    "failed_attempt": (
        "research/97_composition_runtime/data/p4/run_b0_value_screen_v9/"
        "failure.json"
    ),
    "gpu_model_runner": "vllm/v1/worker/gpu_model_runner.py",
    "gpu_worker": "vllm/v1/worker/gpu_worker.py",
    "koff_runtime": "vllm/v1/spec_decode/koff_runtime.py",
    "preservation_script": (
        "research/97_composition_runtime/scripts/"
        "preserve_p4_b0_value_screen_v10_attempt.py"
    ),
    "prompt_bundle": (
        "research/97_composition_runtime/data/p4/p4_b0_prompt_tokens.jsonl.gz"
    ),
    "prompt_manifest": (
        "research/97_composition_runtime/data/p4/p4_b0_prompt_manifest.json"
    ),
    "scheduler": "vllm/v1/core/sched/scheduler.py",
    "v10_authorization": (
        "research/97_composition_runtime/data/p4/"
        "p4_b0_run_authorization_v10.json"
    ),
}


class P4TransitionDiagnosisError(RuntimeError):
    """Raised when the diagnosis cannot preserve its reviewed boundary."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise P4TransitionDiagnosisError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise P4TransitionDiagnosisError(f"cannot load {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON artifact is not an object: {path}")
    return value


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise P4TransitionDiagnosisError(f"refusing to overwrite {path}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def _reference(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _exception_record(exc: BaseException) -> dict[str, Any]:
    formatted = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(formatted),
    }


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError, RuntimeError):
            pass
    return str(value)


def validate_authorization(
    authorization: Mapping[str, Any],
    *,
    authorization_path: Path,
    output_dir: Path,
    require_output_absent: bool,
) -> None:
    """Validate the source-bound, create-only two-GPU diagnosis package."""
    _require(
        authorization_path == AUTHORIZATION_PATH.resolve(),
        "diagnosis must use the reviewed authorization path",
    )
    _require(
        output_dir == OUTPUT_DIR.resolve(),
        "diagnosis output differs from the reviewed create-only path",
    )
    if require_output_absent:
        _require(not output_dir.exists(), "diagnosis output already exists")
    _require(
        authorization.get("schema_version") == 1
        and authorization.get("package_id") == EXPECTED_PACKAGE_ID
        and authorization.get("status")
        == "authorized_gpu0_gpu1_non_scored_transition_diagnosis_only",
        "diagnosis package identity drifted",
    )
    contract = authorization.get("run_contract")
    _require(isinstance(contract, Mapping), "diagnosis run contract is missing")
    _require(contract.get("scored") is False, "diagnosis cannot be scored")
    _require(
        contract.get("output_dir") == _relative(OUTPUT_DIR),
        "diagnosis output contract drifted",
    )
    _require(contract.get("jobs") == JOBS, "diagnosis job matrix drifted")
    engine = contract.get("engine")
    _require(isinstance(engine, Mapping), "diagnosis engine contract is missing")
    _require(
        engine.get("max_num_batched_tokens")
        == CONFIGURED_MAX_NUM_BATCHED_TOKENS
        and engine.get("enable_chunked_prefill") is True
        and engine.get("num_speculative_tokens") == 4
        and engine.get("max_num_seqs") == 32
        and engine.get("gpu_memory_utilization") == 0.9,
        "diagnosis engine geometry drifted",
    )
    action = contract.get("action")
    _require(
        isinstance(action, Mapping)
        and action.get("action_id") == ACTION_ID
        and action.get("dynamic_k_schedule") == [[1, 32, 4]],
        "diagnosis action drifted",
    )
    policy = authorization.get("execution_policy")
    _require(isinstance(policy, Mapping), "diagnosis execution policy is missing")
    _require(
        policy.get("create_new_output_only") is True
        and policy.get("concurrent_physical_boots") == 2
        and policy.get("score_output") is False
        and policy.get("retry_allowed") is False
        and policy.get("resume_allowed") is False
        and policy.get("v10_output_reuse_allowed") is False
        and policy.get("downstream_authority_granted") is False,
        "diagnosis execution policy drifted",
    )
    sources = authorization.get("source_artifacts")
    _require(isinstance(sources, Mapping), "source bindings are missing")
    _require(
        set(sources) == set(REQUIRED_SOURCE_PATHS),
        "diagnosis source binding set drifted",
    )
    for key, expected_path in REQUIRED_SOURCE_PATHS.items():
        reference = sources.get(key)
        _require(isinstance(reference, Mapping), f"source binding {key} is missing")
        _require(
            reference.get("path") == expected_path,
            f"source path {key} drifted",
        )
        path = REPO_ROOT / expected_path
        _require(path.is_file(), f"source artifact is missing: {expected_path}")
        _require(
            reference.get("sha256") == _sha256(path),
            f"source artifact drifted: {expected_path}",
        )
    failure = _load_json(REPO_ROOT / REQUIRED_SOURCE_PATHS["failed_attempt"])
    manifest = _load_json(REPO_ROOT / REQUIRED_SOURCE_PATHS["capture_manifest"])
    _require(
        failure.get("disposition", {}).get("v10_consumed") is True
        and failure.get("disposition", {}).get("scoring_allowed") is False,
        "V10 consumed disposition drifted",
    )
    _require(
        manifest.get("counts", {}).get("complete_captures") == 72
        and manifest.get("counts", {}).get("empty_placeholders") == 1,
        "V10 capture boundary drifted",
    )


class _DiagnosticRecorder:
    capture_id = "p4-b0-r5cot-r8-non-scored-diagnosis"

    def __init__(self) -> None:
        self.event_count = 0
        self.events_by_action: dict[str, int] = {}
        self.last_event: dict[str, Any] | None = None
        self.closed = False

    def record(self, event: Mapping[str, Any], _evidence: Any) -> None:
        self.event_count += 1
        action_id = str(event.get("action_id"))
        self.events_by_action[action_id] = self.events_by_action.get(action_id, 0) + 1
        self.last_event = {
            "engine_step_index": event.get("engine_step_index"),
            "action_id": event.get("action_id"),
            "pure_decode": event.get("pure_decode"),
            "request_ids": [
                row.get("request_id") for row in event.get("request_steps", ())
            ],
            "quality": event.get("quality"),
        }

    def close(self) -> None:
        self.closed = True

    def summary(self) -> dict[str, Any]:
        return {
            "event_count": self.event_count,
            "events_by_action": dict(sorted(self.events_by_action.items())),
            "last_event": self.last_event,
            "closed": self.closed,
        }


def _case_cohorts(
    authorization: Mapping[str, Any], case_id: str
) -> list[tuple[dict[str, Any], tuple[str, ...], str]]:
    plan = _load_json(PLAN_PATH)
    cells = {
        cell["capture_id"]: cell
        for cell in plan.get("cells", ())
        if isinstance(cell, dict)
    }
    jobs = authorization["run_contract"]["jobs"]
    job = jobs[case_id]
    result = []
    for cohort in job["cohorts"]:
        capture_id = cohort["capture_id"]
        _require(capture_id in cells, f"capture cell is missing: {capture_id}")
        cell = cells[capture_id]
        prompt_ids = cell["generation"]["prompt_record_ids"]
        selected = tuple(
            prompt_ids[cohort["slice_start"] : cohort["slice_stop"]]
        )
        expected_count = cohort["slice_stop"] - cohort["slice_start"]
        _require(
            len(selected) == expected_count and len(set(selected)) == len(selected),
            f"cohort slice is malformed: {cohort['label']}",
        )
        result.append((cell, selected, cohort["label"]))
    return result


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
    _require(isinstance(client, InprocClient), "diagnosis did not use InprocClient")
    core = client.engine_core
    executor = core.model_executor
    wrapper = getattr(executor, "driver_worker", None)
    worker = None if wrapper is None else getattr(wrapper, "worker", None)
    _require(worker is not None, "diagnosis cannot inspect the GPU worker")
    return core.scheduler, worker


def _resource_evidence(scheduler: Any, worker: Any) -> dict[str, Any]:
    capacity = scheduler.kv_cache_manager.block_pool.num_gpu_blocks
    configured_budget = scheduler.scheduler_config.max_num_batched_tokens
    effective_budget = scheduler.max_num_scheduled_tokens
    _require(
        configured_budget == CONFIGURED_MAX_NUM_BATCHED_TOKENS,
        "configured scheduler budget drifted",
    )
    _require(
        effective_budget == EFFECTIVE_MAX_NUM_SCHEDULED_TOKENS,
        "effective scheduler budget drifted",
    )
    _require(
        type(capacity) is int and capacity >= MINIMUM_SHARED_KV_BLOCKS,
        "shared target-KV capacity is below the registered floor",
    )
    return {
        "configured_max_num_batched_tokens": configured_budget,
        "effective_max_num_scheduled_tokens": effective_budget,
        "shared_target_kv_block_capacity": capacity,
        "minimum_shared_target_kv_blocks": MINIMUM_SHARED_KV_BLOCKS,
        "actual_cuda_graph_memory_bytes": getattr(
            worker, "cuda_graph_memory_bytes", None
        ),
        "estimated_cuda_graph_memory_bytes": getattr(
            worker, "cudagraph_memory_estimate", None
        ),
    }


def _request_id(value: Any) -> str | None:
    for name in ("req_id", "request_id"):
        candidate = getattr(value, name, None)
        if isinstance(candidate, str):
            return candidate
    if isinstance(value, Mapping):
        for name in ("req_id", "request_id"):
            candidate = value.get(name)
            if isinstance(candidate, str):
                return candidate
    return None


def _scheduler_snapshot(output: Any) -> dict[str, Any]:
    metadata = getattr(output, "koff_runtime", None)
    return {
        "total_num_scheduled_tokens": getattr(
            output, "total_num_scheduled_tokens", None
        ),
        "num_scheduled_tokens": dict(
            getattr(output, "num_scheduled_tokens", {})
        ),
        "new_request_ids": [
            _request_id(row) for row in getattr(output, "scheduled_new_reqs", ())
        ],
        "cached_request_ids": [
            _request_id(row) for row in getattr(output, "scheduled_cached_reqs", ())
        ],
        "finished_req_ids": sorted(getattr(output, "finished_req_ids", ())),
        "koff_runtime": (
            dataclasses.asdict(metadata) if dataclasses.is_dataclass(metadata) else None
        ),
    }


def _install_instrumentation(
    scheduler: Any, worker: Any, state: dict[str, Any]
) -> None:
    import vllm.v1.worker.gpu_model_runner as gpu_model_runner

    scheduler_snapshots: deque[dict[str, Any]] = deque(maxlen=8)
    state["scheduler_snapshots"] = scheduler_snapshots
    original_schedule = scheduler.schedule

    def schedule_wrapper(*args: Any, **kwargs: Any) -> Any:
        output = original_schedule(*args, **kwargs)
        scheduler_snapshots.append(_scheduler_snapshot(output))
        return output

    scheduler.schedule = schedule_wrapper

    original_make_evidence = gpu_model_runner.make_runner_evidence

    def make_evidence_wrapper(*args: Any, **kwargs: Any) -> Any:
        metadata = kwargs.get("metadata")
        output = kwargs.get("output")
        state["runner_evidence_attempt"] = {
            "metadata": (
                dataclasses.asdict(metadata)
                if dataclasses.is_dataclass(metadata)
                else None
            ),
            "proposal_called": kwargs.get("proposal_called"),
            "draft_step0_query_width": kwargs.get("draft_step0_query_width"),
            "draft_step0_runtime_mode": kwargs.get("draft_step0_runtime_mode"),
            "draft_chain_runtime_mode": kwargs.get("draft_chain_runtime_mode"),
            "draft_output_shape": list(getattr(output, "shape", ())),
        }
        try:
            return original_make_evidence(*args, **kwargs)
        except Exception as exc:
            state["runner_evidence_exception"] = _exception_record(exc)
            raise

    gpu_model_runner.make_runner_evidence = make_evidence_wrapper

    drafter = worker.model_runner.drafter
    original_set_inputs = drafter.set_inputs_first_pass

    def set_inputs_wrapper(*args: Any, **kwargs: Any) -> Any:
        result = original_set_inputs(*args, **kwargs)
        num_tokens, _indices, metadata = result
        batch_size = metadata.batch_size()
        state["draft_step0_input"] = {
            "num_tokens": int(num_tokens),
            "batch_size": int(batch_size),
            "num_tokens_mod_batch_size": (
                int(num_tokens) % int(batch_size) if batch_size else None
            ),
            "uniform_query_width": (
                int(num_tokens) // int(batch_size)
                if batch_size and num_tokens % batch_size == 0
                else None
            ),
            "max_query_len": _json_scalar(metadata.max_query_len),
            "num_actual_tokens": _json_scalar(metadata.num_actual_tokens),
        }
        return result

    drafter.set_inputs_first_pass = set_inputs_wrapper
    state["drafter"] = drafter


def _trace_record_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def run_child_case(
    authorization: Mapping[str, Any], case_id: str, case_dir: Path
) -> None:
    """Run one non-scored diagnosis case and preserve success or failure."""
    from run_p4_b0_value_screen import (
        _engine_args,
        _load_prompt_rows,
        _run_prompt_chunk,
    )

    from vllm import LLMEngine

    trace_path = case_dir / "koff_trace.jsonl"
    _require(
        os.environ.get("VLLM_SELF_SPEC_KOFF_TRACE") == str(trace_path),
        "child K/OFF trace path differs from its case output",
    )
    _manifest, prompt_rows = _load_prompt_rows()
    cohorts = _case_cohorts(authorization, case_id)
    engine = None
    scheduler = None
    worker = None
    recorder = _DiagnosticRecorder()
    state: dict[str, Any] = {}
    completed_cohorts: list[str] = []
    active_stage = "engine_initialization"
    primary_exception = None
    shutdown_exception = None
    resources = None
    try:
        engine = LLMEngine.from_engine_args(_engine_args(_engine_spec(authorization)))
        scheduler, worker = _runtime_handles(engine)
        scheduler._p4_capture_recorder = recorder
        resources = _resource_evidence(scheduler, worker)
        _install_instrumentation(scheduler, worker, state)
        for cell, request_ids, label in cohorts:
            active_stage = label
            _run_prompt_chunk(engine, cell, request_ids, prompt_rows)
            completed_cohorts.append(label)
        active_stage = "complete"
    except Exception as exc:
        primary_exception = _exception_record(exc)
    finally:
        drafter = state.pop("drafter", None)
        if drafter is not None:
            state["drafter_post_step"] = {
                "last_step0_query_width": getattr(
                    drafter, "_last_step0_query_width", None
                ),
                "last_step0_runtime_mode": getattr(
                    drafter, "_last_step0_runtime_mode", None
                ),
                "last_chain_runtime_mode": getattr(
                    drafter, "_last_chain_runtime_mode", None
                ),
            }
        if engine is not None:
            try:
                engine.engine_core.shutdown()
            except Exception as exc:
                shutdown_exception = _exception_record(exc)

    snapshots = state.get("scheduler_snapshots")
    if isinstance(snapshots, deque):
        state["scheduler_snapshots"] = list(snapshots)
    expected_failure = (
        primary_exception is not None
        and primary_exception["type"] == "KOffRuntimeError"
        and primary_exception["message"] == EXPECTED_FAILURE_MESSAGE
        and active_stage == "r8-first"
    )
    if expected_failure:
        status = "observed_expected_r8_invariant_failure"
    elif primary_exception is None:
        status = "completed_without_invariant_failure"
    else:
        status = "inconclusive_unexpected_failure"
    job = authorization["run_contract"]["jobs"][case_id]
    result = {
        "schema_version": 1,
        "artifact_id": f"p4-b0-r5cot-r8-diagnosis-{case_id}",
        "status": status,
        "scored": False,
        "authorization_consumed": True,
        "authorization": _reference(AUTHORIZATION_PATH),
        "case_id": case_id,
        "gpu": job["gpu"],
        "active_stage_at_exit": active_stage,
        "completed_cohorts": completed_cohorts,
        "cohorts": job["cohorts"],
        "resources": resources,
        "primary_exception": primary_exception,
        "shutdown_exception": shutdown_exception,
        "instrumentation": state,
        "recorder": recorder.summary(),
        "trace": {
            "path": _relative(trace_path),
            "present": trace_path.is_file(),
            "record_count": _trace_record_count(trace_path),
            "sha256": _sha256(trace_path) if trace_path.is_file() else None,
        },
        "claims": {
            "expected_r8_invariant_failure_observed": expected_failure,
            "performance_claim_allowed": False,
            "score_eligible": False,
        },
        "authorizations": {
            "value_screen_retry": False,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
    }
    _write_exclusive(case_dir / "case_result.json", result)


def classify_results(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Classify whether R8 itself or transition history causes the failure."""
    isolated = results["isolated-r8"]
    transition = results["r5cot-to-r8"]
    isolated_expected = (
        isolated.get("status") == "observed_expected_r8_invariant_failure"
    )
    transition_expected = (
        transition.get("status") == "observed_expected_r8_invariant_failure"
    )
    unexpected = any(
        result.get("status") == "inconclusive_unexpected_failure"
        for result in results.values()
    )
    if unexpected:
        conclusion = "inconclusive_unexpected_execution_failure"
    elif isolated_expected and transition_expected:
        conclusion = "r8_variable_width_prefill_evidence_bug_not_history_contamination"
    elif not isolated_expected and transition_expected:
        conclusion = "r5cot_to_r8_transition_history_contamination"
    elif not isolated_expected and not transition_expected:
        conclusion = "original_failure_not_reproduced"
    else:
        conclusion = "isolated_r8_failure_with_nonreproducing_transition"
    return {
        "isolated_r8_expected_failure": isolated_expected,
        "transition_expected_failure": transition_expected,
        "conclusion": conclusion,
    }


def _child_environment(
    authorization: Mapping[str, Any], case_id: str, case_dir: Path
) -> dict[str, str]:
    from run_p4_b0_value_screen import _boot_child_environment

    contract = authorization["run_contract"]
    environment = dict(contract["environment"])
    gpu = contract["jobs"][case_id]["gpu"]
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu["physical_index"])
    child_environment = _boot_child_environment(environment)
    child_environment["VLLM_SELF_SPEC_KOFF_TRACE"] = str(
        case_dir / "koff_trace.jsonl"
    )
    return child_environment


def _gpu_inventory() -> dict[int, dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.used",
                "--format=csv,noheader,nounits",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise P4TransitionDiagnosisError(f"cannot inventory GPUs: {exc}") from exc
    inventory = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        _require(len(fields) == 4, "unexpected nvidia-smi GPU inventory")
        inventory[int(fields[0])] = {
            "physical_index": int(fields[0]),
            "uuid": fields[1],
            "name": fields[2],
            "memory_used_mib": int(fields[3]),
        }
    return inventory


def _active_compute_processes() -> dict[str, list[dict[str, Any]]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise P4TransitionDiagnosisError(
            f"cannot inspect GPU processes: {exc}"
        ) from exc
    processes: dict[str, list[dict[str, Any]]] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        _require(len(fields) == 4, "unexpected nvidia-smi process inventory")
        processes.setdefault(fields[0], []).append(
            {
                "pid": int(fields[1]),
                "process_name": fields[2],
                "used_memory_mib": int(fields[3]),
            }
        )
    return processes


def _preflight_gpus(authorization: Mapping[str, Any]) -> list[dict[str, Any]]:
    inventory = _gpu_inventory()
    processes = _active_compute_processes()
    evidence = []
    for case_id, job in authorization["run_contract"]["jobs"].items():
        expected = job["gpu"]
        index = expected["physical_index"]
        _require(index in inventory, f"physical GPU {index} is missing")
        observed = inventory[index]
        _require(
            observed["uuid"] == expected["uuid"],
            f"physical GPU {index} UUID drifted",
        )
        active = processes.get(observed["uuid"], [])
        _require(not active, f"physical GPU {index} has active processes: {active}")
        evidence.append({"case_id": case_id, **observed, "active_processes": []})
    return evidence


def execute_parent(
    authorization: Mapping[str, Any],
    authorization_path: Path,
    output_dir: Path,
) -> None:
    """Launch both source-bound cases concurrently on physical GPUs 0 and 1."""
    from run_p4_b0_value_screen import (
        _preflight_inprocess_engine_core,
        _preflight_native_sampler,
    )

    child_environments = {}
    for case_id in JOBS:
        placeholder = output_dir / case_id
        environment = _child_environment(authorization, case_id, placeholder)
        _preflight_native_sampler(environment)
        _preflight_inprocess_engine_core(environment)
        child_environments[case_id] = environment
    gpu_evidence = _preflight_gpus(authorization)

    output_dir.mkdir(parents=False, exist_ok=False)
    for case_id in JOBS:
        (output_dir / case_id).mkdir(exist_ok=False)
    _write_exclusive(
        output_dir / "preparation.json",
        {
            "schema_version": 1,
            "record_type": "p4_b0_r5cot_r8_diagnosis_preparation",
            "authorization": _reference(authorization_path),
            "output_dir": _relative(output_dir),
            "gpu_preflight": gpu_evidence,
            "concurrent_physical_boots": 2,
            "scored": False,
            "v10_output_reused": False,
        },
    )

    processes: dict[str, tuple[subprocess.Popen[str], Any]] = {}
    script_path = Path(__file__).resolve()
    for case_id, environment in child_environments.items():
        case_dir = output_dir / case_id
        _write_exclusive(
            case_dir / "preparation.json",
            {
                "authorization": _reference(authorization_path),
                "case_id": case_id,
                "gpu": JOBS[case_id]["gpu"],
                "scored": False,
            },
        )
        log_stream = (case_dir / "child.log").open("x", encoding="utf-8")
        process = subprocess.Popen(
            [
                sys.executable,
                str(script_path),
                "--authorization",
                str(authorization_path),
                "--output-dir",
                str(output_dir),
                "--child-case",
                case_id,
            ],
            cwd=REPO_ROOT,
            env=environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes[case_id] = (process, log_stream)

    return_codes = {}
    for case_id, (process, log_stream) in processes.items():
        return_codes[case_id] = process.wait()
        log_stream.close()
    _require(
        all(code == 0 for code in return_codes.values()),
        f"diagnosis child process failed: {return_codes}",
    )
    results = {
        case_id: _load_json(output_dir / case_id / "case_result.json")
        for case_id in JOBS
    }
    classification = classify_results(results)
    diagnosis = {
        "schema_version": 1,
        "artifact_id": "p4-b0-r5cot-r8-transition-diagnosis-v1",
        "status": "complete",
        "scored": False,
        "authorization_consumed": True,
        "authorization": _reference(authorization_path),
        "failed_attempt": _reference(
            REPO_ROOT / REQUIRED_SOURCE_PATHS["failed_attempt"]
        ),
        "case_results": {
            case_id: _reference(output_dir / case_id / "case_result.json")
            for case_id in JOBS
        },
        "classification": classification,
        "claims": {
            "diagnostic_only": True,
            "performance_claim_allowed": False,
            "value_screen_retry_authorized": False,
            "action_admission_authorized": False,
        },
        "next_action": (
            "Repair draft step-0 evidence so variable-width pure-prefill records "
            "a positive per-request query-width fact without claiming uniform "
            "decode width, then add isolated and transition regressions."
            if classification["conclusion"]
            == "r8_variable_width_prefill_evidence_bug_not_history_contamination"
            else "Review the two case artifacts before changing runtime code."
        ),
    }
    _write_exclusive(output_dir / "diagnosis.json", diagnosis)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--child-case", choices=tuple(JOBS), help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    authorization_path = args.authorization.resolve()
    output_dir = args.output_dir.resolve()
    authorization = _load_json(authorization_path)
    validate_authorization(
        authorization,
        authorization_path=authorization_path,
        output_dir=output_dir,
        require_output_absent=args.child_case is None,
    )
    if args.prepare_only:
        _require(args.child_case is None, "child cannot be prepare-only")
        print(
            json.dumps(
                {
                    "status": "pass",
                    "gpu_executed": False,
                    "authorization_consumed": False,
                    "jobs": list(JOBS),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.child_case is not None:
        case_dir = output_dir / args.child_case
        _require(case_dir.is_dir(), "child case output is missing")
        preparation = _load_json(case_dir / "preparation.json")
        _require(
            preparation.get("authorization", {}).get("sha256")
            == _sha256(authorization_path),
            "child preparation does not bind the authorization",
        )
        run_child_case(authorization, args.child_case, case_dir)
    else:
        execute_parent(authorization, authorization_path, output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except P4TransitionDiagnosisError as exc:
        print(f"P4 transition diagnosis refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
