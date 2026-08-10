#!/usr/bin/env python3
"""Reproduce and trace the V5 full-prefill capture-ingress rejection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from run_p4_b0_serving_chunked_prefill_diagnosis import (
    GPU_UUID,
    MODEL_SNAPSHOT,
    PHYSICAL_GPU,
    PROMPT_COUNTS,
    PROMPT_IDS,
    REPO_ROOT,
    ServingDiagnosisError,
    _child_environment,
    _load_prompts,
    _preflight,
)

PHASE_DIR = Path(__file__).resolve().parents[1]
AUTHORIZATION_PATH = (
    PHASE_DIR
    / "data"
    / "p4"
    / "p4_b0_full_prefill_ingress_diagnosis_authorization.json"
)
OUTPUT_PATH = PHASE_DIR / "data" / "p4" / "run_b0_full_prefill_ingress_diagnosis_v1"
BOOT_SPEC_PATH = (
    PHASE_DIR
    / "data"
    / "p4"
    / "run_b0_value_screen_v4"
    / "boot_specs"
    / "p4-b0-b1-p1-off.json"
)
PLAN_PATH = (
    PHASE_DIR
    / "data"
    / "p4"
    / "run_b0_value_screen_v4"
    / "plans"
    / "p4-b0-b1-p1-off.json"
)
PACKAGE_ID = "p4-b0-full-prefill-ingress-diagnosis-v1"
MAX_NUM_BATCHED_TOKENS = 114688
EFFECTIVE_SCHEDULER_BUDGET = 114656
GPU_MEMORY_UTILIZATION = 0.96
MAX_OUTPUT_TOKENS = 16
P4_EXCLUSIONS = {
    "prefill_or_mixed_batch",
    "preemption",
    "recomputation",
    "invalid_spec_tokens",
}


class FullPrefillIngressError(ServingDiagnosisError):
    """Raised when the exact V5 ingress diagnosis drifts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FullPrefillIngressError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FullPrefillIngressError(f"cannot load JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON artifact must be an object: {path}")
    return value


def _repo_path(relative_path: str) -> Path:
    path = (REPO_ROOT / relative_path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise FullPrefillIngressError(f"source escapes repository: {path}") from exc
    return path


def _file_ref(relative_path: str) -> dict[str, str]:
    path = _repo_path(relative_path)
    _require(path.is_file(), f"bound source is missing: {path}")
    return {
        "path": relative_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _expected_sources() -> dict[str, str]:
    return {
        "diagnostic_runner": (
            "research/97_composition_runtime/scripts/"
            "run_p4_b0_full_prefill_ingress_diagnosis.py"
        ),
        "diagnostic_helper": (
            "research/97_composition_runtime/scripts/"
            "run_p4_b0_serving_chunked_prefill_diagnosis.py"
        ),
        "v5_boot_spec": (
            "research/97_composition_runtime/data/p4/run_b0_value_screen_v4/"
            "boot_specs/p4-b0-b1-p1-off.json"
        ),
        "v5_capture_plan": (
            "research/97_composition_runtime/data/p4/run_b0_value_screen_v4/"
            "plans/p4-b0-b1-p1-off.json"
        ),
        "v5_failure": (
            "research/97_composition_runtime/data/p4/"
            "run_b0_value_screen_v4/failure.json"
        ),
        "prompt_manifest": (
            "research/97_composition_runtime/data/p4/p4_b0_prompt_manifest.json"
        ),
        "prompt_bundle": (
            "research/97_composition_runtime/data/p4/p4_b0_prompt_tokens.jsonl.gz"
        ),
        "scheduler": "vllm/v1/core/sched/scheduler.py",
        "koff_runtime": "vllm/v1/spec_decode/koff_runtime.py",
        "model_runner": "vllm/v1/worker/gpu_model_runner.py",
        "draft_model": "vllm/v1/spec_decode/draft_model.py",
        "draft_proposer": "vllm/v1/spec_decode/llm_base_proposer.py",
        "sampler_backend": "vllm/v1/sample/ops/topk_topp_sampler.py",
    }


def _expected_invocation() -> list[str]:
    return [
        ".venv/bin/python",
        _expected_sources()["diagnostic_runner"],
        "--authorization",
        (
            "research/97_composition_runtime/data/p4/"
            "p4_b0_full_prefill_ingress_diagnosis_authorization.json"
        ),
        "--output-dir",
        str(OUTPUT_PATH.relative_to(REPO_ROOT)),
    ]


def _expected_run() -> dict[str, Any]:
    return {
        "invocation": {
            "argv": _expected_invocation(),
            "output_dir": str(OUTPUT_PATH.relative_to(REPO_ROOT)),
            "create_new": True,
        },
        "gpu": {
            "physical_index": PHYSICAL_GPU,
            "uuid": GPU_UUID,
            "fallback_authorized": False,
        },
        "model": {
            "snapshot_path": str(MODEL_SNAPSHOT),
            "target_quantization": None,
            "target_kv_dtype": "bfloat16",
            "draft_weight_source": "target_alias",
            "draft_kv_source": "target_shared_cache",
        },
        "engine": {
            "max_model_len": 20480,
            "max_num_seqs": 32,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "effective_scheduler_token_budget": EFFECTIVE_SCHEDULER_BUDGET,
            "enable_chunked_prefill": True,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "async_scheduling": False,
            "enforce_eager": False,
            "prefix_caching": False,
            "native_sampler": True,
            "dynamic_k_schedule": [[1, 32, 0], [33, 33, 4]],
        },
        "workload": {
            "capture_id": "capture-b1-p1-off-r4-s0-r1",
            "prompt_record_ids": list(PROMPT_IDS),
            "prompt_token_counts": list(PROMPT_COUNTS),
            "total_prompt_tokens": sum(PROMPT_COUNTS),
            "max_output_tokens_per_request": MAX_OUTPUT_TOKENS,
            "temperature": 0.0,
            "ignore_eos": True,
            "generation_seed": 0,
        },
    }


def validate_authorization(
    authorization: Mapping[str, Any], *, require_output_absent: bool
) -> dict[str, Any]:
    """Validate the narrow one-boot ingress diagnosis authorization."""
    expected_fields = {
        "schema_version",
        "package_id",
        "date",
        "status",
        "source_artifacts",
        "run_contract",
        "diagnostic_contract",
        "decision",
        "claims",
        "authorizations",
        "execution_policy",
        "next_artifact",
    }
    _require(set(authorization) == expected_fields, "authorization fields drifted")
    _require(
        authorization.get("schema_version") == 1
        and authorization.get("package_id") == PACKAGE_ID
        and authorization.get("status")
        == "authorized_gpu4_non_scored_full_prefill_ingress_diagnosis_only",
        "wrong full-prefill ingress package",
    )
    sources = _expected_sources()
    _require(
        set(authorization["source_artifacts"]) == set(sources),
        "source closure drifted",
    )
    for role, path in sources.items():
        _require(
            authorization["source_artifacts"][role] == _file_ref(path),
            f"source hash drifted for {role}",
        )
    run = _expected_run()
    _require(authorization["run_contract"] == run, "run contract drifted")
    _require(
        authorization["diagnostic_contract"]
        == {
            "p4_capture_plan": str(PLAN_PATH.relative_to(REPO_ROOT)),
            "p4_capture_recorder_enabled": True,
            "passive_trace_enabled": True,
            "trace_flushes_before_recorder": True,
            "expected_recorder_disposition": "fail_closed_ineligible_event",
            "adaptation_allowed": False,
            "scoring_allowed": False,
        },
        "diagnostic contract drifted",
    )
    _require(
        authorization["decision"]
        == {
            "state": "approve",
            "scope": "gpu4_full_prefill_first_event_ingress_diagnosis_only",
            "invalidated_by": [
                "bound_source_hash_drift",
                "output_directory_exists",
                "gpu4_not_idle",
                "gpu_identity_drift",
                "engine_or_capture_plan_drift",
                "adaptation_or_scoring_enablement",
            ],
        },
        "decision boundary drifted",
    )
    _require(
        authorization["claims"]
        == {
            "v5_screen_passed": False,
            "v5_exclusion_identified": False,
            "performance_claim_allowed": False,
            "action_admitted": False,
        },
        "claims overstate evidence",
    )
    _require(
        authorization["authorizations"]
        == {
            "ingress_diagnosis": True,
            "value_screen": False,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "authority exceeds one ingress diagnosis",
    )
    _require(
        authorization["execution_policy"]
        == {
            "physical_boot_count": 1,
            "expected_complete_capture_count": 0,
            "retry_allowed": False,
            "partial_resume_allowed": False,
            "prior_output_reuse_allowed": False,
            "on_any_unexpected_failure": (
                "stop_preserve_trace_require_fresh_authorization"
            ),
        },
        "execution policy drifted",
    )
    _require(
        authorization["next_artifact"]
        == {
            "kind": "p4_b0_full_prefill_ingress_diagnosis",
            "may_score_value_screen": False,
            "may_authorize_v6": False,
            "may_authorize_p4a": False,
            "may_admit_action": False,
        },
        "post-run boundary drifted",
    )
    _require(BOOT_SPEC_PATH.is_file() and PLAN_PATH.is_file(), "V5 inputs missing")
    if require_output_absent:
        _require(not OUTPUT_PATH.exists(), "create-new output already exists")
    return run


def _child_env(trace_path: Path, capture_dir: Path) -> dict[str, str]:
    environment = _child_environment(trace_path)
    spec = _load_json(BOOT_SPEC_PATH)
    environment.update({key: str(value) for key, value in spec["environment"].items()})
    environment.update(
        {
            "VLLM_SELF_SPEC_KOFF_TRACE": str(trace_path.resolve()),
            "VLLM_SELF_SPEC_P4_CAPTURE_CONFIG": str(PLAN_PATH.resolve()),
            "VLLM_SELF_SPEC_P4_CAPTURE_OUTPUT": str(capture_dir.resolve()),
            "PHASE97_FULL_PREFILL_INGRESS_CHILD": "1",
        }
    )
    return environment


def _engine_args(spec: Mapping[str, Any]):
    from vllm import EngineArgs

    engine = spec["engine"]
    speculative_config = {
        "method": "draft_model",
        "model": str(MODEL_SNAPSHOT),
        "num_speculative_tokens": 4,
        "num_speculative_tokens_per_batch_size": spec["dynamic_k_schedule"],
        "draft_tensor_parallel_size": 1,
    }
    return EngineArgs(
        model=str(MODEL_SNAPSHOT),
        speculative_config=speculative_config,
        tensor_parallel_size=engine["tensor_parallel_size"],
        pipeline_parallel_size=engine["pipeline_parallel_size"],
        max_model_len=engine["max_model_len"],
        max_num_seqs=engine["max_num_seqs"],
        max_num_batched_tokens=engine["max_num_batched_tokens"],
        enable_chunked_prefill=engine["enable_chunked_prefill"],
        gpu_memory_utilization=engine["gpu_memory_utilization"],
        enable_prefix_caching=engine["prefix_caching"],
        async_scheduling=engine["async_scheduling"],
        enforce_eager=engine["enforce_eager"],
        enable_flashinfer_autotune=engine["flashinfer_autotune"],
        seed=engine["generation_seed"],
        disable_log_stats=True,
        generation_config="vllm",
    )


def _sampling_params():
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0,
        max_tokens=MAX_OUTPUT_TOKENS,
        ignore_eos=True,
        seed=0,
    )


def _trace_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _analyze(
    trace_path: Path,
    capture_dir: Path,
    observed_error: Mapping[str, str],
) -> dict[str, Any]:
    rows = _trace_rows(trace_path)
    headers = [row for row in rows if row.get("record_type") == "koff_runtime_header"]
    steps = [row for row in rows if row.get("record_type") == "koff_engine_step"]
    _require(len(headers) == 1 and len(steps) >= 2, "diagnostic trace is incomplete")
    _require(
        headers[0]["environment"]["max_num_batched_tokens"]
        == EFFECTIVE_SCHEDULER_BUDGET,
        "effective full-prefill budget drifted",
    )
    decode_steps = [
        step for step in steps if step.get("counters", {}).get("H_target_steps", 0) > 0
    ]
    _require(bool(decode_steps), "trace has no observed decode event")
    first_decode = decode_steps[0]
    p4_exclusions = [
        reason
        for reason in first_decode.get("exclusion_reasons", ())
        if reason in P4_EXCLUSIONS
    ]
    _require(bool(p4_exclusions), "first decode did not reproduce a P4 exclusion")
    placeholders = sorted(capture_dir.glob("*.json"))
    _require(len(placeholders) == 1, "recorder did not leave one fail-closed file")
    _require(placeholders[0].stat().st_size == 0, "placeholder unexpectedly has data")
    _require(
        placeholders[0].name == "capture-b1-p1-off-r4-s0-r1.json",
        "wrong capture cell was activated",
    )
    for step in steps:
        execution = step["execution"]
        _require(
            execution["shared_kv_layer_count"] == 36
            and execution["shared_kv_storage_alias_count"] == 36,
            "diagnosis lost shared target KV",
        )
        _require(
            execution["shared_weight_parameter_count"] == 291
            and execution["target_weight_version_id"]
            == execution["draft_weight_version_id"],
            "diagnosis lost target/draft weight identity",
        )
    diagnostic = first_decode["execution"]["diagnostic"]
    return {
        "schema_version": 1,
        "record_type": "p4_b0_full_prefill_ingress_diagnosis",
        "status": "pass",
        "scored": False,
        "decision": "v5_first_event_exclusion_identified",
        "observed_failure": dict(observed_error),
        "engine": {
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "effective_scheduler_token_budget": EFFECTIVE_SCHEDULER_BUDGET,
            "enable_chunked_prefill": True,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        },
        "trace": {
            "engine_step_count": len(steps),
            "first_decode_step_index": first_decode["engine_step_index"],
            "first_decode_exclusion_reasons": first_decode["exclusion_reasons"],
            "first_decode_p4_exclusions": p4_exclusions,
            "first_decode_request_ids": diagnostic["request_ids"],
            "first_decode_query_widths": diagnostic["num_scheduled_tokens"],
            "first_decode_computed_tokens": diagnostic["num_computed_tokens"],
            "first_decode_prompt_tokens": diagnostic["num_prompt_tokens"],
            "preemptions": first_decode["counters"]["preemptions"],
            "recomputed_tokens": first_decode["counters"]["recomputed_tokens"],
            "shared_target_kv_blocks": first_decode["resources"][
                "shared_target_kv_block_capacity"
            ],
        },
        "capture": {
            "complete_capture_count": 0,
            "empty_placeholder": str(placeholders[0].relative_to(REPO_ROOT)),
            "adapted_rounds_emitted": False,
            "score_emitted": False,
        },
        "invariants": {
            "passive_trace_flushed_before_recorder_failure": True,
            "shared_target_kv_layer_count": 36,
            "private_draft_kv_allocated": False,
            "target_draft_weight_alias_count": 291,
        },
        "claims": {
            "value_screen_repaired": False,
            "v6_authorized": False,
            "performance_claim_allowed": False,
            "action_admitted": False,
        },
    }


def _run_child(output_dir: Path) -> None:
    _require(
        os.environ.get("PHASE97_FULL_PREFILL_INGRESS_CHILD") == "1",
        "unauthorized child",
    )
    trace_path = output_dir / "koff_trace.jsonl"
    capture_dir = output_dir / "captures"
    result_path = output_dir / "diagnosis.json"
    _require(
        not trace_path.exists() and not result_path.exists(),
        "child output exists",
    )
    prompts = _load_prompts()
    spec = _load_json(BOOT_SPEC_PATH)
    from vllm import LLMEngine

    engine = LLMEngine.from_engine_args(_engine_args(spec))
    observed_error: dict[str, str] | None = None
    try:
        params = _sampling_params()
        for record_id in PROMPT_IDS:
            engine.add_request(
                record_id,
                {"prompt_token_ids": prompts[record_id]},
                params,
            )
        steps = 0
        while engine.has_unfinished_requests():
            engine.step()
            steps += 1
            _require(steps <= 8, "recorder did not fail at first decode ingress")
    except Exception as exc:
        observed_error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        engine.engine_core.shutdown()
    _require(
        observed_error is not None,
        "capture recorder unexpectedly accepted ingress",
    )
    result = _analyze(trace_path, capture_dir, observed_error)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _execute(authorization: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    run = validate_authorization(authorization, require_output_absent=True)
    trace_path = output_dir / "koff_trace.jsonl"
    capture_dir = output_dir / "captures"
    environment = _child_env(trace_path, capture_dir)
    preflight = _preflight(environment)
    output_dir.mkdir(parents=False, exist_ok=False)
    capture_dir.mkdir(exist_ok=False)
    preparation = {
        "schema_version": 1,
        "record_type": "p4_b0_full_prefill_ingress_diagnosis_preparation",
        "package_id": PACKAGE_ID,
        "scored": False,
        "gpu_executed": False,
        "preflight": preflight,
        "run_contract": run,
    }
    (output_dir / "preparation.json").write_text(
        json.dumps(preparation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--authorization",
                str(AUTHORIZATION_PATH.resolve()),
                "--output-dir",
                str(output_dir.resolve()),
                "--child",
            ],
            cwd=REPO_ROOT,
            env=environment,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        failure = {
            "schema_version": 1,
            "record_type": "p4_b0_full_prefill_ingress_diagnosis_failure",
            "scored": False,
            "returncode": exc.returncode,
            "preserve_without_retry": True,
        }
        (output_dir / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    return _load_json(output_dir / "diagnosis.json")


def parse_args() -> argparse.Namespace:
    """Parse the create-new ingress diagnosis interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    """Validate or execute one exact full-prefill ingress diagnosis."""
    args = parse_args()
    _require(
        args.authorization.resolve() == AUTHORIZATION_PATH.resolve(),
        "authorization path drifted",
    )
    _require(args.output_dir.resolve() == OUTPUT_PATH.resolve(), "output path drifted")
    authorization = _load_json(args.authorization)
    if args.child:
        validate_authorization(authorization, require_output_absent=False)
        _run_child(args.output_dir)
        return 0
    if args.validate_only:
        run = validate_authorization(authorization, require_output_absent=True)
        print(
            json.dumps(
                {
                    "status": "pass",
                    "gpu_executed": False,
                    "package_id": PACKAGE_ID,
                    "run_contract": run,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = _execute(authorization, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
