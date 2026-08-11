#!/usr/bin/env python3
"""Prove atomic full-prefill ingress with the in-process EngineCore."""

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

from run_p4_b0_full_prefill_ingress_diagnosis import (
    BOOT_SPEC_PATH,
    EFFECTIVE_SCHEDULER_BUDGET,
    GPU_MEMORY_UTILIZATION,
    MAX_NUM_BATCHED_TOKENS,
    _engine_args,
    _sampling_params,
)
from run_p4_b0_serving_chunked_prefill_diagnosis import (
    GPU_UUID,
    MODEL_SNAPSHOT,
    PHYSICAL_GPU,
    PROMPT_COUNTS,
    PROMPT_IDS,
    REPO_ROOT,
    _child_environment,
    _load_prompts,
    _preflight,
)

from vllm.v1.spec_decode.koff_runtime import (
    KOffRuntimeError,
    canonicalize_p4_request_ids,
)

PHASE_DIR = Path(__file__).resolve().parents[1]
AUTHORIZATION_PATH = (
    PHASE_DIR / "data" / "p4" / "p4_b0_atomic_ingress_proof_authorization_v2.json"
)
OUTPUT_PATH = PHASE_DIR / "data" / "p4" / "run_b0_atomic_ingress_proof_v2"
PRIOR_DIAGNOSIS_PATH = (
    PHASE_DIR
    / "data"
    / "p4"
    / "run_b0_full_prefill_ingress_diagnosis_v1"
    / "diagnosis.json"
)
PACKAGE_ID = "p4-b0-atomic-ingress-proof-v2"
ENGINE_CORE_CLASS = "InprocClient"
STEP_COUNT = 2
MINIMUM_SHARED_KV_BLOCKS = 21682


class AtomicIngressProofError(RuntimeError):
    """Raised when the atomic-ingress proof loses its narrow contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AtomicIngressProofError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AtomicIngressProofError(f"cannot load JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON artifact must be an object: {path}")
    return value


def _repo_path(relative_path: str) -> Path:
    path = (REPO_ROOT / relative_path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise AtomicIngressProofError(f"source escapes repository: {path}") from exc
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
        "proof_runner": (
            "research/97_composition_runtime/scripts/run_p4_b0_atomic_ingress_proof.py"
        ),
        "proof_test": (
            "research/97_composition_runtime/tests/test_p4_b0_atomic_ingress_proof.py"
        ),
        "full_prefill_helper": (
            "research/97_composition_runtime/scripts/"
            "run_p4_b0_full_prefill_ingress_diagnosis.py"
        ),
        "serving_helper": (
            "research/97_composition_runtime/scripts/"
            "run_p4_b0_serving_chunked_prefill_diagnosis.py"
        ),
        "v5_boot_spec": (
            "research/97_composition_runtime/data/p4/run_b0_value_screen_v4/"
            "boot_specs/p4-b0-b1-p1-off.json"
        ),
        "prior_authorization": (
            "research/97_composition_runtime/data/p4/"
            "p4_b0_full_prefill_ingress_diagnosis_authorization.json"
        ),
        "prior_diagnosis": (
            "research/97_composition_runtime/data/p4/"
            "run_b0_full_prefill_ingress_diagnosis_v1/diagnosis.json"
        ),
        "prior_trace": (
            "research/97_composition_runtime/data/p4/"
            "run_b0_full_prefill_ingress_diagnosis_v1/koff_trace.jsonl"
        ),
        "prior_proof_authorization": (
            "research/97_composition_runtime/data/p4/"
            "p4_b0_atomic_ingress_proof_authorization.json"
        ),
        "prior_proof_preparation": (
            "research/97_composition_runtime/data/p4/"
            "run_b0_atomic_ingress_proof_v1/preparation.json"
        ),
        "prior_proof_failure": (
            "research/97_composition_runtime/data/p4/"
            "run_b0_atomic_ingress_proof_v1/failure.json"
        ),
        "prior_proof_failure_diagnosis": (
            "research/97_composition_runtime/data/p4/"
            "p4_b0_atomic_ingress_proof_v1_failure_diagnosis.json"
        ),
        "prompt_manifest": (
            "research/97_composition_runtime/data/p4/p4_b0_prompt_manifest.json"
        ),
        "prompt_bundle": (
            "research/97_composition_runtime/data/p4/p4_b0_prompt_tokens.jsonl.gz"
        ),
        "environment": "vllm/envs.py",
        "engine_args": "vllm/engine/arg_utils.py",
        "llm_engine": "vllm/v1/engine/llm_engine.py",
        "engine_core_client": "vllm/v1/engine/core_client.py",
        "uniproc_executor": "vllm/v1/executor/uniproc_executor.py",
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
        _expected_sources()["proof_runner"],
        "--authorization",
        (
            "research/97_composition_runtime/data/p4/"
            "p4_b0_atomic_ingress_proof_authorization_v2.json"
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
            "engine_core_mode": "in_process",
            "engine_core_class": ENGINE_CORE_CLASS,
            "v1_multiprocessing": False,
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
            "prompt_record_ids": list(PROMPT_IDS),
            "prompt_token_counts": list(PROMPT_COUNTS),
            "total_prompt_tokens": sum(PROMPT_COUNTS),
            "max_output_tokens_per_request": 16,
            "temperature": 0.0,
            "ignore_eos": True,
            "generation_seed": 0,
            "engine_steps": STEP_COUNT,
        },
    }


def validate_authorization(
    authorization: Mapping[str, Any], *, require_output_absent: bool
) -> dict[str, Any]:
    """Validate one source-bound, non-scored atomic-ingress proof."""
    expected_fields = {
        "schema_version",
        "package_id",
        "date",
        "status",
        "prior_diagnosis",
        "prior_attempt",
        "source_artifacts",
        "run_contract",
        "proof_contract",
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
        == "authorized_gpu4_non_scored_atomic_ingress_proof_v2_only",
        "wrong atomic-ingress proof package",
    )
    _require(
        authorization["prior_diagnosis"]
        == {
            "authorization": _file_ref(_expected_sources()["prior_authorization"]),
            "diagnosis": _file_ref(_expected_sources()["prior_diagnosis"]),
            "disposition": "mixed_ingress_identified_screen_unscored",
        },
        "prior diagnosis binding drifted",
    )
    _require(
        authorization["prior_attempt"]
        == {
            "authorization": _file_ref(
                _expected_sources()["prior_proof_authorization"]
            ),
            "preparation": _file_ref(_expected_sources()["prior_proof_preparation"]),
            "failure": _file_ref(_expected_sources()["prior_proof_failure"]),
            "failure_diagnosis": _file_ref(
                _expected_sources()["prior_proof_failure_diagnosis"]
            ),
            "disposition": "v1_consumed_zero_steps_no_trace_no_score",
        },
        "prior proof attempt binding drifted",
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
        authorization["proof_contract"]
        == {
            "recording": "append_only_non_scored_koff_trace",
            "p4_capture_recorder_enabled": False,
            "request_admission": "queue_all_before_first_engine_step",
            "expected_step_count": STEP_COUNT,
            "expected_initial_step": "eight_request_pure_prefill",
            "expected_first_decode_step": "eight_request_pure_decode",
            "expected_first_decode_query_widths": [1] * len(PROMPT_IDS),
            "expected_first_decode_exclusions": [],
            "shared_target_kv_required": True,
            "private_draft_kv_allowed": False,
            "adaptation_allowed": False,
            "scoring_allowed": False,
        },
        "proof contract drifted",
    )
    _require(
        authorization["decision"]
        == {
            "state": "approve",
            "scope": "gpu4_inprocess_atomic_ingress_proof_only",
            "invalidated_by": [
                "bound_source_hash_drift",
                "output_directory_exists",
                "gpu4_not_idle",
                "gpu_identity_drift",
                "multiprocess_engine_core",
                "engine_or_workload_drift",
                "capture_adaptation_or_scoring_enablement",
            ],
        },
        "decision boundary drifted",
    )
    _require(
        authorization["claims"]
        == {
            "atomic_ingress_proven": False,
            "value_screen_repaired": False,
            "v6_authorized": False,
            "performance_claim_allowed": False,
            "action_admitted": False,
        },
        "claims overstate evidence",
    )
    _require(
        authorization["authorizations"]
        == {
            "atomic_ingress_proof": True,
            "value_screen": False,
            "v6_execution": False,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "authority exceeds one atomic-ingress proof",
    )
    _require(
        authorization["execution_policy"]
        == {
            "physical_boot_count": 1,
            "engine_step_count": STEP_COUNT,
            "expected_complete_capture_count": 0,
            "retry_allowed": False,
            "partial_resume_allowed": False,
            "prior_output_reuse_allowed": False,
            "on_any_failure": "stop_preserve_trace_require_fresh_authorization",
        },
        "execution policy drifted",
    )
    _require(
        authorization["next_artifact"]
        == {
            "kind": "p4_b0_atomic_ingress_proof",
            "may_request_v6_review_on_pass": True,
            "may_authorize_v6": False,
            "may_score_value_screen": False,
            "may_authorize_p4a": False,
            "may_admit_action": False,
        },
        "post-run boundary drifted",
    )
    _require(MODEL_SNAPSHOT.is_dir(), "bound model snapshot is missing")
    _require(BOOT_SPEC_PATH.is_file(), "bound V5 boot specification is missing")
    _require(PRIOR_DIAGNOSIS_PATH.is_file(), "prior diagnosis is missing")
    if require_output_absent:
        _require(not OUTPUT_PATH.exists(), "create-new output already exists")
    return run


def _child_env(trace_path: Path) -> dict[str, str]:
    environment = _child_environment(trace_path)
    spec = _load_json(BOOT_SPEC_PATH)
    environment.update({key: str(value) for key, value in spec["environment"].items()})
    environment.update(
        {
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_SELF_SPEC_KOFF_TRACE": str(trace_path.resolve()),
            "VLLM_SELF_SPEC_P4_CAPTURE_CONFIG": "",
            "VLLM_SELF_SPEC_P4_CAPTURE_OUTPUT": "",
            "VLLM_SELF_SPEC_P4_BOOT_ACTION": "",
            "VLLM_SELF_SPEC_P4_LOGICAL_WEIGHT_VERSION": "",
            "VLLM_SELF_SPEC_P4_MIN_KV_BLOCKS": "0",
            "PHASE97_ATOMIC_INGRESS_CHILD": "1",
        }
    )
    return environment


def _preflight_atomic(environment: Mapping[str, str]) -> dict[str, Any]:
    preflight = _preflight(environment)
    mode = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from vllm import envs; "
                "from vllm.v1.engine.core_client import InprocClient; "
                "assert not envs.VLLM_ENABLE_V1_MULTIPROCESSING; "
                "assert InprocClient.__name__ == 'InprocClient'"
            ),
        ],
        cwd=REPO_ROOT,
        env=dict(environment),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    _require(mode.returncode == 0, "in-process EngineCore preflight failed")
    return {
        **preflight,
        "v1_multiprocessing": False,
        "expected_engine_core_class": ENGINE_CORE_CLASS,
    }


def _trace_rows(path: Path) -> list[dict[str, Any]]:
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise AtomicIngressProofError(f"cannot read proof trace: {exc}") from exc


def _require_request_ids(actual: list[str], expected: tuple[str, ...]) -> None:
    _require(len(actual) == len(expected), "request count drifted")
    try:
        canonical_ids = canonicalize_p4_request_ids(actual, expected)
    except KOffRuntimeError as exc:
        raise AtomicIngressProofError(f"request identity is invalid: {exc}") from exc
    _require(tuple(canonical_ids) == expected, "request order drifted")


def _analyze(trace_path: Path, engine_core_class: str) -> dict[str, Any]:
    rows = _trace_rows(trace_path)
    headers = [row for row in rows if row.get("record_type") == "koff_runtime_header"]
    steps = [row for row in rows if row.get("record_type") == "koff_engine_step"]
    _require(len(headers) == 1, "proof trace must have one header")
    _require(len(steps) == STEP_COUNT, "proof trace must have exactly two steps")
    _require(
        headers[0]["environment"]["max_num_batched_tokens"]
        == EFFECTIVE_SCHEDULER_BUDGET,
        "effective scheduler budget drifted",
    )
    _require(engine_core_class == ENGINE_CORE_CLASS, "EngineCore was not in-process")
    _require(
        [step["engine_step_index"] for step in steps] == [0, 1],
        "engine step indices drifted",
    )

    prefill, decode = steps
    prefill_diagnostic = prefill["execution"]["diagnostic"]
    decode_diagnostic = decode["execution"]["diagnostic"]
    _require_request_ids(prefill_diagnostic["request_ids"], PROMPT_IDS)
    _require_request_ids(decode_diagnostic["request_ids"], PROMPT_IDS)
    _require(
        prefill["counters"]["H_target_steps"] == 0
        and prefill_diagnostic["num_scheduled_tokens"] == list(PROMPT_COUNTS)
        and prefill_diagnostic["num_computed_tokens"] == [0] * len(PROMPT_IDS)
        and prefill_diagnostic["num_prompt_tokens"] == list(PROMPT_COUNTS),
        "initial step was not the complete eight-request prefill",
    )
    _require(
        prefill["exclusion_reasons"] == ["no_decode_action", "prefill_or_mixed_batch"],
        "initial pure-prefill disposition drifted",
    )
    _require(
        decode["counters"]["H_target_steps"] == len(PROMPT_IDS)
        and decode_diagnostic["num_scheduled_tokens"] == [1] * len(PROMPT_IDS)
        and decode_diagnostic["num_computed_tokens"] == list(PROMPT_COUNTS)
        and decode_diagnostic["num_prompt_tokens"] == list(PROMPT_COUNTS),
        "first decode step was not eight-request q=1 pure decode",
    )
    _require(decode["exclusion_reasons"] == [], "first decode was score-ineligible")
    _require(decode["eligible_for_p3_replay"], "first decode was not eligible")
    _require(
        decode["counters"]["preemptions"] == 0
        and decode["counters"]["recomputed_tokens"] == 0,
        "first decode carried preemption or recomputation",
    )
    _require(
        decode["verified_action_id"] == "off"
        and decode["next_action_id"] == "off"
        and not decode["execution"]["draft_dispatched"],
        "first decode did not remain OFF",
    )
    for step in steps:
        execution = step["execution"]
        _require(
            execution["shared_kv_layer_count"] == 36
            and execution["shared_kv_storage_alias_count"] == 36,
            "proof lost shared target KV",
        )
        _require(
            execution["shared_weight_parameter_count"] == 291
            and execution["target_weight_version_id"]
            == execution["draft_weight_version_id"],
            "proof lost target/draft weight identity",
        )
        _require(
            step["resources"]["shared_target_kv_block_capacity"]
            >= MINIMUM_SHARED_KV_BLOCKS,
            "proof fell below the shared-KV launch floor",
        )
    _require(
        prefill["resources"] == decode["resources"],
        "target-owned KV identity changed between proof steps",
    )

    return {
        "schema_version": 1,
        "record_type": "p4_b0_atomic_ingress_proof",
        "status": "pass",
        "scored": False,
        "decision": "atomic_ingress_gate_passed",
        "engine": {
            "engine_core_mode": "in_process",
            "engine_core_class": engine_core_class,
            "v1_multiprocessing": False,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "effective_scheduler_token_budget": EFFECTIVE_SCHEDULER_BUDGET,
            "enable_chunked_prefill": True,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        },
        "trace": {
            "engine_step_count": len(steps),
            "initial_prefill_step_index": prefill["engine_step_index"],
            "initial_prefill_request_count": len(prefill_diagnostic["request_ids"]),
            "initial_prefill_query_widths": prefill_diagnostic["num_scheduled_tokens"],
            "first_decode_step_index": decode["engine_step_index"],
            "first_decode_request_count": decode["counters"]["H_target_steps"],
            "first_decode_query_widths": decode_diagnostic["num_scheduled_tokens"],
            "first_decode_exclusion_reasons": decode["exclusion_reasons"],
            "preemptions": decode["counters"]["preemptions"],
            "recomputed_tokens": decode["counters"]["recomputed_tokens"],
            "shared_target_kv_blocks": decode["resources"][
                "shared_target_kv_block_capacity"
            ],
        },
        "invariants": {
            "all_requests_queued_before_execution": True,
            "initial_step_is_full_microbatch_prefill": True,
            "first_decode_is_pure_q1_off": True,
            "first_decode_has_no_exclusion": True,
            "shared_target_kv_layer_count": 36,
            "private_draft_kv_allocated": False,
            "target_draft_weight_alias_count": 291,
            "capture_recorder_enabled": False,
        },
        "claims": {
            "atomic_ingress_proven": True,
            "value_screen_repaired": False,
            "v6_authorized": False,
            "performance_claim_allowed": False,
            "action_admitted": False,
        },
    }


def _run_child(output_dir: Path) -> None:
    _require(
        os.environ.get("PHASE97_ATOMIC_INGRESS_CHILD") == "1",
        "unauthorized child",
    )
    trace_path = output_dir / "koff_trace.jsonl"
    result_path = output_dir / "proof.json"
    _require(
        not trace_path.exists() and not result_path.exists(),
        "child output exists",
    )
    from vllm import LLMEngine, envs
    from vllm.v1.engine.core_client import InprocClient

    _require(not envs.VLLM_ENABLE_V1_MULTIPROCESSING, "multiprocessing remained on")
    prompts = _load_prompts()
    spec = _load_json(BOOT_SPEC_PATH)
    engine = LLMEngine.from_engine_args(_engine_args(spec))
    _require(isinstance(engine.engine_core, InprocClient), "wrong EngineCore client")
    engine_core_class = type(engine.engine_core).__name__
    try:
        params = _sampling_params()
        for record_id in PROMPT_IDS:
            engine.add_request(
                record_id,
                {"prompt_token_ids": prompts[record_id]},
                params,
            )
        _require(
            engine.get_num_unfinished_requests() == len(PROMPT_IDS),
            "frontend did not queue the complete microbatch",
        )
        for _ in range(STEP_COUNT):
            engine.step()
    finally:
        engine.engine_core.shutdown()
    result = _analyze(trace_path, engine_core_class)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _execute(authorization: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    run = validate_authorization(authorization, require_output_absent=True)
    trace_path = output_dir / "koff_trace.jsonl"
    environment = _child_env(trace_path)
    preflight = _preflight_atomic(environment)
    output_dir.mkdir(parents=False, exist_ok=False)
    preparation = {
        "schema_version": 1,
        "record_type": "p4_b0_atomic_ingress_proof_preparation",
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
            "record_type": "p4_b0_atomic_ingress_proof_failure",
            "scored": False,
            "returncode": exc.returncode,
            "preserve_without_retry": True,
        }
        (output_dir / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    return _load_json(output_dir / "proof.json")


def parse_args() -> argparse.Namespace:
    """Parse the create-new atomic-ingress proof interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    """Validate or execute one exact atomic-ingress proof."""
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
