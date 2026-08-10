#!/usr/bin/env python3
"""Run the authorized GPU-4 full-prefill memory-envelope repair probe."""

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

from jsonschema import Draft202012Validator
from probe_p4_b0_full_prefill_resource import (
    BASE_AUTHORIZATION_PATH,
    GPU_UUID,
    MODEL_REVISION,
    P4FullPrefillResourceProbeError,
    _probe_environment,
    _validate_physical_gpu,
)
from probe_p4_b0_full_prefill_resource import (
    _probe_spec as _base_probe_spec,
)
from probe_p4_b0_full_prefill_resource import (
    _resource_result as _base_resource_result,
)
from run_p4_b0_value_screen import (
    EXPECTED_MAX_MICROBATCH_PROMPT_TOKENS,
    FULL_PREFILL_MAX_NUM_BATCHED_TOKENS,
    MINIMUM_SHARED_KV_BLOCKS,
    NATIVE_SAMPLER_ENV,
    NATIVE_SAMPLER_VALUE,
    P4RunnerError,
    _engine_args,
    _load_json,
    _preflight_native_sampler,
)

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
AUTHORIZATION_PATH = (
    PHASE_DIR
    / "data"
    / "p4"
    / "p4_b0_full_prefill_resource_repair_probe_authorization.json"
)
OUTPUT_PATH = (
    PHASE_DIR / "data" / "p4" / "p4_b0_full_prefill_resource_repair_probe.json"
)
SCHEMA_PATH = (
    PHASE_DIR / "schemas" / "p4_b0_full_prefill_resource_repair_probe.schema.json"
)
PACKAGE_ID = "p4-b0-full-prefill-resource-repair-probe-v2"
PROBE_GPU_MEMORY_UTILIZATION = 0.96
SERVING_GPU_MEMORY_UTILIZATION = 0.90
SERVING_MAX_NUM_BATCHED_TOKENS = 8192

SOURCE_PATHS = {
    "base_authorization": (
        "research/97_composition_runtime/data/p4/p4_b0_run_authorization_v2.json"
    ),
    "prior_probe_authorization": (
        "research/97_composition_runtime/data/p4/"
        "p4_b0_full_prefill_resource_probe_authorization.json"
    ),
    "prior_probe_result": (
        "research/97_composition_runtime/data/p4/p4_b0_full_prefill_resource_probe.json"
    ),
    "matrix_runner": (
        "research/97_composition_runtime/scripts/run_p4_b0_value_screen.py"
    ),
    "runner_tests": (
        "research/97_composition_runtime/tests/test_p4_b0_capture_runner.py"
    ),
    "base_probe_script": (
        "research/97_composition_runtime/scripts/probe_p4_b0_full_prefill_resource.py"
    ),
    "repair_probe_script": (
        "research/97_composition_runtime/scripts/"
        "probe_p4_b0_full_prefill_resource_repair.py"
    ),
    "repair_probe_tests": (
        "research/97_composition_runtime/tests/"
        "test_p4_b0_full_prefill_resource_repair_probe.py"
    ),
    "repair_probe_schema": (
        "research/97_composition_runtime/schemas/"
        "p4_b0_full_prefill_resource_repair_probe.schema.json"
    ),
}

SERVING_CHUNKED_PREFILL_BOUNDARY = {
    "scope": "post_measurement_real_serving_diagnosis",
    "max_num_batched_tokens": SERVING_MAX_NUM_BATCHED_TOKENS,
    "chunked_prefill_enabled": True,
    "gpu_memory_utilization": SERVING_GPU_MEMORY_UTILIZATION,
    "measurement_override_is_serving_default": False,
    "mixed_prefill_decode_diagnosis_required": True,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise P4FullPrefillResourceProbeError(message)


def _repo_path(relative_path: str) -> Path:
    path = (REPO_ROOT / relative_path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise P4FullPrefillResourceProbeError(
            f"resource-repair source escapes the repository: {path}"
        ) from exc
    return path


def _reference(relative_path: str) -> dict[str, str]:
    path = _repo_path(relative_path)
    _require(path.is_file(), f"resource-repair source is missing: {path}")
    return {
        "path": relative_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _schema_errors(package: Mapping[str, Any]) -> list[Any]:
    schema = _load_json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    return sorted(
        Draft202012Validator(schema).iter_errors(package),
        key=lambda error: [str(part) for part in error.absolute_path],
    )


def validate_repair_probe_authorization(
    package: Mapping[str, Any], *, require_output_absent: bool = True
) -> dict[str, Any]:
    """Validate the one-initialization GPU-4 memory-envelope repair probe."""
    errors = _schema_errors(package)
    if errors:
        raise P4FullPrefillResourceProbeError(
            f"resource-repair schema rejected: {errors[0].message}"
        )
    expected_sources = {role: _reference(path) for role, path in SOURCE_PATHS.items()}
    _require(
        package["source_artifacts"] == expected_sources,
        "resource-repair source closure drifted",
    )

    base_engine = _load_json(BASE_AUTHORIZATION_PATH)["run_contract"]["engine"]
    _require(
        base_engine["max_num_batched_tokens"] == SERVING_MAX_NUM_BATCHED_TOKENS
        and base_engine["gpu_memory_utilization"] == SERVING_GPU_MEMORY_UTILIZATION,
        "resource-repair base serving envelope drifted",
    )

    expected_invocation = [
        ".venv/bin/python",
        SOURCE_PATHS["repair_probe_script"],
        "--authorization",
        str(AUTHORIZATION_PATH.relative_to(REPO_ROOT)),
        "--output",
        str(OUTPUT_PATH.relative_to(REPO_ROOT)),
    ]
    _require(
        package["probe_contract"]
        == {
            "physical_gpu_index": 4,
            "gpu_uuid": GPU_UUID,
            "model_revision": MODEL_REVISION,
            "max_num_batched_tokens": FULL_PREFILL_MAX_NUM_BATCHED_TOKENS,
            "chunked_prefill_enabled": True,
            "gpu_memory_utilization": PROBE_GPU_MEMORY_UTILIZATION,
            "speculative_slot_reserve": 32,
            "effective_scheduler_token_budget": 114656,
            "max_microbatch_prompt_tokens": (EXPECTED_MAX_MICROBATCH_PROMPT_TOKENS),
            "minimum_shared_kv_blocks": MINIMUM_SHARED_KV_BLOCKS,
            "invocation": {
                "argv": expected_invocation,
                "output": str(OUTPUT_PATH.relative_to(REPO_ROOT)),
                "create_new_output": True,
            },
        },
        "resource-repair probe contract drifted",
    )
    _require(
        package["memory_repair"]
        == {
            "changed_engine_field": "gpu_memory_utilization",
            "prior_value": SERVING_GPU_MEMORY_UTILIZATION,
            "probe_value": PROBE_GPU_MEMORY_UTILIZATION,
            "all_other_engine_fields_unchanged": True,
            "prior_num_gpu_blocks": 19928,
            "required_additional_blocks": 1754,
            "projected_num_gpu_blocks": 22102,
            "projection_is_authority": False,
        },
        "resource-repair memory delta drifted",
    )
    _require(
        package["serving_chunked_prefill_boundary"] == SERVING_CHUNKED_PREFILL_BOUNDARY,
        "resource-repair serving boundary drifted",
    )
    _require(
        package["decision"]
        == {
            "state": "approve",
            "scope": "gpu4_full_prefill_memory_repair_probe_only",
            "invalidated_by": [
                "source_hash_drift",
                "output_exists",
                "gpu_identity_drift",
                "probe_contract_drift",
                "memory_repair_drift",
                "serving_boundary_drift",
            ],
        },
        "resource-repair decision drifted",
    )
    _require(
        package["claims"]
        == {
            "full_screen_run_ready": False,
            "resource_capacity_measured": False,
            "serving_configuration_validated": False,
            "action_admitted": False,
            "performance_claim_allowed": False,
        },
        "resource-repair claims overstate evidence",
    )
    _require(
        package["authorizations"]
        == {
            "gpu_resource_probe": True,
            "gpu_value_screen": False,
            "serving_diagnosis": False,
            "p4a_engineering": False,
            "action_admission": False,
        },
        "resource-repair authority is inflated",
    )
    _require(
        package["execution_policy"]
        == {
            "fallback_gpu_authorized": False,
            "model_initializations": 1,
            "requests_executed": 0,
            "create_new_output_required": True,
            "partial_resume_allowed": False,
            "on_any_failure": ("stop_and_require_fresh_resource_probe_authorization"),
        },
        "resource-repair execution policy drifted",
    )
    _require(
        package["next_artifact"]
        == {
            "kind": "p4_b0_full_prefill_resource_repair_probe_result",
            "passing_result_may_support_v5_review": True,
            "authorizes_v5": False,
        },
        "resource-repair next-artifact boundary drifted",
    )
    if require_output_absent:
        _require(not OUTPUT_PATH.exists(), "resource-repair output already exists")
    return dict(package)


def _probe_spec() -> tuple[dict[str, Any], dict[str, Any]]:
    spec, evidence = _base_probe_spec()
    _require(
        spec["engine"]["gpu_memory_utilization"] == SERVING_GPU_MEMORY_UTILIZATION,
        "base probe memory utilization drifted",
    )
    spec["engine"]["gpu_memory_utilization"] = PROBE_GPU_MEMORY_UTILIZATION
    return spec, evidence


def _resource_result(
    engine: Any,
    package: Mapping[str, Any],
    gpu: Mapping[str, Any],
    budget: Mapping[str, Any],
) -> dict[str, Any]:
    actual_utilization = engine.vllm_config.cache_config.gpu_memory_utilization
    _require(
        actual_utilization == PROBE_GPU_MEMORY_UTILIZATION,
        "live engine memory utilization differs from the repair contract",
    )
    result = _base_resource_result(engine, package, gpu, budget)
    result["schema_version"] = 2
    result["record_type"] = "p4_b0_full_prefill_resource_repair_probe_result"
    result["engine"]["gpu_memory_utilization"] = actual_utilization
    result["memory_repair"] = dict(package["memory_repair"])
    result["serving_chunked_prefill_boundary"] = dict(
        package["serving_chunked_prefill_boundary"]
    )
    result["claims"]["serving_configuration_validated"] = False
    return result


def run_child(package: Mapping[str, Any]) -> None:
    validate_repair_probe_authorization(package, require_output_absent=True)
    _require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "4",
        "resource-repair child is not isolated to physical GPU 4",
    )
    _require(
        os.environ.get(NATIVE_SAMPLER_ENV) == NATIVE_SAMPLER_VALUE,
        "resource-repair child lost the native-sampler policy",
    )
    gpu = _validate_physical_gpu()
    spec, budget = _probe_spec()
    from vllm import LLMEngine

    engine = LLMEngine.from_engine_args(_engine_args(spec))
    try:
        result = _resource_result(engine, package, gpu, budget)
    finally:
        engine.engine_core.shutdown()
    with OUTPUT_PATH.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    _require(
        result["decision"]["state"] == "pass",
        "repaired full-prefill shared-KV capacity is below the launch floor",
    )


def execute_probe(package: Mapping[str, Any]) -> None:
    validate_repair_probe_authorization(package, require_output_absent=True)
    environment = _probe_environment()
    _preflight_native_sampler(environment)
    _validate_physical_gpu()
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--authorization",
            str(AUTHORIZATION_PATH.resolve()),
            "--output",
            str(OUTPUT_PATH.resolve()),
            "--child",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse the exact resource-repair probe interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    """Validate and execute one resource-repair probe."""
    args = parse_args()
    _require(
        args.authorization.resolve() == AUTHORIZATION_PATH.resolve(),
        "resource-repair authorization path drifted",
    )
    _require(
        args.output.resolve() == OUTPUT_PATH.resolve(),
        "resource-repair output path drifted",
    )
    package = _load_json(args.authorization)
    if args.child:
        run_child(package)
    else:
        execute_probe(package)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (P4FullPrefillResourceProbeError, P4RunnerError) as exc:
        print(f"P4 resource-repair probe refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
