#!/usr/bin/env python3
"""Run the authorized GPU-4 full-prefill KV-capacity probe."""

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
from run_p4_b0_value_screen import (
    EXPECTED_MAX_MICROBATCH_PROMPT_TOKENS,
    FULL_PREFILL_MAX_NUM_BATCHED_TOKENS,
    MINIMUM_SHARED_KV_BLOCKS,
    NATIVE_SAMPLER_ENV,
    NATIVE_SAMPLER_VALUE,
    P4RunnerError,
    _boot_child_environment,
    _engine_args,
    _load_json,
    _load_prompt_rows,
    _preflight_native_sampler,
    apply_full_prefill_engine_contract,
    full_prefill_budget_evidence,
)

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
AUTHORIZATION_PATH = (
    PHASE_DIR / "data" / "p4" / "p4_b0_full_prefill_resource_probe_authorization.json"
)
OUTPUT_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_full_prefill_resource_probe.json"
BASE_AUTHORIZATION_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization_v2.json"
SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_full_prefill_resource_probe.schema.json"
PACKAGE_ID = "p4-b0-full-prefill-resource-probe-v1"
GPU_UUID = "GPU-c9d19019-5065-2353-80a9-f1797eb19d51"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

SOURCE_PATHS = {
    "base_authorization": (
        "research/97_composition_runtime/data/p4/p4_b0_run_authorization_v2.json"
    ),
    "failed_v4_attempt": (
        "research/97_composition_runtime/data/p4/run_b0_value_screen_v3/failure.json"
    ),
    "matrix_runner": (
        "research/97_composition_runtime/scripts/run_p4_b0_value_screen.py"
    ),
    "runner_tests": (
        "research/97_composition_runtime/tests/test_p4_b0_capture_runner.py"
    ),
    "probe_script": (
        "research/97_composition_runtime/scripts/probe_p4_b0_full_prefill_resource.py"
    ),
    "probe_tests": (
        "research/97_composition_runtime/tests/"
        "test_p4_b0_full_prefill_resource_probe.py"
    ),
    "probe_schema": (
        "research/97_composition_runtime/schemas/"
        "p4_b0_full_prefill_resource_probe.schema.json"
    ),
}


class P4FullPrefillResourceProbeError(RuntimeError):
    """Raised when the resource probe cannot prove its narrow authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise P4FullPrefillResourceProbeError(message)


def _repo_path(relative_path: str) -> Path:
    path = (REPO_ROOT / relative_path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise P4FullPrefillResourceProbeError(
            f"resource-probe source escapes the repository: {path}"
        ) from exc
    return path


def _reference(relative_path: str) -> dict[str, str]:
    path = _repo_path(relative_path)
    _require(path.is_file(), f"resource-probe source is missing: {path}")
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


def validate_probe_authorization(
    package: Mapping[str, Any], *, require_output_absent: bool = True
) -> dict[str, Any]:
    """Validate the exact one-initialization GPU-4 resource probe."""
    errors = _schema_errors(package)
    if errors:
        raise P4FullPrefillResourceProbeError(
            f"resource-probe schema rejected: {errors[0].message}"
        )
    expected_sources = {role: _reference(path) for role, path in SOURCE_PATHS.items()}
    _require(
        package["source_artifacts"] == expected_sources,
        "resource-probe source closure drifted",
    )
    expected_invocation = [
        ".venv/bin/python",
        SOURCE_PATHS["probe_script"],
        "--authorization",
        str(AUTHORIZATION_PATH.relative_to(REPO_ROOT)),
        "--output",
        str(OUTPUT_PATH.relative_to(REPO_ROOT)),
    ]
    contract = package["probe_contract"]
    _require(
        contract
        == {
            "physical_gpu_index": 4,
            "gpu_uuid": GPU_UUID,
            "model_revision": MODEL_REVISION,
            "max_num_batched_tokens": FULL_PREFILL_MAX_NUM_BATCHED_TOKENS,
            "chunked_prefill_enabled": True,
            "speculative_slot_reserve": 32,
            "effective_scheduler_token_budget": 114656,
            "max_microbatch_prompt_tokens": EXPECTED_MAX_MICROBATCH_PROMPT_TOKENS,
            "minimum_shared_kv_blocks": MINIMUM_SHARED_KV_BLOCKS,
            "invocation": {
                "argv": expected_invocation,
                "output": str(OUTPUT_PATH.relative_to(REPO_ROOT)),
                "create_new_output": True,
            },
        },
        "resource-probe contract drifted",
    )
    _require(
        package["decision"]
        == {
            "state": "approve",
            "scope": "gpu4_full_prefill_capacity_probe_only",
            "invalidated_by": [
                "source_hash_drift",
                "output_exists",
                "gpu_identity_drift",
                "probe_contract_drift",
            ],
        },
        "resource-probe decision drifted",
    )
    _require(
        package["claims"]
        == {
            "full_screen_run_ready": False,
            "resource_capacity_measured": False,
            "action_admitted": False,
            "performance_claim_allowed": False,
        },
        "resource-probe claims overstate evidence",
    )
    _require(
        package["authorizations"]
        == {
            "gpu_resource_probe": True,
            "gpu_value_screen": False,
            "p4a_engineering": False,
            "action_admission": False,
        },
        "resource-probe authority is inflated",
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
        "resource-probe execution policy drifted",
    )
    _require(
        package["next_artifact"]
        == {
            "kind": "p4_b0_full_prefill_resource_probe_result",
            "may_authorize_value_screen": False,
        },
        "resource-probe next-artifact boundary drifted",
    )
    if require_output_absent:
        _require(not OUTPUT_PATH.exists(), "resource-probe output already exists")
    return dict(package)


def _probe_spec() -> tuple[dict[str, Any], dict[str, Any]]:
    authorization = apply_full_prefill_engine_contract(
        _load_json(BASE_AUTHORIZATION_PATH)
    )
    manifest, prompt_rows = _load_prompt_rows()
    evidence = full_prefill_budget_evidence(
        manifest,
        prompt_rows,
        authorization["run_contract"]["engine"],
    )
    off = next(
        row
        for row in authorization["run_contract"]["action_boots"]
        if row["action_id"] == "off"
    )
    return (
        {
            "engine": authorization["run_contract"]["engine"],
            "model": authorization["run_contract"]["model"],
            "dynamic_k_schedule": off["dynamic_k_schedule"],
        },
        evidence,
    )


def _probe_environment() -> dict[str, str]:
    authorization = _load_json(BASE_AUTHORIZATION_PATH)
    environment = dict(authorization["run_contract"]["environment"])
    if environment.get("VLLM_SELF_SPEC_DRAFT_PARTIAL_REPLICA") == "0":
        environment["VLLM_SELF_SPEC_DRAFT_PARTIAL_REPLICA"] = ""
    environment.update(
        {
            NATIVE_SAMPLER_ENV: NATIVE_SAMPLER_VALUE,
            "VLLM_SELF_SPEC_DRAFT_KV_WINDOW": "0",
            "VLLM_SELF_SPEC_DRAFT_KV_SINKS": "0",
        }
    )
    _require(
        environment.get("CUDA_VISIBLE_DEVICES") == "4",
        "resource probe is not bound to physical GPU 4",
    )
    try:
        return _boot_child_environment(environment)
    except P4RunnerError as exc:
        raise P4FullPrefillResourceProbeError(str(exc)) from exc


def _validate_physical_gpu() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id=4",
                "--query-gpu=index,uuid,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise P4FullPrefillResourceProbeError(
            f"cannot validate physical GPU 4: {exc}"
        ) from exc
    parts = [part.strip() for part in result.stdout.strip().split(",")]
    _require(len(parts) == 4, "nvidia-smi returned malformed GPU identity")
    index, uuid, name, memory_mib = parts
    _require(index == "4" and uuid == GPU_UUID, "physical GPU identity drifted")
    return {
        "physical_index": 4,
        "uuid": uuid,
        "name": name,
        "memory_mib": int(memory_mib),
    }


def _resource_result(
    engine: Any,
    package: Mapping[str, Any],
    gpu: Mapping[str, Any],
    budget: Mapping[str, Any],
) -> dict[str, Any]:
    cache = engine.vllm_config.cache_config
    num_blocks = cache.num_gpu_blocks
    block_size = cache.block_size
    cache_tokens = cache.kv_cache_size_tokens
    _require(type(num_blocks) is int and num_blocks > 0, "KV block count is invalid")
    _require(type(block_size) is int and block_size > 0, "KV block size is invalid")
    _require(
        type(cache_tokens) is int and cache_tokens == num_blocks * block_size,
        "KV token capacity does not close over blocks",
    )
    decision = "pass" if num_blocks >= MINIMUM_SHARED_KV_BLOCKS else "fail"
    return {
        "schema_version": 1,
        "record_type": "p4_b0_full_prefill_resource_probe_result",
        "package_id": package["package_id"],
        "gpu": dict(gpu),
        "engine": {
            "max_num_batched_tokens": FULL_PREFILL_MAX_NUM_BATCHED_TOKENS,
            "chunked_prefill_enabled": True,
            "sampler_backend": "pytorch_native",
            "requests_executed": 0,
        },
        "full_prefill_budget": dict(budget),
        "shared_kv_capacity": {
            "num_gpu_blocks": num_blocks,
            "block_size_tokens": block_size,
            "capacity_tokens": cache_tokens,
            "minimum_launch_blocks": MINIMUM_SHARED_KV_BLOCKS,
            "headroom_blocks": num_blocks - MINIMUM_SHARED_KV_BLOCKS,
        },
        "decision": {
            "state": decision,
            "value_screen_may_be_separately_authorized": decision == "pass",
            "authority_granted": False,
        },
        "claims": {
            "model_initialized": True,
            "generation_performed": False,
            "action_admitted": False,
            "performance_claim_allowed": False,
        },
    }


def run_child(package: Mapping[str, Any]) -> None:
    validate_probe_authorization(package, require_output_absent=True)
    _require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "4",
        "resource-probe child is not isolated to physical GPU 4",
    )
    _require(
        os.environ.get(NATIVE_SAMPLER_ENV) == NATIVE_SAMPLER_VALUE,
        "resource-probe child lost the native-sampler policy",
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
        "full-prefill shared-KV capacity is below the launch floor",
    )


def execute_probe(package: Mapping[str, Any]) -> None:
    validate_probe_authorization(package, require_output_absent=True)
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
    """Parse the exact resource-probe interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    """Validate and execute one resource probe."""
    args = parse_args()
    _require(
        args.authorization.resolve() == AUTHORIZATION_PATH.resolve(),
        "resource-probe authorization path drifted",
    )
    _require(
        args.output.resolve() == OUTPUT_PATH.resolve(),
        "resource-probe output path drifted",
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
        print(f"P4 resource probe refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
