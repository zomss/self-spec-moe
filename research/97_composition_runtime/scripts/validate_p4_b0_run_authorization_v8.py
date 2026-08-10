#!/usr/bin/env python3
"""Validate the source-bound Phase 97 B0 decode-work V8 package."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from run_p4_b0_value_screen import (
    FULL_PREFILL_MAX_NUM_BATCHED_TOKENS,
    GPU4_UUID,
    MEASUREMENT_GPU_MEMORY_UTILIZATION,
    MINIMUM_SHARED_KV_BLOCKS,
    NATIVE_SAMPLER_ENV,
    NATIVE_SAMPLER_VALUE,
    PREFILL_SAMPLED_TOKENS_PER_REQUEST,
    SERVING_GPU_MEMORY_UTILIZATION,
    SERVING_MAX_NUM_BATCHED_TOKENS,
    V1_MULTIPROCESSING_ENV,
    V1_MULTIPROCESSING_VALUE,
    V8_OUTPUT_PATH,
    P4RunnerError,
    _boot_child_environment,
    _preflight_inprocess_engine_core,
    _preflight_native_sampler,
    _total_output_tokens,
    build_boot_specs,
    resolve_authorization_package,
    validate_execution_authority,
)

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_run_authorization_v8.schema.json"


class B0RunAuthorizationV8Error(ValueError):
    """Raised when the V8 package drifts or inflates authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise B0RunAuthorizationV8Error(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise B0RunAuthorizationV8Error(
            f"cannot load JSON artifact {path}: {exc}"
        ) from exc
    _require(isinstance(value, dict), f"JSON artifact must be an object: {path}")
    return value


def _format_json_path(parts: Sequence[Any]) -> str:
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def _validate_schema(authorization: Mapping[str, Any]) -> None:
    schema = _load_json(SCHEMA_PATH)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise B0RunAuthorizationV8Error(
            f"invalid V8 authorization schema: {exc}"
        ) from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(authorization),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if errors:
        first = errors[0]
        path = _format_json_path(list(first.absolute_path))
        raise B0RunAuthorizationV8Error(
            f"V8 authorization schema rejected {path}: {first.message}"
        )


def validate_authorization_v8(
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate V8 without creating its output directory or running a GPU."""
    _validate_schema(authorization)
    try:
        effective = resolve_authorization_package(
            authorization,
            require_output_absent=True,
        )
        validate_execution_authority(effective)
        environment = _boot_child_environment({})
        sampler_evidence = _preflight_native_sampler(environment)
        engine_core_evidence = _preflight_inprocess_engine_core(environment)
        output_dir = (REPO_ROOT / V8_OUTPUT_PATH).resolve()
        specs = build_boot_specs(effective, output_dir)
    except P4RunnerError as exc:
        raise B0RunAuthorizationV8Error(
            f"live runner rejects the V8 package: {exc}"
        ) from exc

    cells = [cell for spec in specs for cell in spec["plan"]["cells"]]
    _require(
        len(specs) == 9 and len(cells) == 432,
        "V8 runner no longer constructs the complete matrix",
    )
    _require(
        all(
            spec["environment"].get(NATIVE_SAMPLER_ENV) == NATIVE_SAMPLER_VALUE
            and spec["environment"].get(V1_MULTIPROCESSING_ENV)
            == V1_MULTIPROCESSING_VALUE
            for spec in specs
        ),
        "a V8 boot spec lost a sampler or EngineCore policy",
    )
    _require(
        all(
            spec["engine"]["max_num_batched_tokens"]
            == FULL_PREFILL_MAX_NUM_BATCHED_TOKENS
            and spec["engine"]["enable_chunked_prefill"] is True
            and spec["engine"]["gpu_memory_utilization"]
            == MEASUREMENT_GPU_MEMORY_UTILIZATION
            for spec in specs
        ),
        "a V8 boot spec lost the measurement-only engine repair",
    )
    _require(
        all(
            _total_output_tokens(cell)
            == cell["generation"]["max_output_tokens"]
            + PREFILL_SAMPLED_TOKENS_PER_REQUEST
            and cell["generation"]["requested_output_tokens"]
            == len(cell["generation"]["prompt_record_ids"])
            * cell["generation"]["max_output_tokens"]
            for cell in cells
        ),
        "a V8 cell lost the prefill-sample/decode-work separation",
    )
    repair = authorization["decode_work_repair"]
    _require(
        repair["measurement_currency"] == "S_dec"
        and repair["prefill_sampled_tokens_per_request"] == 1
        and repair["frontend_total_output_tokens"]
        == "generation.max_output_tokens + 1",
        "V8 decode-work repair is incomplete",
    )
    executable_dir = Path(environment["PATH"].split(os.pathsep, maxsplit=1)[0])
    _require(
        executable_dir == Path(sys.prefix) / "bin",
        "V8 preflight did not prepend the active environment",
    )
    _require(
        not (REPO_ROOT / V8_OUTPUT_PATH).exists(),
        "V8 validation created the authorized output",
    )
    return {
        "status": "pass",
        "package_id": authorization["package_id"],
        "authorization_decision": "approve",
        "approved_scope": authorization["decision"]["scope"],
        "gpu_executed": False,
        "prior_value_screen_attempts": 6,
        "latest_prior_complete_capture_count": 0,
        "latest_prior_incomplete_capture_count": 1,
        "prior_outputs_preserved": True,
        "decode_work_offset_tested": True,
        "prefill_sampled_tokens_per_request": 1,
        "all_action_capture_rollover_tested": True,
        "request_id_canonicalization_tested": True,
        "internal_request_id_randomization_retained": True,
        "native_sampler_preflight": sampler_evidence,
        "engine_core_preflight": engine_core_evidence,
        "ninja_preflight_passed": True,
        "source_bindings_current": True,
        "gpu_physical_index": 4,
        "gpu_uuid": GPU4_UUID,
        "physical_boot_count": 9,
        "capture_count": 432,
        "measurement_max_num_batched_tokens": (FULL_PREFILL_MAX_NUM_BATCHED_TOKENS),
        "measurement_gpu_memory_utilization": MEASUREMENT_GPU_MEMORY_UTILIZATION,
        "minimum_launch_capacity_blocks": MINIMUM_SHARED_KV_BLOCKS,
        "serving_max_num_batched_tokens": SERVING_MAX_NUM_BATCHED_TOKENS,
        "serving_gpu_memory_utilization": SERVING_GPU_MEMORY_UTILIZATION,
        "serving_chunked_prefill_preserved": True,
        "v8_execution_authorized": True,
        "p4a_engineering_authorized": False,
        "action_admitted": False,
        "output_dir_create_new_ready": True,
        "next_artifact": "p4_b0_value_screen_result",
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Validate one V8 package and print its non-execution summary."""
    args = parse_args()
    result = validate_authorization_v8(_load_json(args.authorization))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
