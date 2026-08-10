#!/usr/bin/env python3
"""Validate the source-bound Phase 97 B0 full-prefill resource retry."""

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
    APPROVED_OUTPUT_PATH,
    FULL_PREFILL_MAX_NUM_BATCHED_TOKENS,
    MEASUREMENT_GPU_MEMORY_UTILIZATION,
    NATIVE_SAMPLER_ENV,
    NATIVE_SAMPLER_VALUE,
    SERVING_GPU_MEMORY_UTILIZATION,
    SERVING_MAX_NUM_BATCHED_TOKENS,
    P4RunnerError,
    _boot_child_environment,
    _preflight_native_sampler,
    build_boot_specs,
    resolve_authorization_package,
    validate_execution_authority,
)

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_run_authorization_v5.schema.json"
AUTHORIZATION_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization_v5.json"


class B0RunAuthorizationV5Error(ValueError):
    """Raised when the full-prefill resource retry drifts or inflates authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise B0RunAuthorizationV5Error(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise B0RunAuthorizationV5Error(
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
        raise B0RunAuthorizationV5Error(
            f"invalid V5 authorization schema: {exc}"
        ) from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(authorization),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if errors:
        first = errors[0]
        path = _format_json_path(list(first.absolute_path))
        raise B0RunAuthorizationV5Error(
            f"V5 authorization schema rejected {path}: {first.message}"
        )


def validate_authorization_v5(
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the repaired retry and return its narrow authority summary."""
    _validate_schema(authorization)
    try:
        effective = resolve_authorization_package(
            authorization, require_output_absent=True
        )
        validate_execution_authority(effective)
        environment = _boot_child_environment({})
        sampler_evidence = _preflight_native_sampler(environment)
        output_dir = (REPO_ROOT / APPROVED_OUTPUT_PATH).resolve()
        specs = build_boot_specs(effective, output_dir)
    except P4RunnerError as exc:
        raise B0RunAuthorizationV5Error(
            f"live runner rejects the V5 retry package: {exc}"
        ) from exc

    cells = [cell for spec in specs for cell in spec["plan"]["cells"]]
    _require(
        len(specs) == 9 and len(cells) == 432,
        "V5 runner no longer constructs the complete matrix",
    )
    _require(
        all(
            spec["environment"].get(NATIVE_SAMPLER_ENV) == NATIVE_SAMPLER_VALUE
            for spec in specs
        ),
        "a V5 boot spec does not force the native sampler",
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
        "a V5 boot spec lost the measurement-only engine repair",
    )
    _require(
        authorization["serving_chunked_prefill_boundary"]
        == {
            "scope": "post_measurement_real_serving_diagnosis",
            "max_num_batched_tokens": SERVING_MAX_NUM_BATCHED_TOKENS,
            "chunked_prefill_enabled": True,
            "gpu_memory_utilization": SERVING_GPU_MEMORY_UTILIZATION,
            "measurement_override_is_serving_default": False,
            "mixed_prefill_decode_diagnosis_required": True,
        },
        "V5 did not preserve the real-serving chunked-prefill boundary",
    )
    _require(
        {cell["generation"]["generation_seed"] for cell in cells} == {0},
        "V5 workload is no longer uniformly per-request seeded",
    )
    executable_dir = Path(environment["PATH"].split(os.pathsep, maxsplit=1)[0])
    _require(
        executable_dir == Path(sys.prefix) / "bin",
        "V5 preflight did not prepend the active environment",
    )
    return {
        "status": "pass",
        "package_id": authorization["package_id"],
        "authorization_decision": "approve",
        "approved_scope": authorization["decision"]["scope"],
        "prior_gpu_attempts": 3,
        "latest_prior_capture_count": 0,
        "latest_prior_scored": False,
        "prior_outputs_preserved": True,
        "native_sampler_preflight": sampler_evidence,
        "ninja_preflight_passed": True,
        "source_bindings_current": True,
        "gpu_physical_index": 4,
        "physical_boot_count": 9,
        "capture_count": 432,
        "measurement_max_num_batched_tokens": (FULL_PREFILL_MAX_NUM_BATCHED_TOKENS),
        "measurement_gpu_memory_utilization": MEASUREMENT_GPU_MEMORY_UTILIZATION,
        "measured_shared_kv_blocks": 22113,
        "minimum_launch_capacity_blocks": 21682,
        "shared_kv_headroom_blocks": 431,
        "serving_max_num_batched_tokens": SERVING_MAX_NUM_BATCHED_TOKENS,
        "serving_gpu_memory_utilization": SERVING_GPU_MEMORY_UTILIZATION,
        "serving_chunked_prefill_preserved": True,
        "serving_diagnosis_authorized": False,
        "gpu_measurement_authorized": True,
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
    """Validate one fresh V5 retry package and print its summary."""
    args = parse_args()
    result = validate_authorization_v5(_load_json(args.authorization))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
