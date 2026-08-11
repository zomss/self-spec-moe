#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the source-bound Phase 97 single-block restart V14 package."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from run_p4_b0_value_screen import (
    INPROCESS_ENGINE_CORE_CLASS,
    MINIMUM_SHARED_KV_BLOCKS,
    SERVING_EFFECTIVE_MAX_NUM_SCHEDULED_TOKENS,
    SERVING_GPU_MEMORY_UTILIZATION,
    SERVING_MAX_NUM_BATCHED_TOKENS,
    V14_AUTHORIZATION_PATH,
    V14_OUTPUT_PATH,
    P4RunnerError,
    _boot_child_environment,
    _preflight_inprocess_engine_core,
    _preflight_native_sampler,
    _reviewed_output_path,
    resolve_authorization_package,
    validate_execution_authority,
)

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
SCHEMA_PATH = (
    PHASE_DIR / "schemas" / "p4_b0_block_restart_authorization_v14.schema.json"
)


class B0RunAuthorizationV14Error(ValueError):
    """Raised when the V14 package cannot prove its execution authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise B0RunAuthorizationV14Error(message)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict), f"{path} is not a JSON object")
    return payload


def _validate_schema(package: Mapping[str, Any]) -> None:
    validator = Draft202012Validator(_load_json(SCHEMA_PATH))
    errors = sorted(validator.iter_errors(package), key=lambda err: list(err.path))
    _require(
        not errors,
        "V14 schema validation failed: "
        + "; ".join(f"{list(err.path)}: {err.message}" for err in errors[:5]),
    )


def validate_authorization_v14(package: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one V14 package without touching a GPU.

    Args:
        package: The parsed V14 authorization document.

    Returns:
        A non-execution summary of the checks that passed.

    Raises:
        B0RunAuthorizationV14Error: If any registered contract failed.
    """
    _validate_schema(package)
    try:
        authorization = resolve_authorization_package(
            package, require_output_absent=False
        )
        validate_execution_authority(authorization)
    except P4RunnerError as exc:
        raise B0RunAuthorizationV14Error(str(exc)) from exc
    policy_lane = package["execution_policy"]["lane"]
    _require(
        _reviewed_output_path(REPO_ROOT / V14_AUTHORIZATION_PATH)
        == REPO_ROOT / V14_OUTPUT_PATH,
        "V14 does not resolve the reviewed create-new output pair",
    )
    try:
        base_child_env = _boot_child_environment({})
        _preflight_native_sampler(base_child_env)
        _preflight_inprocess_engine_core(base_child_env)
    except P4RunnerError as exc:
        raise B0RunAuthorizationV14Error(str(exc)) from exc
    policy = package["execution_policy"]
    return {
        "artifact_id": "p4-b0-run-authorization-v14-validation",
        "status": "pass",
        "gpu_executed": False,
        "package_id": package["package_id"],
        "lane_parallel": False,
        "restart_block_id": policy["restart_block_id"],
        "lane_id": policy_lane["lane_id"],
        "physical_gpu_index": policy_lane["physical_gpu_index"],
        "physical_boot_count": policy["physical_boot_count"],
        "rerun_capture_count": policy["rerun_capture_count"],
        "reused_capture_count": policy["reused_capture_count"],
        "scored_capture_count": policy["scored_capture_count"],
        "block_selection_rule": package["block_selection"]["rule"],
        "declared_before_rerun": package["block_selection"]["declared_before_rerun"],
        "result_used_regardless_of_certification": package["outcome_commitment"][
            "result_used_regardless_of_certification"
        ],
        "rerun_until_pass_forbidden": package["outcome_commitment"][
            "rerun_until_pass_forbidden"
        ],
        "max_num_batched_tokens": SERVING_MAX_NUM_BATCHED_TOKENS,
        "effective_max_num_scheduled_tokens": (
            SERVING_EFFECTIVE_MAX_NUM_SCHEDULED_TOKENS
        ),
        "gpu_memory_utilization": SERVING_GPU_MEMORY_UTILIZATION,
        "minimum_launch_capacity_blocks": MINIMUM_SHARED_KV_BLOCKS,
        "engine_core_class": INPROCESS_ENGINE_CORE_CLASS,
        "source_snapshot_required": policy["source_snapshot_required"],
        "contention_bound_measured": False,
        "v14_execution_authorized": True,
        "value_screen_scoring_authorized": True,
        "score_grants_authority": False,
        "p4a_engineering_authorized": False,
        "action_admitted": False,
        "output_dir_create_new_ready": not (REPO_ROOT / V14_OUTPUT_PATH).exists(),
        "next_artifact": "p4_b0_value_screen_result",
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    """Validate one V14 package and emit its non-execution summary."""
    args = parse_args()
    result = validate_authorization_v14(_load_json(args.authorization))
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(payload, end="")
    else:
        args.out.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except B0RunAuthorizationV14Error as exc:
        print(f"V14 authorization refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
