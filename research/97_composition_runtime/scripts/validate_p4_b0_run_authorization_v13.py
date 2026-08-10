#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the source-bound Phase 97 block-parallel two-lane V13 package."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from run_p4_b0_value_screen import (
    ACTION_ORDERS,
    INPROCESS_ENGINE_CORE_CLASS,
    MINIMUM_SHARED_KV_BLOCKS,
    SERVING_EFFECTIVE_MAX_NUM_SCHEDULED_TOKENS,
    SERVING_GPU_MEMORY_UTILIZATION,
    SERVING_MAX_NUM_BATCHED_TOKENS,
    V13_AUTHORIZATION_PATH,
    V13_OUTPUT_PATH,
    P4RunnerError,
    _boot_child_environment,
    _preflight_inprocess_engine_core,
    _preflight_native_sampler,
    _reviewed_output_path,
    _validate_lane_assignment,
    build_boot_specs,
    resolve_authorization_package,
    validate_execution_authority,
)

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_run_authorization_v13.schema.json"


class B0RunAuthorizationV13Error(ValueError):
    """Raised when the V13 package cannot prove its execution authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise B0RunAuthorizationV13Error(message)


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
        "V13 schema validation failed: "
        + "; ".join(f"{list(err.path)}: {err.message}" for err in errors[:5]),
    )


def _validate_lane_coverage(package: Mapping[str, Any]) -> dict[str, Any]:
    """Prove each block sits on exactly one lane and lanes cover the square."""
    lanes = package["execution_policy"]["lane_assignment"]
    _validate_lane_assignment(lanes)
    owners: dict[int, str] = {}
    for lane in lanes:
        for block_id in lane["block_ids"]:
            _require(
                block_id not in owners,
                f"block {block_id} is claimed by more than one lane",
            )
            owners[block_id] = lane["lane_id"]
    _require(
        sorted(owners) == sorted(ACTION_ORDERS),
        "lane assignment does not cover every capture block",
    )
    boots_per_lane = {
        lane["lane_id"]: sum(
            len(ACTION_ORDERS[block_id]) for block_id in lane["block_ids"]
        )
        for lane in lanes
    }
    _require(
        sum(boots_per_lane.values()) == 9,
        "lane assignment does not close to nine physical boots",
    )
    return {
        "block_owner_by_block": {str(key): value for key, value in owners.items()},
        "boots_per_lane": boots_per_lane,
        "solo_tail_lane": max(boots_per_lane, key=lambda key: boots_per_lane[key]),
    }


def _validate_spec_lanes(package: Mapping[str, Any]) -> dict[str, Any]:
    """Build every boot spec on CPU and prove no block spans two lanes."""
    authorization = resolve_authorization_package(package, require_output_absent=False)
    specs = build_boot_specs(authorization, REPO_ROOT / V13_OUTPUT_PATH)
    _require(len(specs) == 9, "boot specs do not close to nine")
    by_block: dict[int, set[str]] = {}
    pinned_devices: dict[str, str] = {}
    for spec in specs:
        lane = spec.get("lane")
        _require(isinstance(lane, Mapping), f"{spec['boot_id']} has no lane")
        by_block.setdefault(spec["boot_block_id"], set()).add(lane["lane_id"])
        pinned = spec["environment"]["CUDA_VISIBLE_DEVICES"]
        _require(
            pinned == str(lane["physical_gpu_index"]),
            f"{spec['boot_id']} is not pinned to its lane GPU",
        )
        pinned_devices[spec["boot_id"]] = pinned
    for block_id, lane_ids in by_block.items():
        _require(
            len(lane_ids) == 1,
            f"block {block_id} is split across lanes {sorted(lane_ids)}",
        )
    return {
        "boot_count": len(specs),
        "blocks_with_single_lane": len(by_block),
        "pinned_devices": pinned_devices,
    }


def validate_authorization_v13(package: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one V13 package without touching a GPU.

    Args:
        package: The parsed V13 authorization document.

    Returns:
        A non-execution summary of the checks that passed.

    Raises:
        B0RunAuthorizationV13Error: If any registered contract failed.
    """
    _validate_schema(package)
    try:
        authorization = resolve_authorization_package(
            package, require_output_absent=False
        )
        validate_execution_authority(authorization)
    except P4RunnerError as exc:
        raise B0RunAuthorizationV13Error(str(exc)) from exc
    lane_coverage = _validate_lane_coverage(package)
    spec_lanes = _validate_spec_lanes(package)
    _require(
        _reviewed_output_path(REPO_ROOT / V13_AUTHORIZATION_PATH)
        == REPO_ROOT / V13_OUTPUT_PATH,
        "V13 does not resolve the reviewed create-new output pair",
    )
    try:
        base_child_env = _boot_child_environment({})
        _preflight_native_sampler(base_child_env)
        _preflight_inprocess_engine_core(base_child_env)
    except P4RunnerError as exc:
        raise B0RunAuthorizationV13Error(str(exc)) from exc
    policy = package["execution_policy"]
    return {
        "artifact_id": "p4-b0-run-authorization-v13-validation",
        "status": "pass",
        "gpu_executed": False,
        "package_id": package["package_id"],
        "lane_parallel": True,
        "lane_count": len(policy["lane_assignment"]),
        "physical_boot_count": policy["physical_boot_count"],
        "capture_count": policy["capture_count"],
        "lane_coverage": lane_coverage,
        "boot_specs": spec_lanes,
        "max_num_batched_tokens": SERVING_MAX_NUM_BATCHED_TOKENS,
        "effective_max_num_scheduled_tokens": (
            SERVING_EFFECTIVE_MAX_NUM_SCHEDULED_TOKENS
        ),
        "gpu_memory_utilization": SERVING_GPU_MEMORY_UTILIZATION,
        "minimum_launch_capacity_blocks": MINIMUM_SHARED_KV_BLOCKS,
        "engine_core_class": INPROCESS_ENGINE_CORE_CLASS,
        "source_snapshot_required": policy["source_snapshot_required"],
        "block_restart_unit": policy["block_restart_unit"],
        "contention_bound_measured": False,
        "v13_execution_authorized": True,
        "value_screen_scoring_authorized": True,
        "score_grants_authority": False,
        "p4a_engineering_authorized": False,
        "action_admitted": False,
        "output_dir_create_new_ready": not (REPO_ROOT / V13_OUTPUT_PATH).exists(),
        "next_artifact": "p4_b0_value_screen_result",
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    """Validate one V13 package and emit its non-execution summary."""
    args = parse_args()
    result = validate_authorization_v13(_load_json(args.authorization))
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(payload, end="")
    else:
        args.out.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except B0RunAuthorizationV13Error as exc:
        print(f"V13 authorization refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
