#!/usr/bin/env python3
"""Validate the source-bound Phase 97 GPU contention-probe package."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from run_p4_b0_gpu_contention_probe import (
    ACTION_ID,
    AUTHORIZATION_PATH,
    CONTENT_SEED,
    EXPECTED_PACKAGE_ID,
    GPU_ASSIGNMENTS,
    OUTPUT_DIR,
    REGIME_ID,
    ROUNDS,
    RUNS,
    SCHEDULE,
    THRESHOLDS,
    ContentionProbeError,
    build_probe_specs,
    validate_authorization,
)

PHASE_DIR = Path(__file__).resolve().parents[1]
SCHEMA_PATH = (
    PHASE_DIR / "schemas" / "p4_b0_gpu_contention_probe_authorization.schema.json"
)


class ContentionProbeAuthorizationError(ValueError):
    """Raised when the contention-probe package drifts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContentionProbeAuthorizationError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContentionProbeAuthorizationError(
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
        raise ContentionProbeAuthorizationError(
            f"invalid contention-probe schema: {exc}"
        ) from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(authorization),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if errors:
        first = errors[0]
        path = _format_json_path(list(first.absolute_path))
        raise ContentionProbeAuthorizationError(
            f"contention-probe schema rejected {path}: {first.message}"
        )


def validate_contention_probe_authorization(
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the package without creating output or running a GPU command."""
    _validate_schema(authorization)
    try:
        validate_authorization(authorization, require_output_absent=True)
        specs = build_probe_specs(OUTPUT_DIR)
    except ContentionProbeError as exc:
        raise ContentionProbeAuthorizationError(
            f"contention runner rejects its package: {exc}"
        ) from exc

    _require(len(specs) == 4, "contention probe no longer builds four runs")
    by_id = {spec["run_id"]: spec for spec in specs}
    _require(set(by_id) == set(RUNS), "contention run identifiers drifted")
    for run_id, spec in by_id.items():
        expected_run = RUNS[run_id]
        expected_gpu = GPU_ASSIGNMENTS[expected_run["gpu_id"]]
        _require(
            spec["condition"] == expected_run["condition"]
            and spec["gpu_id"] == expected_run["gpu_id"]
            and spec["gpu"]["physical_index"] == expected_gpu["physical_index"]
            and spec["gpu"]["uuid"] == expected_gpu["uuid"]
            and spec["cpu_affinity"] == expected_gpu["cpu_affinity"]
            and spec["port_start"] == expected_run["port_start"],
            f"contention child assignment drifted for {run_id}",
        )
        _require(
            spec["environment"]["CUDA_VISIBLE_DEVICES"]
            == str(expected_gpu["physical_index"])
            and spec["environment"]["VLLM_PORT"] == str(expected_run["port_start"])
            and spec["environment"]["VLLM_CACHE_ROOT"] == expected_gpu["cache_root"],
            f"contention child environment drifted for {run_id}",
        )
        templates = [row["template"] for row in spec["configs"]]
        _require(
            [row["matrix"]["round_index"] for row in templates] == list(ROUNDS)
            and all(
                row["matrix"]["action_id"] == ACTION_ID
                and row["matrix"]["regime_id"] == REGIME_ID
                and row["matrix"]["content_seed"] == CONTENT_SEED
                and row["generation"]["batch"] == 16
                and row["generation"]["max_output_tokens"] == 2048
                and len(row["generation"]["prompt_record_ids"]) == 32
                and row["runner"]["hardware_id"] == expected_gpu["uuid"]
                for row in templates
            ),
            f"contention child workload drifted for {run_id}",
        )
    _require(not OUTPUT_DIR.exists(), "validation created the contention output")
    return {
        "status": "pass",
        "package_id": EXPECTED_PACKAGE_ID,
        "authorization_decision": "approve_probe_only",
        "gpu_executed": False,
        "output_created": False,
        "gpu_assignments": {
            gpu_id: {
                "physical_index": row["physical_index"],
                "uuid": row["uuid"],
                "cpu_affinity": row["cpu_affinity"],
            }
            for gpu_id, row in GPU_ASSIGNMENTS.items()
        },
        "run_ids": list(RUNS),
        "schedule": SCHEDULE,
        "rounds_per_run": len(ROUNDS),
        "total_measured_rounds": len(RUNS) * len(ROUNDS),
        "action_id": ACTION_ID,
        "regime_id": REGIME_ID,
        "content_seed": CONTENT_SEED,
        "thresholds": THRESHOLDS,
        "dual_gpu_value_screen_authorized": False,
        "value_screen_scoring_authorized": False,
        "performance_claim_allowed": False,
        "next_artifact": "p4_b0_gpu_contention_probe_result",
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, default=AUTHORIZATION_PATH)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    """Validate the reviewed package and optionally write the audit."""
    args = parse_args()
    result = validate_contention_probe_authorization(_load_json(args.authorization))
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        try:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        except FileExistsError as exc:
            raise ContentionProbeAuthorizationError(
                f"refusing to overwrite {args.output}"
            ) from exc
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContentionProbeAuthorizationError as exc:
        print(f"P4 contention authorization refused: {exc}")
        raise SystemExit(2) from exc
