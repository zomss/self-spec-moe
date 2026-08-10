#!/usr/bin/env python3
"""Validate the non-ephemeral Phase 97 GPU contention-probe V2 package."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from run_p4_b0_gpu_contention_probe import ContentionProbeError
from run_p4_b0_gpu_contention_probe_v2 import (
    AUTHORIZATION_PATH,
    EXPECTED_PACKAGE_ID,
    GPU_ASSIGNMENTS,
    OUTPUT_DIR,
    RUNS,
    SCHEDULE,
    build_probe_specs,
    validate_authorization,
)

PHASE_DIR = Path(__file__).resolve().parents[1]
SCHEMA_PATH = (
    PHASE_DIR / "schemas" / "p4_b0_gpu_contention_probe_authorization_v2.schema.json"
)


class ContentionProbeAuthorizationV2Error(ValueError):
    """Raised when the V2 package drifts or inflates authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContentionProbeAuthorizationV2Error(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContentionProbeAuthorizationV2Error(
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
        raise ContentionProbeAuthorizationV2Error(
            f"invalid contention-probe V2 schema: {exc}"
        ) from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(authorization),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if errors:
        first = errors[0]
        path = _format_json_path(list(first.absolute_path))
        raise ContentionProbeAuthorizationV2Error(
            f"contention-probe V2 schema rejected {path}: {first.message}"
        )


def validate_contention_probe_authorization_v2(
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate V2 without creating output or running a GPU command."""
    _validate_schema(authorization)
    try:
        validate_authorization(authorization, require_output_absent=True)
        specs = build_probe_specs(OUTPUT_DIR)
    except ContentionProbeError as exc:
        raise ContentionProbeAuthorizationV2Error(
            f"contention runner rejects V2: {exc}"
        ) from exc
    _require(len(specs) == 4, "V2 no longer builds four complete runs")
    by_id = {spec["run_id"]: spec for spec in specs}
    _require(set(by_id) == set(RUNS), "V2 run identifiers drifted")
    for run_id, expected_run in RUNS.items():
        spec = by_id[run_id]
        gpu = GPU_ASSIGNMENTS[expected_run["gpu_id"]]
        _require(
            spec["condition"] == expected_run["condition"]
            and spec["gpu_id"] == expected_run["gpu_id"]
            and spec["gpu"]["physical_index"] == gpu["physical_index"]
            and spec["gpu"]["uuid"] == gpu["uuid"]
            and spec["cpu_affinity"] == gpu["cpu_affinity"]
            and spec["port_start"] == expected_run["port_start"]
            and spec["environment"]["VLLM_PORT"] == str(expected_run["port_start"])
            and spec["environment"]["VLLM_CACHE_ROOT"] == gpu["cache_root"],
            f"V2 child assignment drifted for {run_id}",
        )
        _require(
            len(spec["configs"]) == 4
            and [row["round_index"] for row in spec["configs"]] == [1, 2, 3, 4],
            f"V2 child workload drifted for {run_id}",
        )
    _require(
        all(run["port_start"] < 32768 for run in RUNS.values()),
        "V2 port range reentered the host ephemeral interval",
    )
    _require(not OUTPUT_DIR.exists(), "V2 validation created the output")
    return {
        "status": "pass",
        "package_id": EXPECTED_PACKAGE_ID,
        "authorization_decision": "approve_fresh_probe_only",
        "gpu_executed": False,
        "output_created": False,
        "v1_failure_preserved": True,
        "v1_serial_results_reused": False,
        "port_starts": {run_id: run["port_start"] for run_id, run in RUNS.items()},
        "ports_outside_host_ephemeral_range": True,
        "schedule": SCHEDULE,
        "run_count": len(specs),
        "rounds_per_run": 4,
        "dual_gpu_value_screen_authorized": False,
        "value_screen_scoring_authorized": False,
        "next_artifact": "p4_b0_gpu_contention_probe_result",
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, default=AUTHORIZATION_PATH)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    """Validate V2 and optionally write its audit."""
    args = parse_args()
    result = validate_contention_probe_authorization_v2(_load_json(args.authorization))
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        try:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        except FileExistsError as exc:
            raise ContentionProbeAuthorizationV2Error(
                f"refusing to overwrite {args.output}"
            ) from exc
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContentionProbeAuthorizationV2Error as exc:
        print(f"P4 contention authorization V2 refused: {exc}")
        raise SystemExit(2) from exc
