#!/usr/bin/env python3
"""Run the non-ephemeral-port GPU contention probe V2."""

from __future__ import annotations

import contextlib
import copy
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import run_p4_b0_gpu_contention_probe as base

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
AUTHORIZATION_PATH = (
    PHASE_DIR / "data" / "p4" / "p4_b0_gpu_contention_probe_authorization_v2.json"
)
OUTPUT_DIR = PHASE_DIR / "data" / "p4" / "run_b0_gpu_contention_probe_v2"
EXPECTED_PACKAGE_ID = "p4-b0-gpu-contention-probe-authorization-v2"
EXPECTED_STATUS = "authorized_gpu0_gpu1_non_ephemeral_contention_probe_only"
DECISION_SCOPE = "gpu0_gpu1_non_scored_contention_probe_v2_only"
DECISION_BASIS = [
    "v11_external_reassignment_preserved",
    "gpu0_gpu1_relocation_probe_passed",
    "contention_v1_ephemeral_port_failure_preserved",
    "contention_v1_serial_results_reuse_forbidden",
    "non_ephemeral_ports_and_cpu_affinity_registered",
    "current_execution_sources_hash_bound",
]
GPU_ASSIGNMENTS = copy.deepcopy(base.GPU_ASSIGNMENTS)
GPU_ASSIGNMENTS["gpu0"]["cache_root"] = "/data/smcho/.cache/vllm-p97-contention-v2/gpu0"
GPU_ASSIGNMENTS["gpu1"]["cache_root"] = "/data/smcho/.cache/vllm-p97-contention-v2/gpu1"
RUNS = {
    "serial-gpu0": {
        "condition": "serial",
        "gpu_id": "gpu0",
        "port_start": 20000,
    },
    "serial-gpu1": {
        "condition": "serial",
        "gpu_id": "gpu1",
        "port_start": 20100,
    },
    "concurrent-gpu0": {
        "condition": "concurrent",
        "gpu_id": "gpu0",
        "port_start": 20200,
    },
    "concurrent-gpu1": {
        "condition": "concurrent",
        "gpu_id": "gpu1",
        "port_start": 20300,
    },
}
SCHEDULE = copy.deepcopy(base.SCHEDULE)
REQUIRED_SOURCE_PATHS = dict(base.REQUIRED_SOURCE_PATHS)
REQUIRED_SOURCE_PATHS["contention_probe_base_runner"] = REQUIRED_SOURCE_PATHS[
    "contention_probe_runner"
]
REQUIRED_SOURCE_PATHS.update(
    {
        "contention_probe_runner": (
            "research/97_composition_runtime/scripts/"
            "run_p4_b0_gpu_contention_probe_v2.py"
        ),
        "contention_probe_schema": (
            "research/97_composition_runtime/schemas/"
            "p4_b0_gpu_contention_probe_authorization_v2.schema.json"
        ),
        "contention_probe_tests": (
            "research/97_composition_runtime/tests/"
            "test_p4_b0_gpu_contention_probe_v2.py"
        ),
        "contention_probe_validator": (
            "research/97_composition_runtime/scripts/"
            "validate_p4_b0_gpu_contention_probe_authorization_v2.py"
        ),
        "contention_probe_v1_authorization": (
            "research/97_composition_runtime/data/p4/"
            "p4_b0_gpu_contention_probe_authorization_v1.json"
        ),
        "contention_probe_v1_diagnosis": (
            "research/97_composition_runtime/data/p4/"
            "run_b0_gpu_contention_probe_v1/diagnosis.json"
        ),
        "contention_probe_v1_failure": (
            "research/97_composition_runtime/data/p4/"
            "run_b0_gpu_contention_probe_v1/failure.json"
        ),
        "contention_probe_v1_serial_gpu0": (
            "research/97_composition_runtime/data/p4/"
            "run_b0_gpu_contention_probe_v1/serial-gpu0/child_result.json"
        ),
        "contention_probe_v1_serial_gpu1": (
            "research/97_composition_runtime/data/p4/"
            "run_b0_gpu_contention_probe_v1/serial-gpu1/child_result.json"
        ),
    }
)

_CONFIGURED_FIELDS = {
    "AUTHORIZATION_PATH": AUTHORIZATION_PATH,
    "OUTPUT_DIR": OUTPUT_DIR,
    "SCRIPT_PATH": Path(__file__).resolve(),
    "EXPECTED_PACKAGE_ID": EXPECTED_PACKAGE_ID,
    "EXPECTED_STATUS": EXPECTED_STATUS,
    "DECISION_SCOPE": DECISION_SCOPE,
    "DECISION_BASIS": DECISION_BASIS,
    "PROBE_ARTIFACT_ID": "p4-b0-gpu-contention-probe-v2",
    "FAILURE_ARTIFACT_ID": "p4-b0-gpu-contention-probe-v2-failure",
    "GPU_ASSIGNMENTS": GPU_ASSIGNMENTS,
    "RUNS": RUNS,
    "SCHEDULE": SCHEDULE,
    "REQUIRED_SOURCE_PATHS": REQUIRED_SOURCE_PATHS,
}
_ORIGINAL_VALIDATE = base.validate_authorization


def _load(relative_path: str) -> dict[str, Any]:
    return base._load_json(REPO_ROOT / relative_path)


def _validate_v1_boundary() -> None:
    failure = _load(REQUIRED_SOURCE_PATHS["contention_probe_v1_failure"])
    diagnosis = _load(REQUIRED_SOURCE_PATHS["contention_probe_v1_diagnosis"])
    serial_gpu0 = _load(REQUIRED_SOURCE_PATHS["contention_probe_v1_serial_gpu0"])
    serial_gpu1 = _load(REQUIRED_SOURCE_PATHS["contention_probe_v1_serial_gpu1"])
    base._require(
        failure.get("status") == "failed_without_dual_gpu_authorization"
        and failure.get("exception", {}).get("message")
        == "port range for concurrent-gpu0 is not free"
        and failure.get("child_results_present")
        == {
            "serial-gpu0": True,
            "serial-gpu1": True,
            "concurrent-gpu0": False,
            "concurrent-gpu1": False,
        }
        and failure.get("disposition", {}).get("retry_allowed") is False,
        "contention V1 failure boundary drifted",
    )
    base._require(
        diagnosis.get("classification")
        == "launcher_port_preflight_collision_ephemeral_range"
        and diagnosis.get("diagnostic", {}).get("model_or_runtime_invariant_failure")
        is False
        and diagnosis.get("diagnostic", {}).get("concurrent_gpu_children_started")
        is False
        and diagnosis.get("disposition", {}).get("v1_serial_results_reusable") is False
        and diagnosis.get("repair_boundary", {}).get(
            "use_non_ephemeral_rendezvous_ranges"
        )
        is True
        and diagnosis.get("repair_boundary", {}).get(
            "rerun_complete_serial_and_concurrent_matrix"
        )
        is True,
        "contention V1 diagnosis boundary drifted",
    )
    base._require(
        serial_gpu0.get("status") == "pass"
        and serial_gpu1.get("status") == "pass"
        and serial_gpu0.get("scored") is False
        and serial_gpu1.get("scored") is False
        and serial_gpu0.get("claims", {}).get("score_eligible") is False
        and serial_gpu1.get("claims", {}).get("score_eligible") is False,
        "contention V1 serial evidence drifted",
    )


def _configured_validate(
    authorization: Mapping[str, Any], *, require_output_absent: bool
) -> None:
    _ORIGINAL_VALIDATE(
        authorization,
        require_output_absent=require_output_absent,
    )
    _validate_v1_boundary()


@contextlib.contextmanager
def _configuration() -> Iterator[None]:
    original = {name: getattr(base, name) for name in _CONFIGURED_FIELDS}
    original_validate = base.validate_authorization
    try:
        for name, value in _CONFIGURED_FIELDS.items():
            setattr(base, name, value)
        base.validate_authorization = _configured_validate
        yield
    finally:
        base.validate_authorization = original_validate
        for name, value in original.items():
            setattr(base, name, value)


def validate_authorization(
    authorization: Mapping[str, Any], *, require_output_absent: bool
) -> None:
    """Validate V2 without retaining module-global configuration."""
    with _configuration():
        base.validate_authorization(
            authorization,
            require_output_absent=require_output_absent,
        )


def build_probe_specs(output_dir: Path) -> list[dict[str, Any]]:
    """Build V2 child specs without retaining module-global configuration."""
    with _configuration():
        return base.build_probe_specs(output_dir)


def evaluate_gate(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate V2 with the unchanged registered numeric thresholds."""
    with _configuration():
        return base.evaluate_gate(results)


def main() -> int:
    """Run the fresh V2 package through the shared fail-closed implementation."""
    with _configuration():
        return base.main()


if __name__ == "__main__":
    parsed_output: Path | None = None
    is_child = "--child-spec" in sys.argv
    try:
        if "--output-dir" in sys.argv:
            parsed_output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
        raise SystemExit(main())
    except (base.ContentionProbeError, base.matrix.P4RunnerError) as exc:
        if parsed_output is not None and not is_child:
            with _configuration():
                base._preserve_failure(parsed_output, exc)
        print(f"P4 GPU contention probe V2 refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
