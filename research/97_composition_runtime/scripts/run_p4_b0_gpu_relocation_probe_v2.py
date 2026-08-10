#!/usr/bin/env python3
"""Run the port-isolated GPU-0/GPU-1 relocation probe V2."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import run_p4_b0_gpu_relocation_probe as probe_v1
import run_p4_b0_r5cot_r8_diagnosis as base
import run_p4_b0_r5cot_r8_diagnosis_v2 as diagnosis_v2
import run_p4_b0_variable_prefill_repair_validation as repair

AUTHORIZATION_PATH = (
    base.PHASE_DIR / "data" / "p4" / "p4_b0_gpu_relocation_probe_authorization_v2.json"
)
OUTPUT_DIR = base.PHASE_DIR / "data" / "p4" / "run_b0_gpu_relocation_probe_v2"
EXPECTED_PACKAGE_ID = "p4-b0-gpu-relocation-probe-authorization-v2"
RENDEZVOUS_PORT_STARTS = {
    "isolated-r8": 46000,
    "r5cot-to-r8": 46100,
}
REQUIRED_SOURCE_PATHS = {
    **probe_v1.REQUIRED_SOURCE_PATHS,
    "relocation_probe_runner": (
        "research/97_composition_runtime/scripts/run_p4_b0_gpu_relocation_probe_v2.py"
    ),
    "relocation_probe_v1_authorization": (
        "research/97_composition_runtime/data/p4/"
        "p4_b0_gpu_relocation_probe_authorization_v1.json"
    ),
    "relocation_probe_v1_failure": (
        "research/97_composition_runtime/data/p4/"
        "run_b0_gpu_relocation_probe_v1/failure.json"
    ),
}
_ORIGINAL_CHILD_ENVIRONMENT = base._child_environment


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise probe_v1.RelocationProbeError(message)


def _child_environment(
    authorization: Mapping[str, Any], case_id: str, case_dir: Path
) -> dict[str, str]:
    environment = _ORIGINAL_CHILD_ENVIRONMENT(authorization, case_id, case_dir)
    port_starts = authorization["run_contract"].get("rendezvous_port_starts")
    _require(
        port_starts == RENDEZVOUS_PORT_STARTS,
        "relocation rendezvous port starts drifted",
    )
    environment["VLLM_PORT"] = str(port_starts[case_id])
    return environment


def _validate_v1_boundary(authorization: Mapping[str, Any]) -> None:
    failure = base._load_json(
        base.REPO_ROOT / REQUIRED_SOURCE_PATHS["relocation_probe_v1_failure"]
    )
    _require(
        failure.get("status") == "inconclusive_launcher_port_collision"
        and failure.get("diagnostic", {}).get("exception_marker") == "EADDRINUSE"
        and failure.get("diagnostic", {}).get("model_or_runtime_invariant_failure")
        is False
        and failure.get("disposition", {}).get("v1_authorization_consumed") is True
        and failure.get("disposition", {}).get("retry_allowed") is False,
        "relocation probe V1 failure boundary drifted",
    )
    _require(
        authorization.get("claims")
        == {
            "v11_interruption_preserved": True,
            "relocation_probe_v1_preserved": True,
            "per_child_rendezvous_isolation_tested": True,
            "gpu0_gpu1_relocation_probe_authorized": True,
            "value_screen_execution_authorized": False,
            "performance_claim_allowed": False,
            "action_admission_allowed": False,
        },
        "V2 relocation probe claims exceed the non-scored boundary",
    )
    _require(
        authorization.get("authorizations")
        == {
            "gpu0_isolated_r8_probe": True,
            "gpu1_r5cot_r8_transition_probe": True,
            "value_screen_execution": False,
            "value_screen_scoring": False,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "V2 relocation probe authority is inflated",
    )


def _validate_v11_boundary() -> None:
    manifest = base._load_json(
        base.REPO_ROOT / REQUIRED_SOURCE_PATHS["v11_capture_manifest"]
    )
    interruption = base._load_json(
        base.REPO_ROOT / REQUIRED_SOURCE_PATHS["v11_interruption"]
    )
    _require(
        manifest.get("counts", {}).get("complete_captures") == 48
        and manifest.get("counts", {}).get("empty_placeholders") == 1
        and manifest.get("invariants", {}).get("score_absent") is True,
        "V11 immutable capture boundary drifted",
    )
    _require(
        interruption.get("diagnostic", {}).get("classification")
        == "external_resource_reassignment"
        and interruption.get("disposition", {}).get("v11_consumed") is True
        and interruption.get("disposition", {}).get("scoring_allowed") is False,
        "V11 interruption disposition drifted",
    )


def _configure_base() -> None:
    original_validate = base.validate_authorization
    base.AUTHORIZATION_PATH = AUTHORIZATION_PATH
    base.OUTPUT_DIR = OUTPUT_DIR
    base.EXPECTED_PACKAGE_ID = EXPECTED_PACKAGE_ID
    base.REQUIRED_SOURCE_PATHS = REQUIRED_SOURCE_PATHS
    base._scheduler_snapshot = diagnosis_v2.scheduler_snapshot
    base._install_instrumentation = repair._install_repair_instrumentation
    base._child_environment = _child_environment
    base.__file__ = str(Path(__file__).resolve())
    probe_v1.AUTHORIZATION_PATH = AUTHORIZATION_PATH

    def validate_authorization(*args: Any, **kwargs: Any) -> None:
        original_validate(*args, **kwargs)
        _validate_v11_boundary()
        _validate_v1_boundary(args[0])

    base.validate_authorization = validate_authorization


def main() -> int:
    """Run the create-only, port-isolated relocation probe V2."""
    _configure_base()
    child = "--child-case" in sys.argv
    prepare_only = "--prepare-only" in sys.argv
    result = base.main()
    if not child and not prepare_only:
        probe_v1.finalize_probe(OUTPUT_DIR.resolve())
    return result


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (base.P4TransitionDiagnosisError, probe_v1.RelocationProbeError) as exc:
        print(f"P4 GPU relocation probe V2 refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
