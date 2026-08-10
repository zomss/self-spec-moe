"""CPU tests for the physical GPU-0/GPU-1 relocation probe."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

import run_p4_b0_gpu_relocation_probe_v2 as probe_v2  # noqa: E402
from run_p4_b0_gpu_relocation_probe import (  # noqa: E402
    REQUIRED_SOURCE_PATHS,
    RelocationProbeError,
    _validate_v11_boundary,
    validate_case_result,
)

PRIOR_CASE_ROOT = (
    PHASE_DIR / "data" / "p4" / "run_b0_variable_prefill_repair_validation_v1"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _probe_authority() -> dict:
    return {
        "claims": {
            "v11_interruption_preserved": True,
            "gpu0_gpu1_relocation_probe_authorized": True,
            "value_screen_execution_authorized": False,
            "performance_claim_allowed": False,
            "action_admission_allowed": False,
        },
        "authorizations": {
            "gpu0_isolated_r8_probe": True,
            "gpu1_r5cot_r8_transition_probe": True,
            "value_screen_execution": False,
            "value_screen_scoring": False,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
    }


def test_preserved_v11_boundary_allows_only_non_scored_probe() -> None:
    _validate_v11_boundary(_probe_authority())


def test_v11_boundary_rejects_value_screen_authority() -> None:
    authority = _probe_authority()
    authority["authorizations"]["value_screen_execution"] = True
    with pytest.raises(RelocationProbeError, match="authority is inflated"):
        _validate_v11_boundary(authority)


def test_checked_in_gpu_cases_close_exact_relocation_signatures() -> None:
    isolated = validate_case_result(
        "isolated-r8",
        _load(PRIOR_CASE_ROOT / "isolated-r8" / "case_result.json"),
    )
    transition = validate_case_result(
        "r5cot-to-r8",
        _load(PRIOR_CASE_ROOT / "r5cot-to-r8" / "case_result.json"),
    )

    assert isolated["gpu"]["physical_index"] == 0
    assert isolated["armed_prefill_observation_count"] == 1
    assert transition["gpu"]["physical_index"] == 1
    assert transition["armed_prefill_observation_count"] == 9
    assert isolated["shared_target_kv_block_capacity"] == 24527
    assert transition["shared_target_kv_block_capacity"] == 24527


def test_probe_source_closure_binds_v11_preservation() -> None:
    assert REQUIRED_SOURCE_PATHS["v11_capture_manifest"].endswith(
        "run_b0_value_screen_v10/capture_manifest.json"
    )
    assert REQUIRED_SOURCE_PATHS["v11_interruption"].endswith(
        "run_b0_value_screen_v10/failure.json"
    )
    assert REQUIRED_SOURCE_PATHS["v11_preservation_script"].endswith(
        "preserve_p4_b0_value_screen_v11_attempt.py"
    )


def test_v2_assigns_disjoint_rendezvous_ranges(monkeypatch) -> None:
    monkeypatch.setattr(
        probe_v2,
        "_ORIGINAL_CHILD_ENVIRONMENT",
        lambda authorization, case_id, case_dir: {
            "CUDA_VISIBLE_DEVICES": str(
                authorization["run_contract"]["jobs"][case_id]["gpu_index"]
            )
        },
    )
    authorization = {
        "run_contract": {
            "rendezvous_port_starts": probe_v2.RENDEZVOUS_PORT_STARTS,
            "jobs": {
                "isolated-r8": {"gpu_index": 0},
                "r5cot-to-r8": {"gpu_index": 1},
            },
        }
    }

    gpu0 = probe_v2._child_environment(authorization, "isolated-r8", Path("/tmp/gpu0"))
    gpu1 = probe_v2._child_environment(authorization, "r5cot-to-r8", Path("/tmp/gpu1"))

    assert gpu0["CUDA_VISIBLE_DEVICES"] == "0"
    assert gpu1["CUDA_VISIBLE_DEVICES"] == "1"
    assert gpu0["VLLM_PORT"] == "46000"
    assert gpu1["VLLM_PORT"] == "46100"
    assert int(gpu1["VLLM_PORT"]) - int(gpu0["VLLM_PORT"]) == 100


def test_v2_binds_consumed_v1_port_collision() -> None:
    authority = _probe_authority()
    authority["claims"] = {
        "v11_interruption_preserved": True,
        "relocation_probe_v1_preserved": True,
        "per_child_rendezvous_isolation_tested": True,
        "gpu0_gpu1_relocation_probe_authorized": True,
        "value_screen_execution_authorized": False,
        "performance_claim_allowed": False,
        "action_admission_allowed": False,
    }
    probe_v2._validate_v1_boundary(authority)
