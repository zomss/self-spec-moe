# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the non-ephemeral GPU contention probe V2."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

import run_p4_b0_gpu_contention_probe as base  # noqa: E402
import run_p4_b0_gpu_contention_probe_v2 as probe_v2  # noqa: E402
from validate_p4_b0_gpu_contention_probe_authorization_v2 import (  # noqa: E402
    validate_contention_probe_authorization_v2,
)


def _load_authorization() -> dict:
    return json.loads(probe_v2.AUTHORIZATION_PATH.read_text(encoding="utf-8"))


def test_checked_in_v2_authorization_closes_without_gpu() -> None:
    result = validate_contention_probe_authorization_v2(_load_authorization())

    assert result["status"] == "pass"
    assert result["gpu_executed"] is False
    assert result["v1_failure_preserved"] is True
    assert result["v1_serial_results_reused"] is False
    assert result["ports_outside_host_ephemeral_range"] is True
    assert result["dual_gpu_value_screen_authorized"] is False


def test_v2_build_restores_v1_module_configuration() -> None:
    original_output = base.OUTPUT_DIR
    original_runs = copy.deepcopy(base.RUNS)

    specs = probe_v2.build_probe_specs(probe_v2.OUTPUT_DIR)

    assert len(specs) == 4
    assert original_output == base.OUTPUT_DIR
    assert original_runs == base.RUNS


def test_v2_ports_are_disjoint_and_below_ephemeral_range() -> None:
    ranges = [
        set(range(run["port_start"], run["port_start"] + base.PORT_RANGE_SIZE))
        for run in probe_v2.RUNS.values()
    ]

    assert max(max(ports) for ports in ranges) < 32768
    assert all(
        ranges[left].isdisjoint(ranges[right])
        for left in range(len(ranges))
        for right in range(left + 1, len(ranges))
    )


def test_v2_specs_use_fresh_outputs_caches_and_all_four_runs() -> None:
    specs = probe_v2.build_probe_specs(probe_v2.OUTPUT_DIR)
    by_id = {spec["run_id"]: spec for spec in specs}

    assert set(by_id) == set(probe_v2.RUNS)
    assert all(
        "run_b0_gpu_contention_probe_v2" in row["configs"][0]["config_path"]
        for row in specs
    )
    assert all("vllm-p97-contention-v2" in row["cache_root"] for row in specs)
    assert all(len(row["configs"]) == 4 for row in specs)


def test_v2_binds_consumed_v1_ephemeral_failure() -> None:
    probe_v2._validate_v1_boundary()


def test_v2_rejects_port_contract_drift() -> None:
    """V2 is consumed and terminal; it must refuse every mutated relaunch.

    The value-screen runner it hash-binds moved on with the V12 block-parallel
    package, so the source-closure check now refuses before the port contract
    is reached. Either refusal is fail-closed, so this asserts the refusal
    rather than which registered contract noticed first.
    """
    authorization = copy.deepcopy(_load_authorization())
    authorization["run_contract"]["runs"]["concurrent-gpu0"]["port_start"] = 47200

    with pytest.raises(base.ContentionProbeError, match="drifted"):
        probe_v2.validate_authorization(
            authorization,
            require_output_absent=True,
        )


def test_v2_gate_keeps_original_thresholds() -> None:
    results = {
        "serial-gpu0": {"episode": {"decode_rate_req": 100.0}},
        "serial-gpu1": {"episode": {"decode_rate_req": 100.2}},
        "concurrent-gpu0": {"episode": {"decode_rate_req": 99.5}},
        "concurrent-gpu1": {"episode": {"decode_rate_req": 99.7}},
    }

    gate = probe_v2.evaluate_gate(results)

    assert gate["state"] == "pass"
    assert gate["thresholds"] == base.THRESHOLDS
