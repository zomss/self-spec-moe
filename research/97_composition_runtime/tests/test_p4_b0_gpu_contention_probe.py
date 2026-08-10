"""CPU tests for the GPU-0/GPU-1 contention probe."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

import run_p4_b0_gpu_contention_probe as probe  # noqa: E402


def _load_authorization() -> dict:
    return json.loads(probe.AUTHORIZATION_PATH.read_text(encoding="utf-8"))


def _fake_results(rates: dict[str, float]) -> dict[str, dict]:
    return {
        run_id: {"episode": {"decode_rate_req": rate}} for run_id, rate in rates.items()
    }


def _round(round_index: int, rate: float) -> dict:
    committed = 1000
    return {
        "matrix": {"round_index": round_index},
        "counters": {"E_committed": committed},
        "timing": {"request_decode_time_s": committed / rate},
        "estimands": {"decode_rate_req": rate},
    }


def test_consumed_v1_is_preserved_without_dual_gpu_authority() -> None:
    diagnosis = json.loads(
        (probe.OUTPUT_DIR / "diagnosis.json").read_text(encoding="utf-8")
    )
    failure = json.loads(
        (probe.OUTPUT_DIR / "failure.json").read_text(encoding="utf-8")
    )

    assert diagnosis["classification"] == (
        "launcher_port_preflight_collision_ephemeral_range"
    )
    assert diagnosis["disposition"]["v1_serial_results_reusable"] is False
    assert failure["status"] == "failed_without_dual_gpu_authorization"
    assert failure["disposition"]["dual_gpu_value_screen_authorized"] is False


def test_probe_specs_keep_gpu_ports_cpu_and_cache_disjoint() -> None:
    specs = probe.build_probe_specs(probe.OUTPUT_DIR)
    by_id = {spec["run_id"]: spec for spec in specs}

    assert by_id["serial-gpu0"]["environment"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert by_id["serial-gpu1"]["environment"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert by_id["concurrent-gpu0"]["environment"]["VLLM_PORT"] == "47200"
    assert by_id["concurrent-gpu1"]["environment"]["VLLM_PORT"] == "47300"
    assert (
        by_id["concurrent-gpu0"]["environment"]["VLLM_CACHE_ROOT"]
        != by_id["concurrent-gpu1"]["environment"]["VLLM_CACHE_ROOT"]
    )
    gpu0_cpus = probe._parse_cpu_affinity(by_id["serial-gpu0"]["cpu_affinity"])
    gpu1_cpus = probe._parse_cpu_affinity(by_id["serial-gpu1"]["cpu_affinity"])
    assert len(gpu0_cpus) == 96
    assert len(gpu1_cpus) == 96
    assert gpu0_cpus.isdisjoint(gpu1_cpus)


def test_probe_specs_use_four_exact_registered_r8_k4_rounds() -> None:
    specs = probe.build_probe_specs(probe.OUTPUT_DIR)
    for spec in specs:
        templates = [row["template"] for row in spec["configs"]]
        assert [row["matrix"]["round_index"] for row in templates] == [1, 2, 3, 4]
        assert all(
            row["matrix"]["action_id"] == "target-matching-k4"
            and row["matrix"]["regime_id"] == "R8"
            and row["generation"]["batch"] == 16
            and row["generation"]["max_output_tokens"] == 2048
            for row in templates
        )


def test_episode_summary_reuses_five_percent_filter() -> None:
    summary = probe._episode_summary(
        [_round(1, 100.0), _round(2, 99.0), _round(3, 94.0), _round(4, 98.0)]
    )

    assert summary["reference_rate"] == 100.0
    assert summary["surviving_round_indices"] == [1, 2, 4]
    assert summary["surviving_round_count"] == 3


def test_gate_passes_small_symmetric_contention() -> None:
    gate = probe.evaluate_gate(
        _fake_results(
            {
                "serial-gpu0": 100.0,
                "serial-gpu1": 100.5,
                "concurrent-gpu0": 99.5,
                "concurrent-gpu1": 100.0,
            }
        )
    )

    assert gate["state"] == "pass"
    assert gate["dual_gpu_block_parallelism_supportable"] is True
    assert all(gate["checks"].values())


def test_gate_holds_on_more_than_one_percent_slowdown() -> None:
    gate = probe.evaluate_gate(
        _fake_results(
            {
                "serial-gpu0": 100.0,
                "serial-gpu1": 100.0,
                "concurrent-gpu0": 97.0,
                "concurrent-gpu1": 100.0,
            }
        )
    )

    assert gate["state"] == "hold"
    assert gate["checks"]["gpu0_slowdown_within_one_percent"] is False
    assert gate["dual_gpu_block_parallelism_supportable"] is False


def test_authorization_rejects_gpu_uuid_drift() -> None:
    authorization = copy.deepcopy(_load_authorization())
    authorization["source_artifacts"] = {
        role: probe._source_reference(path)
        for role, path in probe.REQUIRED_SOURCE_PATHS.items()
    }
    authorization["run_contract"]["gpu_assignments"]["gpu0"]["uuid"] = (
        "GPU-00000000-0000-0000-0000-000000000000"
    )

    with pytest.raises(probe.ContentionProbeError, match="run contract drifted"):
        probe.validate_authorization(authorization, require_output_absent=False)


def test_registered_port_ranges_are_disjoint() -> None:
    ranges = [
        set(range(run["port_start"], run["port_start"] + probe.PORT_RANGE_SIZE))
        for run in probe.RUNS.values()
    ]
    assert all(
        ranges[left].isdisjoint(ranges[right])
        for left in range(len(ranges))
        for right in range(left + 1, len(ranges))
    )
