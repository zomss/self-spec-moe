#!/usr/bin/env python3
"""Run the source-bound GPU-0/GPU-1 P4 contention probe."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import run_p4_b0_r5cot_r8_diagnosis as diagnosis
import run_p4_b0_value_screen as matrix

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
SCRIPT_PATH = Path(__file__).resolve()
AUTHORIZATION_PATH = (
    PHASE_DIR / "data" / "p4" / "p4_b0_gpu_contention_probe_authorization_v1.json"
)
OUTPUT_DIR = PHASE_DIR / "data" / "p4" / "run_b0_gpu_contention_probe_v1"
BASE_AUTHORIZATION_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization_v2.json"
EXPECTED_PACKAGE_ID = "p4-b0-gpu-contention-probe-authorization-v1"
EXPECTED_STATUS = "authorized_gpu0_gpu1_non_scored_contention_probe_only"
DECISION_SCOPE = "gpu0_gpu1_non_scored_contention_probe_v1_only"
DECISION_BASIS = [
    "v11_external_reassignment_preserved",
    "gpu0_gpu1_relocation_probe_passed",
    "single_cell_same_event_measurement_reused",
    "fixed_ports_and_cpu_affinity_registered",
    "current_execution_sources_hash_bound",
]
PROBE_ARTIFACT_ID = "p4-b0-gpu-contention-probe-v1"
FAILURE_ARTIFACT_ID = "p4-b0-gpu-contention-probe-v1-failure"
ACTION_ID = "target-matching-k4"
SOURCE_BOOT_ID = "p4-b0-b1-p2-k4"
REGIME_ID = "R8"
CONTENT_SEED = 0
ROUNDS = (1, 2, 3, 4)
PORT_RANGE_SIZE = 100
GPU_ASSIGNMENTS = {
    "gpu0": {
        "physical_index": 0,
        "uuid": "GPU-4938442e-5508-9249-0fa6-37baa1985703",
        "cpu_affinity": "0-47,96-143",
        "cache_root": "/data/smcho/.cache/vllm-p97-contention-v1/gpu0",
    },
    "gpu1": {
        "physical_index": 1,
        "uuid": "GPU-ba39f4f0-61fe-34ca-c1af-ffe565b70923",
        "cpu_affinity": "48-95,144-191",
        "cache_root": "/data/smcho/.cache/vllm-p97-contention-v1/gpu1",
    },
}
RUNS = {
    "serial-gpu0": {
        "condition": "serial",
        "gpu_id": "gpu0",
        "port_start": 47000,
    },
    "serial-gpu1": {
        "condition": "serial",
        "gpu_id": "gpu1",
        "port_start": 47100,
    },
    "concurrent-gpu0": {
        "condition": "concurrent",
        "gpu_id": "gpu0",
        "port_start": 47200,
    },
    "concurrent-gpu1": {
        "condition": "concurrent",
        "gpu_id": "gpu1",
        "port_start": 47300,
    },
}
SCHEDULE = [
    ["serial-gpu0"],
    ["serial-gpu1"],
    ["concurrent-gpu0", "concurrent-gpu1"],
]
THRESHOLDS = {
    "episode_floor_fraction": 0.95,
    "minimum_surviving_rounds": 2,
    "maximum_per_gpu_concurrent_slowdown_fraction": 0.01,
    "maximum_contention_ratio_disagreement_fraction": 0.01,
    "maximum_cross_gpu_rate_disagreement_fraction": 0.02,
}
REQUIRED_SOURCE_PATHS = {
    "base_authorization": (
        "research/97_composition_runtime/data/p4/p4_b0_run_authorization_v2.json"
    ),
    "capture_adapter": (
        "research/97_composition_runtime/scripts/adapt_p4_b0_same_event.py"
    ),
    "contention_probe_runner": (
        "research/97_composition_runtime/scripts/run_p4_b0_gpu_contention_probe.py"
    ),
    "contention_probe_schema": (
        "research/97_composition_runtime/schemas/"
        "p4_b0_gpu_contention_probe_authorization.schema.json"
    ),
    "contention_probe_tests": (
        "research/97_composition_runtime/tests/test_p4_b0_gpu_contention_probe.py"
    ),
    "contention_probe_validator": (
        "research/97_composition_runtime/scripts/"
        "validate_p4_b0_gpu_contention_probe_authorization.py"
    ),
    "draft_proposer": "vllm/v1/spec_decode/llm_base_proposer.py",
    "gpu_model_runner": "vllm/v1/worker/gpu_model_runner.py",
    "gpu_worker": "vllm/v1/worker/gpu_worker.py",
    "koff_runtime": "vllm/v1/spec_decode/koff_runtime.py",
    "prompt_bundle": (
        "research/97_composition_runtime/data/p4/p4_b0_prompt_tokens.jsonl.gz"
    ),
    "prompt_manifest": (
        "research/97_composition_runtime/data/p4/p4_b0_prompt_manifest.json"
    ),
    "relocation_probe_authorization": (
        "research/97_composition_runtime/data/p4/"
        "p4_b0_gpu_relocation_probe_authorization_v2.json"
    ),
    "relocation_probe_result": (
        "research/97_composition_runtime/data/p4/"
        "run_b0_gpu_relocation_probe_v2/probe_result.json"
    ),
    "scheduler": "vllm/v1/core/sched/scheduler.py",
    "v11_authorization": (
        "research/97_composition_runtime/data/p4/p4_b0_run_authorization_v11.json"
    ),
    "v11_capture_manifest": (
        "research/97_composition_runtime/data/p4/"
        "run_b0_value_screen_v10/capture_manifest.json"
    ),
    "v11_interruption": (
        "research/97_composition_runtime/data/p4/run_b0_value_screen_v10/failure.json"
    ),
    "value_screen_runner": (
        "research/97_composition_runtime/scripts/run_p4_b0_value_screen.py"
    ),
}


class ContentionProbeError(RuntimeError):
    """Raised when the bounded contention probe cannot close exactly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContentionProbeError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContentionProbeError(f"cannot load JSON artifact {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON artifact must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def _reference(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise ContentionProbeError(f"refusing to overwrite {path}") from exc


def _source_reference(relative_path: str) -> dict[str, str]:
    path = REPO_ROOT / relative_path
    _require(path.is_file(), f"required source artifact is missing: {relative_path}")
    return {"path": relative_path, "sha256": _sha256(path)}


def _artifact_reference(relative_path: str) -> dict[str, Any]:
    path = REPO_ROOT / relative_path
    _require(path.is_file(), f"required evidence artifact is missing: {relative_path}")
    return _reference(path)


def _expected_run_contract() -> dict[str, Any]:
    return {
        "scope": "source_bound_non_scored_serial_vs_concurrent_aa_probe_only",
        "scored": False,
        "output_dir": _relative(OUTPUT_DIR),
        "source_value_screen": {
            "base_authorization": _artifact_reference(
                REQUIRED_SOURCE_PATHS["base_authorization"]
            ),
            "consumed_v11_authorization": _artifact_reference(
                REQUIRED_SOURCE_PATHS["v11_authorization"]
            ),
        },
        "engine": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "max_model_len": 20480,
            "max_num_batched_tokens": 8192,
            "effective_max_num_scheduled_tokens": 8160,
            "max_num_seqs": 32,
            "gpu_memory_utilization": 0.9,
            "num_speculative_tokens": 4,
            "enable_chunked_prefill": True,
            "async_scheduling": False,
            "prefix_caching": False,
            "flashinfer_autotune": False,
            "enforce_eager": False,
            "generation_seed": 0,
        },
        "workload": {
            "source_boot_id": SOURCE_BOOT_ID,
            "action_id": ACTION_ID,
            "boot_block_id": 1,
            "action_position": 2,
            "regime_id": REGIME_ID,
            "content_seed": CONTENT_SEED,
            "rounds": list(ROUNDS),
            "prompts_per_round": 32,
            "batch": 16,
            "max_output_tokens": 2048,
            "measurement_currency": "S_dec",
        },
        "gpu_assignments": copy.deepcopy(GPU_ASSIGNMENTS),
        "runs": copy.deepcopy(RUNS),
        "schedule": copy.deepcopy(SCHEDULE),
        "port_range_size": PORT_RANGE_SIZE,
        "thresholds": copy.deepcopy(THRESHOLDS),
        "attempt_policy": {
            "create_new_output_required": True,
            "partial_resume_allowed": False,
            "retry_allowed": False,
            "score_input_allowed": False,
            "on_any_failure": "preserve_without_dual_gpu_authorization",
        },
    }


def validate_authorization(
    authorization: Mapping[str, Any], *, require_output_absent: bool
) -> None:
    """Validate the exact non-scored contention-probe authority."""
    expected_keys = {
        "schema_version",
        "package_id",
        "date",
        "status",
        "consumed_v11",
        "relocation_probe",
        "source_artifacts",
        "run_contract",
        "decision",
        "claims",
        "authorizations",
        "next_artifact",
    }
    _require(set(authorization) == expected_keys, "authorization fields drifted")
    _require(
        authorization.get("schema_version") == 1
        and authorization.get("package_id") == EXPECTED_PACKAGE_ID
        and authorization.get("status") == EXPECTED_STATUS,
        "contention-probe identity drifted",
    )
    expected_v11 = {
        "authorization": _artifact_reference(
            REQUIRED_SOURCE_PATHS["v11_authorization"]
        ),
        "capture_manifest": _artifact_reference(
            REQUIRED_SOURCE_PATHS["v11_capture_manifest"]
        ),
        "interruption": _artifact_reference(REQUIRED_SOURCE_PATHS["v11_interruption"]),
        "disposition": "consumed_external_resource_reassignment",
        "complete_capture_count": 48,
        "empty_placeholder_count": 1,
        "score_emitted": False,
        "reuse_allowed": False,
    }
    _require(
        authorization.get("consumed_v11") == expected_v11,
        "consumed V11 boundary drifted",
    )
    interruption = _load_json(REPO_ROOT / REQUIRED_SOURCE_PATHS["v11_interruption"])
    manifest = _load_json(REPO_ROOT / REQUIRED_SOURCE_PATHS["v11_capture_manifest"])
    _require(
        interruption.get("diagnostic", {}).get("classification")
        == "external_resource_reassignment"
        and interruption.get("disposition", {}).get("v11_consumed") is True
        and interruption.get("disposition", {}).get("scoring_allowed") is False
        and manifest.get("counts", {}).get("complete_captures") == 48
        and manifest.get("counts", {}).get("empty_placeholders") == 1,
        "V11 preservation evidence no longer closes",
    )
    expected_relocation = {
        "authorization": _artifact_reference(
            REQUIRED_SOURCE_PATHS["relocation_probe_authorization"]
        ),
        "result": _artifact_reference(REQUIRED_SOURCE_PATHS["relocation_probe_result"]),
        "status": "pass",
        "scored": False,
        "gpu0_exact_r8_passed": True,
        "gpu1_r5cot_to_r8_passed": True,
        "shared_kv_floor_passed_on_both": True,
        "performance_claim_allowed": False,
    }
    _require(
        authorization.get("relocation_probe") == expected_relocation,
        "relocation-probe evidence drifted",
    )
    relocation = _load_json(
        REPO_ROOT / REQUIRED_SOURCE_PATHS["relocation_probe_result"]
    )
    _require(
        relocation.get("status") == "pass"
        and relocation.get("scored") is False
        and relocation.get("claims")
        == {
            "gpu0_exact_r8_passed": True,
            "gpu1_r5cot_to_r8_passed": True,
            "shared_kv_floor_passed_on_both": True,
            "full_value_screen_authorized": False,
            "performance_claim_allowed": False,
        },
        "relocation result no longer supports a bounded contention probe",
    )
    references = authorization.get("source_artifacts")
    _require(isinstance(references, Mapping), "source closure is missing")
    _require(
        set(references) == set(REQUIRED_SOURCE_PATHS),
        "source closure is incomplete or inflated",
    )
    for role, relative_path in REQUIRED_SOURCE_PATHS.items():
        _require(
            references[role] == _source_reference(relative_path),
            f"source hash drifted for {role}",
        )
    _require(
        authorization.get("run_contract") == _expected_run_contract(),
        "contention run contract drifted",
    )
    _require(
        authorization.get("decision")
        == {
            "state": "approve",
            "scope": DECISION_SCOPE,
            "basis": DECISION_BASIS,
            "invalidated_by": [
                "source_hash_drift",
                "output_directory_exists",
                "gpu_identity_drift",
                "gpu_not_idle",
                "port_range_not_free",
                "cpu_affinity_drift",
                "capture_or_episode_failure",
                "threshold_contract_drift",
            ],
        },
        "contention-probe decision drifted",
    )
    _require(
        authorization.get("claims")
        == {
            "v11_interruption_preserved": True,
            "gpu0_gpu1_functional_relocation_passed": True,
            "contention_probe_executable": True,
            "contention_bound_measured": False,
            "dual_gpu_value_screen_authorized": False,
            "value_screen_scoring_authorized": False,
            "performance_claim_allowed": False,
            "action_admitted": False,
        },
        "contention-probe claims are inflated",
    )
    _require(
        authorization.get("authorizations")
        == {
            "contention_probe_execution": True,
            "dual_gpu_value_screen_execution": False,
            "value_screen_scoring": False,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "contention-probe authority is inflated",
    )
    _require(
        authorization.get("next_artifact")
        == {
            "kind": "p4_b0_gpu_contention_probe_result",
            "may_support_dual_gpu_authorization": True,
            "direct_value_screen_authority": False,
            "direct_scoring_authority": False,
        },
        "contention-probe next-artifact boundary drifted",
    )
    if require_output_absent:
        _require(not OUTPUT_DIR.exists(), "contention-probe output already exists")


def _effective_matrix_authorization() -> dict[str, Any]:
    base = _load_json(BASE_AUTHORIZATION_PATH)
    _require(
        base.get("package_id") == matrix.BASE_PACKAGE_ID
        and base.get("schema_version") == 2,
        "frozen base value-screen package drifted",
    )
    effective = matrix.apply_chunked_prefill_engine_contract(base)
    effective["package_id"] = matrix.V11_PACKAGE_ID
    effective["schema_version"] = 11
    effective["run_contract"]["environment"][matrix.V1_MULTIPROCESSING_ENV] = (
        matrix.V1_MULTIPROCESSING_VALUE
    )
    return effective


def _selected_source_spec() -> dict[str, Any]:
    effective = _effective_matrix_authorization()
    specs = matrix.build_boot_specs(effective, OUTPUT_DIR / "unmaterialized")
    matches = [spec for spec in specs if spec["boot_id"] == SOURCE_BOOT_ID]
    _require(len(matches) == 1, "registered K4 source boot is missing")
    source = matches[0]
    cells = [
        cell
        for cell in source["plan"]["cells"]
        if cell["matrix"]["regime_id"] == REGIME_ID
        and cell["matrix"]["content_seed"] == CONTENT_SEED
    ]
    _require(
        [cell["matrix"]["round_index"] for cell in cells] == list(ROUNDS),
        "registered R8/K4 rounds drifted",
    )
    _require(
        all(
            cell["matrix"]["action_id"] == ACTION_ID
            and cell["generation"]["batch"] == 16
            and cell["generation"]["max_output_tokens"] == 2048
            and len(cell["generation"]["prompt_record_ids"]) == 32
            for cell in cells
        ),
        "registered R8/K4 work drifted",
    )
    source["probe_cells"] = cells
    return source


def build_probe_specs(output_dir: Path) -> list[dict[str, Any]]:
    """Build all four exact child specs without creating files."""
    source = _selected_source_spec()
    specs = []
    for run_id, run in RUNS.items():
        gpu = GPU_ASSIGNMENTS[run["gpu_id"]]
        run_dir = output_dir / run_id
        configs = []
        for cell in source["probe_cells"]:
            round_index = cell["matrix"]["round_index"]
            capture_id = f"contention-{run_id}-r{round_index}"
            config_path = run_dir / "configs" / f"{capture_id}.json"
            capture_path = run_dir / "captures" / f"{capture_id}.json"
            template = copy.deepcopy(cell)
            template["capture_id"] = capture_id
            template["matrix"]["boot_id"] = f"p4-b0-contention-{run_id}"
            template["runner"]["hardware_id"] = gpu["uuid"]
            configs.append(
                {
                    "round_index": round_index,
                    "config_path": str(config_path.resolve()),
                    "capture_path": str(capture_path.resolve()),
                    "template": template,
                }
            )
        environment = dict(source["environment"])
        environment.update(
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(gpu["physical_index"]),
                "VLLM_CACHE_ROOT": gpu["cache_root"],
                "TORCHINDUCTOR_CACHE_DIR": str(Path(gpu["cache_root"]) / "inductor"),
                "TRITON_CACHE_DIR": str(Path(gpu["cache_root"]) / "triton"),
                "VLLM_PORT": str(run["port_start"]),
                "VLLM_SELF_SPEC_P4_CAPTURE_CONFIG": configs[0]["config_path"],
                "VLLM_SELF_SPEC_P4_CAPTURE_OUTPUT": configs[0]["capture_path"],
            }
        )
        specs.append(
            {
                "schema_version": 1,
                "run_id": run_id,
                "condition": run["condition"],
                "gpu_id": run["gpu_id"],
                "gpu": {
                    "physical_index": gpu["physical_index"],
                    "uuid": gpu["uuid"],
                },
                "cpu_affinity": gpu["cpu_affinity"],
                "cache_root": gpu["cache_root"],
                "port_start": run["port_start"],
                "authorization": _relative(AUTHORIZATION_PATH),
                "logical_draft_weight_version": source["logical_draft_weight_version"],
                "minimum_shared_kv_blocks": source["minimum_shared_kv_blocks"],
                "dynamic_k_schedule": source["dynamic_k_schedule"],
                "engine": source["engine"],
                "model": source["model"],
                "environment": environment,
                "configs": configs,
            }
        )
    return specs


def _parse_cpu_affinity(value: str) -> set[int]:
    cpus: set[int] = set()
    for item in value.split(","):
        bounds = item.split("-", maxsplit=1)
        start = int(bounds[0])
        stop = int(bounds[-1])
        _require(0 <= start <= stop, "CPU affinity range is invalid")
        cpus.update(range(start, stop + 1))
    _require(cpus, "CPU affinity is empty")
    return cpus


def _ports_available(port_start: int) -> bool:
    sockets: list[socket.socket] = []
    try:
        for port in range(port_start, port_start + PORT_RANGE_SIZE):
            handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            handle.bind(("127.0.0.1", port))
            sockets.append(handle)
    except OSError:
        return False
    finally:
        for handle in sockets:
            handle.close()
    return True


def _preflight_runs(run_ids: Sequence[str]) -> list[dict[str, Any]]:
    evidence = []
    for run_id in run_ids:
        run = RUNS[run_id]
        gpu = GPU_ASSIGNMENTS[run["gpu_id"]]
        gpu_evidence = matrix._preflight_gpu_identity_and_idle(
            gpu["physical_index"], gpu["uuid"]
        )
        _require(
            _ports_available(run["port_start"]),
            f"port range for {run_id} is not free",
        )
        evidence.append(
            {
                "run_id": run_id,
                "gpu": gpu_evidence,
                "port_start": run["port_start"],
                "port_range_size": PORT_RANGE_SIZE,
                "ports_available": True,
            }
        )
    return evidence


def _materialize_run(
    authorization_path: Path,
    authorization: Mapping[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    specs = build_probe_specs(output_dir)
    output_dir.mkdir(parents=False, exist_ok=False)
    materialized = []
    for spec in specs:
        run_dir = output_dir / spec["run_id"]
        (run_dir / "configs").mkdir(parents=True)
        (run_dir / "captures").mkdir()
        for config in spec["configs"]:
            _write_exclusive(Path(config["config_path"]), config["template"])
        child_spec = copy.deepcopy(spec)
        for config in child_spec["configs"]:
            config.pop("template")
        child_spec_path = run_dir / "child_spec.json"
        _write_exclusive(child_spec_path, child_spec)
        materialized.append(child_spec)
    _write_exclusive(
        output_dir / "preparation.json",
        {
            "schema_version": 1,
            "record_type": "p4_b0_gpu_contention_probe_preparation",
            "authorization": _reference(authorization_path),
            "output_dir": _relative(output_dir),
            "scored": False,
            "gpu_executed": False,
            "schedule": copy.deepcopy(SCHEDULE),
            "run_count": len(specs),
            "rounds_per_run": len(ROUNDS),
            "thresholds": copy.deepcopy(THRESHOLDS),
            "claims": {
                "dual_gpu_value_screen_authorized": False,
                "value_screen_scoring_authorized": False,
            },
        },
    )
    return materialized


def _episode_summary(rounds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(len(rounds) == len(ROUNDS), "probe episode round count drifted")
    rates = [float(row["estimands"]["decode_rate_req"]) for row in rounds]
    _require(
        all(math.isfinite(rate) and rate > 0 for rate in rates),
        "probe episode contains an invalid rate",
    )
    reference = max(rates)
    threshold = reference * THRESHOLDS["episode_floor_fraction"]
    survivors = [
        row for row in rounds if float(row["estimands"]["decode_rate_req"]) >= threshold
    ]
    _require(
        len(survivors) >= THRESHOLDS["minimum_surviving_rounds"],
        "probe episode retained too few rounds",
    )
    committed = sum(int(row["counters"]["E_committed"]) for row in survivors)
    request_time = sum(
        float(row["timing"]["request_decode_time_s"]) for row in survivors
    )
    return {
        "round_rates": rates,
        "reference_rate": reference,
        "retention_threshold": threshold,
        "surviving_round_indices": [
            int(row["matrix"]["round_index"]) for row in survivors
        ],
        "surviving_round_count": len(survivors),
        "committed_tokens": committed,
        "request_decode_time_s": request_time,
        "decode_rate_req": committed / request_time,
    }


def run_child(spec_path: Path) -> None:
    """Run four canonical R8/K4 rounds in one physical engine boot."""
    spec = _load_json(spec_path)
    run_id = spec.get("run_id")
    _require(run_id in RUNS, "child run id is not authorized")
    _require(
        spec_path.resolve() == (OUTPUT_DIR / str(run_id) / "child_spec.json").resolve(),
        "child spec is outside the authorized output",
    )
    expected = next(
        item for item in build_probe_specs(OUTPUT_DIR) if item["run_id"] == run_id
    )
    expected_clean = copy.deepcopy(expected)
    for config in expected_clean["configs"]:
        config.pop("template")
    _require(spec == expected_clean, "child spec drifted from its authorization")
    for key, value in spec["environment"].items():
        _require(os.environ.get(key) == str(value), f"child environment drifted: {key}")

    requested_affinity = _parse_cpu_affinity(spec["cpu_affinity"])
    os.sched_setaffinity(0, requested_affinity)
    _require(
        os.sched_getaffinity(0) == requested_affinity,
        "child CPU affinity did not bind exactly",
    )

    from adapt_p4_b0_same_event import adapt_capture

    from vllm import LLMEngine, envs
    from vllm.v1.engine.core_client import InprocClient
    from vllm.v1.spec_decode.koff_runtime import P4SameEventRecorder

    _require(
        not envs.VLLM_ENABLE_V1_MULTIPROCESSING,
        "contention child did not construct in-process EngineCore",
    )
    _manifest, prompt_rows = matrix._load_prompt_rows()
    engine = LLMEngine.from_engine_args(matrix._engine_args(spec))
    _require(
        isinstance(engine.engine_core, InprocClient),
        "contention child did not construct InprocClient",
    )
    scheduler, _worker = diagnosis._runtime_handles(engine)
    try:
        for index, config in enumerate(spec["configs"]):
            if index:
                _require(
                    scheduler._p4_capture_cohort is None,
                    "prior capture cohort remained active",
                )
                scheduler._p4_capture_recorder = P4SameEventRecorder(
                    config["config_path"],
                    config["capture_path"],
                    boot_action_id=ACTION_ID,
                    logical_weight_version=spec["logical_draft_weight_version"],
                    minimum_shared_kv_blocks=spec["minimum_shared_kv_blocks"],
                )
            template = _load_json(Path(config["config_path"]))
            prompt_ids = template["generation"]["prompt_record_ids"]
            batch = template["generation"]["batch"]
            for start in range(0, len(prompt_ids), batch):
                matrix._run_prompt_chunk(
                    engine,
                    template,
                    prompt_ids[start : start + batch],
                    prompt_rows,
                )
            _require(
                scheduler._p4_capture_recorder.completed_capture_count == 1,
                "contention child did not close one exact capture",
            )
    finally:
        engine.engine_core.shutdown()

    captures = [_load_json(Path(row["capture_path"])) for row in spec["configs"]]
    adapted = [adapt_capture(capture) for capture in captures]
    _require(
        [row["matrix"]["round_index"] for row in adapted] == list(ROUNDS)
        and all(row["matrix"]["action_id"] == ACTION_ID for row in adapted),
        "contention child adapted another workload",
    )
    result_path = spec_path.parent / "child_result.json"
    _write_exclusive(
        result_path,
        {
            "schema_version": 1,
            "record_type": "p4_b0_gpu_contention_probe_child_result",
            "status": "pass",
            "scored": False,
            "authorization": _reference(AUTHORIZATION_PATH),
            "run_id": run_id,
            "condition": spec["condition"],
            "gpu_id": spec["gpu_id"],
            "gpu": spec["gpu"],
            "cpu_affinity": {
                "registered": spec["cpu_affinity"],
                "logical_cpu_count": len(requested_affinity),
                "bound_exactly": True,
            },
            "port_start": spec["port_start"],
            "cache_root": spec["cache_root"],
            "captures": [
                _reference(Path(row["capture_path"])) for row in spec["configs"]
            ],
            "episode": _episode_summary(adapted),
            "claims": {
                "performance_claim_allowed": False,
                "score_eligible": False,
                "dual_gpu_value_screen_authorized": False,
            },
        },
    )


def evaluate_gate(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate the registered contention and device-equivalence bounds."""
    _require(set(results) == set(RUNS), "contention result set is incomplete")
    rates = {
        run_id: float(result["episode"]["decode_rate_req"])
        for run_id, result in results.items()
    }
    _require(
        all(math.isfinite(rate) and rate > 0 for rate in rates.values()),
        "contention result contains an invalid episode rate",
    )
    ratios = {
        gpu_id: rates[f"concurrent-{gpu_id}"] / rates[f"serial-{gpu_id}"]
        for gpu_id in GPU_ASSIGNMENTS
    }
    slowdowns = {gpu_id: max(0.0, 1.0 - ratio) for gpu_id, ratio in ratios.items()}
    ratio_disagreement = abs(ratios["gpu0"] - ratios["gpu1"]) / max(ratios.values())

    def disagreement(left: float, right: float) -> float:
        return abs(left - right) / max(left, right)

    serial_disagreement = disagreement(rates["serial-gpu0"], rates["serial-gpu1"])
    concurrent_disagreement = disagreement(
        rates["concurrent-gpu0"], rates["concurrent-gpu1"]
    )
    checks = {
        "gpu0_slowdown_within_one_percent": (
            slowdowns["gpu0"]
            <= THRESHOLDS["maximum_per_gpu_concurrent_slowdown_fraction"]
        ),
        "gpu1_slowdown_within_one_percent": (
            slowdowns["gpu1"]
            <= THRESHOLDS["maximum_per_gpu_concurrent_slowdown_fraction"]
        ),
        "contention_ratio_disagreement_within_one_percent": (
            ratio_disagreement
            <= THRESHOLDS["maximum_contention_ratio_disagreement_fraction"]
        ),
        "serial_cross_gpu_rate_within_two_percent": (
            serial_disagreement
            <= THRESHOLDS["maximum_cross_gpu_rate_disagreement_fraction"]
        ),
        "concurrent_cross_gpu_rate_within_two_percent": (
            concurrent_disagreement
            <= THRESHOLDS["maximum_cross_gpu_rate_disagreement_fraction"]
        ),
    }
    passed = all(checks.values())
    return {
        "state": "pass" if passed else "hold",
        "rates": rates,
        "concurrent_to_serial_ratios": ratios,
        "concurrent_slowdown_fractions": slowdowns,
        "contention_ratio_disagreement_fraction": ratio_disagreement,
        "serial_cross_gpu_rate_disagreement_fraction": serial_disagreement,
        "concurrent_cross_gpu_rate_disagreement_fraction": concurrent_disagreement,
        "thresholds": copy.deepcopy(THRESHOLDS),
        "checks": checks,
        "dual_gpu_block_parallelism_supportable": passed,
    }


def _launch_group(
    run_ids: Sequence[str], authorization_path: Path, output_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    preflight = _preflight_runs(run_ids)
    processes: dict[str, tuple[subprocess.Popen[str], Any]] = {}
    specs = {spec["run_id"]: spec for spec in build_probe_specs(output_dir)}
    for run_id in run_ids:
        spec = specs[run_id]
        run_dir = output_dir / run_id
        child_env = matrix._boot_child_environment(spec["environment"])
        log_stream = (run_dir / "child.log").open("x", encoding="utf-8")
        process = subprocess.Popen(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--authorization",
                str(authorization_path),
                "--output-dir",
                str(output_dir),
                "--child-spec",
                str(run_dir / "child_spec.json"),
            ],
            cwd=REPO_ROOT,
            env=child_env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes[run_id] = (process, log_stream)
    return_codes = {}
    for run_id, (process, log_stream) in processes.items():
        return_codes[run_id] = process.wait()
        log_stream.close()
    return preflight, return_codes


def execute_parent(
    authorization_path: Path,
    authorization: Mapping[str, Any],
    output_dir: Path,
) -> None:
    """Execute the registered serial runs, then the concurrent pair once."""
    _require(
        authorization_path.resolve() == AUTHORIZATION_PATH.resolve(),
        "execution must use the reviewed contention authorization path",
    )
    _require(
        output_dir.resolve() == OUTPUT_DIR.resolve(),
        "execution must use the reviewed contention output path",
    )
    validate_authorization(authorization, require_output_absent=True)
    base_environment = matrix._boot_child_environment({})
    matrix._preflight_native_sampler(base_environment)
    matrix._preflight_inprocess_engine_core(base_environment)
    for gpu in GPU_ASSIGNMENTS.values():
        cache_root = Path(gpu["cache_root"])
        _require(
            not cache_root.exists(),
            f"contention cache already exists: {cache_root}",
        )
    _materialize_run(authorization_path, authorization, output_dir)
    for gpu in GPU_ASSIGNMENTS.values():
        cache_root = Path(gpu["cache_root"])
        (cache_root / "inductor").mkdir(parents=True)
        (cache_root / "triton").mkdir()

    group_evidence = []
    for run_ids in SCHEDULE:
        preflight, return_codes = _launch_group(run_ids, authorization_path, output_dir)
        group_evidence.append(
            {
                "run_ids": list(run_ids),
                "gpu_preflight": preflight,
                "return_codes": return_codes,
            }
        )
        _require(
            all(code == 0 for code in return_codes.values()),
            f"contention child process failed: {return_codes}",
        )
    results = {
        run_id: _load_json(output_dir / run_id / "child_result.json") for run_id in RUNS
    }
    gate = evaluate_gate(results)
    _write_exclusive(
        output_dir / "probe_result.json",
        {
            "schema_version": 1,
            "artifact_id": PROBE_ARTIFACT_ID,
            "status": "pass" if gate["state"] == "pass" else "hold",
            "scored": False,
            "authorization_consumed": True,
            "authorization": _reference(authorization_path),
            "preflight_groups": group_evidence,
            "child_results": {
                run_id: _reference(output_dir / run_id / "child_result.json")
                for run_id in RUNS
            },
            "gate": gate,
            "claims": {
                "performance_claim_allowed": False,
                "score_eligible": False,
                "dual_gpu_value_screen_authorized": False,
                "dual_gpu_authorization_package_may_be_prepared": (
                    gate["state"] == "pass"
                ),
            },
            "next_action": (
                "Prepare a separate source-bound dual-GPU block authorization."
                if gate["state"] == "pass"
                else "Retain single-GPU execution; do not authorize parallel scoring."
            ),
        },
    )


def _preserve_failure(output_dir: Path, exc: BaseException) -> None:
    if not output_dir.is_dir() or (output_dir / "failure.json").exists():
        return
    child_results = {
        run_id: (output_dir / run_id / "child_result.json").is_file() for run_id in RUNS
    }
    _write_exclusive(
        output_dir / "failure.json",
        {
            "schema_version": 1,
            "artifact_id": FAILURE_ARTIFACT_ID,
            "status": "failed_without_dual_gpu_authorization",
            "scored": False,
            "authorization_consumed": True,
            "exception": {"type": type(exc).__name__, "message": str(exc)},
            "child_results_present": child_results,
            "disposition": {
                "partial_resume_allowed": False,
                "retry_allowed": False,
                "dual_gpu_value_screen_authorized": False,
                "value_screen_scoring_authorized": False,
            },
        },
    )


def parse_args() -> argparse.Namespace:
    """Parse the contention-probe interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--child-spec", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    """Validate, prepare, or consume the one-shot contention authority."""
    args = parse_args()
    authorization = _load_json(args.authorization)
    child = args.child_spec is not None
    validate_authorization(authorization, require_output_absent=not child)
    _require(
        args.output_dir.resolve() == OUTPUT_DIR.resolve(),
        "contention-probe output path drifted",
    )
    if args.prepare_only:
        _require(not child, "child mode cannot prepare")
        specs = build_probe_specs(args.output_dir)
        print(
            json.dumps(
                {
                    "status": "pass",
                    "mode": "prepare_only",
                    "gpu_executed": False,
                    "output_created": False,
                    "run_ids": [spec["run_id"] for spec in specs],
                    "rounds_per_run": len(ROUNDS),
                },
                sort_keys=True,
            )
        )
        return 0
    if child:
        assert args.child_spec is not None
        run_child(args.child_spec)
    else:
        execute_parent(args.authorization, authorization, args.output_dir)
    return 0


if __name__ == "__main__":
    parsed_output: Path | None = None
    is_child = "--child-spec" in sys.argv
    try:
        if "--output-dir" in sys.argv:
            parsed_output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
        raise SystemExit(main())
    except (ContentionProbeError, matrix.P4RunnerError) as exc:
        if parsed_output is not None and not is_child:
            _preserve_failure(parsed_output, exc)
        print(f"P4 GPU contention probe refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
