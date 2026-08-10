#!/usr/bin/env python3
"""Validate the separate Phase 97 B0 run-authorization review."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from score_p4_b0 import validate_runner_scorer_contract
from validate_p4_b0_resource_bound import validate_bound
from validate_p4_b0_value_screen import validate_preregistration
from validate_p4_prompt_manifest import validate_manifest
from validate_p4_w512_equivalence import validate_proof

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_run_authorization.schema.json"

EXPECTED_SOURCE_ARTIFACTS = {
    "value_screen_preregistration": (
        "research/97_composition_runtime/data/p4/p4_b0_value_screen_prereg.json",
        "536f2ba56b162ce38569bcbbbe1b43b0fc40fe50b04dedbdf5f576313092b234",
    ),
    "runner_scorer_contract": (
        "research/97_composition_runtime/data/p4/p4_b0_runner_scorer_contract.json",
        "6bb1de6a2b13725c5b8d3637e024da2dd52e18212a504b9bd384e73a90274736",
    ),
    "prompt_manifest": (
        "research/97_composition_runtime/data/p4/p4_b0_prompt_manifest.json",
        "10de0f73897c08b278d532a5764d3e45ccb67dbc52c1dc9d1e85a3da4c54ad93",
    ),
    "live_recorder_wiring": (
        "research/97_composition_runtime/data/p4/p4_live_recorder_wiring.json",
        "188dec31061cde98e6534f497e940757f96b073192b4c37fa1d1d5711805e745",
    ),
    "w512_equivalence": (
        "research/97_composition_runtime/data/p4/p4_w512_acceptance_equivalence.json",
        "67f6ac822acf208eb7147e386fdd67f7559d9c07460d9c82dd7d1d9297bdd5a1",
    ),
    "resource_bound": (
        "research/97_composition_runtime/data/p4/"
        "p4_b0_conservative_resource_bound.json",
        "228da050563759abb4b4acc2d1915f91b77f1fa69e2d485a18c68a202fdf6609",
    ),
    "measurement_adapter": (
        "research/97_composition_runtime/scripts/adapt_p4_b0_same_event.py",
        "32b1ad037d4bc7cfc288bfb50e3a8d50a8c3246f14544e33c04a587f6f70e789",
    ),
    "scorer": (
        "research/97_composition_runtime/scripts/score_p4_b0.py",
        "092a689aa9c60619e51dac6531aca58991644e6f5fdffad4ae880876b4a76fb4",
    ),
    "live_runtime": (
        "vllm/v1/spec_decode/koff_runtime.py",
        "37c84eda8ce974688b39b4b8f51934b9a3ccd3e48c862aef204e70049bdc9592",
    ),
    "live_scheduler": (
        "vllm/v1/core/sched/scheduler.py",
        "66381595d235192c10ad15f02b5dfec5b7442f95e45a4ed9ab31eb999490960f",
    ),
}

EXPECTED_EVIDENCE = {
    "research_objective_and_weights_frozen",
    "same_event_contract_registered",
    "matched_action_set_frozen",
    "legacy_w14d_disposition_fixed",
    "exact_prompt_manifest_frozen",
    "same_event_measurement_adapter_frozen",
    "runner_scorer_frozen",
    "same_event_live_recorder_wiring",
    "w512_mask_equivalence_proven",
    "conservative_resource_bound_complete",
}

EXPECTED_BLOCKERS = (
    "w512_boot_contract_blocked",
    "w512_recorder_action_blocked",
    "multi_cell_same_boot_capture_missing",
    "boot_static_action_relabel_missing",
    "logical_weight_version_unstable",
    "runtime_resource_floor_guard_missing",
)

EXPECTED_BLOCKER_EVIDENCE = {
    "w512_boot_contract_blocked": (
        "validate_boot_config requires draft_kv_window == 0, so the registered "
        "window=512 boot is rejected."
    ),
    "w512_recorder_action_blocked": (
        "P4SameEventRecorder._ACTION_REALIZATIONS contains only OFF and K4 and "
        "rejects the w512 action id."
    ),
    "multi_cell_same_boot_capture_missing": (
        "The recorder owns one config/output cell and the required "
        "48-cell-per-boot matrix runner does not exist."
    ),
    "boot_static_action_relabel_missing": (
        "Same-event action ids are resolved from the OFF/K4 live registry; no "
        "fail-closed boot-static w512 relabel binds K4 events to the w512 "
        "realization."
    ),
    "logical_weight_version_unstable": (
        "The recorder replaces the logical draft-weight version with a "
        "storage-pointer-derived alias id that cannot remain stable across "
        "nine boots."
    ),
    "runtime_resource_floor_guard_missing": (
        "The live capture path records KV capacity but does not stop "
        "launch/capture below the registered 21,682-block conservative floor."
    ),
}

EXPECTED_COMMIT = "cc8ed50f22568a4f6b88b299e9c47ebba7a8046a"
EXPECTED_MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
EXPECTED_MODEL_PATH = (
    "/data/smcho/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/"
    + EXPECTED_MODEL_REVISION
)
EXPECTED_RUNNER_PATH = (
    "research/97_composition_runtime/scripts/run_p4_b0_value_screen.py"
)
EXPECTED_OUTPUT_DIR = "research/97_composition_runtime/data/p4/run_b0_value_screen_v1"
EXPECTED_AUTHORIZATION_PATH = (
    "research/97_composition_runtime/data/p4/p4_b0_run_authorization.json"
)

EXPECTED_ENGINE = {
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "max_model_len": 20480,
    "max_num_batched_tokens": 8192,
    "max_num_seqs": 32,
    "gpu_memory_utilization": 0.9,
    "num_speculative_tokens": 4,
    "async_scheduling": False,
    "prefix_caching": False,
    "flashinfer_autotune": False,
    "enforce_eager": False,
    "generation_seed": 0,
}

EXPECTED_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "4",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "VLLM_SELF_SPEC_KOFF_RUNTIME": "1",
    "VLLM_SELF_SPEC_SHARED_KV": "1",
    "VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE": "1",
    "VLLM_SELF_SPEC_SHARE_WEIGHTS": "1",
    "VLLM_SELF_SPEC_DRAFT_KV_DTYPE": "",
    "VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS": "",
    "VLLM_SELF_SPEC_DRAFT_PARTIAL_REPLICA": "0",
    "VLLM_SELF_SPEC_AHEAD_CHAIN": "0",
    "VLLM_SELF_SPEC_CONSUME_AHEAD": "0",
    "VLLM_SELF_SPEC_DRAFT_FULL_CG": "1",
    "VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE": "1",
    "VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD": "1",
    "VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG": "0",
    "VLLM_SELF_SPEC_DRAFT_WHOLECHAIN": "0",
    "VLLM_SELF_SPEC_DRAFT_FULLCG": "0",
}

EXPECTED_ACTION_BOOTS = [
    {
        "action_id": "off",
        "window_tokens": 0,
        "sink_tokens": 0,
        "dynamic_k_schedule": [[1, 32, 0], [33, 33, 4]],
        "cost_credit_allowed": False,
    },
    {
        "action_id": "target-matching-k4",
        "window_tokens": 0,
        "sink_tokens": 0,
        "dynamic_k_schedule": [[1, 32, 4]],
        "cost_credit_allowed": False,
    },
    {
        "action_id": "target-matching-w512-masked-k4",
        "window_tokens": 512,
        "sink_tokens": 16,
        "dynamic_k_schedule": [[1, 32, 4]],
        "cost_credit_allowed": False,
    },
]


class B0RunAuthorizationError(ValueError):
    """Raised when the B0 run-authorization review drifts or overclaims."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise B0RunAuthorizationError(message)


def _format_json_path(parts: Sequence[Any]) -> str:
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise B0RunAuthorizationError(
            f"cannot load JSON artifact {path}: {exc}"
        ) from exc
    _require(isinstance(value, dict), f"JSON artifact must be an object: {path}")
    return value


def _repository_path(relative_path: str) -> Path:
    path = (REPO_ROOT / relative_path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise B0RunAuthorizationError(f"artifact escapes repository: {path}") from exc
    return path


def _validate_schema(authorization: Mapping[str, Any]) -> None:
    schema = _load_json(SCHEMA_PATH)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise B0RunAuthorizationError(
            f"invalid run-authorization schema: {exc}"
        ) from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(authorization),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if errors:
        first = errors[0]
        path = _format_json_path(list(first.absolute_path))
        raise B0RunAuthorizationError(
            f"run-authorization schema rejected {path}: {first.message}"
        )


def _validate_sources(
    authorization: Mapping[str, Any],
    *,
    enforce_current_sources: bool,
) -> dict[str, str | dict[str, Any]]:
    references = authorization["source_artifacts"]
    _require(
        set(references) == set(EXPECTED_SOURCE_ARTIFACTS),
        "authorization must bind exactly the registered source roles",
    )
    loaded: dict[str, str | dict[str, Any]] = {}
    json_roles = {
        "value_screen_preregistration",
        "runner_scorer_contract",
        "prompt_manifest",
        "live_recorder_wiring",
        "w512_equivalence",
        "resource_bound",
    }
    for role, (expected_path, expected_hash) in EXPECTED_SOURCE_ARTIFACTS.items():
        reference = references[role]
        _require(
            reference == {"path": expected_path, "sha256": expected_hash},
            f"source role {role} differs from the reviewed artifact",
        )
        path = _repository_path(expected_path)
        _require(path.is_file(), f"missing referenced artifact: {path}")
        content = path.read_bytes()
        actual_hash = hashlib.sha256(content).hexdigest()
        if enforce_current_sources or role not in {"live_runtime", "live_scheduler"}:
            _require(
                actual_hash == expected_hash,
                f"artifact hash mismatch for {expected_path}: "
                f"{actual_hash} != {expected_hash}",
            )
        loaded[role] = _load_json(path) if role in json_roles else content.decode()
    return loaded


def _validate_readiness(
    authorization: Mapping[str, Any],
    sources: Mapping[str, str | Mapping[str, Any]],
    *,
    enforce_current_sources: bool,
) -> None:
    readiness = authorization["evidence_readiness"]
    _require(
        set(readiness["satisfied"]) == EXPECTED_EVIDENCE,
        "evidence-readiness closure differs from the additive proof chain",
    )
    preregistration = sources["value_screen_preregistration"]
    runner_scorer = sources["runner_scorer_contract"]
    manifest = sources["prompt_manifest"]
    recorder = sources["live_recorder_wiring"]
    equivalence = sources["w512_equivalence"]
    bound = sources["resource_bound"]
    assert isinstance(preregistration, Mapping)
    assert isinstance(runner_scorer, Mapping)
    assert isinstance(manifest, Mapping)
    assert isinstance(recorder, Mapping)
    assert isinstance(equivalence, Mapping)
    assert isinstance(bound, Mapping)
    if not enforce_current_sources:
        _require(
            preregistration["readiness"]["state"] == "blocked"
            and runner_scorer["readiness"]["runner_scorer_frozen"]
            and manifest["authorizations"]["exact_prompt_manifest_frozen"]
            and recorder["status"] == "pass_synchronous_cpu_contract"
            and equivalence["claims"]["boot_static_acceptance_surrogate_eligible"]
            and bound["result"]["decision"] == "pass",
            "historical readiness artifacts no longer match the reviewed hashes",
        )
        return
    try:
        prereg_result = validate_preregistration(preregistration)
        runner_result = validate_runner_scorer_contract(runner_scorer)
        manifest_result = validate_manifest(manifest)
        equivalence_result = validate_proof(equivalence)
        bound_result = validate_bound(bound)
    except Exception as exc:
        raise B0RunAuthorizationError(
            f"an upstream readiness artifact no longer validates: {exc}"
        ) from exc
    _require(
        prereg_result["readiness"] == "blocked"
        and not prereg_result["gpu_measurement_authorized"],
        "the frozen preregistration was rewritten or granted authority",
    )
    _require(
        runner_result["runner_scorer_frozen"]
        and runner_result["expected_round_records"] == 432,
        "the frozen runner/scorer contract is not intact",
    )
    _require(
        manifest_result["exact_prompt_manifest_frozen"]
        and manifest_result["record_count"] == 384,
        "the exact prompt manifest is not intact",
    )
    _require(
        recorder["status"] == "pass_synchronous_cpu_contract"
        and recorder["readiness_update"]["cleared"] == ["same_event_recorder_unwired"]
        and not any(recorder["authorizations"].values()),
        "the additive recorder-wiring evidence drifted or grants authority",
    )
    _require(
        equivalence_result["cleared_blocker"] == "w512_mask_equivalence_unproven"
        and not equivalence_result["latency_transfer_allowed"],
        "w512 equivalence no longer proves only acceptance semantics",
    )
    _require(
        bound_result["resource_decision"] == "pass"
        and bound_result["remaining_readiness_blockers"] == []
        and not bound_result["gpu_measurement_authorized"],
        "the conservative resource bound no longer closes readiness narrowly",
    )


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise B0RunAuthorizationError(f"cannot resolve repository HEAD: {exc}") from exc
    return result.stdout.strip()


def _validate_run_contract(
    authorization: Mapping[str, Any],
    sources: Mapping[str, str | Mapping[str, Any]],
    *,
    enforce_current_sources: bool,
) -> None:
    run = authorization["run_contract"]
    _require(
        run["repository_commit"] == EXPECTED_COMMIT == _git_head(),
        "authorization repository commit differs from the reviewed HEAD",
    )
    model = run["model"]
    _require(
        model
        == {
            "model_id": "Qwen/Qwen3-8B",
            "revision": EXPECTED_MODEL_REVISION,
            "snapshot_path": EXPECTED_MODEL_PATH,
            "target_quantization": None,
            "target_kv_dtype": "bfloat16",
            "draft_weight_path": "target-matching-alias",
        },
        "model realization differs from the reviewed target-matching B0",
    )
    _require(
        Path(EXPECTED_MODEL_PATH).is_dir(),
        "the pinned local model snapshot is unavailable",
    )
    gpu = run["gpu_assignment"]
    _require(
        gpu
        == {
            "cuda_visible_devices": "4",
            "physical_index": 4,
            "uuid": "GPU-c9d19019-5065-2353-80a9-f1797eb19d51",
            "model": "NVIDIA H100 80GB HBM3",
            "memory_mib": 81559,
            "all_nine_boots_same_gpu": True,
            "fallback_gpu": {
                "physical_index": 5,
                "uuid": "GPU-318614e3-fd51-31a9-399d-9e94cb767967",
                "authorized": False,
            },
        },
        "GPU assignment differs from the reviewed GPU-4-only matrix",
    )
    _require(run["engine"] == EXPECTED_ENGINE, "engine geometry or mode drifted")
    _require(
        run["environment"] == EXPECTED_ENVIRONMENT,
        "common launch environment drifted",
    )
    _require(
        run["action_boots"] == EXPECTED_ACTION_BOOTS,
        "boot-static action realization or K schedule drifted",
    )

    contract = sources["runner_scorer_contract"]
    assert isinstance(contract, Mapping)
    source_matrix = contract["matrix"]
    matrix = run["matrix"]
    expected_cells = (
        len(source_matrix["regimes"])
        * len(source_matrix["content_seeds"])
        * source_matrix["rounds_per_boot"]
    )
    expected_boots = len(source_matrix["boot_blocks"]) * len(source_matrix["actions"])
    _require(
        matrix["boot_blocks"] == source_matrix["boot_blocks"],
        "counterbalanced physical-boot order drifted",
    )
    _require(
        matrix["physical_boot_count"] == expected_boots == 9
        and matrix["cells_per_boot"] == expected_cells == 48
        and matrix["raw_capture_count"] == expected_boots * expected_cells == 432
        and matrix["adapted_round_count"]
        == source_matrix["expected_round_records"]
        == 432
        and matrix["rounds_per_regime_seed"] == source_matrix["rounds_per_boot"] == 4
        and matrix["prompts_per_cell"] == source_matrix["prompts_per_round"] == 32,
        "the 9-boot, 48-cell, 432-round matrix does not close",
    )

    invocation = run["invocation"]
    expected_argv = [
        ".venv/bin/python",
        EXPECTED_RUNNER_PATH,
        "--authorization",
        EXPECTED_AUTHORIZATION_PATH,
        "--output-dir",
        EXPECTED_OUTPUT_DIR,
    ]
    _require(
        invocation
        == {
            "runner_path": EXPECTED_RUNNER_PATH,
            "argv": expected_argv,
            "output_dir": EXPECTED_OUTPUT_DIR,
            "overwrite_allowed": False,
            "runner_exists": False,
            "launchable_now": False,
        },
        "registered held invocation drifted",
    )
    if enforce_current_sources:
        _require(
            not _repository_path(EXPECTED_RUNNER_PATH).exists(),
            "the package is stale because the missing matrix runner now exists",
        )
    if enforce_current_sources:
        _require(
            not _repository_path(EXPECTED_OUTPUT_DIR).exists(),
            "held authorization cannot point at an existing run output directory",
        )


def _validate_resource_gates(
    authorization: Mapping[str, Any],
    sources: Mapping[str, str | Mapping[str, Any]],
) -> None:
    bound = sources["resource_bound"]
    assert isinstance(bound, Mapping)
    result = bound["result"]
    base = bound["base_capacity"]
    expected = {
        "base_available_shared_kv_blocks": base["available_shared_kv_blocks"],
        "conservative_floor_blocks": result["lower_bound_shared_kv_blocks"],
        "required_live_kv_blocks": result["required_live_kv_blocks"],
        "minimum_launch_capacity_blocks": result["lower_bound_shared_kv_blocks"],
        "single_target_owned_pool_required": True,
        "preemption_allowed": False,
        "recomputation_allowed": False,
        "on_violation": "stop_without_scoring",
    }
    _require(
        authorization["resource_gates"] == expected,
        "launch resource gates differ from the conservative B0 bound",
    )
    _require(
        expected["conservative_floor_blocks"] == 21682
        and expected["required_live_kv_blocks"] == 21000,
        "reviewed resource-floor arithmetic drifted",
    )


def _validate_current_blockers(
    authorization: Mapping[str, Any],
    sources: Mapping[str, str | Mapping[str, Any]],
    *,
    enforce_current_sources: bool,
) -> None:
    checks = authorization["implementation_audit"]["checks"]
    observed_codes = tuple(check["code"] for check in checks)
    _require(
        observed_codes == EXPECTED_BLOCKERS,
        "implementation audit must retain exactly the six reviewed blockers",
    )
    _require(
        all(
            check["status"] == "fail"
            and check["evidence"] == EXPECTED_BLOCKER_EVIDENCE[check["code"]]
            for check in checks
        ),
        "implementation blocker status or evidence was weakened",
    )
    if not enforce_current_sources:
        return
    runtime = sources["live_runtime"]
    scheduler = sources["live_scheduler"]
    assert isinstance(runtime, str)
    assert isinstance(scheduler, str)
    _require(
        '(options.draft_kv_window == 0, "no draft KV window")' in runtime,
        "the reviewed w512 boot-contract blocker is no longer present",
    )
    _require(
        'OFF_ACTION_ID: "live-b0-forced-off"' in runtime
        and 'K4_ACTION_ID: "live-b0-target-matching-k4"' in runtime
        and "the synchronous recorder currently permits only live B0 OFF/K4" in runtime,
        "the reviewed recorder action blocker is no longer present",
    )
    _require(
        '"""Collect one capture cell and emit the frozen raw P4 JSON contract."""'
        in runtime
        and "def __init__(self, config_path: str, output_path: str)" in runtime
        and not _repository_path(EXPECTED_RUNNER_PATH).exists(),
        "the reviewed multi-cell capture blocker is no longer present",
    )
    _require(
        "action = action_for_id(metadata.verified_action_id)" in runtime
        and '"action_id": action.action_id' in runtime
        and "W512_ACTION_ID" not in runtime,
        "the reviewed boot-static relabel blocker is no longer present",
    )
    _require(
        "storage.data_ptr()" in runtime
        and "version_parts.append" in runtime
        and 'runner["draft_weight_version"] = evidence.draft_weight_version_id'
        in runtime,
        "the reviewed logical-weight-version blocker is no longer present",
    )
    _require(
        "shared_target_kv_block_capacity" in runtime
        and "shared_target_kv_block_capacity" in scheduler
        and "21682" not in runtime
        and "minimum_launch_capacity_blocks" not in runtime,
        "the reviewed runtime resource-floor blocker is no longer present",
    )


def _validate_decision_boundary(authorization: Mapping[str, Any]) -> None:
    _require(
        tuple(authorization["decision"]["reason_codes"]) == EXPECTED_BLOCKERS,
        "HOLD reason codes differ from the failed implementation checks",
    )
    _require(
        tuple(authorization["next_artifact"]["must_clear"]) == EXPECTED_BLOCKERS,
        "conformance handoff does not require all six blockers",
    )
    expected_claims = {
        "evidence_readiness_complete": True,
        "executable_run_ready": False,
        "exact_invocation_registered": True,
        "gpu_run_performed": False,
        "runtime_w512_switching_implemented": False,
        "action_admitted": False,
        "performance_claim_allowed": False,
    }
    expected_authorizations = {
        "capture_runner_conformance_engineering": True,
        "gpu_measurement": False,
        "p4a_engineering": False,
        "action_admission": False,
        "production_value_claim": False,
    }
    _require(
        authorization["claims"] == expected_claims,
        "authorization claims inflate executable, measurement, or value status",
    )
    _require(
        authorization["authorizations"] == expected_authorizations,
        "only narrowly scoped capture-runner conformance may be authorized",
    )


def validate_authorization(
    authorization: Mapping[str, Any],
    *,
    enforce_current_sources: bool = True,
) -> dict[str, Any]:
    """Validate the HOLD package and return its narrow decision summary.

    Args:
        authorization: Parsed run-authorization review.

    Returns:
        Machine-readable evidence, blocker, and authority summary.

    Raises:
        B0RunAuthorizationError: If evidence, contract, or authority drifts.
    """
    _validate_schema(authorization)
    sources = _validate_sources(
        authorization,
        enforce_current_sources=enforce_current_sources,
    )
    _validate_decision_boundary(authorization)
    _validate_run_contract(
        authorization,
        sources,
        enforce_current_sources=enforce_current_sources,
    )
    _validate_resource_gates(authorization, sources)
    _validate_current_blockers(
        authorization,
        sources,
        enforce_current_sources=enforce_current_sources,
    )
    _validate_readiness(
        authorization,
        sources,
        enforce_current_sources=enforce_current_sources,
    )
    return {
        "status": "pass",
        "package_id": authorization["package_id"],
        "authorization_decision": "hold",
        "reviewed_sources_current": enforce_current_sources,
        "evidence_readiness": "complete",
        "executable_run_ready": False,
        "failed_implementation_checks": list(EXPECTED_BLOCKERS),
        "gpu_physical_index": 4,
        "physical_boot_count": 9,
        "capture_count": 432,
        "minimum_launch_capacity_blocks": 21682,
        "capture_runner_conformance_engineering_authorized": True,
        "gpu_measurement_authorized": False,
        "p4a_engineering_authorized": False,
        "action_admitted": False,
        "next_artifact": "p4_b0_capture_runner_conformance",
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Validate the requested authorization review and print JSON."""
    args = parse_args()
    result = validate_authorization(_load_json(args.authorization))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
