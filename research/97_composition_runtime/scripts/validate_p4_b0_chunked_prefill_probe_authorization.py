#!/usr/bin/env python3
"""Validate the one-shot non-scored P4 chunked-prefill probe authority."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
AUTHORIZATION_PATH = (
    PHASE_DIR / "data" / "p4" / "p4_b0_chunked_prefill_probe_authorization.json"
)
OUTPUT_DIR = PHASE_DIR / "data" / "p4" / "run_b0_chunked_prefill_probe_v1"
VALIDATION_PATH = (
    PHASE_DIR
    / "data"
    / "p4"
    / "p4_b0_chunked_prefill_probe_authorization_validation.json"
)
GPU4_UUID = "GPU-c9d19019-5065-2353-80a9-f1797eb19d51"


class P4ChunkedPrefillAuthorizationError(ValueError):
    """Raised when probe authority is missing, stale, or overbroad."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise P4ChunkedPrefillAuthorizationError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise P4ChunkedPrefillAuthorizationError(
            f"cannot load JSON artifact {path}: {exc}"
        ) from exc
    _require(isinstance(value, dict), f"JSON artifact must be an object: {path}")
    return value


def _repo_path(relative_path: str) -> Path:
    path = (REPO_ROOT / relative_path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise P4ChunkedPrefillAuthorizationError(
            f"source artifact escapes repository: {relative_path}"
        ) from exc
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_sources(value: Mapping[str, Any]) -> dict[str, str]:
    expected_roles = {
        "authorization_schema",
        "authorization_tests",
        "authorization_validator",
        "capture_runner",
        "capture_runner_tests",
        "cohort_barrier_cpu_proof",
        "cohort_barrier_implementation",
        "gpu_worker",
        "prompt_bundle",
        "prompt_manifest",
        "probe_runner",
        "scheduler",
        "scheduler_tests",
        "transient_bound",
        "v9_consumed_failure",
    }
    _require(set(value) == expected_roles, "probe source roles drifted")
    observed = {}
    for role, reference in value.items():
        _require(
            isinstance(reference, Mapping) and set(reference) == {"path", "sha256"},
            f"probe source reference is malformed: {role}",
        )
        path_value = reference["path"]
        digest = reference["sha256"]
        _require(
            isinstance(path_value, str)
            and isinstance(digest, str)
            and len(digest) == 64,
            f"probe source reference is invalid: {role}",
        )
        path = _repo_path(path_value)
        _require(path.is_file(), f"probe source is missing: {role}")
        actual = _sha256(path)
        _require(actual == digest, f"probe source hash drifted: {role}")
        observed[role] = actual
    return observed


def _validate_contract(
    contract: Mapping[str, Any],
    *,
    authorization_path: Path = AUTHORIZATION_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    _require(
        set(contract)
        == {
            "scope",
            "scored",
            "physical_boot_count",
            "gpu",
            "model",
            "engine",
            "action",
            "workload",
            "environment",
            "invocation",
            "pass_gates",
        },
        "probe run-contract fields drifted",
    )
    _require(
        contract["scope"] == "gpu4_chunked_prefill_cohort_barrier_probe_only"
        and contract["scored"] is False
        and contract["physical_boot_count"] == 1,
        "probe scope is not one non-scored physical boot",
    )
    _require(
        contract["gpu"]
        == {
            "cuda_visible_devices": "4",
            "physical_index": 4,
            "uuid": GPU4_UUID,
            "fallback_gpu_authorized": False,
        },
        "probe GPU assignment drifted",
    )
    _require(
        contract["model"]
        == {
            "model_id": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "snapshot_path": (
                "/data/smcho/huggingface/hub/models--Qwen--Qwen3-8B/"
                "snapshots/b968826d9c46dd6066d109eabc6255188de91218"
            ),
            "target_quantization": None,
            "target_kv_dtype": "bfloat16",
            "draft_weight_path": "target-matching-alias",
        },
        "probe model contract drifted",
    )
    _require(
        contract["engine"]
        == {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "max_model_len": 20480,
            "max_num_batched_tokens": 8192,
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
        "probe must retain the bounded 8192-token compiled engine",
    )
    _require(
        contract["action"]
        == {
            "action_id": "target-matching-k4",
            "dynamic_k_schedule": [[1, 32, 4]],
            "window_tokens": 0,
            "sink_tokens": 0,
        },
        "probe action is not the target-matching K4 boot",
    )
    _require(
        contract["workload"]
        == {
            "regimes": ["R4", "R5", "R5cot"],
            "content_seed": 0,
            "prompts_per_cohort": 8,
            "unmeasured_prefill_tokens_per_request": 1,
            "measured_decode_tokens_per_request": 1,
        },
        "probe workload no longer covers the three ingress cohorts",
    )
    expected_environment = {
        "CUDA_VISIBLE_DEVICES": "4",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_SELF_SPEC_AHEAD_CHAIN": "0",
        "VLLM_SELF_SPEC_CONSUME_AHEAD": "0",
        "VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD": "1",
        "VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE": "1",
        "VLLM_SELF_SPEC_DRAFT_FULLCG": "0",
        "VLLM_SELF_SPEC_DRAFT_FULL_CG": "1",
        "VLLM_SELF_SPEC_DRAFT_KV_DTYPE": "",
        "VLLM_SELF_SPEC_DRAFT_KV_SINKS": "0",
        "VLLM_SELF_SPEC_DRAFT_KV_WINDOW": "0",
        "VLLM_SELF_SPEC_DRAFT_PARTIAL_REPLICA": "",
        "VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS": "",
        "VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG": "0",
        "VLLM_SELF_SPEC_DRAFT_WHOLECHAIN": "0",
        "VLLM_SELF_SPEC_KOFF_RUNTIME": "1",
        "VLLM_SELF_SPEC_SHARED_KV": "1",
        "VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE": "1",
        "VLLM_SELF_SPEC_SHARE_WEIGHTS": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }
    _require(
        contract["environment"] == expected_environment,
        "probe child environment drifted",
    )
    authorization_relative = authorization_path.resolve().relative_to(REPO_ROOT)
    output_relative = output_dir.resolve().relative_to(REPO_ROOT)
    _require(
        contract["invocation"]
        == {
            "runner_path": (
                "research/97_composition_runtime/scripts/"
                "run_p4_b0_chunked_prefill_probe.py"
            ),
            "argv": [
                ".venv/bin/python",
                "research/97_composition_runtime/scripts/"
                "run_p4_b0_chunked_prefill_probe.py",
                "--authorization",
                str(authorization_relative),
                "--output-dir",
                str(output_relative),
            ],
            "output_dir": str(output_relative),
            "overwrite_allowed": False,
        },
        "probe invocation drifted",
    )
    _require(
        contract["pass_gates"]
        == {
            "actual_cuda_graph_memory_bytes_positive": True,
            "minimum_shared_target_kv_blocks": 21682,
            "pure_first_measured_decode_every_cohort": True,
            "zero_preemption": True,
            "zero_recomputation": True,
            "zero_invalid_spec_tokens": True,
            "exact_one_plus_one_token_accounting": True,
        },
        "probe pass gates drifted",
    )


def validate_authorization(
    authorization: Mapping[str, Any],
    *,
    authorization_path: Path = AUTHORIZATION_PATH,
    output_dir: Path = OUTPUT_DIR,
    require_output_absent: bool = True,
) -> dict[str, Any]:
    """Validate exact source-bound authority without executing a GPU command."""
    _require(
        authorization_path.resolve() == AUTHORIZATION_PATH.resolve(),
        "probe must use the reviewed authorization path",
    )
    _require(
        output_dir.resolve() == OUTPUT_DIR.resolve(),
        "probe must use the reviewed create-new output path",
    )
    if require_output_absent:
        _require(
            not output_dir.exists(), "probe output already exists; retry forbidden"
        )
    _require(
        set(authorization)
        == {
            "schema_version",
            "package_id",
            "date",
            "status",
            "source_artifacts",
            "run_contract",
            "execution_policy",
            "decision",
            "claims",
            "authorizations",
            "next_artifact",
        },
        "probe authorization fields drifted",
    )
    _require(
        authorization["schema_version"] == 1
        and authorization["package_id"]
        == "p4-b0-chunked-prefill-probe-authorization-v1"
        and authorization["date"] == "2026-08-09"
        and authorization["status"]
        == "authorized_gpu4_non_scored_chunked_prefill_probe_only",
        "probe authorization identity drifted",
    )
    sources = _validate_sources(authorization["source_artifacts"])
    _validate_contract(authorization["run_contract"])
    _require(
        authorization["execution_policy"]
        == {
            "create_new_output_only": True,
            "single_parent_launch": True,
            "single_child_boot": True,
            "retry_allowed": False,
            "resume_allowed": False,
            "reuse_v9_captures_allowed": False,
            "fallback_gpu_allowed": False,
            "score_output": False,
        },
        "probe execution policy permits retry, reuse, fallback, or scoring",
    )
    _require(
        authorization["decision"]
        == {
            "decision": "authorize",
            "scope": "one_non_scored_gpu4_probe",
            "reason": "live_wiring_cpu_regressions_passed",
        },
        "probe decision is not narrowly authorized",
    )
    _require(
        authorization["claims"]
        == {
            "live_engine_wired": True,
            "ordinary_serving_unchanged": True,
            "gpu_resource_fit_proven": False,
            "value_screen_run_ready": False,
            "p4a_ready": False,
            "performance_claim_allowed": False,
        },
        "probe authorization claims drifted",
    )
    _require(
        authorization["authorizations"]
        == {
            "gpu4_non_scored_probe": True,
            "v10_value_screen": False,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "probe package grants authority beyond one non-scored GPU-4 probe",
    )
    _require(
        authorization["next_artifact"]
        == {
            "kind": "p4_b0_chunked_prefill_gpu_probe_result",
            "path": (
                "research/97_composition_runtime/data/p4/"
                "run_b0_chunked_prefill_probe_v1/probe_result.json"
            ),
            "on_pass": "draft_separate_source_bound_v10_authorization",
            "on_fail": "diagnose_without_retry_or_fallback",
            "v10_authorized_here": False,
        },
        "probe next-artifact boundary drifted",
    )
    return {
        "artifact_id": authorization["package_id"],
        "status": "pass",
        "source_hashes": sources,
        "gpu_authority_granted": True,
        "gpu_executed": False,
        "scoring_authorized": False,
        "v10_authorized": False,
        "output_absent": not output_dir.exists(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, default=AUTHORIZATION_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = validate_authorization(
            _load_json(args.authorization),
            authorization_path=args.authorization,
            output_dir=args.output_dir,
        )
    except P4ChunkedPrefillAuthorizationError as exc:
        raise SystemExit(
            f"P4 chunked-prefill probe authorization rejected: {exc}"
        ) from exc
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(payload, end="")
    else:
        args.out.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
