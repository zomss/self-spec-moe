#!/usr/bin/env python3
"""Run the non-scored Phase 97 real-serving chunked-prefill diagnosis."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
AUTHORIZATION_PATH = (
    PHASE_DIR
    / "data"
    / "p4"
    / "p4_b0_serving_chunked_prefill_diagnosis_authorization.json"
)
OUTPUT_PATH = PHASE_DIR / "data" / "p4" / "run_b0_serving_chunked_prefill_diagnosis_v1"
PROMPT_MANIFEST_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_prompt_manifest.json"
PROMPT_BUNDLE_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_prompt_tokens.jsonl.gz"
PACKAGE_ID = "p4-b0-real-serving-chunked-prefill-diagnosis-v1"
MODEL_SNAPSHOT = Path(
    "/data/smcho/huggingface/hub/"
    "models--Qwen--Qwen3-8B/snapshots/"
    "b968826d9c46dd6066d109eabc6255188de91218"
)
PROMPT_IDS = tuple(f"R4-s0-p{index:03d}" for index in range(8))
PROMPT_COUNTS = (8077, 7839, 8368, 8435, 7836, 7876, 8597, 8172)
WORKLOAD_SHA256 = "204ed66b50bf7d0b8d0487be1d5a892e20550b4d14ce968edcb07a7dc06af631"
MAX_OUTPUT_TOKENS = 16
PHYSICAL_GPU = 4
GPU_UUID = "GPU-c9d19019-5065-2353-80a9-f1797eb19d51"
MAX_NUM_BATCHED_TOKENS = 8192
EFFECTIVE_SCHEDULER_BUDGET = 8160
GPU_MEMORY_UTILIZATION = 0.90


class ServingDiagnosisError(RuntimeError):
    """Raised when the diagnosis loses its narrow execution contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ServingDiagnosisError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ServingDiagnosisError(f"cannot load JSON artifact {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON artifact must be an object: {path}")
    return value


def _repo_path(relative_path: str) -> Path:
    path = (REPO_ROOT / relative_path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ServingDiagnosisError(f"source escapes repository: {path}") from exc
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_ref(relative_path: str) -> dict[str, str]:
    path = _repo_path(relative_path)
    _require(path.is_file(), f"bound source is missing: {path}")
    return {"path": relative_path, "sha256": _sha256(path)}


def _expected_sources() -> dict[str, str]:
    return {
        "diagnostic_runner": (
            "research/97_composition_runtime/scripts/"
            "run_p4_b0_serving_chunked_prefill_diagnosis.py"
        ),
        "prompt_manifest": (
            "research/97_composition_runtime/data/p4/p4_b0_prompt_manifest.json"
        ),
        "prompt_bundle": (
            "research/97_composition_runtime/data/p4/p4_b0_prompt_tokens.jsonl.gz"
        ),
        "scheduler": "vllm/v1/core/sched/scheduler.py",
        "koff_runtime": "vllm/v1/spec_decode/koff_runtime.py",
        "model_runner": "vllm/v1/worker/gpu_model_runner.py",
        "draft_model": "vllm/v1/spec_decode/draft_model.py",
        "draft_proposer": "vllm/v1/spec_decode/llm_base_proposer.py",
        "sampler_backend": "vllm/v1/sample/ops/topk_topp_sampler.py",
    }


def _expected_invocation() -> list[str]:
    return [
        ".venv/bin/python",
        _expected_sources()["diagnostic_runner"],
        "--authorization",
        (
            "research/97_composition_runtime/data/p4/"
            "p4_b0_serving_chunked_prefill_diagnosis_authorization.json"
        ),
        "--output-dir",
        (
            "research/97_composition_runtime/data/p4/"
            "run_b0_serving_chunked_prefill_diagnosis_v1"
        ),
    ]


def validate_authorization(
    authorization: Mapping[str, Any], *, require_output_absent: bool
) -> dict[str, Any]:
    """Validate exact source, workload, resource, and authority boundaries."""
    expected_fields = {
        "schema_version",
        "package_id",
        "date",
        "status",
        "prior_measurement_attempt",
        "source_artifacts",
        "run_contract",
        "diagnostic_contract",
        "decision",
        "claims",
        "authorizations",
        "execution_policy",
        "next_artifact",
    }
    _require(set(authorization) == expected_fields, "authorization fields drifted")
    _require(
        authorization.get("schema_version") == 1
        and authorization.get("package_id") == PACKAGE_ID
        and authorization.get("status")
        == "authorized_gpu4_non_scored_serving_diagnosis_only",
        "wrong serving-diagnosis package",
    )
    prior = authorization["prior_measurement_attempt"]
    _require(
        prior
        == {
            "authorization": _file_ref(
                "research/97_composition_runtime/data/p4/"
                "p4_b0_run_authorization_v5.json"
            ),
            "failure": _file_ref(
                "research/97_composition_runtime/data/p4/"
                "run_b0_value_screen_v4/failure.json"
            ),
            "disposition": "v5_consumed_zero_captures_no_score",
        },
        "V5 failure binding drifted",
    )
    sources = _expected_sources()
    _require(
        set(authorization["source_artifacts"]) == set(sources),
        "source closure drifted",
    )
    for role, path in sources.items():
        _require(
            authorization["source_artifacts"][role] == _file_ref(path),
            f"source hash drifted for {role}",
        )

    expected_run = {
        "invocation": {
            "argv": _expected_invocation(),
            "output_dir": str(OUTPUT_PATH.relative_to(REPO_ROOT)),
            "create_new": True,
        },
        "gpu": {
            "physical_index": PHYSICAL_GPU,
            "uuid": GPU_UUID,
            "fallback_authorized": False,
        },
        "model": {
            "model_id": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "snapshot_path": str(MODEL_SNAPSHOT),
            "target_quantization": None,
            "target_kv_dtype": "bfloat16",
            "draft_weight_source": "target_alias",
            "draft_kv_source": "target_shared_cache",
        },
        "engine": {
            "max_model_len": 20480,
            "max_num_seqs": 32,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "effective_scheduler_token_budget": EFFECTIVE_SCHEDULER_BUDGET,
            "enable_chunked_prefill": True,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "async_scheduling": False,
            "enforce_eager": False,
            "prefix_caching": False,
            "native_sampler": True,
            "dynamic_k_schedule": [[1, 32, 0], [33, 33, 4]],
        },
        "workload": {
            "regime_id": "R4",
            "content_seed": 0,
            "prompt_record_ids": list(PROMPT_IDS),
            "prompt_token_counts": list(PROMPT_COUNTS),
            "total_prompt_tokens": sum(PROMPT_COUNTS),
            "canonical_workload_sha256": WORKLOAD_SHA256,
            "max_output_tokens_per_request": MAX_OUTPUT_TOKENS,
            "temperature": 0.0,
            "ignore_eos": True,
            "generation_seed": 0,
        },
    }
    _require(authorization["run_contract"] == expected_run, "run contract drifted")
    _require(
        authorization["diagnostic_contract"]
        == {
            "recording": "append_only_non_scored_koff_trace",
            "p4_capture_recorder_enabled": False,
            "expected_boundary": "natural_mixed_prefill_decode",
            "mixed_decode_policy": "q1_off_force_off",
            "require_later_pure_decode": True,
            "shared_target_kv_required": True,
            "private_draft_kv_allowed": False,
            "score_eligible_output_allowed": False,
        },
        "diagnostic contract drifted",
    )
    _require(
        authorization["decision"]
        == {
            "state": "approve",
            "scope": "gpu4_real_serving_8192_chunked_prefill_diagnosis_only",
            "invalidated_by": [
                "bound_source_hash_drift",
                "output_directory_exists",
                "gpu4_not_idle",
                "gpu_identity_drift",
                "engine_or_workload_drift",
                "scoring_or_capture_enablement",
            ],
        },
        "decision boundary drifted",
    )
    _require(
        authorization["claims"]
        == {
            "v5_screen_passed": False,
            "serving_chunked_prefill_validated": False,
            "performance_claim_allowed": False,
            "action_admitted": False,
        },
        "claims overstate evidence",
    )
    _require(
        authorization["authorizations"]
        == {
            "serving_diagnosis": True,
            "value_screen": False,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "authority exceeds one serving diagnosis",
    )
    _require(
        authorization["execution_policy"]
        == {
            "physical_boot_count": 1,
            "retry_allowed": False,
            "partial_resume_allowed": False,
            "prior_output_reuse_allowed": False,
            "on_any_failure": "stop_preserve_trace_require_fresh_authorization",
        },
        "execution policy drifted",
    )
    _require(
        authorization["next_artifact"]
        == {
            "kind": "p4_b0_real_serving_chunked_prefill_diagnosis",
            "may_score_value_screen": False,
            "may_authorize_p4a": False,
            "may_admit_action": False,
        },
        "post-run boundary drifted",
    )
    _require(MODEL_SNAPSHOT.is_dir(), "bound model snapshot is missing")
    if require_output_absent:
        _require(not OUTPUT_PATH.exists(), "create-new output already exists")
    return expected_run


def _load_prompts() -> dict[str, list[int]]:
    manifest = _load_json(PROMPT_MANIFEST_PATH)
    _require(
        manifest.get("manifest_id") == "p4-b0-six-regime-prompts-v1", "wrong manifest"
    )
    try:
        with gzip.open(PROMPT_BUNDLE_PATH, "rt", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ServingDiagnosisError(f"cannot read prompt bundle: {exc}") from exc
    by_id = {row["record_id"]: row for row in rows}
    _require(set(PROMPT_IDS) <= set(by_id), "diagnostic prompts are missing")
    prompts = {record_id: by_id[record_id]["token_ids"] for record_id in PROMPT_IDS}
    _require(
        tuple(len(prompts[record_id]) for record_id in PROMPT_IDS) == PROMPT_COUNTS,
        "diagnostic prompt lengths drifted",
    )
    canonical = [
        {"record_id": record_id, "token_ids": prompts[record_id]}
        for record_id in PROMPT_IDS
    ]
    payload = json.dumps(
        canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    _require(hashlib.sha256(payload).hexdigest() == WORKLOAD_SHA256, "workload drifted")
    return prompts


def _child_environment(trace_path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(PHYSICAL_GPU),
            "PYTHONDONTWRITEBYTECODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_SELF_SPEC_KOFF_RUNTIME": "1",
            "VLLM_SELF_SPEC_KOFF_TRACE": str(trace_path.resolve()),
            "VLLM_SELF_SPEC_SHARED_KV": "1",
            "VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE": "1",
            "VLLM_SELF_SPEC_SHARE_WEIGHTS": "1",
            "VLLM_SELF_SPEC_DRAFT_KV_DTYPE": "",
            "VLLM_SELF_SPEC_DRAFT_KV_WINDOW": "0",
            "VLLM_SELF_SPEC_DRAFT_KV_SINKS": "0",
            "VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS": "",
            "VLLM_SELF_SPEC_DRAFT_PARTIAL_REPLICA": "",
            "VLLM_SELF_SPEC_AHEAD_CHAIN": "0",
            "VLLM_SELF_SPEC_CONSUME_AHEAD": "0",
            "VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD": "1",
            "VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE": "1",
            "VLLM_SELF_SPEC_DRAFT_FULLCG": "0",
            "VLLM_SELF_SPEC_DRAFT_FULL_CG": "1",
            "VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG": "0",
            "VLLM_SELF_SPEC_DRAFT_WHOLECHAIN": "0",
            "VLLM_SELF_SPEC_P4_CAPTURE_CONFIG": "",
            "VLLM_SELF_SPEC_P4_CAPTURE_OUTPUT": "",
            "PHASE97_SERVING_DIAG_CHILD": "1",
        }
    )
    executable_dir = Path(sys.prefix) / "bin"
    environment["PATH"] = os.pathsep.join(
        (str(executable_dir), environment.get("PATH", ""))
    )
    return environment


def _preflight(environment: Mapping[str, str]) -> dict[str, Any]:
    expected_prefix = (REPO_ROOT / ".venv").resolve()
    _require(Path(sys.prefix).resolve() == expected_prefix, "not running in .venv")
    ninja = expected_prefix / "bin" / "ninja"
    _require(ninja.is_file() and os.access(ninja, os.X_OK), "venv ninja is missing")
    subprocess.run(
        [str(ninja), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    sampler = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from vllm import envs; "
                "from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler; "
                "s=TopKTopPSampler(); "
                "assert not envs.VLLM_USE_FLASHINFER_SAMPLER; "
                "assert getattr(s.forward, '__func__', None) is "
                "TopKTopPSampler.forward_native"
            ),
        ],
        cwd=REPO_ROOT,
        env=dict(environment),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    _require(sampler.returncode == 0, "native sampler preflight failed")
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(PHYSICAL_GPU),
            "--query-gpu=uuid,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    uuid, used = [part.strip() for part in gpu.split(",", maxsplit=1)]
    _require(uuid == GPU_UUID, "GPU 4 identity drifted")
    _require(int(used) == 0, f"GPU 4 is not idle ({used} MiB used)")
    return {"gpu_uuid": uuid, "gpu_memory_used_mib": int(used), "sampler": "native"}


def _engine_args():
    from vllm import EngineArgs

    speculative_config = {
        "method": "draft_model",
        "model": str(MODEL_SNAPSHOT),
        "num_speculative_tokens": 4,
        "num_speculative_tokens_per_batch_size": [[1, 32, 0], [33, 33, 4]],
        "draft_tensor_parallel_size": 1,
    }
    return EngineArgs(
        model=str(MODEL_SNAPSHOT),
        speculative_config=speculative_config,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=20480,
        max_num_seqs=32,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enable_chunked_prefill=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enable_prefix_caching=False,
        async_scheduling=False,
        enforce_eager=False,
        enable_flashinfer_autotune=False,
        seed=0,
        disable_log_stats=True,
        generation_config="vllm",
    )


def _sampling_params():
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0,
        max_tokens=MAX_OUTPUT_TOKENS,
        ignore_eos=True,
        seed=0,
    )


def _read_trace(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _analyze_trace(
    trace_path: Path, output_counts: Mapping[str, int]
) -> dict[str, Any]:
    records = _read_trace(trace_path)
    headers = [
        row for row in records if row.get("record_type") == "koff_runtime_header"
    ]
    steps = [row for row in records if row.get("record_type") == "koff_engine_step"]
    _require(len(headers) == 1 and bool(steps), "trace is incomplete")
    header = headers[0]
    _require(header.get("scored") is False, "trace attempted to score")
    _require(
        header.get("environment", {}).get("max_num_batched_tokens")
        == EFFECTIVE_SCHEDULER_BUDGET,
        "effective scheduler budget drifted",
    )
    mixed = [
        step
        for step in steps
        if "prefill_or_mixed_batch" in step.get("exclusion_reasons", ())
        and step.get("counters", {}).get("H_target_steps", 0) > 0
    ]
    pure_decode = [
        step
        for step in steps
        if step.get("eligible_for_p3_replay")
        and step.get("verified_action_id") == "off"
    ]
    _require(bool(mixed), "real chunked prefill produced no mixed boundary")
    _require(bool(pure_decode), "trace never reached a later pure decode step")
    for step in mixed:
        _require(
            step.get("verified_action_id") == "off"
            and step.get("next_action_id") == "off"
            and step.get("selection_intent") == "force_off",
            "mixed boundary did not force q1/OFF",
        )
        _require(
            step.get("exclusion_reasons") == ["prefill_or_mixed_batch"],
            "mixed boundary has an additional eligibility failure",
        )
        execution = step["execution"]
        diagnostic = execution["diagnostic"]
        computed = diagnostic["num_computed_tokens"]
        prompts = diagnostic["num_prompt_tokens"]
        widths = diagnostic["num_scheduled_tokens"]
        decode_rows = [
            index
            for index, (done, prompt) in enumerate(zip(computed, prompts, strict=True))
            if done >= prompt
        ]
        prefill_rows = [
            index for index in range(len(widths)) if index not in decode_rows
        ]
        _require(
            bool(decode_rows) and bool(prefill_rows), "mixed row classes are absent"
        )
        _require(
            all(widths[index] == 1 for index in decode_rows),
            "mixed decode row is not q1",
        )
        _require(
            all(widths[index] > 1 for index in prefill_rows),
            "mixed prefill row is not a chunk",
        )
        _require(
            execution["target_query_width"] == 1
            and not execution["draft_dispatched"]
            and execution["produced_draft_width"] == 0
            and not any(diagnostic["num_draft_tokens"]),
            "mixed boundary dispatched a draft",
        )
    for step in steps:
        execution = step["execution"]
        _require(
            execution["shared_kv_layer_count"] == 36
            and execution["shared_kv_storage_alias_count"] == 36,
            "trace lost shared target KV",
        )
        _require(
            execution["shared_weight_parameter_count"] == 291
            and execution["target_weight_version_id"]
            == execution["draft_weight_version_id"],
            "trace lost target/draft weight identity",
        )
    _require(
        output_counts == dict.fromkeys(PROMPT_IDS, MAX_OUTPUT_TOKENS),
        "requests did not complete exact diagnostic work",
    )
    capacities = {
        step["resources"]["shared_target_kv_block_capacity"] for step in steps
    }
    _require(len(capacities) == 1, "shared-KV capacity changed during diagnosis")
    first_mixed = mixed[0]
    first_diagnostic = first_mixed["execution"]["diagnostic"]
    return {
        "schema_version": 1,
        "record_type": "p4_b0_real_serving_chunked_prefill_diagnosis",
        "status": "pass",
        "scored": False,
        "decision": "mixed_boundary_observed_and_safely_forced_off",
        "engine": {
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "effective_scheduler_token_budget": EFFECTIVE_SCHEDULER_BUDGET,
            "enable_chunked_prefill": True,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        },
        "workload": {
            "prompt_count": len(PROMPT_IDS),
            "total_prompt_tokens": sum(PROMPT_COUNTS),
            "output_tokens_per_request": MAX_OUTPUT_TOKENS,
        },
        "trace": {
            "engine_step_count": len(steps),
            "mixed_step_count": len(mixed),
            "pure_decode_step_count": len(pure_decode),
            "first_mixed_step_index": first_mixed["engine_step_index"],
            "first_mixed_request_ids": first_diagnostic["request_ids"],
            "first_mixed_query_widths": first_diagnostic["num_scheduled_tokens"],
            "first_mixed_computed_tokens": first_diagnostic["num_computed_tokens"],
            "first_mixed_prompt_tokens": first_diagnostic["num_prompt_tokens"],
            "shared_target_kv_blocks": next(iter(capacities)),
        },
        "invariants": {
            "mixed_steps_are_score_ineligible": True,
            "mixed_steps_force_q1_off": True,
            "mixed_steps_dispatch_no_draft": True,
            "later_pure_decode_observed": True,
            "shared_target_kv_layer_count": 36,
            "private_draft_kv_allocated": False,
            "target_draft_weight_alias_count": 291,
        },
        "claims": {
            "value_screen_repaired": False,
            "performance_claim_allowed": False,
            "action_admitted": False,
        },
    }


def _run_child(output_dir: Path) -> None:
    _require(os.environ.get("PHASE97_SERVING_DIAG_CHILD") == "1", "unauthorized child")
    trace_path = output_dir / "koff_trace.jsonl"
    result_path = output_dir / "diagnosis.json"
    outputs_path = output_dir / "request_outputs.json"
    _require(
        not trace_path.exists() and not result_path.exists(), "child output exists"
    )
    prompts = _load_prompts()
    from vllm import LLMEngine

    engine = LLMEngine.from_engine_args(_engine_args())
    outputs: dict[str, list[int]] = {}
    try:
        params = _sampling_params()
        for record_id in PROMPT_IDS:
            engine.add_request(
                record_id,
                {"prompt_token_ids": prompts[record_id]},
                params,
            )
        steps = 0
        while engine.has_unfinished_requests():
            for request_output in engine.step():
                if request_output.finished:
                    _require(request_output.outputs, "finished request has no output")
                    outputs[request_output.request_id] = list(
                        request_output.outputs[0].token_ids
                    )
            steps += 1
            _require(steps <= 256, "serving diagnosis exceeded 256 engine steps")
    finally:
        engine.engine_core.shutdown()
    output_counts = {
        record_id: len(token_ids) for record_id, token_ids in outputs.items()
    }
    outputs_path.write_text(
        json.dumps(
            {"scored": False, "request_token_ids": outputs}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    result = _analyze_trace(trace_path, output_counts)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _execute(authorization: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    run = validate_authorization(authorization, require_output_absent=True)
    trace_path = output_dir / "koff_trace.jsonl"
    environment = _child_environment(trace_path)
    preflight = _preflight(environment)
    output_dir.mkdir(parents=False, exist_ok=False)
    preparation = {
        "schema_version": 1,
        "record_type": "p4_b0_serving_chunked_prefill_diagnosis_preparation",
        "package_id": PACKAGE_ID,
        "scored": False,
        "gpu_executed": False,
        "preflight": preflight,
        "run_contract": run,
    }
    (output_dir / "preparation.json").write_text(
        json.dumps(preparation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--authorization",
                str(AUTHORIZATION_PATH.resolve()),
                "--output-dir",
                str(output_dir.resolve()),
                "--child",
            ],
            cwd=REPO_ROOT,
            env=environment,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        failure = {
            "schema_version": 1,
            "record_type": "p4_b0_serving_chunked_prefill_diagnosis_failure",
            "scored": False,
            "returncode": exc.returncode,
            "preserve_without_retry": True,
        }
        (output_dir / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    return _load_json(output_dir / "diagnosis.json")


def parse_args() -> argparse.Namespace:
    """Parse the create-new diagnosis interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    """Validate or execute exactly one authorized GPU-4 diagnosis."""
    args = parse_args()
    _require(
        args.authorization.resolve() == AUTHORIZATION_PATH.resolve(),
        "authorization path drifted",
    )
    _require(args.output_dir.resolve() == OUTPUT_PATH.resolve(), "output path drifted")
    authorization = _load_json(args.authorization)
    if args.child:
        validate_authorization(authorization, require_output_absent=False)
        _run_child(args.output_dir)
        return 0
    if args.validate_only:
        run = validate_authorization(authorization, require_output_absent=True)
        print(
            json.dumps(
                {
                    "status": "pass",
                    "gpu_executed": False,
                    "package_id": PACKAGE_ID,
                    "run_contract": run,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = _execute(authorization, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
