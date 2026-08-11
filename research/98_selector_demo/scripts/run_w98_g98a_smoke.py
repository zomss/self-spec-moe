#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""G98-A smoke: prove every required boot class initializes. Non-scored.

Five boots, no measurement claim. The gate exists for one reason above the
others: the preregistration records that **a quantized draft booting with
shared target KV is UNVERIFIED**, and the lattice halves from 30 to 15 if it
fails.

That path is genuinely untested. ``DraftModel._weight_sharing_enabled``
raises when the draft checkpoint or quantization differs from the target, so
the quantized boots must run ``SHARE_WEIGHTS=0`` with ``SHARED_KV=1``: a
co-resident 5.7 GiB W4A16 draft alongside the 15.27 GiB bf16 target, sharing
the target's KV cache but not its weights. Phase 97 never exercised this — its
screen was target-matching with weight aliasing on — so both the boot and the
resulting shared-KV block capacity are open questions.

Emits no captures and no acceptance numbers. Grants no Round-1 or Round-2
authority.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "research/97_composition_runtime/scripts"))

import run_p4_b0_value_screen as matrix  # noqa: E402

PACKAGE_ID = "w98-g98a-smoke-authorization-v4"
AUTHORIZATION_PATH = "research/98_selector_demo/data/w98_g98a_authorization_v4.json"
OUTPUT_PATH = "research/98_selector_demo/data/g98_a_v4"
V1_FINDING_PATH = "research/98_selector_demo/data/g98_a/finding.json"
V2_RESULT_PATH = "research/98_selector_demo/data/g98_a_v2/g98a_result.json"
PREREG_MATRIX = "research/98_selector_demo/data/prereg/w98_prereg_matrix.json"
PROMPT_MANIFEST = "research/98_selector_demo/data/prereg/w98_prompt_manifest.json"
TARGET_MODEL = (
    "/data/smcho/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/"
    "b968826d9c46dd6066d109eabc6255188de91218"
)
QUANT_DRAFT_CKPT = "/data/smcho/ckpts/Qwen3-8B-W4A16-INT4"

# Five boot classes. A1-A3 cover the target-matching axes Phase 97 already
# exercised; A4 and A5 are the ones that can invalidate the frozen lattice.
BOOT_CLASSES = (
    {
        "boot_id": "A1-target-matching-woff-skip0",
        "quant": "target-matching",
        "window": 0,
        "skip_count": 0,
        "share_weights": True,
        "purpose": "baseline; the path the Phase 97 screen ran",
    },
    {
        "boot_id": "A2-target-matching-w512-skip0",
        "quant": "target-matching",
        "window": 512,
        "skip_count": 0,
        "share_weights": True,
        "purpose": "window axis",
    },
    {
        "boot_id": "A3-target-matching-woff-skip4",
        "quant": "target-matching",
        "window": 0,
        "skip_count": 4,
        "share_weights": True,
        "purpose": "skip axis",
    },
    {
        "boot_id": "A4-quantized-woff-skip0",
        "quant": "w4a16-quantized",
        "window": 0,
        "skip_count": 0,
        "share_weights": False,
        "purpose": "THE unverified assumption: quantized draft + shared KV",
    },
    {
        "boot_id": "A5-quantized-w512-skip4",
        "quant": "w4a16-quantized",
        "window": 512,
        "skip_count": 4,
        "share_weights": False,
        "purpose": "riskiest composition: quant x window x skip together",
    },
)
SKIP_SETS = {0: "", 4: "2,4,7,16", 8: "2,4,7,11,16,20,25,30"}
LANE = matrix.lane_for_block(1)


class G98ASmokeError(RuntimeError):
    """Raised when the smoke gate cannot prove its authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G98ASmokeError(message)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict), f"{path} is not a JSON object")
    return payload


def expected_authorization() -> dict[str, Any]:
    """Return the exact package this gate will execute under."""
    return {
        "schema_version": 1,
        "package_id": PACKAGE_ID,
        "gate": "G98-A",
        "status": "authorized_non_scored_smoke_only",
        "source_artifacts": {
            role: matrix._file_reference(path)
            for role, path in {
                "prereg_matrix": PREREG_MATRIX,
                "prompt_manifest": PROMPT_MANIFEST,
                "smoke_runner": (
                    "research/98_selector_demo/scripts/run_w98_g98a_smoke.py"
                ),
                "smoke_tests": ("research/98_selector_demo/tests/test_w98_g98a.py"),
                "matrix_runner": (
                    "research/97_composition_runtime/scripts/run_p4_b0_value_screen.py"
                ),
                "koff_runtime": "vllm/v1/spec_decode/koff_runtime.py",
                "draft_model": "vllm/v1/spec_decode/draft_model.py",
            }.items()
        },
        "models": {
            "target": TARGET_MODEL,
            "quantized_draft": QUANT_DRAFT_CKPT,
            "quantized_draft_present": Path(QUANT_DRAFT_CKPT).is_dir(),
        },
        "consumed_v2": {
            "result": matrix._file_reference(V2_RESULT_PATH),
            "boots_passed": 3,
            "passing_boots": [
                "A1",
                "A2 (a window minimal-b0 forbade)",
                "A3 (skip-4, the subset alias proof)",
            ],
            "shared_kv_blocks_observed": {
                "target_matching": 24772,
                "skip4": 24497,
                "quantized": [21946, 22084],
                "phase_97_floor": 21682,
            },
            "reported_5_of_5_by_a_status_bug": True,
            "quantized_draft_loaded": True,
            "quant_assumption_disproven": False,
            "blocked_by": "target-matching alias proof, now scope-aware",
        },
        "consumed_v1": {
            "finding": matrix._file_reference(V1_FINDING_PATH),
            "boots_passed": 0,
            "quant_assumption_disproven": False,
            "causes_repaired": [
                "C1 operator bug: PARTIAL_REPLICA '0' now mapped to ''",
                "C2 contract wall: w98-lattice boot scope registered",
            ],
            "lattice_unchanged_at": 30,
        },
        "boot_classes": list(BOOT_CLASSES),
        "assumption_under_test": {
            "claim": "a quantized draft can boot with shared target KV",
            "prereg_status": "unverified",
            "weight_aliasing_impossible_for_quantized_draft": True,
            "reason": (
                "DraftModel._weight_sharing_enabled raises when the draft "
                "checkpoint or quantization differs from the target"
            ),
            "quantized_boots_use": {
                "SHARED_KV": 1,
                "SHARE_WEIGHTS": 0,
                "BOOT_SCOPE": "w98-lattice",
            },
            "on_failure": (
                "the quant axis drops, the lattice halves from 30 to 15, and "
                "the D1 held-out split must be re-frozen before any Round-1 "
                "measurement"
            ),
        },
        "checks_per_boot": [
            "engine_initializes",
            "in_process_engine_core",
            "native_sampler_bound",
            "single_kv_cache_group",
            "shared_kv_block_capacity_recorded",
            "draft_chain_dispatches",
            "generation_completes",
            "chain_runtime_mode_observed",
            "observed_device_recorded",
        ],
        "deferred_checks": {
            "checks": ["shared_kv_alias_proven", "true_slot_identity_proven"],
            "reason": (
                "these are capture-time evidence produced by the P4 recorder "
                "during a scored run; a smoke gate that emits no captures "
                "cannot produce them, and claiming them here would overstate "
                "what five boots prove"
            ),
            "produced_at": "G98-B",
        },
        "execution_policy": {
            "lane": LANE,
            "physical_boot_count": len(BOOT_CLASSES),
            "captures_emitted": 0,
            "create_new_output_required": True,
            "retry_allowed": False,
            "on_any_failure": "preserve_and_require_fresh_authorization",
        },
        "authorizations": {
            "gpu_measurement": True,
            "g98a_smoke_execution": True,
            "round1_execution": False,
            "round2_execution": False,
            "scoring": False,
            "acceptance_claim": False,
            "production_value_claim": False,
        },
        "next_artifact": {
            "kind": "w98_g98a_smoke_result",
            "scored": False,
            "may_authorize_round1": False,
        },
    }


def validate_authorization(package: Mapping[str, Any]) -> None:
    """Refuse anything but the exact registered non-scored smoke gate.

    Args:
        package: The parsed G98-A authorization.

    Raises:
        G98ASmokeError: If any registered field or source hash drifted.
    """
    expected = expected_authorization()
    _require(set(package) == set(expected), "G98-A authorization fields drifted")
    for key, value in expected.items():
        _require(package[key] == value, f"G98-A authorization drifted at {key}")
    prereg = _load_json(matrix._repository_path(PREREG_MATRIX))
    _require(
        prereg["status"] == "frozen_before_any_scored_data_manifest_committed",
        "G98-A requires the completed preregistration",
    )
    _require(
        prereg["content_seeds"] == [2, 3],
        "G98-A prereg content seeds drifted",
    )
    quants = set(prereg["lattice"]["quant"])
    _require(
        {b["quant"] for b in BOOT_CLASSES} <= quants,
        "a boot class names a quant path outside the frozen lattice",
    )
    _require(
        Path(QUANT_DRAFT_CKPT).is_dir(),
        f"quantized draft checkpoint is missing: {QUANT_DRAFT_CKPT}",
    )


def boot_environment(spec: Mapping[str, Any]) -> dict[str, str]:
    """Build the env for one boot class on the reserved lane."""
    base = _load_json(matrix._repository_path(matrix.BASE_AUTHORIZATION_PATH))[
        "run_contract"
    ]["environment"]
    env = {k: str(v) for k, v in base.items()}
    env.pop(matrix.DEVICE_PIN_ENV, None)
    # PARTIAL_REPLICA parses as a STRING, so the base "0" is truthy and the
    # boot contract reads it as a declared replica. Phase 97's build_boot_specs
    # maps it to "" for the same reason; omitting this is what failed all five
    # boots on the first G98-A attempt.
    if env.get("VLLM_SELF_SPEC_DRAFT_PARTIAL_REPLICA") == "0":
        env["VLLM_SELF_SPEC_DRAFT_PARTIAL_REPLICA"] = ""
    env.update(
        {
            matrix.DEVICE_PIN_ENV: str(LANE["physical_gpu_index"]),
            matrix.CACHE_ROOT_ENV: LANE["cache_root"],
            matrix.V1_MULTIPROCESSING_ENV: matrix.V1_MULTIPROCESSING_VALUE,
            matrix.NATIVE_SAMPLER_ENV: matrix.NATIVE_SAMPLER_VALUE,
            "VLLM_SELF_SPEC_BOOT_SCOPE": "w98-lattice",
            "VLLM_SELF_SPEC_SHARED_KV": "1",
            "VLLM_SELF_SPEC_SHARE_WEIGHTS": "1" if spec["share_weights"] else "0",
            "VLLM_SELF_SPEC_DRAFT_KV_WINDOW": str(spec["window"]),
            "VLLM_SELF_SPEC_DRAFT_KV_SINKS": "16" if spec["window"] else "0",
            "VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS": SKIP_SETS[spec["skip_count"]],
        }
    )
    return env


def _engine_args(spec: Mapping[str, Any]):
    """Build EngineArgs for one boot class.

    A4/A5 point ``speculative_config.model`` at the quantized checkpoint, which
    is what makes weight aliasing impossible and the KV sharing the open
    question.

    Args:
        spec: One registered boot class.

    Returns:
        The EngineArgs for that boot.
    """
    from vllm import EngineArgs

    base = _load_json(matrix._repository_path(matrix.BASE_AUTHORIZATION_PATH))[
        "run_contract"
    ]["engine"]
    draft = TARGET_MODEL if spec["quant"] == "target-matching" else QUANT_DRAFT_CKPT
    return EngineArgs(
        model=TARGET_MODEL,
        speculative_config={
            "method": "draft_model",
            "model": draft,
            "num_speculative_tokens": 4,
            "draft_tensor_parallel_size": 1,
        },
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=base["max_model_len"],
        max_num_batched_tokens=matrix.SERVING_MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=base["max_num_seqs"],
        enable_chunked_prefill=True,
        gpu_memory_utilization=matrix.SERVING_GPU_MEMORY_UTILIZATION,
        enable_prefix_caching=False,
        async_scheduling=False,
        enforce_eager=False,
        enable_flashinfer_autotune=False,
        seed=0,
        disable_log_stats=True,
        generation_config="vllm",
    )


def _kv_block_capacity(engine: Any) -> tuple[int | None, int | None]:
    """Read the live shared-KV pool size and its group count.

    Returns:
        ``(num_gpu_blocks, kv_cache_group_count)``; either may be None when the
        internals are not reachable, which is recorded rather than guessed.
    """
    blocks = groups = None
    try:
        scheduler = engine.engine_core.engine_core.scheduler
        blocks = int(scheduler.kv_cache_manager.block_pool.num_gpu_blocks)
        groups = len(scheduler.kv_cache_config.kv_cache_groups)
    except Exception:  # noqa: BLE001 - absence is evidence, not a crash
        pass
    return blocks, groups


def run_boot(boot_id: str, output_dir: Path) -> int:
    """Boot one class, exercise the draft chain, and record what happened.

    Args:
        boot_id: The registered boot-class id.
        output_dir: The create-only gate output directory.

    Returns:
        Zero when every performed check passed.
    """
    from vllm import LLMEngine, SamplingParams
    from vllm.v1.engine.core_client import InprocClient
    from vllm.v1.spec_decode.koff_runtime import (
        _RUNTIME_OBSERVATION,
        observed_hardware_id,
    )

    spec = next(b for b in BOOT_CLASSES if b["boot_id"] == boot_id)
    record: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "w98_g98a_boot_record",
        "boot_id": boot_id,
        "quant": spec["quant"],
        "window": spec["window"],
        "skip_count": spec["skip_count"],
        "share_weights": spec["share_weights"],
        "scored": False,
        "checks": {},
    }
    checks = record["checks"]
    engine = None
    try:
        engine = LLMEngine.from_engine_args(_engine_args(spec))
        checks["engine_initializes"] = True
        checks["in_process_engine_core"] = isinstance(engine.engine_core, InprocClient)
        checks["native_sampler_bound"] = (
            os.environ.get(matrix.NATIVE_SAMPLER_ENV) == matrix.NATIVE_SAMPLER_VALUE
        )
        blocks, groups = _kv_block_capacity(engine)
        record["shared_kv_block_capacity"] = blocks
        record["kv_cache_group_count"] = groups
        checks["shared_kv_block_capacity_recorded"] = blocks is not None
        checks["single_kv_cache_group"] = groups == 1 if groups else None

        engine.add_request(
            "g98a-0",
            {"prompt_token_ids": list(range(10, 74))},
            SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True),
        )
        emitted = 0
        for _ in range(4096):
            for out in engine.step():
                emitted += len(out.outputs[0].token_ids) if out.outputs else 0
            if not engine.has_unfinished_requests():
                break
        record["generated_tokens"] = emitted
        checks["generation_completes"] = emitted > 0
        mode = _RUNTIME_OBSERVATION.get("chain_runtime_mode", "unobserved")
        record["chain_runtime_mode"] = mode
        checks["chain_runtime_mode_observed"] = mode != "unobserved"
        checks["draft_chain_dispatches"] = mode != "unobserved"
        record["observed_hardware_id"] = observed_hardware_id()
        checks["observed_device_recorded"] = (
            record["observed_hardware_id"] != "unavailable"
        )
    except Exception as exc:  # noqa: BLE001 - a failed boot IS the result
        record["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        checks.setdefault("engine_initializes", False)
    finally:
        if engine is not None:
            with contextlib.suppress(Exception):
                engine.engine_core.shutdown()
    performed = {k: v for k, v in checks.items() if v is not None}
    # An exception forces a fail. The first version computed status from the
    # checks alone, so a boot that threw after its early checks passed was
    # recorded as "pass" with an exception attached -- which is how A4/A5 were
    # briefly reported as verifying the quant assumption while generating no
    # tokens at all.
    record["status"] = (
        "pass"
        if performed and all(performed.values()) and "exception" not in record
        else "fail"
    )
    (output_dir / f"{boot_id}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if record["status"] == "pass" else 1


def execute(authorization_path: Path, output_dir: Path) -> int:
    """Preflight, run all five boots on the reserved lane, then aggregate."""
    validate_authorization(_load_json(authorization_path))
    _require(not output_dir.exists(), f"refusing to overwrite {output_dir}")
    base_env = matrix._boot_child_environment({})
    matrix._preflight_native_sampler(base_env)
    matrix._preflight_inprocess_engine_core(base_env)
    identity = matrix._preflight_gpu_identity_and_idle(
        LANE["physical_gpu_index"], LANE["physical_gpu_uuid"]
    )
    output_dir.mkdir(parents=True)
    lo, hi = LANE["cpu_affinity"].split("-")
    affinity = sorted(range(int(lo), int(hi) + 1))
    Path(LANE["cache_root"]).mkdir(parents=True, exist_ok=True)

    results = []
    for spec in BOOT_CLASSES:
        env = matrix._boot_child_environment(boot_environment(spec))
        log = output_dir / f"{spec['boot_id']}.log"
        with log.open("w", encoding="utf-8") as handle:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--authorization",
                    str(authorization_path.resolve()),
                    "--output-dir",
                    str(output_dir.resolve()),
                    "--boot-id",
                    spec["boot_id"],
                ],
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                preexec_fn=lambda: os.sched_setaffinity(0, affinity),
            )
        path = output_dir / f"{spec['boot_id']}.json"
        results.append(
            _load_json(path)
            if path.is_file()
            else {"boot_id": spec["boot_id"], "status": "no_record", "checks": {}}
        )

    quant_ok = all(
        r["status"] == "pass" for r in results if r.get("quant") == "w4a16-quantized"
    )
    summary = {
        "schema_version": 1,
        "record_type": "w98_g98a_smoke_result",
        "package_id": PACKAGE_ID,
        "scored": False,
        "may_authorize_round1": False,
        "gpu_identity": identity,
        "lane": LANE,
        "boots": results,
        "passed": sum(1 for r in results if r["status"] == "pass"),
        "total": len(results),
        "quantized_draft_with_shared_kv_verified": quant_ok,
        "lattice_consequence": (
            "quant axis retained; lattice stays at 30"
            if quant_ok
            else "quant axis FAILS: lattice halves to 15 and the D1 held-out "
            "split must be re-frozen before any Round-1 measurement"
        ),
    }
    (output_dir / "g98a_result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] == summary["total"] else 1


def parse_args() -> argparse.Namespace:
    """Parse the smoke-gate interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--boot-id", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    """Validate the gate, then run its five boots."""
    args = parse_args()
    package = _load_json(args.authorization.resolve())
    validate_authorization(package)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "mode": "validate_only",
                    "gpu_executed": False,
                    "boot_classes": len(BOOT_CLASSES),
                    "quantized_draft_present": Path(QUANT_DRAFT_CKPT).is_dir(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.boot_id is not None:
        return run_boot(args.boot_id, args.output_dir.resolve())
    return execute(args.authorization.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (G98ASmokeError, matrix.P4RunnerError) as exc:
        print(f"G98-A refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
