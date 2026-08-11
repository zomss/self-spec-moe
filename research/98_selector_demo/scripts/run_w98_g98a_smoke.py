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
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "research/97_composition_runtime/scripts"))

import run_p4_b0_value_screen as matrix  # noqa: E402

PACKAGE_ID = "w98-g98a-smoke-authorization-v1"
AUTHORIZATION_PATH = "research/98_selector_demo/data/w98_g98a_authorization_v1.json"
OUTPUT_PATH = "research/98_selector_demo/data/g98_a"
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
        "boot_classes": list(BOOT_CLASSES),
        "assumption_under_test": {
            "claim": "a quantized draft can boot with shared target KV",
            "prereg_status": "unverified",
            "weight_aliasing_impossible_for_quantized_draft": True,
            "reason": (
                "DraftModel._weight_sharing_enabled raises when the draft "
                "checkpoint or quantization differs from the target"
            ),
            "quantized_boots_use": {"SHARED_KV": 1, "SHARE_WEIGHTS": 0},
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
            "one_target_owned_kv_cache",
            "shared_kv_alias_proven",
            "true_slot_identity_proven",
            "shared_kv_block_capacity_recorded",
            "kmax8_position_counters_reachable",
            "chain_runtime_mode_observed",
            "observed_device_recorded",
        ],
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
    env.update(
        {
            matrix.DEVICE_PIN_ENV: str(LANE["physical_gpu_index"]),
            matrix.CACHE_ROOT_ENV: LANE["cache_root"],
            matrix.V1_MULTIPROCESSING_ENV: matrix.V1_MULTIPROCESSING_VALUE,
            matrix.NATIVE_SAMPLER_ENV: matrix.NATIVE_SAMPLER_VALUE,
            "VLLM_SELF_SPEC_SHARED_KV": "1",
            "VLLM_SELF_SPEC_SHARE_WEIGHTS": "1" if spec["share_weights"] else "0",
            "VLLM_SELF_SPEC_DRAFT_KV_WINDOW": str(spec["window"]),
            "VLLM_SELF_SPEC_DRAFT_KV_SINKS": "16" if spec["window"] else "0",
            "VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS": SKIP_SETS[spec["skip_count"]],
        }
    )
    return env


def parse_args() -> argparse.Namespace:
    """Parse the smoke-gate interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Validate the gate; execution is a separate, later step."""
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
    raise G98ASmokeError(
        "execution is not wired in this package; G98-A is authorized and "
        "validated, and its boot loop is the next implementation step"
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (G98ASmokeError, matrix.P4RunnerError) as exc:
        print(f"G98-A refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
