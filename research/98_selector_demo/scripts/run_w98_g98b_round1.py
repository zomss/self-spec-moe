#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""G98-B Round 1: single-lever profiles, factored fit, held-out reveal.

D1 claims the factored cost model eliminates *soundly*. That claim is only
testable if the predictions for the held-out composed set were fixed before
those configurations were measured, so this gate is built around a
**commitment barrier**:

    B1 singles (GPU)  ->  B2 fit + commit (CPU)  ||  B3 reveal (GPU)  ->  B4 score

B3 refuses to run unless B2's predictions exist and are hash-registered, and
refuses if any held-out measurement already exists. B4 refuses to score unless
the predictions it compares against are the committed ones. The barrier is the
gate's reason for existing; without it D1 is unfalsifiable.

Round 1 reads the PASSIVE step trace (VLLM_SELF_SPEC_KOFF_TRACE ->
build_live_step_record), which is scope-aware, not the P4 capture recorder.
The capture path still asserts target/draft weight equality in two places and
would refuse a quantized boot; that is Round 2's problem, not this gate's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "research/97_composition_runtime/scripts"))
sys.path.insert(0, str(PHASE_DIR / "scripts"))

import run_p4_b0_value_screen as matrix  # noqa: E402

PACKAGE_ID = "w98-g98b-round1-authorization-v1"
AUTHORIZATION_PATH = "research/98_selector_demo/data/w98_g98b_authorization_v1.json"
OUTPUT_PATH = "research/98_selector_demo/data/g98_b"
PREREG_MATRIX = "research/98_selector_demo/data/prereg/w98_prereg_matrix.json"
PREREG_HELDOUT = "research/98_selector_demo/data/prereg/w98_d1_heldout.json"
PROMPT_MANIFEST = "research/98_selector_demo/data/prereg/w98_prompt_manifest.json"
G98A_RESULT = "research/98_selector_demo/data/g98_a_v4/g98a_result.json"
PREDICTIONS_NAME = "d1_predictions.json"
SINGLES_DIR = "singles"
HELDOUT_DIR = "heldout"
EPSILON_ARM = 0.015
# Measured at G98-A: a resident W4A16 draft leaves far less shared-KV headroom
# than the target-matching path, so Round 1's shapes are sized against the
# quantized figure rather than the comfortable one.
QUANTIZED_KV_BLOCKS = 22190
PHASE_97_KV_FLOOR = 21682


class G98BError(RuntimeError):
    """Raised when Round 1 cannot prove its authority or its barrier."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G98BError(message)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict), f"{path} is not a JSON object")
    return payload


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def single_lever_profiles() -> list[dict[str, Any]]:
    """Return the single-lever configurations the factored model fits from."""
    prereg = _load_json(matrix._repository_path(PREREG_MATRIX))
    lattice = prereg["lattice"]
    base_quant = lattice["quant"][0]
    singles = [{"quant": base_quant, "window": "off", "skip_count": 0}]
    singles += [
        {"quant": base_quant, "window": w, "skip_count": 0}
        for w in lattice["window"]
        if w != "off"
    ]
    singles += [
        {"quant": base_quant, "window": "off", "skip_count": s}
        for s in lattice["skip_counts"]
        if s
    ]
    singles += [{"quant": lattice["quant"][1], "window": "off", "skip_count": 0}]
    return singles


def expected_authorization() -> dict[str, Any]:
    """Return the exact package this gate will execute under."""
    heldout = _load_json(matrix._repository_path(PREREG_HELDOUT))
    g98a = _load_json(matrix._repository_path(G98A_RESULT))
    return {
        "schema_version": 1,
        "package_id": PACKAGE_ID,
        "gate": "G98-B",
        "status": "authorized_round1_profiles_and_heldout_reveal",
        "source_artifacts": {
            role: matrix._file_reference(path)
            for role, path in {
                "prereg_matrix": PREREG_MATRIX,
                "prereg_heldout": PREREG_HELDOUT,
                "prompt_manifest": PROMPT_MANIFEST,
                "g98a_result": G98A_RESULT,
                "round1_runner": (
                    "research/98_selector_demo/scripts/run_w98_g98b_round1.py"
                ),
                "round1_tests": ("research/98_selector_demo/tests/test_w98_g98b.py"),
                "cost_model": ("research/98_selector_demo/scripts/w98_cost_model.py"),
                "koff_runtime": "vllm/v1/spec_decode/koff_runtime.py",
            }.items()
        },
        "prerequisite": {
            "gate": "G98-A",
            "passed": g98a["passed"] == g98a["total"],
            "quantized_draft_with_shared_kv_verified": g98a[
                "quantized_draft_with_shared_kv_verified"
            ],
        },
        "commitment_barrier": {
            "why": (
                "D1 claims sound elimination, which is testable only if the "
                "held-out predictions were fixed before those configurations "
                "were measured"
            ),
            "predictions_committed_before_reveal": True,
            "reveal_refuses_without_committed_predictions": True,
            "reveal_refuses_if_heldout_measurements_exist": True,
            "score_refuses_uncommitted_predictions": True,
        },
        "stages": [
            {
                "id": "B1",
                "name": "singles",
                "gpu": True,
                "boots": len(single_lever_profiles()),
            },
            {"id": "B2", "name": "fit_and_commit", "gpu": False, "boots": 0},
            {
                "id": "B3",
                "name": "heldout_reveal",
                "gpu": True,
                "boots": len(heldout["heldout"]),
            },
            {"id": "B4", "name": "score", "gpu": False, "boots": 0},
        ],
        "d1": {
            "epsilon_arm": EPSILON_ARM,
            "elimination_rule": "(K+1)/q_lo < 1 + epsilon_arm",
            "false_elimination_budget": 0,
            "heldout_count": heldout["heldout_count"],
            "fit_uses_single_lever_profiles_only": True,
            "on_coverage_failure": "local non-pruning mask, not global failure",
        },
        "w14d_fallback": {
            "phase_96_w14d_scored_surface_present": False,
            "verified_on": "2026-08-11",
            "consequence": (
                "affected strata run Round 1 in measure-everything mode and D1 "
                "is reported as NOT EXERCISED there, per the preregistration"
            ),
        },
        "trace_path": {
            "uses": "passive koff step trace (VLLM_SELF_SPEC_KOFF_TRACE)",
            "scope_aware": True,
            "uses_p4_capture_recorder": False,
            "note": (
                "the P4 capture path still asserts target/draft weight equality "
                "at koff_runtime.py:1927 and :2503 and would refuse a quantized "
                "boot; Round 2 must resolve that, Round 1 does not touch it"
            ),
        },
        "resource_note": {
            "quantized_shared_kv_blocks": QUANTIZED_KV_BLOCKS,
            "phase_97_floor": PHASE_97_KV_FLOOR,
            "headroom_fraction": round(QUANTIZED_KV_BLOCKS / PHASE_97_KV_FLOOR - 1, 4),
            "shapes_sized_against": "quantized",
        },
        "execution_policy": {
            "lane": matrix.lane_for_block(1),
            "boot_scope": "w98-lattice",
            "create_new_output_required": True,
            "retry_allowed": False,
            "on_any_failure": "preserve_and_require_fresh_authorization",
        },
        "authorizations": {
            "gpu_measurement": True,
            "round1_execution": True,
            "round1_scoring": True,
            "round2_execution": False,
            "round2_scoring": False,
            "d3_claim": False,
            "production_value_claim": False,
        },
        "next_artifact": {
            "kind": "w98_round1_result",
            "may_authorize_round2": False,
        },
    }


def validate_authorization(package: Mapping[str, Any]) -> None:
    """Refuse anything but the exact registered Round-1 gate.

    Args:
        package: The parsed G98-B authorization.

    Raises:
        G98BError: If any registered field or source hash drifted.
    """
    expected = expected_authorization()
    _require(set(package) == set(expected), "G98-B authorization fields drifted")
    for key, value in expected.items():
        _require(package[key] == value, f"G98-B authorization drifted at {key}")
    _require(
        package["prerequisite"]["passed"],
        "G98-B requires a passing G98-A",
    )
    _require(
        package["prerequisite"]["quantized_draft_with_shared_kv_verified"],
        "G98-B requires the verified quantized-draft assumption",
    )
    prereg = _load_json(matrix._repository_path(PREREG_MATRIX))
    _require(
        prereg["status"] == "frozen_before_any_scored_data_manifest_committed",
        "G98-B requires the completed preregistration",
    )
    _require(
        prereg["tolerances"]["epsilon_arm"] == EPSILON_ARM,
        "G98-B epsilon_arm differs from the preregistration",
    )


def commit_predictions(
    output_dir: Path, predictions: Mapping[str, Any]
) -> dict[str, Any]:
    """Freeze the held-out predictions before any held-out boot runs.

    Args:
        output_dir: The create-only Round-1 output directory.
        predictions: One entry per held-out triple.

    Returns:
        The committed record, including its own digest.

    Raises:
        G98BError: If predictions already exist, or any held-out measurement
            has already been taken.
    """
    output_dir = output_dir.resolve()
    path = output_dir / PREDICTIONS_NAME
    _require(
        not path.exists(),
        "held-out predictions are already committed and are immutable",
    )
    heldout_dir = output_dir / HELDOUT_DIR
    existing = sorted(heldout_dir.glob("*.json")) if heldout_dir.is_dir() else []
    _require(
        not existing,
        "refusing to commit predictions after held-out measurements exist: "
        f"{[p.name for p in existing]}",
    )
    registered = _load_json(matrix._repository_path(PREREG_HELDOUT))["heldout"]
    keys = {_triple_key(t) for t in registered}
    _require(
        set(predictions) == keys,
        "predictions must cover exactly the registered held-out set",
    )
    record = {
        "schema_version": 1,
        "record_type": "w98_d1_committed_predictions",
        "package_id": PACKAGE_ID,
        "committed_before_reveal": True,
        "heldout_count": len(keys),
        "predictions": dict(predictions),
    }
    record["digest"] = _digest(record["predictions"])
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def require_committed_predictions(output_dir: Path) -> dict[str, Any]:
    """Refuse the reveal unless predictions were committed first.

    Args:
        output_dir: The Round-1 output directory.

    Returns:
        The committed predictions record.

    Raises:
        G98BError: If predictions are missing or their digest drifted.
    """
    path = output_dir.resolve() / PREDICTIONS_NAME
    _require(
        path.is_file(),
        "held-out reveal refused: no committed predictions. D1 is "
        "unfalsifiable without them.",
    )
    record = _load_json(path)
    _require(
        record.get("committed_before_reveal") is True,
        "committed predictions do not claim pre-reveal commitment",
    )
    _require(
        record.get("digest") == _digest(record.get("predictions", {})),
        "committed predictions were modified after commitment",
    )
    return record


def _triple_key(triple: Mapping[str, Any]) -> str:
    return f"{triple['quant']}/w{triple['window']}/skip{triple['skip_count']}"


def parse_args() -> argparse.Namespace:
    """Parse the Round-1 interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Validate the gate; the measurement stages are wired separately."""
    args = parse_args()
    validate_authorization(_load_json(args.authorization.resolve()))
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "mode": "validate_only",
                    "gpu_executed": False,
                    "singles": len(single_lever_profiles()),
                    "heldout": len(
                        _load_json(matrix._repository_path(PREREG_HELDOUT))["heldout"]
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise G98BError(
        "Round-1 measurement stages are not wired in this package; the "
        "authorization and its commitment barrier are validated and the "
        "B1/B3 measurement loops are the next implementation step"
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (G98BError, matrix.P4RunnerError) as exc:
        print(f"G98-B refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
