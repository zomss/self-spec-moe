#!/usr/bin/env python3
"""Validate the fail-closed Phase 96/F decision for the P4 window action."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from validate_p4_window_entry import validate_entry

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_f_value.schema.json"
W14D_DIR = REPO_ROOT / "research" / "96_selector_foundations" / "data" / "w14"

EXPECTED_SOURCE_PATHS = {
    "w3_value_threshold": "research/96_selector_foundations/w3_preregistration.md",
    "phase96_f_plan": "research/96_selector_foundations/w14_plan.md",
    "p4_entry": (
        "research/97_composition_runtime/data/p4/p4_window_entry_w512_masked.json"
    ),
    "workload": (
        "research/97_composition_runtime/data/preflight/workload_rl_capacity_v1.json"
    ),
    "resource_plan": (
        "research/97_composition_runtime/data/preflight/plan_rl_capacity_v1.json"
    ),
    "resource_candidate": (
        "research/97_composition_runtime/data/p4/"
        "candidate_b0_window512_masked_projection.json"
    ),
    "w14d_preregistration": (
        "research/96_selector_foundations/data/w14/w14d_prereg.json"
    ),
    "w14d_scorer": "research/96_selector_foundations/scripts/score_w14d.py",
}
EXPECTED_BLOCKERS = {
    "workload_objective_unscored",
    "workload_weights_not_frozen",
    "exact_action_acceptance_missing",
    "exact_action_cost_missing",
    "phase96_d_result_missing",
    "phase96_d_accounting_invalid",
    "phase96_e_result_missing",
    "phase96_f_portfolio_result_missing",
    "resource_evidence_not_exact",
    "transition_overhead_unmeasured",
}
EXPECTED_NEXT_REQUIREMENTS = {
    "frozen_scored_workload_objective",
    "declared_workload_weights",
    "repaired_target_step_accounting",
    "matched_off_k4_w512_controls",
}


class P4FValueError(ValueError):
    """Raised when the F decision omits a blocker or claims approval."""


def _format_json_path(parts: Sequence[Any]) -> str:
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise P4FValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise P4FValueError(f"cannot load JSON artifact {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON artifact must be an object: {path}")
    return value


def _validate_schema(decision: Mapping[str, Any]) -> None:
    schema = _load_json(SCHEMA_PATH)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise P4FValueError(f"invalid F decision schema: {exc}") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(decision),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if errors:
        first = errors[0]
        path = _format_json_path(list(first.absolute_path))
        raise P4FValueError(f"F decision schema rejected {path}: {first.message}")


def _repository_path(relative_path: str) -> Path:
    path = (REPO_ROOT / relative_path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise P4FValueError(f"artifact escapes repository: {path}") from exc
    return path


def _artifact_path(reference: Mapping[str, Any]) -> Path:
    path = _repository_path(str(reference["path"]))
    _require(path.is_file(), f"missing referenced artifact: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = str(reference["sha256"])
    _require(
        actual == expected,
        f"artifact hash mismatch for {reference['path']}: {actual} != {expected}",
    )
    return path


def _validate_sources(decision: Mapping[str, Any]) -> dict[str, Path]:
    sources = decision["source_artifacts"]
    _require(
        set(sources) == set(EXPECTED_SOURCE_PATHS),
        "F decision must bind exactly the required source roles",
    )
    paths: dict[str, Path] = {}
    for role, expected_path in EXPECTED_SOURCE_PATHS.items():
        reference = sources[role]
        _require(
            reference["path"] == expected_path,
            f"F source role {role} points to an unexpected artifact",
        )
        paths[role] = _artifact_path(reference)
    return paths


def _expected_w14d_files(prereg: Mapping[str, Any]) -> dict[str, tuple[int, str]]:
    expected: dict[str, tuple[int, str]] = {}
    for block, actions in enumerate(prereg["block_order"], start=1):
        for action in actions:
            name = f"w14d_block{block}_{action}.json"
            expected[name] = (block, str(action))
    return expected


def _bundle_sha256(files: Sequence[Path]) -> str:
    """Match `sha256sum files | sort | sha256sum` from the repository root."""
    rows = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(REPO_ROOT).as_posix()
        rows.append(f"{digest}  {relative}\n")
    return hashlib.sha256("".join(sorted(rows)).encode()).hexdigest()


def _validate_w14d_bundle(
    audit: Mapping[str, Any],
    prereg: Mapping[str, Any],
) -> dict[str, Any]:
    files = sorted(W14D_DIR.glob("w14d_block*.json"))
    expected = _expected_w14d_files(prereg)
    _require(
        {path.name for path in files} == set(expected),
        "W14/D bundle does not match the 21 registered boot slots",
    )

    prompt_hashes = {key: value["sha256"] for key, value in prereg["prompts"].items()}
    complete_count = 0
    round_count = 0
    negative_unarmed = 0
    minimum_unarmed: int | None = None
    for path in files:
        payload = _load_json(path)
        block, action = expected[path.name]
        config, expected_k = ("AR", 0)
        if action != "AR":
            config, k_text = action.rsplit("-K", maxsplit=1)
            expected_k = int(k_text)
        _require(payload["block"] == block, f"wrong block in {path.name}")
        _require(payload["config"] == config, f"wrong configuration in {path.name}")
        _require(payload["K"] == expected_k, f"wrong K in {path.name}")
        _require(
            not payload["smoke"],
            f"scored W14/D file is marked smoke: {path.name}",
        )
        _require(
            payload["prereg_git_rev"] == prereg["git_rev"],
            f"registration revision drift in {path.name}",
        )
        _require(
            payload["prereg_sha_index"] == prompt_hashes,
            f"prompt hash index drift in {path.name}",
        )
        if payload["complete"]:
            complete_count += 1
        _require(len(payload["cells"]) == 20, f"incomplete cell matrix in {path.name}")
        for cell in payload["cells"]:
            _require(
                len(cell["rounds"]) == prereg["iters"],
                f"wrong round count in {path.name}",
            )
            for row in cell["rounds"]:
                round_count += 1
                unarmed = row["H_target_steps"] - row["D_armed"]
                _require(
                    row["U_unarmed"] == unarmed,
                    f"stored U_unarmed disagrees with H-D in {path.name}",
                )
                minimum_unarmed = (
                    unarmed
                    if minimum_unarmed is None
                    else min(minimum_unarmed, unarmed)
                )
                if unarmed < 0:
                    negative_unarmed += 1

    stats = {
        "observed_file_count": len(files),
        "complete_file_count": complete_count,
        "round_count": round_count,
        "negative_unarmed_round_count": negative_unarmed,
        "minimum_unarmed_steps": minimum_unarmed,
        "bundle_sha256": _bundle_sha256(files),
    }
    for field, observed in stats.items():
        _require(
            audit[field] == observed,
            f"W14/D audit drift for {field}: {audit[field]} != {observed}",
        )
    _require(
        negative_unarmed > 0,
        "this not-approved decision requires the recorded D accounting defect",
    )
    return stats


def _validate_absent_result(audit: Mapping[str, Any]) -> None:
    path = _repository_path(str(audit["result_path"]))
    _require(not path.exists(), f"decision is stale because result now exists: {path}")
    _require(not audit["result_present"], f"absent result marked present: {path}")
    _require(not audit["pass"], f"absent result cannot pass: {path}")


def _validate_prerequisites(
    decision: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    audit = decision["prerequisite_audit"]
    entry = _load_json(paths["p4_entry"])
    entry_result = validate_entry(entry)
    workload = _load_json(paths["workload"])
    resource_plan = _load_json(paths["resource_plan"])
    candidate = _load_json(paths["resource_candidate"])
    prereg = _load_json(paths["w14d_preregistration"])

    scope = decision["scope"]
    _require(scope["entry_package_id"] == entry["package_id"], "entry id drift")
    _require(
        scope["candidate_action_id"] == entry["selected_action"]["action_id"],
        "F decision evaluates the wrong candidate action",
    )
    _require(
        scope["proper_subset_action_id"] == entry["selected_action"]["base_action_id"],
        "F decision evaluates the wrong proper subset",
    )
    _require(
        entry_result["not_approved_promotion_gates"] == ["phase96_f_resident_value"],
        "P4 entry no longer records the failed F gate",
    )

    workload_audit = audit["workload_value_contract"]
    _require(
        workload["evidence_grade"] == workload_audit["evidence_grade"],
        "workload evidence grade drift",
    )
    _require(workload["scored"] is False, "engineering workload became scored")
    _require(
        workload["service_objective"]["kind"] == workload_audit["objective_kind"],
        "workload objective drift",
    )
    _require(
        "workload_weights" not in workload,
        "decision is stale because workload weights now exist",
    )
    _require(
        resource_plan["scored"] is False
        and resource_plan["workload_evidence_grade"] == "engineering_assumption",
        "resource plan cannot supply a scored value objective",
    )

    exact = audit["exact_action_value"]
    _require(
        entry["selection"]["phase96_exact_action_match"]
        == exact["phase96_exact_action_match"],
        "exact-action evidence state drift",
    )
    _require(
        all(
            exact[field] is None
            for field in (
                "acceptance_result",
                "cost_result",
                "robust_gain_lcb",
            )
        ),
        "missing exact-action evidence must stay null, not zero",
    )

    d_stats = _validate_w14d_bundle(audit["phase96_d"], prereg)
    _validate_absent_result(audit["phase96_d"])
    _validate_absent_result(audit["phase96_e"])
    _validate_absent_result(audit["phase96_f_portfolio"])
    portfolio = audit["phase96_f_portfolio"]
    _require(
        all(
            portfolio[field] is None
            for field in (
                "workload_weights",
                "robust_objective_value",
                "robust_gain_lcb",
                "sensitivity_result",
            )
        ),
        "unmeasured portfolio values must stay null, not zero",
    )

    resource = audit["resource_feasibility"]
    evidence = candidate["evidence"]
    _require(
        evidence["candidate_realization_match"]
        == resource["candidate_realization_match"],
        "resource realization-match state drift",
    )
    _require(
        evidence["capacity"]["measurement_relation"]
        == resource["measurement_relation"],
        "resource measurement relation drift",
    )
    _require(
        entry_result["resource_decision"] == resource["decision"] == "reject",
        "non-exact P4 resource evidence must be rejected",
    )
    _require(
        set(entry_result["resource_reason_codes"]) == set(resource["reason_codes"]),
        "resource reason-code drift",
    )

    transition = audit["transition_overhead"]
    _require(
        entry["selected_action"]["transition_cost"]["status"] == "unmeasured",
        "entry transition cost is no longer unmeasured",
    )
    _require(
        all(value is None for key, value in transition.items() if key != "pass"),
        "unmeasured transition values must stay null, not zero",
    )
    return d_stats


def validate_decision(decision: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one F decision and return its fail-closed approval summary.

    Args:
        decision: Parsed Phase 96/F decision artifact.

    Returns:
        Machine-readable validation and approval state.

    Raises:
        P4FValueError: If evidence drifts or the artifact claims approval.
    """
    _validate_schema(decision)
    paths = _validate_sources(decision)
    d_stats = _validate_prerequisites(decision, paths)

    result = decision["decision"]
    _require(
        set(result["blocking_reason_codes"]) == EXPECTED_BLOCKERS,
        "F decision must retain every observed blocking reason",
    )
    _require(
        not any(result["authorizations"].values()),
        "a not-approved F decision cannot authorize downstream work",
    )
    _require(
        set(result["next_artifact"]["requires"]) == EXPECTED_NEXT_REQUIREMENTS,
        "next value-screen preregistration omits a prerequisite",
    )
    return {
        "status": "pass",
        "decision_id": decision["decision_id"],
        "approval_state": result["state"],
        "value_interpretation": result["value_interpretation"],
        "blocking_reason_codes": sorted(result["blocking_reason_codes"]),
        "p4a_engineering_authorized": result["authorizations"]["p4a_engineering"],
        "gpu_measurement_authorized": result["authorizations"]["gpu_measurement"],
        "action_admission_authorized": result["authorizations"]["action_admission"],
        "w14d_audit": d_stats,
        "next_artifact": result["next_artifact"]["kind"],
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Validate an F decision file and print a machine-readable result."""
    args = parse_args()
    result = validate_decision(_load_json(args.decision))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
