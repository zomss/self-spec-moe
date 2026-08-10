#!/usr/bin/env python3
"""Validate the blocked Phase 97 matched B0 value-screen preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from score_p4_b0 import validate_runner_scorer_contract
from validate_p4_prompt_manifest import validate_manifest as validate_prompt_manifest
from validate_p4_same_event_accounting import validate_contract

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_value_screen.schema.json"

EXPECTED_SOURCE_PATHS = {
    "w3_value_threshold": ("research/96_selector_foundations/w3_preregistration.md"),
    "phase96_f_plan": "research/96_selector_foundations/w14_plan.md",
    "regime_loader": "research/88_regime_eval/scripts/regime_datasets.py",
    "prompt_manifest": (
        "research/97_composition_runtime/data/p4/p4_b0_prompt_manifest.json"
    ),
    "p4_entry": (
        "research/97_composition_runtime/data/p4/p4_window_entry_w512_masked.json"
    ),
    "f_decision": ("research/97_composition_runtime/data/p4/p4_f_value_decision.json"),
    "environment": (
        "research/97_composition_runtime/data/preflight/"
        "environment_qwen3_8b_h100_tp1.json"
    ),
    "accounting_contract": (
        "research/97_composition_runtime/data/p4/p4_same_event_accounting_contract.json"
    ),
    "runner_scorer_contract": (
        "research/97_composition_runtime/data/p4/p4_b0_runner_scorer_contract.json"
    ),
    "resource_candidate": (
        "research/97_composition_runtime/data/p4/"
        "candidate_b0_window512_masked_projection.json"
    ),
    "p3b_result": ("research/97_composition_runtime/results_p3b_mixed_abort.md"),
}
EXPECTED_ACTION_IDS = {
    "off",
    "target-matching-k4",
    "target-matching-w512-masked-k4",
}
EXPECTED_BASE_POOL = {"off", "target-matching-k4"}
EXPECTED_REGIMES = {
    "R4": (8, 512, 0.0),
    "R5": (8, 512, 0.0),
    "R5cot": (8, 3072, 0.0),
    "R8": (16, 2048, 1.0),
    "R1": (1, 1024, 0.0),
    "R6": (32, 256, 0.0),
}
EXPECTED_MATCHED_FIELDS = {
    "target_checkpoint_revision",
    "target_quantization",
    "target_kv_dtype",
    "draft_weight_version",
    "shared_kv_binding",
    "true_slot_mapping",
    "hardware",
    "parallel_layout",
    "prompt_token_ids",
    "generation_seed",
    "batch",
    "generated_suffix",
    "k",
    "kernel_backend",
    "graph_grade",
    "warmup_policy",
    "measurement_currency",
}
EXPECTED_BLOCKERS = {
    "same_event_recorder_unwired",
    "w512_mask_equivalence_unproven",
    "conservative_resource_bound_missing",
}
EXPECTED_NEXT_REQUIREMENTS = {
    "same_event_live_recorder_wiring",
    "w512_mask_equivalence_proof",
    "conservative_resource_bound",
}
EXPECTED_SATISFIED = {
    "research_objective_and_weights_frozen",
    "same_event_contract_registered",
    "matched_action_set_frozen",
    "legacy_w14d_disposition_fixed",
    "exact_prompt_manifest_frozen",
    "same_event_measurement_adapter_frozen",
    "runner_scorer_frozen",
}


class B0ValueScreenError(ValueError):
    """Raised when the B0 value screen is incomplete or over-authorized."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise B0ValueScreenError(message)


def _format_json_path(parts: Sequence[Any]) -> str:
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise B0ValueScreenError(f"cannot load JSON artifact {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON artifact must be an object: {path}")
    return value


def _validate_schema(preregistration: Mapping[str, Any]) -> None:
    schema = _load_json(SCHEMA_PATH)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise B0ValueScreenError(f"invalid B0 value-screen schema: {exc}") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(preregistration),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if errors:
        first = errors[0]
        path = _format_json_path(list(first.absolute_path))
        raise B0ValueScreenError(
            f"B0 value-screen schema rejected {path}: {first.message}"
        )


def _repository_path(relative_path: str) -> Path:
    path = (REPO_ROOT / relative_path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise B0ValueScreenError(f"artifact escapes repository: {path}") from exc
    return path


def _validate_sources(preregistration: Mapping[str, Any]) -> dict[str, Path]:
    sources = preregistration["source_artifacts"]
    _require(
        set(sources) == set(EXPECTED_SOURCE_PATHS),
        "B0 preregistration must bind exactly the required source roles",
    )
    paths: dict[str, Path] = {}
    for role, expected_path in EXPECTED_SOURCE_PATHS.items():
        reference = sources[role]
        _require(
            reference["path"] == expected_path,
            f"B0 source role {role} points to an unexpected artifact",
        )
        path = _repository_path(expected_path)
        _require(path.is_file(), f"missing referenced artifact: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(
            actual == reference["sha256"],
            f"artifact hash mismatch for {expected_path}: "
            f"{actual} != {reference['sha256']}",
        )
        paths[role] = path
    return paths


def _validate_upstream(
    paths: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    entry = _load_json(paths["p4_entry"])
    decision = _load_json(paths["f_decision"])
    environment = _load_json(paths["environment"])
    accounting = _load_json(paths["accounting_contract"])
    runner_scorer = _load_json(paths["runner_scorer_contract"])
    prompt_manifest = _load_json(paths["prompt_manifest"])
    resource = _load_json(paths["resource_candidate"])

    _require(
        entry["selected_action"]["action_id"] == "target-matching-w512-masked-k4",
        "P4 entry candidate action changed",
    )
    _require(
        entry["selected_action"]["base_action_id"] == "target-matching-k4",
        "P4 entry proper subset changed",
    )
    _require(
        entry["selection"]["cost_grade"] == "acceptance_only"
        and not entry["selection"]["cost_credit_allowed"],
        "P4 entry no longer forbids window cost credit",
    )
    _require(
        decision["decision"]["state"] == "not_approved",
        "the source F decision is no longer not-approved",
    )
    _require(
        not any(decision["decision"]["authorizations"].values()),
        "the source F decision unexpectedly grants authority",
    )
    _require(
        environment["self_spec_contract"]["shared_target_kv_required"],
        "fixed environment no longer requires shared target KV",
    )
    accounting_result = validate_contract(accounting)
    runner_scorer_result = validate_runner_scorer_contract(runner_scorer)
    prompt_result = validate_prompt_manifest(prompt_manifest)
    _require(
        accounting_result["contract_status"] == "registered_unwired",
        "same-event accounting is unexpectedly marked wired",
    )
    _require(
        runner_scorer_result["same_event_measurement_adapter_frozen"]
        and runner_scorer_result["runner_scorer_frozen"],
        "same-event adapter or B0 scorer is not frozen",
    )
    _require(
        not runner_scorer_result["same_event_live_recorder_wired"],
        "the live same-event recorder must remain an explicit blocker",
    )
    evidence = resource["evidence"]
    _require(
        evidence["grade"] == "static_proxy_projection"
        and not evidence["candidate_realization_match"],
        "resource proxy no longer matches the registered blocked state",
    )
    _require(
        evidence["capacity"]["measurement_relation"] == "optimistic_proxy_ceiling",
        "resource proxy relation changed",
    )
    return accounting_result, prompt_result, runner_scorer_result


def _validate_objective(preregistration: Mapping[str, Any]) -> dict[str, float]:
    scope = preregistration["scope"]
    _require(scope["scored"], "research value screen must be scored")
    _require(
        not scope["production_workload_evidence"],
        "research suite cannot claim production workload evidence",
    )
    _require(
        not scope["production_value_claim_allowed"],
        "equal research weights cannot authorize a production value claim",
    )

    objective = preregistration["objective"]
    _require(
        set(objective["base_pool"]) == EXPECTED_BASE_POOL,
        "base pool must be exactly OFF plus target-matching K4",
    )
    _require(
        set(objective["candidate_pool"]) == EXPECTED_ACTION_IDS,
        "candidate pool must be exactly OFF, K4, and masked w512",
    )
    regimes = {row["regime_id"]: row for row in objective["regimes"]}
    _require(
        set(regimes) == set(EXPECTED_REGIMES),
        "research objective must use the fixed six-regime W3 suite",
    )
    weights: dict[str, float] = {}
    for regime_id, expected in EXPECTED_REGIMES.items():
        row = regimes[regime_id]
        observed = (row["batch"], row["max_output_tokens"], row["temperature"])
        _require(observed == expected, f"regime contract drift for {regime_id}")
        weights[regime_id] = float(row["weight"])
    _require(
        math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12),
        "research workload weights do not sum to one",
    )
    first_weight = next(iter(weights.values()))
    _require(
        all(
            math.isclose(value, first_weight, abs_tol=1e-12)
            for value in weights.values()
        ),
        "W3 research-suite weights must remain equal",
    )
    return weights


def _validate_actions(preregistration: Mapping[str, Any]) -> None:
    actions = {row["action_id"]: row for row in preregistration["actions"]}
    _require(set(actions) == EXPECTED_ACTION_IDS, "registered action set drifted")

    off = actions["off"]
    _require(
        off["kind"] == "off" and off["k"] == 0,
        "OFF action must remain non-speculative",
    )
    _require(
        off["draft_weight_path_id"] is None,
        "OFF action cannot select a draft-weight path",
    )

    k4 = actions["target-matching-k4"]
    _require(k4["k"] == 4, "target-matching K4 depth changed")
    _require(k4["window"]["mode"] == "off", "K4 proper subset gained a window")
    _require(
        k4["draft_weight_path_id"] == "target-matching",
        "K4 proper subset changed draft weights",
    )

    w512 = actions["target-matching-w512-masked-k4"]
    _require(
        w512["proper_subset_action_id"] == "target-matching-k4",
        "w512 proper subset changed",
    )
    _require(
        w512["window"]
        == {
            "mode": "masked",
            "value": 512,
            "sink_tokens": 16,
            "cost_grade": "acceptance_only",
            "cost_credit_allowed": False,
        },
        "masked w512 semantics or cost grade changed",
    )
    measurement = w512["measurement"]
    _require(
        measurement["equivalence_status"] == "pending",
        "this blocked preregistration requires pending mask equivalence",
    )
    _require(
        measurement["cost_source"] == "target-matching-k4-no-window-credit",
        "w512 screen must borrow no separately booted window-cost credit",
    )
    _require(
        measurement["surrogate_latency_use"] == "diagnostic_only",
        "w512 surrogate latency cannot enter the value score",
    )
    for action in actions.values():
        _require(action["kv_path"] == "shared_target", "private KV entered P4")
        _require(action["skip_set"] == [], "layer skip entered the B0 window screen")


def _validate_measurement_design(
    preregistration: Mapping[str, Any], prompt_result: Mapping[str, Any]
) -> None:
    design = preregistration["measurement_design"]
    prompt = design["prompt_manifest"]
    _require(prompt["status"] == "frozen", "prompt manifest is not frozen")
    _require(prompt["required_before_gpu"], "prompt hashes must precede GPU work")
    _require(
        prompt
        == {
            "status": "frozen",
            "required_before_gpu": True,
            "manifest_source_role": "prompt_manifest",
            "manifest_id": "p4-b0-six-regime-prompts-v1",
            "loader_source_role": "regime_loader",
            "tokenizer_model": "Qwen/Qwen3-8B",
            "tokenizer_revision_status": "frozen_exact_revision",
            "tokenizer_revision": ("b968826d9c46dd6066d109eabc6255188de91218"),
            "content_seeds": [0, 1],
            "prompts_per_seed_and_regime": 32,
            "token_id_hashes_required": True,
            "bundle_record_count": 384,
            "bundle_sha256": (
                "0615cf0174bc476dcc744cb947440b09be4b61043685a262e62f573821de1506"
            ),
        },
        "prompt-manifest binding drifted",
    )
    _require(
        prompt_result["exact_prompt_manifest_frozen"]
        and prompt_result["manifest_id"] == prompt["manifest_id"]
        and prompt_result["tokenizer_revision"] == prompt["tokenizer_revision"]
        and prompt_result["record_count"] == prompt["bundle_record_count"]
        and prompt_result["bundle_sha256"] == prompt["bundle_sha256"],
        "prompt-manifest validation does not match the preregistration",
    )
    _require(
        design["generation"]["equal_work"] and design["generation"]["ignore_eos"],
        "matched screen must use fixed equal output work",
    )
    _require(
        not design["engine"]["async_scheduling"],
        "the first same-event screen is registered with synchronous scheduling",
    )
    _require(
        not design["engine"]["enable_flashinfer_autotune"],
        "autotune must stay disabled under the W3 protocol",
    )
    _require(
        set(design["matched_fields"]) == EXPECTED_MATCHED_FIELDS,
        "matched-ablation fields drifted",
    )
    orders = [block["action_order"] for block in design["paired_boot_blocks"]]
    _require(
        all(set(order) == EXPECTED_ACTION_IDS for order in orders),
        "every paired block must contain all three actions exactly once",
    )
    _require(
        [order[0] for order in orders]
        == [
            "off",
            "target-matching-k4",
            "target-matching-w512-masked-k4",
        ],
        "paired-block action order is no longer counterbalanced",
    )
    _require(
        design["accounting"]["prometheus_use"] == "diagnostic_only",
        "Prometheus interval deltas cannot score the value screen",
    )
    _require(
        design["accounting"]["timing_source"] == "same_scheduler_event_monotonic",
        "decode timing must share the scheduler-event boundary",
    )


def _validate_decision_and_authority(preregistration: Mapping[str, Any]) -> None:
    rule = preregistration["decision_rule"]
    gate = rule["pre_engineering_gate"]
    _require(gate["use_lower_confidence_bound"], "value gate must use an LCB")
    _require(
        gate["mean_gain_fraction"] == 0.02
        and gate["single_regime_gain_fraction"] == 0.05
        and gate["single_branch_mean_floor"] == 0.0,
        "W3 pre-engineering thresholds drifted",
    )
    _require(
        rule["proper_subset_non_regression"]["tolerance_fraction"] == 0.01,
        "proper-subset non-regression tolerance drifted",
    )
    _require(
        rule["resource_gate"]["current_state"] == "reject",
        "projection-only resource evidence must remain rejected",
    )
    _require(
        rule["resource_gate"]["evidence_grade"]
        == "measured_base_plus_conservative_delta_bound"
        and rule["resource_gate"]["measurement_relation"]
        == "conservative_candidate_upper_bound",
        "pre-engineering resources require a conservative candidate bound",
    )
    post_gate = rule["post_engineering_gate"]
    _require(
        post_gate["exact_resource_evidence_grade"] == "measured_exact"
        and post_gate["exact_resource_measurement_relation"] == "exact_candidate",
        "exact resource evidence must remain a post-engineering gate",
    )
    readiness = preregistration["readiness"]
    _require(
        set(readiness["satisfied"]) == EXPECTED_SATISFIED,
        "readiness satisfied set drifted",
    )
    _require(
        set(readiness["blocking_reason_codes"]) == EXPECTED_BLOCKERS,
        "blocked preregistration must retain every run-readiness blocker",
    )
    _require(
        set(preregistration["next_artifact"]["requires"]) == EXPECTED_NEXT_REQUIREMENTS,
        "next run-ready package omits a required artifact",
    )
    _require(
        not any(preregistration["authorizations"].values()),
        "a blocked preregistration cannot grant downstream authority",
    )


def validate_preregistration(preregistration: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen-but-blocked B0 value-screen package.

    Args:
        preregistration: Parsed B0 value-screen artifact.

    Returns:
        Machine-readable readiness and authority summary.

    Raises:
        B0ValueScreenError: If the package drifts or grants authority.
    """
    _validate_schema(preregistration)
    paths = _validate_sources(preregistration)
    accounting, prompt_result, runner_scorer = _validate_upstream(paths)
    weights = _validate_objective(preregistration)
    _validate_actions(preregistration)
    _validate_measurement_design(preregistration, prompt_result)
    _validate_decision_and_authority(preregistration)
    return {
        "status": "pass",
        "preregistration_id": preregistration["preregistration_id"],
        "readiness": preregistration["readiness"]["state"],
        "scored_research_objective_frozen": preregistration["scope"]["scored"],
        "production_workload_evidence": preregistration["scope"][
            "production_workload_evidence"
        ],
        "regime_weights": weights,
        "action_ids": sorted(EXPECTED_ACTION_IDS),
        "exact_prompt_manifest_frozen": prompt_result["exact_prompt_manifest_frozen"],
        "prompt_record_count": prompt_result["record_count"],
        "tokenizer_revision": prompt_result["tokenizer_revision"],
        "same_event_measurement_adapter_frozen": runner_scorer[
            "same_event_measurement_adapter_frozen"
        ],
        "runner_scorer_frozen": runner_scorer["runner_scorer_frozen"],
        "blocking_reason_codes": sorted(EXPECTED_BLOCKERS),
        "legacy_w14d_status": accounting["legacy_w14d_status"],
        "gpu_measurement_authorized": preregistration["authorizations"][
            "gpu_measurement"
        ],
        "p4a_engineering_authorized": preregistration["authorizations"][
            "p4a_engineering"
        ],
        "next_artifact": preregistration["next_artifact"]["kind"],
    }


def evaluate_value_gate(
    preregistration: Mapping[str, Any],
    regime_results: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Evaluate synthetic or future result intervals under the frozen gate.

    Args:
        preregistration: Validated B0 value-screen preregistration.
        regime_results: Per-regime ``tau_k4_lcb``, ``tau_w512_ucb``, and
            ``portfolio_gain_lcb`` values.

    Returns:
        Dominance and W3 branch decision. This does not grant authority.

    Raises:
        B0ValueScreenError: If a registered regime or finite value is missing.
    """
    weights = _validate_objective(preregistration)
    _require(
        set(regime_results) == set(weights),
        "gate results must contain exactly the registered regimes",
    )
    required = {"tau_k4_lcb", "tau_w512_ucb", "portfolio_gain_lcb"}
    for regime_id, row in regime_results.items():
        _require(
            set(row) == required,
            f"gate result fields are incomplete for {regime_id}",
        )
        _require(
            all(math.isfinite(float(value)) for value in row.values()),
            f"gate result contains a non-finite value for {regime_id}",
        )

    dominated = all(
        row["tau_w512_ucb"] <= row["tau_k4_lcb"] for row in regime_results.values()
    )
    weighted_gain = sum(
        weights[regime_id] * row["portfolio_gain_lcb"]
        for regime_id, row in regime_results.items()
    )
    best_single = max(row["portfolio_gain_lcb"] for row in regime_results.values())
    gate = preregistration["decision_rule"]["pre_engineering_gate"]
    mean_branch = weighted_gain >= gate["mean_gain_fraction"]
    single_branch = (
        best_single >= gate["single_regime_gain_fraction"]
        and weighted_gain >= gate["single_branch_mean_floor"]
    )
    value_pass = not dominated and (mean_branch or single_branch)
    return {
        "dominance_short_circuit": dominated,
        "weighted_gain_lcb": weighted_gain,
        "best_single_regime_gain_lcb": best_single,
        "mean_branch": mean_branch,
        "single_branch": single_branch,
        "value_gate_pass": value_pass,
        "authority_granted": False,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    """Validate one B0 value-screen preregistration and print JSON."""
    args = parse_args()
    result = validate_preregistration(_load_json(args.preregistration))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
