#!/usr/bin/env python3
"""Validate and score the frozen Phase 97 B0 same-event round matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from validate_p4_prompt_manifest import validate_manifest
from validate_p4_same_event_accounting import (
    validate_contract as validate_accounting_contract,
)

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
CONTRACT_SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_runner_scorer.schema.json"
ROUND_SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_adapted_round.schema.json"

EXPECTED_SOURCE_PATHS = {
    "contract_schema": (
        "research/97_composition_runtime/schemas/p4_b0_runner_scorer.schema.json"
    ),
    "capture_schema": (
        "research/97_composition_runtime/schemas/p4_b0_same_event_capture.schema.json"
    ),
    "round_schema": (
        "research/97_composition_runtime/schemas/p4_b0_adapted_round.schema.json"
    ),
    "measurement_adapter": (
        "research/97_composition_runtime/scripts/adapt_p4_b0_same_event.py"
    ),
    "scorer": "research/97_composition_runtime/scripts/score_p4_b0.py",
    "accounting_contract": (
        "research/97_composition_runtime/data/p4/p4_same_event_accounting_contract.json"
    ),
    "prompt_manifest": (
        "research/97_composition_runtime/data/p4/p4_b0_prompt_manifest.json"
    ),
}
EXPECTED_ACTIONS = [
    "off",
    "target-matching-k4",
    "target-matching-w512-masked-k4",
]
EXPECTED_ACTION_ORDERS = {
    1: ["off", "target-matching-k4", "target-matching-w512-masked-k4"],
    2: ["target-matching-k4", "target-matching-w512-masked-k4", "off"],
    3: ["target-matching-w512-masked-k4", "off", "target-matching-k4"],
}
EXPECTED_REGIMES = {
    "R4": (8, 512, 0.0),
    "R5": (8, 512, 0.0),
    "R5cot": (8, 3072, 0.0),
    "R8": (16, 2048, 1.0),
    "R1": (1, 1024, 0.0),
    "R6": (32, 256, 0.0),
}


class B0ScorerError(ValueError):
    """Raised when a scorer contract or adapted matrix fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise B0ScorerError(message)


def _format_json_path(parts: Sequence[Any]) -> str:
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise B0ScorerError(f"cannot load JSON artifact {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON artifact must be an object: {path}")
    return value


@cache
def _schema_validator(schema_path: Path) -> Draft202012Validator:
    schema = _load_json(schema_path)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise B0ScorerError(f"invalid JSON schema {schema_path}: {exc}") from exc
    return Draft202012Validator(schema)


def _validate_schema(
    instance: Mapping[str, Any], schema_path: Path, label: str
) -> None:
    errors = sorted(
        _schema_validator(schema_path).iter_errors(instance),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if errors:
        first = errors[0]
        path = _format_json_path(list(first.absolute_path))
        raise B0ScorerError(f"{label} schema rejected {path}: {first.message}")


def _repository_path(relative_path: str) -> Path:
    path = (REPO_ROOT / relative_path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise B0ScorerError(f"artifact escapes repository: {path}") from exc
    return path


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_sources(contract: Mapping[str, Any]) -> dict[str, Path]:
    sources = contract["source_artifacts"]
    _require(
        set(sources) == set(EXPECTED_SOURCE_PATHS),
        "runner/scorer contract must bind exactly its seven source roles",
    )
    paths: dict[str, Path] = {}
    for role, expected_path in EXPECTED_SOURCE_PATHS.items():
        reference = sources[role]
        _require(
            reference["path"] == expected_path,
            f"runner/scorer source role {role} points to an unexpected artifact",
        )
        path = _repository_path(expected_path)
        _require(path.is_file(), f"missing runner/scorer source artifact: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(
            actual == reference["sha256"],
            f"runner/scorer source hash mismatch for {role}: "
            f"{actual} != {reference['sha256']}",
        )
        paths[role] = path
    return paths


def _validate_matrix_contract(
    contract: Mapping[str, Any], prompt_manifest: Mapping[str, Any]
) -> None:
    matrix = contract["matrix"]
    _require(matrix["actions"] == EXPECTED_ACTIONS, "scorer action order drifted")
    blocks = {row["block_id"]: row["action_order"] for row in matrix["boot_blocks"]}
    _require(blocks == EXPECTED_ACTION_ORDERS, "paired block orders drifted")
    regimes = {row["regime_id"]: row for row in matrix["regimes"]}
    _require(
        set(regimes) == set(EXPECTED_REGIMES),
        "scorer must retain exactly the six W3 regimes",
    )
    for regime_id, expected in EXPECTED_REGIMES.items():
        row = regimes[regime_id]
        observed = (row["batch"], row["max_output_tokens"], row["temperature"])
        _require(observed == expected, f"scorer regime drift for {regime_id}")
        _require(
            row["weight"] == 1 / 6,
            f"scorer regime weight is not equal for {regime_id}",
        )
    manifest_regimes = {
        row["regime_id"]: (
            row["batch"],
            row["max_output_tokens"],
            row["temperature"],
        )
        for row in prompt_manifest["prompt_plan"]["regimes"]
    }
    _require(
        manifest_regimes == EXPECTED_REGIMES,
        "scorer matrix differs from the frozen prompt manifest",
    )
    expected_count = (
        len(blocks)
        * len(EXPECTED_ACTIONS)
        * len(EXPECTED_REGIMES)
        * len(matrix["content_seeds"])
        * matrix["rounds_per_boot"]
    )
    _require(
        matrix["expected_round_records"] == expected_count == 432,
        "scorer matrix record count does not close",
    )


def validate_runner_scorer_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the frozen, still-unwired B0 runner/scorer contract.

    Args:
        contract: Parsed runner/scorer contract.

    Returns:
        Machine-readable freeze and authority summary.

    Raises:
        B0ScorerError: If a source, rule, or authority has drifted.
    """
    _validate_schema(contract, CONTRACT_SCHEMA_PATH, "runner/scorer contract")
    paths = _validate_sources(contract)
    accounting = _load_json(paths["accounting_contract"])
    accounting_result = validate_accounting_contract(accounting)
    prompt_manifest = _load_json(paths["prompt_manifest"])
    prompt_result = validate_manifest(prompt_manifest)
    _require(
        accounting_result["contract_status"] == "registered_unwired",
        "runner/scorer must remain bound to the unwired accounting contract",
    )
    _require(
        prompt_result["exact_prompt_manifest_frozen"],
        "runner/scorer prompt manifest is not exactly frozen",
    )
    _validate_matrix_contract(contract, prompt_manifest)
    _require(
        contract["input_contract"]["legacy_p3_score_reuse"] is False,
        "legacy P3 traces cannot enter the value score",
    )
    _require(
        not contract["readiness"]["same_event_live_recorder_wired"],
        "this CPU-only contract cannot claim live recorder wiring",
    )
    _require(
        not any(contract["authorizations"].values()),
        "the frozen runner/scorer contract cannot grant authority",
    )
    return {
        "status": "pass",
        "contract_id": contract["contract_id"],
        "contract_status": contract["status"],
        "same_event_measurement_adapter_frozen": contract["readiness"][
            "same_event_measurement_adapter_frozen"
        ],
        "runner_scorer_frozen": contract["readiness"]["runner_scorer_frozen"],
        "same_event_live_recorder_wired": contract["readiness"][
            "same_event_live_recorder_wired"
        ],
        "expected_round_records": contract["matrix"]["expected_round_records"],
        "bootstrap_draws": contract["uncertainty"]["draws"],
        "bootstrap_seed": contract["uncertainty"]["seed"],
        "legacy_p3_status": contract["input_contract"]["legacy_p3_status"],
        "gpu_measurement_authorized": contract["authorizations"]["gpu_measurement"],
        "p4a_engineering_authorized": contract["authorizations"]["p4a_engineering"],
    }


def _prompt_id_hashes(
    prompt_manifest: Mapping[str, Any],
) -> dict[tuple[str, int], str]:
    grouped: defaultdict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in prompt_manifest["prompts"]:
        grouped[(row["regime_id"], row["content_seed"])].append(row)
    result = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda row: row["prompt_index"])
        _require(len(rows) == 32, f"prompt group {key} does not contain 32 rows")
        result[key] = _canonical_sha256([row["record_id"] for row in rows])
    return result


def _round_key(row: Mapping[str, Any]) -> tuple[int, str, str, int, int]:
    matrix = row["matrix"]
    return (
        matrix["boot_block_id"],
        matrix["action_id"],
        matrix["regime_id"],
        matrix["content_seed"],
        matrix["round_index"],
    )


def _validate_round_matrix(
    contract: Mapping[str, Any],
    rounds: Sequence[Mapping[str, Any]],
    prompt_manifest: Mapping[str, Any],
) -> None:
    expected_count = contract["matrix"]["expected_round_records"]
    _require(
        len(rounds) == expected_count,
        f"adapted matrix must contain exactly {expected_count} rounds",
    )
    manifest_sha = hashlib.sha256(
        _repository_path(EXPECTED_SOURCE_PATHS["prompt_manifest"]).read_bytes()
    ).hexdigest()
    bundle_sha = prompt_manifest["bundle"]["sha256"]
    prompt_hashes = _prompt_id_hashes(prompt_manifest)
    seen_keys: set[tuple[int, str, str, int, int]] = set()
    capture_ids: set[str] = set()
    stable_hashes: set[str] = set()
    boot_ids: defaultdict[tuple[int, str], set[str]] = defaultdict(set)
    proof_ids: defaultdict[tuple[int, str], set[tuple[str, str]]] = defaultdict(set)

    for index, row in enumerate(rounds):
        _validate_schema(row, ROUND_SCHEMA_PATH, f"adapted round {index}")
        key = _round_key(row)
        _require(key not in seen_keys, f"duplicate adapted matrix key {key}")
        seen_keys.add(key)
        _require(
            row["capture_id"] not in capture_ids,
            f"duplicate adapted capture id {row['capture_id']}",
        )
        capture_ids.add(row["capture_id"])
        block_id, action_id, regime_id, content_seed, _ = key
        matrix = row["matrix"]
        expected_position = EXPECTED_ACTION_ORDERS[block_id].index(action_id) + 1
        _require(
            matrix["action_position"] == expected_position,
            f"adapted action position drift for {key}",
        )
        stable_hashes.add(row["match"]["stable_config_sha256"])
        boot_key = (block_id, action_id)
        boot_ids[boot_key].add(matrix["boot_id"])
        proof_ids[boot_key].add(
            (
                row["proofs"]["shared_kv_binding_id"],
                row["proofs"]["true_slot_mapping_id"],
            )
        )

        match = row["match"]
        _require(
            match["prompt_manifest_sha256"] == manifest_sha,
            f"adapted prompt-manifest hash drift for {key}",
        )
        _require(
            match["prompt_bundle_sha256"] == bundle_sha,
            f"adapted prompt-bundle hash drift for {key}",
        )
        _require(
            match["prompt_record_ids_sha256"]
            == prompt_hashes[(regime_id, content_seed)],
            f"adapted prompt-id hash drift for {key}",
        )
        expected_regime = EXPECTED_REGIMES[regime_id]
        observed_regime = (
            match["batch"],
            match["max_output_tokens"],
            match["temperature"],
        )
        _require(observed_regime == expected_regime, f"generation drift for {key}")
        requested = 32 * expected_regime[1]
        _require(
            match["requested_output_tokens"] == requested,
            f"requested equal work drift for {key}",
        )

        counters = row["counters"]
        h_steps = counters["H_target_steps"]
        d_armed = counters["D_armed"]
        accepted = counters["A_accepted"]
        clipped = counters["C_clipped"]
        committed = counters["E_committed"]
        _require(committed == requested, f"committed equal work drift for {key}")
        _require(
            committed + clipped == accepted + h_steps,
            f"adapted token closure failed for {key}",
        )
        _require(
            counters["U_unarmed"] == h_steps - d_armed,
            f"adapted unarmed count failed for {key}",
        )
        _require(0 <= d_armed <= h_steps, f"draft subset failed for {key}")
        _require(accepted <= 4 * d_armed, f"draft acceptance bound failed for {key}")
        if action_id == "off":
            _require(
                d_armed == 0 and accepted == 0,
                f"OFF round contains draft work for {key}",
            )
        request_time = row["timing"]["request_decode_time_s"]
        rate = committed / request_time
        tau = 1 + accepted / h_steps
        _require(
            math.isclose(
                row["estimands"]["decode_rate_req"],
                rate,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ),
            f"adapted decode rate is not raw E/time for {key}",
        )
        _require(
            math.isclose(
                row["estimands"]["tau_raw"],
                tau,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ),
            f"adapted tau is not raw 1+A/H for {key}",
        )

    _require(len(stable_hashes) == 1, "matched stable configuration hash drifted")
    _require(
        all(len(ids) == 1 for ids in boot_ids.values()),
        "one paired-block action contains multiple boot ids",
    )
    unique_boots = {next(iter(ids)) for ids in boot_ids.values()}
    _require(len(unique_boots) == 9, "the 3x3 matrix must use nine distinct boots")
    _require(
        all(len(ids) == 1 for ids in proof_ids.values()),
        "shared-KV or true-slot proof id changed within one boot",
    )

    expected_keys = {
        (block, action, regime, seed, round_index)
        for block in EXPECTED_ACTION_ORDERS
        for action in EXPECTED_ACTIONS
        for regime in EXPECTED_REGIMES
        for seed in contract["matrix"]["content_seeds"]
        for round_index in range(1, contract["matrix"]["rounds_per_boot"] + 1)
    }
    _require(seen_keys == expected_keys, "adapted matrix is not exactly complete")


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    counters = {
        name: sum(row["counters"][name] for row in rows)
        for name in (
            "H_target_steps",
            "D_armed",
            "A_accepted",
            "C_clipped",
            "E_committed",
            "U_unarmed",
        )
    }
    request_time = sum(row["timing"]["request_decode_time_s"] for row in rows)
    return {
        **counters,
        "request_decode_time_s": request_time,
        "decode_rate_req": counters["E_committed"] / request_time,
        "tau_raw": 1 + counters["A_accepted"] / counters["H_target_steps"],
    }


def _merge_aggregates(rows: Sequence[Mapping[str, float | int]]) -> dict[str, float]:
    names = (
        "H_target_steps",
        "D_armed",
        "A_accepted",
        "C_clipped",
        "E_committed",
        "U_unarmed",
    )
    counters = {name: sum(float(row[name]) for row in rows) for name in names}
    request_time = sum(float(row["request_decode_time_s"]) for row in rows)
    return {
        **counters,
        "request_decode_time_s": request_time,
        "decode_rate_req": counters["E_committed"] / request_time,
        "tau_raw": 1 + counters["A_accepted"] / counters["H_target_steps"],
    }


def _certify_rounds(
    contract: Mapping[str, Any], rounds: Sequence[Mapping[str, Any]]
) -> tuple[dict[tuple[int, str, str], dict[str, float]], list[dict[str, Any]]]:
    grouped: defaultdict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in rounds:
        matrix = row["matrix"]
        grouped[
            (matrix["action_id"], matrix["regime_id"], matrix["content_seed"])
        ].append(row)

    per_seed_boot: dict[tuple[int, str, str, int], dict[str, float | int]] = {}
    audit: list[dict[str, Any]] = []
    episode = contract["episode_protocol"]
    for (action_id, regime_id, content_seed), cell_rows in sorted(grouped.items()):
        rates = sorted(
            (row["estimands"]["decode_rate_req"] for row in cell_rows),
            reverse=True,
        )
        _require(len(rates) == 12, "episode cell must contain 3x4 rounds")
        batch = EXPECTED_REGIMES[regime_id][0]
        reference = rates[1] if batch == 1 else rates[0]
        threshold = (1 - episode["rejection_fraction"]) * reference
        boot_rates = []
        total_rejected = 0
        kept_counts = {}
        for block_id in EXPECTED_ACTION_ORDERS:
            boot_rows = [
                row for row in cell_rows if row["matrix"]["boot_block_id"] == block_id
            ]
            kept = [
                row
                for row in boot_rows
                if row["estimands"]["decode_rate_req"] >= threshold
            ]
            _require(
                len(kept) >= episode["minimum_surviving_rounds_per_boot"],
                "episode rejection left fewer than two rounds for "
                f"{action_id}/{regime_id}/seed{content_seed}/block{block_id}",
            )
            summary = _aggregate_rows(kept)
            per_seed_boot[(block_id, action_id, regime_id, content_seed)] = summary
            boot_rates.append(float(summary["decode_rate_req"]))
            rejected = len(boot_rows) - len(kept)
            total_rejected += rejected
            kept_counts[str(block_id)] = len(kept)
        disagreement = (max(boot_rates) - min(boot_rates)) / max(boot_rates)
        _require(
            disagreement <= episode["cross_boot_agreement_fraction"],
            "cross-boot certification exceeded 2% for "
            f"{action_id}/{regime_id}/seed{content_seed}: {disagreement:.6f}",
        )
        audit.append(
            {
                "action_id": action_id,
                "regime_id": regime_id,
                "content_seed": content_seed,
                "reference_rate": reference,
                "kept_rounds_by_block": kept_counts,
                "rejected_rounds": total_rejected,
                "cross_boot_disagreement_fraction": disagreement,
                "certified": True,
            }
        )

    block_summaries = {}
    for block_id in EXPECTED_ACTION_ORDERS:
        for action_id in EXPECTED_ACTIONS:
            for regime_id in EXPECTED_REGIMES:
                seed_rows = [
                    per_seed_boot[(block_id, action_id, regime_id, seed)]
                    for seed in contract["matrix"]["content_seeds"]
                ]
                block_summaries[(block_id, action_id, regime_id)] = _merge_aggregates(
                    seed_rows
                )
    return block_summaries, audit


def _draw_metrics(
    block_summaries: Mapping[tuple[int, str, str], Mapping[str, float]],
    selected_blocks: Sequence[int],
) -> dict[str, dict[str, float]]:
    output = {}
    for regime_id in EXPECTED_REGIMES:
        actions = {}
        for action_id in EXPECTED_ACTIONS:
            summaries = [
                block_summaries[(block_id, action_id, regime_id)]
                for block_id in selected_blocks
            ]
            actions[action_id] = _merge_aggregates(summaries)
        rate_off = actions["off"]["decode_rate_req"]
        rate_k4 = actions["target-matching-k4"]["decode_rate_req"]
        tau_k4 = actions["target-matching-k4"]["tau_raw"]
        tau_w512 = actions["target-matching-w512-masked-k4"]["tau_raw"]
        s_k4 = rate_k4 / rate_off
        w512_ratio = tau_w512 / tau_k4
        s_w512 = s_k4 * w512_ratio
        base = max(1.0, s_k4)
        candidate = max(1.0, s_k4, s_w512)
        output[regime_id] = {
            "rate_off": rate_off,
            "rate_k4": rate_k4,
            "tau_k4": tau_k4,
            "tau_w512": tau_w512,
            "s_k4": s_k4,
            "w512_over_k4": w512_ratio,
            "s_w512_no_cost_credit": s_w512,
            "base_score": base,
            "candidate_score": candidate,
            "raw_regime_gain": candidate - base,
        }
    return output


def _percentile_interval(values: Sequence[float]) -> dict[str, float]:
    _require(bool(values), "cannot form an interval from no bootstrap draws")
    ordered = sorted(values)
    count = len(ordered)
    lower_index = min(int(0.025 * count), count - 1)
    upper_index = min(int(0.975 * count), count - 1)
    return {
        "lcb": ordered[lower_index],
        "ucb": ordered[upper_index],
    }


def score_rounds(
    contract: Mapping[str, Any],
    rounds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score one exact adapted-round matrix under the frozen contract.

    Args:
        contract: Frozen B0 runner/scorer contract.
        rounds: Exact 432-record adapted round matrix.

    Returns:
        Point estimates, paired-block intervals, and a non-authorizing gate.

    Raises:
        B0ScorerError: If the matrix is incomplete, uncertified, or invalid.
    """
    validate_runner_scorer_contract(contract)
    prompt_path = _repository_path(EXPECTED_SOURCE_PATHS["prompt_manifest"])
    prompt_manifest = _load_json(prompt_path)
    _validate_round_matrix(contract, rounds, prompt_manifest)
    block_summaries, episode_audit = _certify_rounds(contract, rounds)

    block_ids = sorted(EXPECTED_ACTION_ORDERS)
    point = _draw_metrics(block_summaries, block_ids)
    uncertainty = contract["uncertainty"]
    rng = random.Random(uncertainty["seed"])
    draws: dict[str, defaultdict[str, list[float]]] = {
        regime_id: defaultdict(list) for regime_id in EXPECTED_REGIMES
    }
    for _ in range(uncertainty["draws"]):
        selected = [rng.choice(block_ids) for _ in block_ids]
        metrics = _draw_metrics(block_summaries, selected)
        for regime_id, row in metrics.items():
            for name, value in row.items():
                draws[regime_id][name].append(value)

    rule = contract["decision_rule"]
    regimes = {}
    retained = {}
    for regime_id in EXPECTED_REGIMES:
        ratio_interval = _percentile_interval(draws[regime_id]["w512_over_k4"])
        retained[regime_id] = ratio_interval["lcb"] >= (
            1 - rule["proper_subset_tolerance_fraction"]
        )
        gain_draws = (
            draws[regime_id]["raw_regime_gain"]
            if retained[regime_id]
            else [0.0] * uncertainty["draws"]
        )
        regimes[regime_id] = {
            "point": {
                **point[regime_id],
                "regime_gain": (
                    point[regime_id]["raw_regime_gain"] if retained[regime_id] else 0.0
                ),
            },
            "intervals": {
                "tau_k4": _percentile_interval(draws[regime_id]["tau_k4"]),
                "tau_w512": _percentile_interval(draws[regime_id]["tau_w512"]),
                "s_k4": _percentile_interval(draws[regime_id]["s_k4"]),
                "w512_over_k4": ratio_interval,
                "s_w512_no_cost_credit": _percentile_interval(
                    draws[regime_id]["s_w512_no_cost_credit"]
                ),
                "portfolio_gain": _percentile_interval(gain_draws),
            },
            "w512_retained_after_non_regression": retained[regime_id],
            "w512_benefit_lcb_certified": ratio_interval["lcb"]
            >= 1 + rule["proper_subset_benefit_lcb_fraction"],
        }

    weighted_draws = []
    for draw_index in range(uncertainty["draws"]):
        weighted_draws.append(
            sum(
                (
                    draws[regime_id]["raw_regime_gain"][draw_index]
                    if retained[regime_id]
                    else 0.0
                )
                / 6
                for regime_id in EXPECTED_REGIMES
            )
        )
    weighted_interval = _percentile_interval(weighted_draws)
    weighted_point = sum(
        regimes[regime_id]["point"]["regime_gain"] / 6 for regime_id in EXPECTED_REGIMES
    )
    dominated = all(
        regimes[regime_id]["intervals"]["tau_w512"]["ucb"]
        <= regimes[regime_id]["intervals"]["tau_k4"]["lcb"]
        for regime_id in EXPECTED_REGIMES
    )
    best_single_lcb = max(
        regimes[regime_id]["intervals"]["portfolio_gain"]["lcb"]
        for regime_id in EXPECTED_REGIMES
    )
    mean_branch = weighted_interval["lcb"] >= rule["mean_gain_fraction"]
    single_branch = (
        best_single_lcb >= rule["single_regime_gain_fraction"]
        and weighted_interval["lcb"] >= rule["single_branch_mean_floor"]
    )
    value_pass = not dominated and (mean_branch or single_branch)
    return {
        "status": "pass",
        "contract_id": contract["contract_id"],
        "input_round_records": len(rounds),
        "episode_audit": episode_audit,
        "episode_rejected_rounds": sum(row["rejected_rounds"] for row in episode_audit),
        "certified_episode_cells": len(episode_audit),
        "bootstrap": {
            "method": uncertainty["method"],
            "draws": uncertainty["draws"],
            "seed": uncertainty["seed"],
            "resampling_unit": uncertainty["resampling_unit"],
        },
        "regimes": regimes,
        "portfolio": {
            "weighted_gain_point": weighted_point,
            "weighted_gain_lcb": weighted_interval["lcb"],
            "weighted_gain_ucb": weighted_interval["ucb"],
            "best_single_regime_gain_lcb": best_single_lcb,
        },
        "decision": {
            "dominance_short_circuit": dominated,
            "mean_branch": mean_branch,
            "single_branch": single_branch,
            "value_gate_pass": value_pass,
            "authority_granted": False,
        },
    }


def _load_rounds(path: Path) -> list[dict[str, Any]]:
    try:
        raw = path.read_text()
    except OSError as exc:
        raise B0ScorerError(f"cannot read adapted rounds {path}: {exc}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        rows = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise B0ScorerError(
                    f"invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            _require(
                isinstance(row, dict),
                f"adapted JSONL row {line_number} is not an object",
            )
            rows.append(row)
        return rows
    _require(isinstance(value, list), "adapted rounds JSON must be an array")
    _require(
        all(isinstance(row, dict) for row in value),
        "every adapted rounds array entry must be an object",
    )
    return value


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--rounds", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    """Validate the contract and optionally score an adapted matrix."""
    args = parse_args()
    contract = _load_json(args.contract)
    if args.rounds:
        result = score_rounds(contract, _load_rounds(args.rounds))
    else:
        result = validate_runner_scorer_contract(contract)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
