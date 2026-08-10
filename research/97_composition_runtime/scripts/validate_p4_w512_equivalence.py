#!/usr/bin/env python3
"""Validate the Phase 97 boot-static w512 acceptance-equivalence proof."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_w512_equivalence.schema.json"

ACTION_ID = "target-matching-w512-masked-k4"
K4_ACTION_ID = "target-matching-k4"
WINDOW = 512
SINKS = 16
K = 4
BLOCK_SIZE = 16

EXPECTED_SOURCE_PATHS = {
    "p4_entry": (
        "research/97_composition_runtime/data/p4/p4_window_entry_w512_masked.json"
    ),
    "base_boot": ("research/97_composition_runtime/data/p3/boot_b0_minimal_k4.json"),
    "base_registry": (
        "research/97_composition_runtime/data/p3/actions_b0_minimal_k4.json"
    ),
    "runtime_snapshot": (
        "research/97_composition_runtime/data/p3/runtime_snapshot_b0_synthetic.json"
    ),
    "environment": (
        "research/97_composition_runtime/data/preflight/"
        "environment_qwen3_8b_h100_tp1.json"
    ),
    "value_screen_preregistration": (
        "research/97_composition_runtime/data/p4/p4_b0_value_screen_prereg.json"
    ),
    "live_recorder_wiring": (
        "research/97_composition_runtime/data/p4/p4_live_recorder_wiring.json"
    ),
}

EXPECTED_SYMBOLS = {
    "window_step0_gate": (
        "vllm/v1/spec_decode/llm_base_proposer.py",
        "SpecDecodeBaseProposer._kv_window_step0_active",
    ),
    "window_compaction": (
        "vllm/v1/spec_decode/llm_base_proposer.py",
        "SpecDecodeBaseProposer._apply_draft_kv_window",
    ),
    "step0_rejection_trim": (
        "vllm/v1/spec_decode/llm_base_proposer.py",
        "SpecDecodeBaseProposer._compact_step0_decode",
    ),
    "draft_slot_buffer": (
        "vllm/v1/spec_decode/llm_base_proposer.py",
        "SpecDecodeBaseProposer._get_slot_mapping",
    ),
    "target_slot_mapping": (
        "vllm/v1/worker/gpu_model_runner.py",
        "GPUModelRunner._get_slot_mappings",
    ),
    "shared_kv_binding": (
        "vllm/v1/worker/gpu_model_runner.py",
        "GPUModelRunner.initialize_kv_cache_tensors",
    ),
    "live_kv_alias_validator": (
        "vllm/v1/spec_decode/koff_runtime.py",
        "validate_shared_kv_aliases",
    ),
    "live_slot_identity": (
        "vllm/v1/spec_decode/koff_runtime.py",
        "slot_mapping_identity",
    ),
    "live_weight_alias_validator": (
        "vllm/v1/spec_decode/koff_runtime.py",
        "validate_shared_weight_aliases",
    ),
}

EXPECTED_ENVIRONMENT = {
    "VLLM_SELF_SPEC_SHARED_KV": "1",
    "VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE": "1",
    "VLLM_SELF_SPEC_SHARE_WEIGHTS": "1",
    "VLLM_SELF_SPEC_DRAFT_KV_DTYPE": "",
    "VLLM_SELF_SPEC_DRAFT_KV_WINDOW": "512",
    "VLLM_SELF_SPEC_DRAFT_KV_SINKS": "16",
    "VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS": "",
    "VLLM_SELF_SPEC_DRAFT_FULL_CG": "1",
    "VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE": "1",
    "VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD": "1",
    "VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG": "0",
    "VLLM_SELF_SPEC_DRAFT_WHOLECHAIN": "0",
    "VLLM_SELF_SPEC_DRAFT_FULLCG": "0",
}


class W512EquivalenceError(ValueError):
    """Raised when the w512 proof drifts or makes an unsupported claim."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise W512EquivalenceError(message)


def _format_json_path(parts: Sequence[Any]) -> str:
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise W512EquivalenceError(f"cannot load JSON artifact {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON artifact must be an object: {path}")
    return value


def _validate_schema(proof: Mapping[str, Any]) -> None:
    schema = _load_json(SCHEMA_PATH)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise W512EquivalenceError(f"invalid w512 proof schema: {exc}") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(proof),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if errors:
        first = errors[0]
        path = _format_json_path(list(first.absolute_path))
        raise W512EquivalenceError(
            f"w512 proof schema rejected {path}: {first.message}"
        )


def _repository_path(relative_path: str) -> Path:
    path = (REPO_ROOT / relative_path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise W512EquivalenceError(f"artifact escapes repository: {path}") from exc
    return path


def _validate_sources(proof: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    sources = proof["source_artifacts"]
    _require(
        set(sources) == set(EXPECTED_SOURCE_PATHS),
        "w512 proof must bind exactly the registered source roles",
    )
    loaded: dict[str, dict[str, Any]] = {}
    for role, expected_path in EXPECTED_SOURCE_PATHS.items():
        reference = sources[role]
        _require(
            reference["path"] == expected_path,
            f"source role {role} points to an unexpected artifact",
        )
        path = _repository_path(expected_path)
        _require(path.is_file(), f"missing referenced artifact: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(
            actual == reference["sha256"],
            f"artifact hash mismatch for {expected_path}: "
            f"{actual} != {reference['sha256']}",
        )
        loaded[role] = _load_json(path)
    return loaded


def _find_symbol(tree: ast.Module, qualname: str) -> ast.AST:
    names = qualname.split(".")
    body: Sequence[ast.stmt] = tree.body
    node: ast.AST | None = None
    for name in names:
        matches = [
            child
            for child in body
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and child.name == name
        ]
        _require(len(matches) == 1, f"cannot resolve implementation symbol {qualname}")
        node = matches[0]
        body = node.body
    assert node is not None
    return node


def symbol_source_sha256(path: Path, qualname: str) -> str:
    """Hash one normalized function or method body from a Python source file."""
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise W512EquivalenceError(
            f"cannot parse implementation source {path}"
        ) from exc
    node = _find_symbol(tree, qualname)
    segment = ast.get_source_segment(source, node)
    _require(segment is not None, f"cannot extract implementation symbol {qualname}")
    normalized = textwrap.dedent(segment).strip() + "\n"
    return hashlib.sha256(normalized.encode()).hexdigest()


def _validate_implementation_symbols(
    proof: Mapping[str, Any],
    *,
    enforce_current_symbols: bool,
) -> None:
    bindings = {row["role"]: row for row in proof["implementation_symbols"]}
    _require(
        len(bindings) == len(proof["implementation_symbols"]),
        "implementation symbol roles must be unique",
    )
    _require(
        set(bindings) == set(EXPECTED_SYMBOLS),
        "implementation symbol set is incomplete or inflated",
    )
    for role, (expected_path, expected_qualname) in EXPECTED_SYMBOLS.items():
        binding = bindings[role]
        _require(
            (binding["path"], binding["qualname"])
            == (expected_path, expected_qualname),
            f"implementation symbol binding changed for {role}",
        )
        if enforce_current_symbols:
            path = _repository_path(expected_path)
            actual = symbol_source_sha256(path, expected_qualname)
            _require(
                actual == binding["source_sha256"],
                f"implementation symbol hash mismatch for {expected_qualname}: "
                f"{actual} != {binding['source_sha256']}",
            )


def _action_by_id(registry: Mapping[str, Any], action_id: str) -> Mapping[str, Any]:
    matches = [row for row in registry["actions"] if row["action_id"] == action_id]
    _require(len(matches) == 1, f"expected exactly one action {action_id}")
    return matches[0]


def _validate_upstream(
    sources: Mapping[str, Mapping[str, Any]],
) -> tuple[int, str]:
    entry = sources["p4_entry"]
    selected = entry["selected_action"]
    selection = entry["selection"]
    _require(selected["action_id"] == ACTION_ID, "P4 action id drifted")
    _require(selected["base_action_id"] == K4_ACTION_ID, "P4 proper subset drifted")
    _require(
        selected["k"] == K
        and selected["draft_weight_path_id"] == "target-matching"
        and selected["skip_set"] == [],
        "P4 action is no longer target-matching window-only K4",
    )
    _require(
        selected["window"]
        == {"mode": "masked", "value": WINDOW, "cost_grade": "acceptance_only"},
        "P4 masked-window contract drifted",
    )
    _require(
        selection["sink_tokens"] == SINKS and not selection["cost_credit_allowed"],
        "P4 sink count or no-cost-credit rule drifted",
    )

    boot = sources["base_boot"]
    base_environment = boot["environment"]
    _require(
        base_environment
        == {
            "VLLM_SELF_SPEC_SHARED_KV": "1",
            "VLLM_SELF_SPEC_DRAFT_KV_DTYPE": "",
            "VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE": "1",
            "VLLM_SELF_SPEC_SHARE_WEIGHTS": "1",
        },
        "minimal-B0 shared-KV environment drifted",
    )
    kv = boot["kv"]
    _require(
        kv["path"] == "shared_target"
        and kv["owner"] == "target"
        and kv["pool_count"] == 1,
        "minimal-B0 no longer has one target-owned shared-KV pool",
    )
    _require(
        kv["cache_spec"]["block_size"] == BLOCK_SIZE,
        "minimal-B0 KV block size changed",
    )
    _require(
        len(kv["expected_layer_bindings"]) == 36,
        "minimal-B0 must retain all 36 draft/target KV layer twins",
    )
    _require(
        boot["draft_weight_paths"]
        == [
            {
                "path_id": "target-matching",
                "kind": "target_matching",
                "quantization": None,
                "version_policy": "target_alias",
                "resident_bytes": 0,
                "refresh_peak_bytes": 0,
            }
        ],
        "minimal-B0 target-matching weight path drifted",
    )

    registry = sources["base_registry"]
    _require(
        {row["action_id"] for row in registry["actions"]} == {"off", K4_ACTION_ID},
        "base executable registry is no longer the closed OFF/K4 pair",
    )
    base_k4 = _action_by_id(registry, K4_ACTION_ID)
    _require(
        base_k4["k"] == K
        and base_k4["window"] == {"mode": "off", "value": 0, "cost_grade": "none"},
        "base K4 action drifted",
    )

    snapshot = sources["runtime_snapshot"]
    _require(
        snapshot["allocated_pool_ids"] == [snapshot["target_pool_id"]],
        "runtime snapshot contains a non-target KV pool",
    )
    layer_bindings = snapshot["layer_bindings"]
    _require(len(layer_bindings) == 36, "runtime snapshot layer count changed")
    _require(
        all(
            row["same_storage"]
            and row["pool_id"] == snapshot["target_pool_id"]
            and row["draft_cache_spec"] == row["target_cache_spec"]
            for row in layer_bindings
        ),
        "runtime snapshot no longer proves exact target-KV aliases",
    )

    environment = sources["environment"]
    max_model_len = environment["target"]["max_model_len"]
    _require(max_model_len == 20480, "fixed P4 maximum model length changed")
    _require(
        environment["kv"]["block_size_tokens"] == BLOCK_SIZE,
        "fixed P4 environment block size changed",
    )

    preregistration = sources["value_screen_preregistration"]
    action = _action_by_id(preregistration, ACTION_ID)
    measurement = action["measurement"]
    _require(
        measurement["realization"] == "boot-static-mask-equivalent-surrogate"
        and measurement["equivalence_status"] == "pending",
        "frozen value screen no longer awaits this exact surrogate proof",
    )
    _require(
        measurement["cost_source"] == "target-matching-k4-no-window-credit"
        and measurement["surrogate_latency_use"] == "diagnostic_only",
        "frozen value screen permits unsupported w512 cost transfer",
    )
    _require(
        "w512_mask_equivalence_unproven"
        in preregistration["readiness"]["blocking_reason_codes"],
        "frozen value-screen blocker snapshot changed",
    )

    wiring = sources["live_recorder_wiring"]
    _require(
        wiring["readiness_update"]["cleared"] == ["same_event_recorder_unwired"],
        "live-recorder prerequisite is not cleared",
    )
    _require(
        set(wiring["readiness_update"]["remaining"])
        == {
            "w512_mask_equivalence_unproven",
            "conservative_resource_bound_missing",
        },
        "live-recorder readiness handoff drifted",
    )
    return max_model_len, snapshot["canonical_true_slot_mapping_id"]


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _boot_static_intervals(
    seq_len: int,
    query_offset: int,
    rejected_suffix: int,
) -> tuple[tuple[int, int], ...]:
    """Mirror the checked-in page-compaction and step-0 trim equations."""
    n_total = _ceil_div(seq_len, BLOCK_SIZE)
    n_sink = _ceil_div(SINKS, BLOCK_SIZE)
    n_last = _ceil_div(WINDOW + query_offset, BLOCK_SIZE)
    start_last = min(max(n_total - n_last, n_sink), n_total)
    dropped = max(start_last - n_sink, 0)
    visible_end = seq_len - rejected_suffix
    if dropped == 0:
        return ((0, visible_end),)
    return ((0, n_sink * BLOCK_SIZE), (start_last * BLOCK_SIZE, visible_end))


def _registered_mask_intervals(
    seq_len: int,
    query_offset: int,
    rejected_suffix: int,
) -> tuple[tuple[int, int], ...]:
    """Apply the independently stated sink-or-trailing-page mask predicate."""
    total_pages = _ceil_div(seq_len, BLOCK_SIZE)
    sink_pages = _ceil_div(SINKS, BLOCK_SIZE)
    trailing_pages = _ceil_div(WINDOW + query_offset, BLOCK_SIZE)
    first_trailing_page = max(sink_pages, total_pages - trailing_pages)
    first_trailing_page = min(first_trailing_page, total_pages)
    visible_end = seq_len - rejected_suffix
    if first_trailing_page <= sink_pages:
        return ((0, visible_end),)
    return (
        (0, min(sink_pages * BLOCK_SIZE, visible_end)),
        (first_trailing_page * BLOCK_SIZE, visible_end),
    )


def compute_mask_coverage(max_model_len: int) -> tuple[int, str]:
    """Exhaustively compare every legal K4 query/length/rejection mask."""
    digest = hashlib.sha256()
    case_count = 0
    for seq_len in range(1, max_model_len + 1):
        for query_offset in range(K):
            rejection_counts = (
                range(min(K, seq_len - 1) + 1) if query_offset == 0 else (0,)
            )
            for rejected_suffix in rejection_counts:
                boot = _boot_static_intervals(seq_len, query_offset, rejected_suffix)
                registered = _registered_mask_intervals(
                    seq_len, query_offset, rejected_suffix
                )
                _require(
                    boot == registered,
                    "mask mismatch at "
                    f"seq_len={seq_len}, query_offset={query_offset}, "
                    f"rejected_suffix={rejected_suffix}: {boot} != {registered}",
                )
                payload = (
                    f"{seq_len}:{query_offset}:{rejected_suffix}:"
                    f"{json.dumps(boot, separators=(',', ':'))}\n"
                )
                digest.update(payload.encode())
                case_count += 1
    return case_count, digest.hexdigest()


def _validate_contract(
    proof: Mapping[str, Any],
    max_model_len: int,
    canonical_slot_mapping_id: str,
) -> tuple[int, str]:
    contract = proof["surrogate_contract"]
    _require(
        contract["required_environment"] == EXPECTED_ENVIRONMENT,
        "boot-static surrogate environment is not the fail-closed w512 B0 set",
    )
    _require(
        contract["max_model_len"] == max_model_len
        and contract["block_size_tokens"] == BLOCK_SIZE,
        "surrogate geometry differs from the fixed environment",
    )
    domain = contract["scored_domain"]
    _require(
        domain["sequence_length_max"] == max_model_len,
        "mask proof does not cover the full fixed model length",
    )

    mask = proof["proof"]["mask"]
    _require(
        mask["sequence_lengths_checked"] == [1, max_model_len],
        "mask proof length interval is incomplete",
    )
    case_count, coverage_sha256 = compute_mask_coverage(max_model_len)
    _require(
        mask["case_count"] == case_count,
        f"mask case count changed: {mask['case_count']} != {case_count}",
    )
    _require(
        mask["coverage_sha256"] == coverage_sha256,
        "mask coverage digest does not match exhaustive equivalence",
    )
    for row in mask["boundary_observations"]:
        observed = tuple(tuple(pair) for pair in row["visible_key_intervals"])
        expected = _boot_static_intervals(
            row["true_seq_len"],
            row["draft_query_offset"],
            row["rejected_suffix"],
        )
        _require(observed == expected, f"boundary observation drifted: {row}")

    shared_kv = proof["proof"]["shared_target_kv"]
    _require(
        shared_kv["layer_count"] == 36
        and shared_kv["exact_tensor_object_alias_required"],
        "w512 equivalence must retain all exact target-KV aliases",
    )
    slots = proof["proof"]["canonical_true_slots"]
    _require(
        slots["mapping_id"] == canonical_slot_mapping_id,
        "w512 equivalence names a noncanonical true-slot mapping",
    )

    claims = proof["claims"]
    _require(
        claims["boot_static_acceptance_surrogate_eligible"]
        and not any(
            claims[field]
            for field in (
                "latency_transfer_allowed",
                "window_cost_credit_allowed",
                "cost_equivalence_proven",
                "executable_masked_action_exists",
                "action_admitted",
                "performance_claim_allowed",
            )
        ),
        "w512 proof overclaims cost, implementation, admission, or performance",
    )
    _require(
        not any(proof["authorizations"].values()),
        "equivalence proof cannot authorize GPU or P4a work",
    )
    return case_count, coverage_sha256


def validate_proof(
    proof: Mapping[str, Any],
    *,
    enforce_current_symbols: bool = True,
) -> dict[str, Any]:
    """Validate one proof and return its current readiness summary."""
    _validate_schema(proof)
    sources = _validate_sources(proof)
    _validate_implementation_symbols(
        proof,
        enforce_current_symbols=enforce_current_symbols,
    )
    max_model_len, canonical_slot_mapping_id = _validate_upstream(sources)
    case_count, coverage_sha256 = _validate_contract(
        proof, max_model_len, canonical_slot_mapping_id
    )
    return {
        "status": "pass",
        "proof_id": proof["proof_id"],
        "implementation_symbols_current": enforce_current_symbols,
        "action_id": ACTION_ID,
        "acceptance_surrogate_eligible": True,
        "mask_cases_checked": case_count,
        "mask_coverage_sha256": coverage_sha256,
        "shared_kv_layer_count": 36,
        "canonical_true_slot_mapping_id": canonical_slot_mapping_id,
        "latency_transfer_allowed": False,
        "cost_credit_allowed": False,
        "gpu_measurement_authorized": False,
        "p4a_engineering_authorized": False,
        "cleared_blocker": "w512_mask_equivalence_unproven",
        "remaining_blockers": ["conservative_resource_bound_missing"],
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Validate the requested proof and print a machine-readable summary."""
    args = parse_args()
    result = validate_proof(_load_json(args.proof))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
