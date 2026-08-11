#!/usr/bin/env python3
"""Adapt one explicit same-event B0 capture into one scoreable round."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from validate_p4_prompt_manifest import validate_manifest
from validate_p4_same_event_accounting import (
    SameEventAccountingError,
    validate_event,
)

PHASE_DIR = Path(__file__).resolve().parents[1]
CAPTURE_SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_same_event_capture.schema.json"
ROUND_SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_adapted_round.schema.json"
PROMPT_MANIFEST_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_prompt_manifest.json"

ACTION_ORDERS = {
    1: ["off", "target-matching-k4", "target-matching-w512-masked-k4"],
    2: ["target-matching-k4", "target-matching-w512-masked-k4", "off"],
    3: ["target-matching-w512-masked-k4", "off", "target-matching-k4"],
}
ACTION_REALIZATIONS = {
    "off": "live-b0-forced-off",
    "target-matching-k4": "live-b0-target-matching-k4",
    "target-matching-w512-masked-k4": ("boot-static-mask-equivalent-surrogate"),
}
STABLE_RUNNER_FIELDS = (
    "target_checkpoint_revision",
    "target_quantization",
    "target_kv_dtype",
    "draft_weight_version",
    "hardware_id",
    "parallel_layout",
    "kernel_backend",
    "graph_grade",
    "warmup_policy",
    "measurement_currency",
)


class B0AdapterError(ValueError):
    """Raised when a raw capture cannot safely become score evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise B0AdapterError(message)


def _format_json_path(parts: Sequence[Any]) -> str:
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise B0AdapterError(f"cannot load JSON artifact {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON artifact must be an object: {path}")
    return value


@cache
def _schema_validator(schema_path: Path) -> Draft202012Validator:
    schema = _load_json(schema_path)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise B0AdapterError(f"invalid JSON schema {schema_path}: {exc}") from exc
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
        raise B0AdapterError(f"{label} schema rejected {path}: {first.message}")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@cache
def _frozen_prompt_manifest() -> dict[str, Any]:
    manifest = _load_json(PROMPT_MANIFEST_PATH)
    result = validate_manifest(manifest)
    _require(
        result["exact_prompt_manifest_frozen"],
        "the checked prompt manifest is not frozen",
    )
    return manifest


def _prompt_group(
    manifest: Mapping[str, Any], regime_id: str, content_seed: int
) -> tuple[list[str], Mapping[str, Any]]:
    prompts = [
        row
        for row in manifest["prompts"]
        if row["regime_id"] == regime_id and row["content_seed"] == content_seed
    ]
    prompts.sort(key=lambda row: row["prompt_index"])
    _require(len(prompts) == 32, "prompt manifest group is not exactly 32 records")
    _require(
        [row["prompt_index"] for row in prompts] == list(range(32)),
        "prompt manifest group indices are not canonical",
    )
    regimes = {row["regime_id"]: row for row in manifest["prompt_plan"]["regimes"]}
    _require(regime_id in regimes, f"unknown prompt regime {regime_id}")
    return [row["record_id"] for row in prompts], regimes[regime_id]


def _validate_capture_binding(
    capture: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[list[str], Mapping[str, Any]]:
    _require(capture["complete"], "incomplete capture cannot be adapted")
    _require(capture["warmup_complete"], "pre-warmup capture cannot be adapted")
    runner = capture["runner"]
    manifest_sha = hashlib.sha256(PROMPT_MANIFEST_PATH.read_bytes()).hexdigest()
    _require(
        runner["prompt_manifest_sha256"] == manifest_sha,
        "capture prompt-manifest hash does not match the frozen manifest",
    )
    _require(
        runner["prompt_bundle_sha256"] == manifest["bundle"]["sha256"],
        "capture prompt-bundle hash does not match the frozen bundle",
    )

    matrix = capture["matrix"]
    block_id = matrix["boot_block_id"]
    position = matrix["action_position"]
    action_id = matrix["action_id"]
    _require(
        ACTION_ORDERS[block_id][position - 1] == action_id,
        "capture action does not match the counterbalanced block position",
    )
    _require(
        matrix["action_realization"] == ACTION_REALIZATIONS[action_id],
        "capture action realization does not match the frozen runner contract",
    )

    prompt_ids, regime = _prompt_group(
        manifest, matrix["regime_id"], matrix["content_seed"]
    )
    generation = capture["generation"]
    _require(
        generation["prompt_record_ids"] == prompt_ids,
        "capture prompt ids differ from the exact frozen manifest group",
    )
    expected_generation = {
        "generation_seed": manifest["prompt_plan"]["generation_seed"],
        "batch": regime["batch"],
        "max_output_tokens": regime["max_output_tokens"],
        "temperature": regime["temperature"],
        "ignore_eos": manifest["prompt_plan"]["ignore_eos"],
        "requested_output_tokens": len(prompt_ids) * regime["max_output_tokens"],
    }
    for field, expected in expected_generation.items():
        _require(
            generation[field] == expected,
            f"capture generation field {field} drifted: "
            f"{generation[field]} != {expected}",
        )
    return prompt_ids, regime


def adapt_capture(
    capture: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert one complete explicit capture into one scoreable round.

    Args:
        capture: Raw capture containing canonical target-step events.

    Returns:
        One schema-valid adapted-round record.

    Raises:
        B0AdapterError: If any source, event, or equal-work invariant fails.
    """
    _validate_schema(capture, CAPTURE_SCHEMA_PATH, "B0 capture")
    prompt_manifest = _frozen_prompt_manifest()
    prompt_ids, _ = _validate_capture_binding(capture, prompt_manifest)

    events = capture["events"]
    event_ids = [event.get("event_id") for event in events]
    _require(
        all(isinstance(event_id, str) and event_id for event_id in event_ids),
        "every capture event must expose an event id",
    )
    _require(
        len(event_ids) == len(set(event_ids)),
        "capture contains duplicate scheduler event ids",
    )
    step_indices = [event.get("engine_step_index") for event in events]
    _require(
        all(isinstance(index, int) and index >= 0 for index in step_indices),
        "every capture event must expose a nonnegative engine-step index",
    )
    _require(
        all(left < right for left, right in zip(step_indices, step_indices[1:])),
        "capture engine-step indices must be strictly increasing",
    )

    action_id = capture["matrix"]["action_id"]
    totals = {
        "H_target_steps": 0,
        "D_armed": 0,
        "A_accepted": 0,
        "C_clipped": 0,
        "E_committed": 0,
        "U_unarmed": 0,
    }
    committed_by_request: defaultdict[str, int] = defaultdict(int)
    request_time = 0.0
    for event in events:
        try:
            result = validate_event(event)
        except SameEventAccountingError as exc:
            raise B0AdapterError(
                f"capture event {event.get('event_id', '<missing>')} is not "
                f"canonical: {exc}"
            ) from exc
        _require(
            result["action_id"] == action_id,
            f"capture event {result['event_id']} belongs to another action",
        )
        _require(
            result["score_eligible"],
            f"capture event {result['event_id']} is not score eligible",
        )
        for name in totals:
            totals[name] += result["counters"][name]
        request_time += result["request_decode_time_s"]
        for row in event["request_steps"]:
            committed_by_request[row["request_id"]] += row["committed_tokens"]

    _require(
        set(committed_by_request) == set(prompt_ids),
        "capture request ids do not exactly match the frozen prompt ids",
    )
    expected_per_request = capture["generation"]["max_output_tokens"]
    wrong_work = {
        request_id: tokens
        for request_id, tokens in committed_by_request.items()
        if tokens != expected_per_request
    }
    _require(
        not wrong_work,
        "capture does not commit the exact fixed output for every prompt",
    )
    requested = capture["generation"]["requested_output_tokens"]
    _require(
        totals["E_committed"] == requested,
        "capture committed-token total does not equal requested equal work",
    )
    _require(
        totals["E_committed"] + totals["C_clipped"]
        == totals["A_accepted"] + totals["H_target_steps"],
        "adapted-round token closure E+C=A+H failed",
    )
    _require(
        totals["U_unarmed"] == totals["H_target_steps"] - totals["D_armed"],
        "adapted-round unarmed count does not close",
    )
    _require(
        0 <= totals["D_armed"] <= totals["H_target_steps"],
        "adapted-round draft rows are not a target-row subset",
    )
    _require(
        math.isfinite(request_time) and request_time > 0,
        "adapted-round request decode time is not positive and finite",
    )

    runner = capture["runner"]
    generation = capture["generation"]
    stable = {field: runner[field] for field in STABLE_RUNNER_FIELDS}
    match = {
        "preregistration_id": runner["preregistration_id"],
        "prompt_manifest_id": runner["prompt_manifest_id"],
        "prompt_manifest_sha256": runner["prompt_manifest_sha256"],
        "prompt_bundle_sha256": runner["prompt_bundle_sha256"],
        "prompt_record_ids_sha256": _canonical_sha256(prompt_ids),
        "prompt_count": len(prompt_ids),
        "generation_seed": generation["generation_seed"],
        "batch": generation["batch"],
        "max_output_tokens": generation["max_output_tokens"],
        "temperature": generation["temperature"],
        "ignore_eos": generation["ignore_eos"],
        "requested_output_tokens": requested,
        "stable_config_sha256": _canonical_sha256(stable),
    }
    round_record = {
        "schema_version": 1,
        "round_contract_id": "p4-b0-adapted-round-v1",
        "adapter_contract_id": "p4-b0-same-event-adapter-v1",
        "capture_id": capture["capture_id"],
        "complete": True,
        "score_eligible": True,
        "matrix": dict(capture["matrix"]),
        "match": match,
        "proofs": {
            "shared_kv_binding_id": runner["shared_kv_binding_id"],
            "shared_kv_alias_proven": runner["shared_kv_alias_proven"],
            "true_slot_mapping_id": runner["true_slot_mapping_id"],
            "true_slot_identity_proven": runner["true_slot_identity_proven"],
        },
        "source": {
            "event_count": len(events),
            "rejected_event_count": 0,
            "event_ids_sha256": _canonical_sha256(event_ids),
            "first_engine_step_index": step_indices[0],
            "last_engine_step_index": step_indices[-1],
        },
        "counters": {
            **totals,
            "closure_holds": True,
            "draft_subset_holds": True,
            "equal_work_holds": True,
        },
        "timing": {
            "request_decode_time_s": request_time,
            "source": "sum_same_scheduler_event_request_decode_time",
        },
        "estimands": {
            "decode_rate_req": totals["E_committed"] / request_time,
            "tau_raw": 1 + totals["A_accepted"] / totals["H_target_steps"],
        },
    }
    _validate_schema(round_record, ROUND_SCHEMA_PATH, "adapted B0 round")
    return round_record


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    """Adapt captures and write canonical JSONL or stdout."""
    args = parse_args()
    rounds = [adapt_capture(_load_json(path)) for path in args.capture]
    rendered = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
        for row in rounds
    )
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
