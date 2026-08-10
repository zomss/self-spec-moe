#!/usr/bin/env python3
"""Validate the Phase 97 same-event target-step accounting contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
CONTRACT_SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_same_event_accounting.schema.json"
EXPECTED_EVENT_SCHEMA_PATH = (
    "research/97_composition_runtime/schemas/p4_target_step_event.schema.json"
)
W14D_DIR = REPO_ROOT / "research" / "96_selector_foundations" / "data" / "w14"


class SameEventAccountingError(ValueError):
    """Raised when an accounting contract or event is not fail-closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SameEventAccountingError(message)


def _format_json_path(parts: Sequence[Any]) -> str:
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SameEventAccountingError(
            f"cannot load JSON artifact {path}: {exc}"
        ) from exc
    _require(isinstance(value, dict), f"JSON artifact must be an object: {path}")
    return value


def _validate_schema(
    instance: Mapping[str, Any],
    schema_path: Path,
    label: str,
) -> None:
    schema = _load_json(schema_path)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise SameEventAccountingError(f"invalid {label} schema: {exc}") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(instance),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if errors:
        first = errors[0]
        path = _format_json_path(list(first.absolute_path))
        raise SameEventAccountingError(
            f"{label} schema rejected {path}: {first.message}"
        )


def _repository_path(relative_path: str) -> Path:
    path = (REPO_ROOT / relative_path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise SameEventAccountingError(f"artifact escapes repository: {path}") from exc
    return path


def _validate_event_schema_reference(contract: Mapping[str, Any]) -> Path:
    reference = contract["event_schema"]
    _require(
        reference["path"] == EXPECTED_EVENT_SCHEMA_PATH,
        "accounting contract points to an unexpected event schema",
    )
    path = _repository_path(str(reference["path"]))
    _require(path.is_file(), f"missing event schema: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    _require(
        actual == reference["sha256"],
        f"event schema hash mismatch: {actual} != {reference['sha256']}",
    )
    return path


def _bundle_sha256(files: Sequence[Path]) -> str:
    rows = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(REPO_ROOT).as_posix()
        rows.append(f"{digest}  {relative}\n")
    return hashlib.sha256("".join(sorted(rows)).encode()).hexdigest()


def _audit_w14d(contract: Mapping[str, Any]) -> dict[str, Any]:
    files = sorted(W14D_DIR.glob("w14d_block*.json"))
    _require(len(files) == 21, "legacy W14/D bundle must contain 21 files")
    rounds = 0
    negative = 0
    minimum: int | None = None
    for path in files:
        payload = _load_json(path)
        _require(payload.get("complete") is True, f"incomplete W14/D file: {path}")
        for cell in payload["cells"]:
            for row in cell["rounds"]:
                rounds += 1
                unarmed = row["H_target_steps"] - row["D_armed"]
                minimum = unarmed if minimum is None else min(minimum, unarmed)
                negative += int(unarmed < 0)
    observed = {
        "bundle_sha256": _bundle_sha256(files),
        "round_count": rounds,
        "negative_unarmed_round_count": negative,
        "minimum_unarmed_steps": minimum,
    }
    registered = contract["legacy_w14d_disposition"]
    for name, value in observed.items():
        _require(
            registered[name] == value,
            f"legacy W14/D audit drift for {name}: {registered[name]} != {value}",
        )
    _require(negative > 0, "legacy W14/D defect unexpectedly disappeared")
    return observed


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the registered accounting contract.

    Args:
        contract: Parsed same-event accounting contract.

    Returns:
        Machine-readable contract and legacy-bundle status.

    Raises:
        SameEventAccountingError: If the contract or evidence has drifted.
    """
    _validate_schema(contract, CONTRACT_SCHEMA_PATH, "accounting contract")
    _validate_event_schema_reference(contract)
    legacy = _audit_w14d(contract)
    _require(
        not any(contract["authorizations"].values()),
        "the unwired accounting contract cannot grant downstream authority",
    )
    return {
        "status": "pass",
        "contract_id": contract["contract_id"],
        "contract_status": contract["status"],
        "legacy_w14d_status": contract["legacy_w14d_disposition"]["status"],
        "legacy_w14d_audit": legacy,
        "gpu_recollection_authorized": contract["authorizations"]["gpu_recollection"],
        "p4a_engineering_authorized": contract["authorizations"]["p4a_engineering"],
    }


def _expected_exclusions(event: Mapping[str, Any]) -> list[str]:
    reasons = []
    if not event["complete"]:
        reasons.append("incomplete_event")
    if not event["pure_decode"]:
        reasons.append("prefill_or_mixed_batch")
    quality = event["quality"]
    if quality["preemptions"]:
        reasons.append("preemption")
    if quality["recomputed_tokens"]:
        reasons.append("recomputation")
    if quality["invalid_spec_tokens"]:
        reasons.append("invalid_spec_tokens")
    return reasons


def validate_event(
    event: Mapping[str, Any],
    event_schema_path: Path | None = None,
) -> dict[str, Any]:
    """Validate one target-step record derived from one scheduler event.

    Args:
        event: Parsed target-step event.
        event_schema_path: Optional schema override used by tests.

    Returns:
        Derived counters and scoring eligibility.

    Raises:
        SameEventAccountingError: If sources, counters, or eligibility differ.
    """
    schema_path = event_schema_path or _repository_path(EXPECTED_EVENT_SCHEMA_PATH)
    _validate_schema(event, schema_path, "target-step event")
    _require(
        event["source"]["scheduler_event_id"] == event["event_id"],
        "all observations must bind to the same scheduler event id",
    )

    action_id = str(event["action_id"])
    _require(
        event["source"]["verified_action_id"] == action_id
        and event["source"]["next_action_id"] == action_id,
        "value events must be steady-state under one verified and next action",
    )
    k = int(event["k"])
    if action_id == "off":
        _require(k == 0, "OFF must declare K=0")
    else:
        _require(k == 4, f"{action_id} must declare K=4")

    rows = event["request_steps"]
    request_ids = [row["request_id"] for row in rows]
    _require(
        len(request_ids) == len(set(request_ids)),
        "request ids must be unique within one scheduler event",
    )
    for row in rows:
        accepted = int(row["accepted_draft_tokens"])
        armed = bool(row["draft_armed"])
        _require(
            accepted == 0 or armed,
            f"unarmed request {row['request_id']} has accepted draft tokens",
        )
        _require(
            accepted <= (k if armed else 0),
            f"request {row['request_id']} exceeds its action acceptance bound",
        )
        _require(
            row["raw_generated_tokens"] == accepted + 1,
            f"request {row['request_id']} raw output is not accepted+target",
        )
        _require(
            row["committed_tokens"] + row["clipped_tokens"]
            == row["raw_generated_tokens"],
            f"request {row['request_id']} commit/clipping does not close",
        )
        if action_id == "off":
            _require(
                not armed and accepted == 0,
                "OFF event contains an armed or accepted draft row",
            )

    derived = {
        "H_target_steps": len(rows),
        "D_armed": sum(bool(row["draft_armed"]) for row in rows),
        "A_accepted": sum(row["accepted_draft_tokens"] for row in rows),
        "C_clipped": sum(row["clipped_tokens"] for row in rows),
        "E_committed": sum(row["committed_tokens"] for row in rows),
    }
    derived["U_unarmed"] = derived["H_target_steps"] - derived["D_armed"]
    counters = event["counters"]
    for name, value in derived.items():
        _require(
            counters[name] == value,
            f"event counter {name} differs from same-event rows: "
            f"{counters[name]} != {value}",
        )
    _require(
        0 <= derived["D_armed"] <= derived["H_target_steps"],
        "armed draft rows are not a subset of target rows",
    )
    _require(
        derived["E_committed"] + derived["C_clipped"]
        == derived["A_accepted"] + derived["H_target_steps"],
        "same-event token closure E+C=A+H failed",
    )
    timing = event["timing"]
    expected_request_time = timing["engine_event_elapsed_s"] * derived["H_target_steps"]
    _require(
        math.isclose(
            timing["request_decode_time_s"],
            expected_request_time,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        "request decode time must be derived from the same engine event",
    )

    expected_exclusions = _expected_exclusions(event)
    _require(
        sorted(event["exclusion_reasons"]) == sorted(expected_exclusions),
        "event exclusion reasons do not match its source quality",
    )
    expected_eligible = not expected_exclusions
    _require(
        event["score_eligible"] is expected_eligible,
        "event scoring eligibility does not fail closed",
    )
    return {
        "event_id": event["event_id"],
        "action_id": action_id,
        "score_eligible": expected_eligible,
        "counters": derived,
        "request_decode_time_s": timing["request_decode_time_s"],
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--event", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    """Validate the contract and optional target-step event artifacts."""
    args = parse_args()
    contract = _load_json(args.contract)
    result = validate_contract(contract)
    event_schema = _repository_path(contract["event_schema"]["path"])
    result["events"] = [
        validate_event(_load_json(path), event_schema) for path in args.event
    ]
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
