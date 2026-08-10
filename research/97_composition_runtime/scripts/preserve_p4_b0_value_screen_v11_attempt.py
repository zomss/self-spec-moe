#!/usr/bin/env python3
"""Preserve the interrupted V11 GPU-4 value-screen attempt."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
OUTPUT_DIR = PHASE_DIR / "data" / "p4" / "run_b0_value_screen_v10"
AUTHORIZATION_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization_v11.json"
TERMINAL_LOG_PATH = PHASE_DIR / "logs" / "p4_b0_value_screen_v11_gpu4_terminal.log"
EXPECTED_AUTHORIZATION_SHA256 = (
    "143c831383d9e7b2f9b1a4f020c8bbf8c185152f9c94537fb2d2c657eb72f60f"
)
EXPECTED_TERMINAL_LOG_SHA256 = (
    "22f64f1945f96b5c4b72236f47819f417de338f1aa611efd357a73d4450060dd"
)
FAILED_CAPTURE_ID = "capture-b1-p2-k4-r4-s0-r1"
LAST_COMPLETE_CAPTURE_ID = "capture-b1-p1-off-r6-s1-r4"
LAST_COMPLETE_ENGINE_STEP = 429487
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
REGIME_ORDER = ("R4", "R5", "R5cot", "R8", "R1", "R6")
CONTENT_SEEDS = (0, 1)
ROUNDS = (1, 2, 3, 4)


class PreservationError(RuntimeError):
    """Raised when the stopped attempt differs from the observed boundary."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreservationError(message)


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reference(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreservationError(f"cannot load {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON artifact is not an object: {path}")
    return value


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise PreservationError(f"refusing to overwrite {path}") from exc


def _capture_record(path: Path) -> dict[str, Any]:
    capture = _load_json(path)
    matrix = capture.get("matrix")
    events = capture.get("events")
    _require(capture.get("complete") is True, f"capture is incomplete: {path}")
    _require(capture.get("scored") is False, f"capture is scored: {path}")
    _require(isinstance(matrix, dict), f"capture matrix is missing: {path}")
    _require(isinstance(events, list) and events, f"capture events are missing: {path}")
    committed_tokens = 0
    for event in events:
        _require(isinstance(event, dict), f"capture event is malformed: {path}")
        counters = event.get("counters")
        _require(isinstance(counters, dict), f"event counters are missing: {path}")
        committed = counters.get("E_committed")
        _require(type(committed) is int and committed >= 0, "bad commit counter")
        committed_tokens += committed
    return {
        **_reference(path),
        "capture_id": capture.get("capture_id"),
        "boot_id": matrix.get("boot_id"),
        "action_id": matrix.get("action_id"),
        "regime_id": matrix.get("regime_id"),
        "content_seed": matrix.get("content_seed"),
        "round_index": matrix.get("round_index"),
        "complete": True,
        "scored": False,
        "event_count": len(events),
        "committed_tokens": committed_tokens,
        "first_engine_step_index": events[0].get("engine_step_index"),
        "last_engine_step_index": events[-1].get("engine_step_index"),
    }


def _package_references(directory: str) -> list[dict[str, Any]]:
    paths = sorted((OUTPUT_DIR / directory).glob("*.json"))
    _require(len(paths) == 9, f"expected nine {directory} artifacts")
    return [_reference(path) for path in paths]


def _expected_complete_capture_ids() -> set[str]:
    return {
        f"capture-b1-p1-off-{regime.lower()}-s{seed}-r{round_index}"
        for regime in REGIME_ORDER
        for seed in CONTENT_SEEDS
        for round_index in ROUNDS
    }


def build_capture_manifest() -> dict[str, Any]:
    """Build a hash-bound inventory of every raw V11 capture file."""
    capture_paths = sorted((OUTPUT_DIR / "captures").glob("*/*.json"))
    _require(len(capture_paths) == 49, "raw capture count differs from 49")
    nonempty = [path for path in capture_paths if path.stat().st_size]
    empty = [path for path in capture_paths if not path.stat().st_size]
    _require(len(nonempty) == 48, "complete capture count differs from 48")
    _require(len(empty) == 1, "empty placeholder count differs from one")
    _require(empty[0].stem == FAILED_CAPTURE_ID, "stopped placeholder changed")

    captures = [_capture_record(path) for path in nonempty]
    capture_ids = {row["capture_id"] for row in captures}
    _require(
        capture_ids == _expected_complete_capture_ids(),
        "complete OFF capture set drifted",
    )
    by_boot = Counter(row["boot_id"] for row in captures)
    by_action = Counter(row["action_id"] for row in captures)
    _require(
        by_boot == {"p4-b0-b1-p1-off": 48},
        "complete captures extend beyond the first OFF boot",
    )
    _require(by_action == {"off": 48}, "action capture counts drifted")
    last_complete = next(
        row for row in captures if row["capture_id"] == LAST_COMPLETE_CAPTURE_ID
    )
    _require(
        last_complete["last_engine_step_index"] == LAST_COMPLETE_ENGINE_STEP,
        "last complete engine step changed",
    )
    return {
        "schema_version": 1,
        "artifact_id": "p4-b0-value-screen-v11-interrupted-capture-manifest",
        "status": "immutable_partial_attempt",
        "authorization": {
            **_reference(AUTHORIZATION_PATH),
            "package_id": "p4-b0-value-screen-run-authorization-v11",
        },
        "output_dir": _relative(OUTPUT_DIR),
        "counts": {
            "raw_capture_files": len(capture_paths),
            "complete_captures": len(captures),
            "empty_placeholders": len(empty),
            "complete_events": sum(row["event_count"] for row in captures),
            "complete_committed_tokens": sum(
                row["committed_tokens"] for row in captures
            ),
            "complete_by_boot": dict(sorted(by_boot.items())),
            "complete_by_action": dict(sorted(by_action.items())),
        },
        "captures": captures,
        "empty_placeholders": [
            {
                **_reference(empty[0]),
                "capture_id": FAILED_CAPTURE_ID,
            }
        ],
        "package_artifacts": {
            "preparation": _reference(OUTPUT_DIR / "preparation.json"),
            "plans": _package_references("plans"),
            "boot_specs": _package_references("boot_specs"),
            "terminal_log": _reference(TERMINAL_LOG_PATH),
        },
        "invariants": {
            "all_nonempty_captures_complete": True,
            "all_captures_unscored": True,
            "adapted_rounds_absent": not (OUTPUT_DIR / "adapted_rounds.jsonl").exists(),
            "score_absent": not (OUTPUT_DIR / "score.json").exists(),
            "preserve_without_overwrite_resume_or_reuse": True,
        },
    }


def build_failure(manifest_path: Path) -> dict[str, Any]:
    """Build the consumed-attempt disposition for the operator interruption."""
    return {
        "schema_version": 1,
        "record_type": "p4_b0_value_screen_execution_interruption",
        "authorization": {
            **_reference(AUTHORIZATION_PATH),
            "package_id": "p4-b0-value-screen-run-authorization-v11",
        },
        "attempt": {
            "physical_gpu_index": 4,
            "physical_gpu_uuid": "GPU-c9d19019-5065-2353-80a9-f1797eb19d51",
            "gpu_model_executed": True,
            "completed_physical_boots": 1,
            "interrupted_boot_id": "p4-b0-b1-p2-k4",
            "complete_captures_emitted": 48,
            "empty_capture_placeholders": 1,
            "interrupted_capture_id": FAILED_CAPTURE_ID,
            "interrupted_regime": "R4",
            "interrupted_content_seed": 0,
            "interrupted_round_index": 1,
            "adapted_rounds_emitted": False,
            "score_emitted": False,
        },
        "diagnostic": {
            "classification": "external_resource_reassignment",
            "reason": "physical_gpu4_reserved_by_another_user",
            "termination": {
                "requested_by_user": True,
                "signal": "SIGINT",
                "parent_exception_type": "KeyboardInterrupt",
                "runtime_fault_observed": False,
            },
            "evidence_provenance": {
                "terminal_transcript": _reference(TERMINAL_LOG_PATH),
                "last_complete_capture_id": LAST_COMPLETE_CAPTURE_ID,
                "last_complete_event_engine_step_index": (LAST_COMPLETE_ENGINE_STEP),
                "interrupted_placeholder_sha256": EMPTY_SHA256,
            },
        },
        "capture_evidence": {
            **_reference(manifest_path),
            "complete_capture_count": 48,
            "empty_placeholder_count": 1,
        },
        "disposition": {
            "v11_consumed": True,
            "requires_fresh_authorization": True,
            "retry_attempted": False,
            "partial_resume_attempted": False,
            "fallback_gpu_used": False,
            "scoring_allowed": False,
            "post_interruption_runner_processes": 0,
            "post_interruption_gpu4_compute_processes": 0,
        },
        "output": {
            "path": _relative(OUTPUT_DIR),
            "adapted_rounds_present": False,
            "score_present": False,
            "preserve_without_overwrite_or_resume": True,
        },
    }


def main() -> int:
    """Write the V11 preservation records exactly once."""
    _require(OUTPUT_DIR.is_dir(), "V11 attempt output is missing")
    _require(
        _sha256(AUTHORIZATION_PATH) == EXPECTED_AUTHORIZATION_SHA256,
        "V11 authorization hash drifted",
    )
    _require(
        _sha256(TERMINAL_LOG_PATH) == EXPECTED_TERMINAL_LOG_SHA256,
        "V11 terminal log hash drifted",
    )
    transcript = TERMINAL_LOG_PATH.read_text(encoding="utf-8")
    _require("KeyboardInterrupt" in transcript, "operator interrupt is absent")
    manifest_path = OUTPUT_DIR / "capture_manifest.json"
    failure_path = OUTPUT_DIR / "failure.json"
    _require(not manifest_path.exists(), "capture manifest already exists")
    _require(not failure_path.exists(), "failure record already exists")
    manifest = build_capture_manifest()
    _write_exclusive(manifest_path, manifest)
    try:
        failure = build_failure(manifest_path)
        _write_exclusive(failure_path, failure)
    except Exception:
        manifest_path.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "status": "pass",
                "complete_capture_count": 48,
                "empty_placeholder_count": 1,
                "v11_consumed": True,
                "scoring_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreservationError as exc:
        print(f"V11 preservation refused: {exc}")
        raise SystemExit(2) from exc
