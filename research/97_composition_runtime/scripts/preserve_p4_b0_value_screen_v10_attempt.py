#!/usr/bin/env python3
"""Preserve the consumed V10 partial value-screen attempt."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
OUTPUT_DIR = PHASE_DIR / "data" / "p4" / "run_b0_value_screen_v9"
AUTHORIZATION_PATH = (
    PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization_v10.json"
)
EXPECTED_AUTHORIZATION_SHA256 = (
    "d772e983718cd8e4ccb16c506da1a7a0ef15908221b59c0ea98b014b5d37bc65"
)
FAILED_CAPTURE_ID = "capture-b1-p2-k4-r8-s0-r1"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


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


def build_capture_manifest() -> dict[str, Any]:
    """Build a hash-bound inventory of every raw capture file."""
    capture_paths = sorted((OUTPUT_DIR / "captures").glob("*/*.json"))
    _require(len(capture_paths) == 73, "raw capture count differs from 73")
    nonempty = [path for path in capture_paths if path.stat().st_size]
    empty = [path for path in capture_paths if not path.stat().st_size]
    _require(len(nonempty) == 72, "complete capture count differs from 72")
    _require(len(empty) == 1, "empty placeholder count differs from one")
    _require(empty[0].stem == FAILED_CAPTURE_ID, "failed placeholder changed")

    captures = [_capture_record(path) for path in nonempty]
    by_boot = Counter(row["boot_id"] for row in captures)
    by_action = Counter(row["action_id"] for row in captures)
    _require(
        by_boot
        == {
            "p4-b0-b1-p1-off": 48,
            "p4-b0-b1-p2-k4": 24,
        },
        "complete captures do not stop at the first K4/R8 cell",
    )
    _require(
        by_action == {"off": 48, "target-matching-k4": 24},
        "action capture counts drifted",
    )
    _require(
        captures[-1]["capture_id"] == "capture-b1-p2-k4-r5cot-s1-r4",
        "last complete capture changed",
    )
    _require(
        captures[-1]["last_engine_step_index"] == 27699,
        "last complete engine step changed",
    )
    return {
        "schema_version": 1,
        "artifact_id": "p4-b0-value-screen-v10-consumed-capture-manifest",
        "status": "immutable_partial_attempt",
        "authorization": {
            **_reference(AUTHORIZATION_PATH),
            "package_id": "p4-b0-value-screen-run-authorization-v10",
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
        },
        "invariants": {
            "all_nonempty_captures_complete": True,
            "all_captures_unscored": True,
            "adapted_rounds_absent": not (
                OUTPUT_DIR / "adapted_rounds.jsonl"
            ).exists(),
            "score_absent": not (OUTPUT_DIR / "score.json").exists(),
            "preserve_without_overwrite_resume_or_reuse": True,
        },
    }


def build_failure(manifest_path: Path) -> dict[str, Any]:
    """Build the consumed-attempt disposition and observed exception record."""
    return {
        "schema_version": 1,
        "record_type": "p4_b0_value_screen_execution_failure",
        "authorization": {
            **_reference(AUTHORIZATION_PATH),
            "package_id": "p4-b0-value-screen-run-authorization-v10",
        },
        "attempt": {
            "physical_gpu_index": 4,
            "gpu_model_executed": True,
            "completed_physical_boots": 1,
            "failed_boot_id": "p4-b0-b1-p2-k4",
            "complete_captures_emitted": 72,
            "empty_capture_placeholders": 1,
            "failed_capture_id": FAILED_CAPTURE_ID,
            "failed_regime": "R8",
            "failed_content_seed": 0,
            "failed_round_index": 1,
            "adapted_rounds_emitted": False,
            "score_emitted": False,
        },
        "diagnostic": {
            "scope": "k4_variable_width_pure_prefill_step0_evidence",
            "last_stage": "first_k4_r8_pure_prefill_draft_evidence_validation",
            "primary_exception": {
                "type": "KOffRuntimeError",
                "message": (
                    "target-matching-k4 non-decode dispatch requires a positive "
                    "draft step-0 query width and pure-prefill cohort arming"
                ),
            },
            "secondary_shutdown_exception": {
                "type": "KOffRuntimeError",
                "message": (
                    "P4 capture cohort "
                    "'capture-b1-p2-k4-r8-s0-r1:cohort:c727fa542ff6239f' "
                    "closed before exact work completion"
                ),
            },
            "failed_step": {
                "engine_step_index": 27700,
                "next_action_id": "target-matching-k4",
                "capture_cohort_arm": True,
                "decode_req_ids": [],
                "pure_decode": False,
                "total_num_scheduled_tokens": 2100,
                "finished_req_ids": ["R5cot-s1-p030", "R5cot-s1-p031"],
                "new_request_ids": [f"R8-s0-p{index:03d}" for index in range(16)],
                "preemptions": 0,
                "recomputed_tokens": 0,
                "shared_target_kv_block_capacity": 24527,
            },
            "evidence_provenance": {
                "terminal_exception_observed_during_consumed_run": True,
                "terminal_transcript_file_retained": False,
                "artifact_boundary_independently_hash_bound": True,
                "last_complete_event_engine_step_index": 27699,
                "failed_placeholder_sha256": EMPTY_SHA256,
            },
        },
        "capture_evidence": {
            **_reference(manifest_path),
            "complete_capture_count": 72,
            "empty_placeholder_count": 1,
        },
        "disposition": {
            "v10_consumed": True,
            "requires_fresh_authorization": True,
            "retry_attempted": False,
            "partial_resume_attempted": False,
            "fallback_gpu_used": False,
            "scoring_allowed": False,
            "post_failure_runner_processes": 0,
            "post_failure_gpu4_compute_processes": 0,
        },
        "output": {
            "path": _relative(OUTPUT_DIR),
            "adapted_rounds_present": False,
            "score_present": False,
            "preserve_without_overwrite_or_resume": True,
        },
    }


def main() -> int:
    """Write the two preservation records exactly once."""
    _require(OUTPUT_DIR.is_dir(), "V10 attempt output is missing")
    _require(
        _sha256(AUTHORIZATION_PATH) == EXPECTED_AUTHORIZATION_SHA256,
        "V10 authorization drifted",
    )
    manifest_path = OUTPUT_DIR / "capture_manifest.json"
    failure_path = OUTPUT_DIR / "failure.json"
    _require(not manifest_path.exists(), "capture manifest already exists")
    _require(not failure_path.exists(), "failure record already exists")
    manifest = build_capture_manifest()
    _write_exclusive(manifest_path, manifest)
    _write_exclusive(failure_path, build_failure(manifest_path))
    print(
        json.dumps(
            {
                "status": "preserved",
                "capture_manifest": _reference(manifest_path),
                "failure": _reference(failure_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
