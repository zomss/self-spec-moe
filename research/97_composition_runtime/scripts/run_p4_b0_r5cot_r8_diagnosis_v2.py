#!/usr/bin/env python3
"""Run V2 of the R5cot-to-R8 diagnosis with a corrected observer."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import run_p4_b0_r5cot_r8_diagnosis as base

AUTHORIZATION_PATH = (
    base.PHASE_DIR
    / "data"
    / "p4"
    / "p4_b0_r5cot_r8_diagnosis_authorization_v2.json"
)
OUTPUT_DIR = base.PHASE_DIR / "data" / "p4" / "run_b0_r5cot_r8_diagnosis_v2"
EXPECTED_PACKAGE_ID = "p4-b0-r5cot-r8-diagnosis-authorization-v2"
REQUIRED_SOURCE_PATHS = {
    **base.REQUIRED_SOURCE_PATHS,
    "diagnosis_runner_base": base.REQUIRED_SOURCE_PATHS["diagnosis_runner"],
    "diagnosis_runner": (
        "research/97_composition_runtime/scripts/"
        "run_p4_b0_r5cot_r8_diagnosis_v2.py"
    ),
    "diagnosis_tests_v1": base.REQUIRED_SOURCE_PATHS["diagnosis_tests"],
    "diagnosis_tests": (
        "research/97_composition_runtime/tests/"
        "test_p4_b0_r5cot_r8_diagnosis_v2.py"
    ),
    "diagnosis_v1_failure": (
        "research/97_composition_runtime/data/p4/"
        "run_b0_r5cot_r8_diagnosis_v1/failure.json"
    ),
}


def scheduler_snapshot(output: Any) -> dict[str, Any]:
    """Summarize SchedulerOutput without iterating CachedRequestData."""
    metadata = getattr(output, "koff_runtime", None)
    cached = getattr(output, "scheduled_cached_reqs", None)
    cached_req_ids = () if cached is None else getattr(cached, "req_ids", ())
    return {
        "total_num_scheduled_tokens": getattr(
            output, "total_num_scheduled_tokens", None
        ),
        "num_scheduled_tokens": dict(
            getattr(output, "num_scheduled_tokens", {})
        ),
        "new_request_ids": [
            base._request_id(row)
            for row in getattr(output, "scheduled_new_reqs", ())
        ],
        "cached_request_ids": list(cached_req_ids),
        "finished_req_ids": sorted(getattr(output, "finished_req_ids", ())),
        "koff_runtime": (
            dataclasses.asdict(metadata) if dataclasses.is_dataclass(metadata) else None
        ),
    }


def _configure_base() -> None:
    base.AUTHORIZATION_PATH = AUTHORIZATION_PATH
    base.OUTPUT_DIR = OUTPUT_DIR
    base.EXPECTED_PACKAGE_ID = EXPECTED_PACKAGE_ID
    base.REQUIRED_SOURCE_PATHS = REQUIRED_SOURCE_PATHS
    base._scheduler_snapshot = scheduler_snapshot
    base.__file__ = str(Path(__file__).resolve())


def main() -> int:
    _configure_base()
    return base.main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except base.P4TransitionDiagnosisError as exc:
        print(f"P4 transition diagnosis V2 refused: {exc}", file=base.sys.stderr)
        raise SystemExit(2) from exc
