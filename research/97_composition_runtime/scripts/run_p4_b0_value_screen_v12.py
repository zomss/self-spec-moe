# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-parallel two-lane launcher for the P4 B0 matched value screen.

The screen's nine boots form a Latin square of three blocks by three actions.
This launcher runs one COMPLETE block per lane at a time and never splits a
block across GPUs, so a lane-wide scale factor cancels in the scored ratio and
any residual stays a block effect that the paired complete-boot-block
bootstrap already resamples.

Two fail-closed properties distinguish it from the serial parent in
``run_p4_b0_value_screen.py``:

- registered sources are snapshotted and hashed once, before any child, and
  every child verifies against that immutable snapshot rather than the live
  worktree; and
- the capture block is the restart unit, so an interrupted block is preserved
  alone and its siblings stay valid.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import run_p4_b0_value_screen as matrix

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
LAUNCHER_ARTIFACT_ID = "p4-b0-value-screen-v12-lane-launcher"
SNAPSHOT_DIRNAME = "source_snapshot"
SNAPSHOT_RECORD = "source_snapshot.json"
LANE_PACKAGE_IDS = frozenset({matrix.V12_PACKAGE_ID, matrix.V13_PACKAGE_ID})


class LaneLauncherError(RuntimeError):
    """Raised when the lane launcher cannot prove its authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LaneLauncherError(message)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict), f"{path} is not a JSON object")
    return payload


def _digest(path: Path) -> str:
    """Hash a file directly, independent of where it sits on disk."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record_path(path: Path) -> str:
    """Record a snapshot copy repository-relative when it is inside the repo."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def snapshot_sources(output_dir: Path, authorization: Mapping[str, Any]) -> Path:
    """Copy every registered source into the run and verify it once.

    The V2 contention probe was consumed by a concurrent worktree write that
    changed a registered file between two children. Children therefore verify
    against this snapshot, which no later worktree write can reach.

    Args:
        output_dir: The create-only run directory.
        authorization: The resolved V12 authorization.

    Returns:
        The snapshot directory path.

    Raises:
        LaneLauncherError: If any registered source has already drifted.
    """
    output_dir = output_dir.resolve()
    references = authorization["source_artifacts"]
    snapshot_dir = output_dir / SNAPSHOT_DIRNAME
    _require(not snapshot_dir.exists(), "source snapshot already exists")
    _require(
        not os.environ.get(matrix.SOURCE_SNAPSHOT_ENV),
        "a snapshot cannot be built while another snapshot is active",
    )
    snapshot_dir.mkdir(parents=True)
    rows = {}
    for role, reference in sorted(references.items()):
        relative = reference["path"]
        source = REPO_ROOT / relative
        _require(source.is_file(), f"registered source is missing: {relative}")
        observed = matrix._file_reference(relative)
        _require(
            observed == reference,
            f"registered source drifted before snapshot: {role}",
        )
        target = snapshot_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        _require(
            _digest(target) == reference["sha256"],
            f"snapshot copy does not match the registered hash: {role}",
        )
        rows[role] = {**reference, "snapshot_path": _record_path(target)}
    _write_json(
        output_dir / SNAPSHOT_RECORD,
        {
            "schema_version": 1,
            "record_type": "p4_b0_value_screen_source_snapshot",
            "artifact_id": LAUNCHER_ARTIFACT_ID,
            "package_id": matrix.V12_PACKAGE_ID,
            "verified_against_worktree": True,
            "source_count": len(rows),
            "sources": rows,
        },
    )
    return snapshot_dir


def verify_against_snapshot(output_dir: Path) -> dict[str, Any]:
    """Re-verify registered sources against the run's immutable snapshot.

    Args:
        output_dir: The run directory holding the snapshot.

    Returns:
        The snapshot record.

    Raises:
        LaneLauncherError: If a snapshot entry no longer matches its copy.
    """
    record = _load_json(output_dir.resolve() / SNAPSHOT_RECORD)
    for role, reference in record["sources"].items():
        target = REPO_ROOT / reference["snapshot_path"]
        _require(target.is_file(), f"snapshot copy is missing: {role}")
        _require(
            _digest(target) == reference["sha256"],
            f"snapshot copy drifted after preparation: {role}",
        )
    return record


def _lane_child_environment(
    lane: Mapping[str, Any],
    spec_environment: Mapping[str, Any],
    snapshot_dir: Path,
) -> dict[str, str]:
    """Build a lane-pinned child environment bound to the source snapshot."""
    child_env = matrix._boot_child_environment(spec_environment)
    _require(
        child_env.get(matrix.DEVICE_PIN_ENV) == str(lane["physical_gpu_index"]),
        "boot spec does not pin its lane GPU",
    )
    _require(
        child_env.get(matrix.CACHE_ROOT_ENV) == lane["cache_root"],
        "boot spec does not pin its lane compile-cache root",
    )
    child_env[matrix.SOURCE_SNAPSHOT_ENV] = str(snapshot_dir.resolve())
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    return child_env


def _parse_affinity(affinity: str) -> list[int]:
    cpus: list[int] = []
    for group in affinity.split(","):
        if "-" in group:
            first, last = group.split("-", maxsplit=1)
            cpus.extend(range(int(first), int(last) + 1))
        else:
            cpus.append(int(group))
    _require(bool(cpus), "lane CPU affinity is empty")
    return cpus


def run_lane(
    lane: Mapping[str, Any],
    authorization_path: Path,
    output_dir: Path,
    specs_by_block: Mapping[int, Sequence[Mapping[str, Any]]],
    snapshot_dir: Path,
) -> dict[str, Any]:
    """Run every block owned by one lane, sequentially, on one GPU.

    Args:
        lane: The registered lane record.
        authorization_path: The reviewed V12 authorization path.
        output_dir: The create-only run directory.
        specs_by_block: Boot specs grouped by capture block.
        snapshot_dir: The immutable source snapshot every child verifies.

    Returns:
        A lane record naming each completed and each failed block.
    """
    script_path = Path(matrix.__file__).resolve()
    affinity = _parse_affinity(lane["cpu_affinity"])
    blocks: list[dict[str, Any]] = []
    for block_id in lane["block_ids"]:
        block_dir = output_dir / "blocks" / f"block{block_id}"
        block_dir.mkdir(parents=True, exist_ok=True)
        boots: list[dict[str, Any]] = []
        failure: str | None = None
        for spec in specs_by_block[block_id]:
            spec_path = output_dir / "boot_specs" / f"{spec['boot_id']}.json"
            child_env = _lane_child_environment(lane, spec["environment"], snapshot_dir)
            log_path = block_dir / f"{spec['boot_id']}.log"
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(script_path),
                        "--authorization",
                        str(authorization_path.resolve()),
                        "--output-dir",
                        str(output_dir.resolve()),
                        "--child-spec",
                        str(spec_path.resolve()),
                    ],
                    cwd=REPO_ROOT,
                    env=child_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    preexec_fn=lambda: os.sched_setaffinity(0, affinity),
                )
            captures = sorted(Path(spec["capture_dir"]).glob("*.json"))
            complete = [path for path in captures if path.stat().st_size > 0]
            boots.append(
                {
                    "boot_id": spec["boot_id"],
                    "action_id": spec["action_id"],
                    "action_position": spec["action_position"],
                    "returncode": completed.returncode,
                    "complete_capture_count": len(complete),
                    "empty_placeholder_count": len(captures) - len(complete),
                    "log_path": str(log_path.relative_to(REPO_ROOT)),
                }
            )
            if completed.returncode != 0 or len(complete) != 48:
                failure = spec["boot_id"]
                break
        block_record = {
            "schema_version": 1,
            "record_type": "p4_b0_value_screen_block_result",
            "block_id": block_id,
            "lane_id": lane["lane_id"],
            "physical_gpu_index": lane["physical_gpu_index"],
            "physical_gpu_uuid": lane["physical_gpu_uuid"],
            "cpu_affinity": lane["cpu_affinity"],
            "expected_boot_count": len(specs_by_block[block_id]),
            "completed_boot_count": sum(
                1 for boot in boots if boot["complete_capture_count"] == 48
            ),
            "complete_capture_count": sum(
                boot["complete_capture_count"] for boot in boots
            ),
            "boots": boots,
            "status": "complete" if failure is None else "failed",
            "failed_boot_id": failure,
            "scored": False,
            "reuse_allowed": False,
            "restart_unit": "block",
            "sibling_blocks_invalidated": False,
        }
        _write_json(block_dir / "block_result.json", block_record)
        blocks.append(block_record)
        if failure is not None:
            break
    return {
        "lane_id": lane["lane_id"],
        "physical_gpu_index": lane["physical_gpu_index"],
        "physical_gpu_uuid": lane["physical_gpu_uuid"],
        "blocks": blocks,
        "status": (
            "complete"
            if len(blocks) == len(lane["block_ids"])
            and all(block["status"] == "complete" for block in blocks)
            else "failed"
        ),
    }


def preflight_worktree_quiescent(
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Refuse to start while a registered source is uncommitted or dirty.

    The V2 contention probe died because a staging pass rewrote a registered
    file between two children. Making worktree quiescence an explicit
    precondition turns that into a refusal before any GPU work, instead of a
    surprise four boots in.

    Args:
        authorization: The resolved V12 authorization.

    Returns:
        The recorded git head and the checked source count.

    Raises:
        LaneLauncherError: If git is unusable or a registered source is dirty.
    """
    registered = {
        reference["path"] for reference in authorization["source_artifacts"].values()
    }
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", *sorted(registered)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        raise LaneLauncherError(f"worktree preflight failed: {exc}") from exc
    dirty = [row.strip() for row in status if row.strip()]
    _require(
        not dirty,
        f"registered sources are uncommitted or modified: {dirty}",
    )
    return {
        "git_head": head,
        "registered_source_count": len(registered),
        "dirty_registered_sources": 0,
    }


def _preflight_lanes(lanes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        matrix._preflight_gpu_identity_and_idle(
            lane["physical_gpu_index"], lane["physical_gpu_uuid"]
        )
        for lane in lanes
    ]


def execute_run(
    authorization_path: Path,
    authorization: Mapping[str, Any],
    output_dir: Path,
) -> int:
    """Prepare once, then run both lanes concurrently and score if complete.

    Args:
        authorization_path: The reviewed V12 authorization path.
        authorization: The resolved V12 authorization.
        output_dir: The registered create-only output directory.

    Returns:
        Zero when all three blocks complete and the frozen score is emitted.
    """
    output_dir = output_dir.resolve()
    authorization_path = authorization_path.resolve()
    _require(
        output_dir == matrix._reviewed_output_path(authorization_path),
        "execution output differs from the reviewed create-new path",
    )
    matrix.validate_execution_authority(authorization)
    base_child_env = matrix._boot_child_environment({})
    matrix._preflight_native_sampler(base_child_env)
    matrix._preflight_inprocess_engine_core(base_child_env)
    lanes = authorization["execution_policy"]["lane_assignment"]
    matrix._validate_lane_assignment(lanes)
    worktree = preflight_worktree_quiescent(authorization)
    identities = _preflight_lanes(lanes)

    specs = matrix.prepare_run(authorization, output_dir)
    snapshot_dir = snapshot_sources(output_dir, authorization)
    verify_against_snapshot(output_dir)
    specs_by_block: dict[int, list[dict[str, Any]]] = {}
    for spec in specs:
        specs_by_block.setdefault(spec["boot_block_id"], []).append(spec)
    for block_id, block_specs in specs_by_block.items():
        _require(
            len({spec["lane"]["lane_id"] for spec in block_specs}) == 1,
            f"block {block_id} is split across lanes",
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        lane_records = list(
            pool.map(
                lambda lane: run_lane(
                    lane, authorization_path, output_dir, specs_by_block, snapshot_dir
                ),
                lanes,
            )
        )

    completed_blocks = [
        block
        for record in lane_records
        for block in record["blocks"]
        if block["status"] == "complete"
    ]
    complete_captures = sum(
        block["complete_capture_count"]
        for record in lane_records
        for block in record["blocks"]
    )
    scored = len(completed_blocks) == len(matrix.ACTION_ORDERS)
    _write_json(
        output_dir / "lane_result.json",
        {
            "schema_version": 1,
            "record_type": "p4_b0_value_screen_lane_result",
            "artifact_id": LAUNCHER_ARTIFACT_ID,
            "package_id": matrix.V12_PACKAGE_ID,
            "authorization_consumed": True,
            "worktree_preflight": worktree,
            "gpu_identities": identities,
            "lanes": lane_records,
            "complete_block_count": len(completed_blocks),
            "complete_capture_count": complete_captures,
            "status": "complete" if scored else "incomplete",
            "scored": scored,
            "restart_unit": "block",
            "complete_blocks_reusable_under_block_authorization": True,
            "disposition": (
                "all_blocks_complete"
                if scored
                else "preserve_and_require_fresh_block_authorization"
            ),
        },
    )
    if not scored:
        raise LaneLauncherError(
            "value screen did not complete every block; "
            f"{len(completed_blocks)}/{len(matrix.ACTION_ORDERS)} blocks complete"
        )
    matrix._adapt_and_score(output_dir)
    return 0


def parse_args() -> argparse.Namespace:
    """Parse the launcher interface and its CPU-only preparation option."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Prepare without GPU, or launch the two authorized lanes."""
    args = parse_args()
    authorization_path = args.authorization.resolve()
    output_dir = args.output_dir.resolve()
    package = _load_json(authorization_path)
    _require(
        package.get("package_id") in LANE_PACKAGE_IDS,
        "the lane launcher accepts only a registered block-parallel authorization",
    )
    authorization = matrix.resolve_authorization_package(
        package, require_output_absent=True
    )
    if args.prepare_only:
        matrix.validate_preparation_contract(authorization)
        matrix.prepare_run(authorization, output_dir)
        snapshot_sources(output_dir, authorization)
        print(
            json.dumps(
                {
                    "status": "pass",
                    "mode": "prepare_only",
                    "gpu_executed": False,
                    "gpu_authority_granted": False,
                    "output_dir": str(output_dir),
                },
                sort_keys=True,
            )
        )
        return 0
    return execute_run(authorization_path, authorization, output_dir)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LaneLauncherError, matrix.P4RunnerError) as exc:
        print(f"P4 lane launcher refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
