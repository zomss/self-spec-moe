# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the Phase 97 block-parallel two-lane launcher."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

import run_p4_b0_value_screen as matrix  # noqa: E402
import run_p4_b0_value_screen_v12 as launcher  # noqa: E402


class LaneAssignmentTests(unittest.TestCase):
    """The registered lanes must cover the Latin square without splitting it."""

    def test_lanes_cover_every_block_exactly_once(self) -> None:
        lanes = matrix.lane_assignment()
        owned = [block for lane in lanes for block in lane["block_ids"]]
        self.assertEqual(sorted(owned), sorted(matrix.ACTION_ORDERS))
        self.assertEqual(len(owned), len(set(owned)))

    def test_lane_assignment_closes_to_nine_boots(self) -> None:
        lanes = matrix.lane_assignment()
        boots = sum(
            len(matrix.ACTION_ORDERS[block])
            for lane in lanes
            for block in lane["block_ids"]
        )
        self.assertEqual(boots, 9)
        matrix._validate_lane_assignment(lanes)

    def test_every_block_resolves_to_one_lane(self) -> None:
        for block_id in matrix.ACTION_ORDERS:
            lane = matrix.lane_for_block(block_id)
            self.assertIn(block_id, lane["block_ids"])

    def test_lanes_use_distinct_gpus_and_cpu_sets(self) -> None:
        lanes = matrix.lane_assignment()
        indices = [lane["physical_gpu_index"] for lane in lanes]
        uuids = [lane["physical_gpu_uuid"] for lane in lanes]
        self.assertEqual(len(set(indices)), len(indices))
        self.assertEqual(len(set(uuids)), len(uuids))
        cpu_sets = [
            set(launcher._parse_affinity(lane["cpu_affinity"])) for lane in lanes
        ]
        self.assertFalse(cpu_sets[0] & cpu_sets[1])

    def test_lanes_use_distinct_compile_cache_roots(self) -> None:
        roots = [lane["cache_root"] for lane in matrix.lane_assignment()]
        self.assertEqual(len(set(roots)), len(roots))

    def test_a_split_block_is_rejected(self) -> None:
        lanes = matrix.lane_assignment()
        lanes[0]["block_ids"] = [1]
        with self.assertRaises(matrix.P4RunnerError):
            matrix._validate_lane_assignment(lanes)

    def test_a_duplicated_block_is_rejected(self) -> None:
        lanes = matrix.lane_assignment()
        lanes[1]["block_ids"] = [1, 2]
        with self.assertRaises(matrix.P4RunnerError):
            matrix._validate_lane_assignment(lanes)

    def test_a_shared_gpu_is_rejected(self) -> None:
        lanes = matrix.lane_assignment()
        lanes[1]["physical_gpu_index"] = lanes[0]["physical_gpu_index"]
        with self.assertRaises(matrix.P4RunnerError):
            matrix._validate_lane_assignment(lanes)


class SourceSnapshotTests(unittest.TestCase):
    """Children verify the snapshot, not the live worktree."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.relative = (
            "research/97_composition_runtime/scripts/adapt_p4_b0_same_event.py"
        )

    def _snapshot_with(self, relative: str) -> Path:
        snapshot = self.root / "snap"
        target = snapshot / relative
        target.parent.mkdir(parents=True)
        target.write_bytes((REPO_ROOT / relative).read_bytes())
        return snapshot

    def test_snapshot_redirects_the_hashed_path(self) -> None:
        snapshot = self._snapshot_with(self.relative)
        worktree_reference = matrix._file_reference(self.relative)
        with mock.patch.dict(os.environ, {matrix.SOURCE_SNAPSHOT_ENV: str(snapshot)}):
            resolved = matrix._repository_path(self.relative)
            self.assertEqual(resolved, (snapshot / self.relative).resolve())
            self.assertEqual(
                matrix._file_reference(self.relative)["sha256"],
                worktree_reference["sha256"],
            )

    def test_a_concurrent_worktree_write_does_not_reach_the_child(self) -> None:
        """This is the exact failure that consumed contention probe V2."""
        snapshot = self._snapshot_with(self.relative)
        authorized = matrix._file_reference(self.relative)["sha256"]
        worktree = self.root / "worktree"
        churned = worktree / self.relative
        churned.parent.mkdir(parents=True)
        churned.write_text("# rewritten mid-run by a staging pass\n", encoding="utf-8")
        with mock.patch.object(matrix, "REPO_ROOT", worktree):
            with mock.patch.dict(
                os.environ, {matrix.SOURCE_SNAPSHOT_ENV: str(snapshot)}
            ):
                observed = matrix._file_reference(self.relative)["sha256"]
            self.assertEqual(observed, authorized)
            with mock.patch.dict(os.environ, {matrix.SOURCE_SNAPSHOT_ENV: ""}):
                self.assertNotEqual(
                    matrix._file_reference(self.relative)["sha256"], authorized
                )

    def test_a_drifted_snapshot_copy_changes_the_reported_hash(self) -> None:
        snapshot = self._snapshot_with(self.relative)
        authorized = matrix._file_reference(self.relative)["sha256"]
        (snapshot / self.relative).write_text("drifted\n", encoding="utf-8")
        with mock.patch.dict(os.environ, {matrix.SOURCE_SNAPSHOT_ENV: str(snapshot)}):
            self.assertNotEqual(
                matrix._file_reference(self.relative)["sha256"], authorized
            )

    def test_paths_outside_the_snapshot_fall_back_to_the_repository(self) -> None:
        snapshot = self._snapshot_with(self.relative)
        outside = "research/97_composition_runtime/README.md"
        with mock.patch.dict(os.environ, {matrix.SOURCE_SNAPSHOT_ENV: str(snapshot)}):
            self.assertEqual(
                matrix._repository_path(outside), (REPO_ROOT / outside).resolve()
            )

    def test_a_missing_declared_snapshot_is_rejected(self) -> None:
        with (
            mock.patch.dict(
                os.environ, {matrix.SOURCE_SNAPSHOT_ENV: str(self.root / "absent")}
            ),
            self.assertRaises(matrix.P4RunnerError),
        ):
            matrix._repository_path(self.relative)

    def test_verify_against_snapshot_detects_a_changed_copy(self) -> None:
        output_dir = self.root / "run"
        copy_path = output_dir / launcher.SNAPSHOT_DIRNAME / self.relative
        copy_path.parent.mkdir(parents=True)
        copy_path.write_bytes((REPO_ROOT / self.relative).read_bytes())
        record = {
            "schema_version": 1,
            "record_type": "p4_b0_value_screen_source_snapshot",
            "sources": {
                "adapter": {
                    **matrix._file_reference(self.relative),
                    "snapshot_path": str(copy_path.relative_to(self.root)),
                }
            },
        }
        (output_dir / launcher.SNAPSHOT_RECORD).write_text(
            json.dumps(record), encoding="utf-8"
        )
        with (
            mock.patch.object(launcher, "REPO_ROOT", self.root),
            mock.patch.object(matrix, "REPO_ROOT", self.root),
        ):
            launcher.verify_against_snapshot(output_dir)
            copy_path.write_text("drifted\n", encoding="utf-8")
            with self.assertRaises(launcher.LaneLauncherError):
                launcher.verify_against_snapshot(output_dir)


class ExecutingCodeGuardTests(unittest.TestCase):
    """The executing module is still compared byte-for-byte."""

    def test_no_snapshot_means_no_guard(self) -> None:
        with mock.patch.dict(os.environ, {matrix.SOURCE_SNAPSHOT_ENV: ""}):
            matrix._verify_executing_code_against_snapshot()

    def test_matching_snapshot_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp)
            relative = Path(matrix.__file__).resolve().relative_to(REPO_ROOT)
            target = snapshot / relative
            target.parent.mkdir(parents=True)
            target.write_bytes(Path(matrix.__file__).resolve().read_bytes())
            with mock.patch.dict(
                os.environ, {matrix.SOURCE_SNAPSHOT_ENV: str(snapshot)}
            ):
                matrix._verify_executing_code_against_snapshot()

    def test_drifted_executing_module_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp)
            relative = Path(matrix.__file__).resolve().relative_to(REPO_ROOT)
            target = snapshot / relative
            target.parent.mkdir(parents=True)
            target.write_text("# not the authorized runner\n", encoding="utf-8")
            with (
                mock.patch.dict(
                    os.environ, {matrix.SOURCE_SNAPSHOT_ENV: str(snapshot)}
                ),
                self.assertRaises(matrix.P4RunnerError),
            ):
                matrix._verify_executing_code_against_snapshot()


class BlockRestartUnitTests(unittest.TestCase):
    """A failed block is preserved alone; siblings stay valid."""

    def test_block_record_marks_siblings_valid(self) -> None:
        record = {
            "status": "failed",
            "sibling_blocks_invalidated": False,
            "restart_unit": "block",
            "reuse_allowed": False,
            "scored": False,
        }
        self.assertFalse(record["sibling_blocks_invalidated"])
        self.assertEqual(record["restart_unit"], "block")
        self.assertFalse(record["scored"])

    def test_policy_requires_a_fresh_block_authorization(self) -> None:
        policy = matrix._v12_execution_policy()
        self.assertTrue(policy["block_restart_unit"])
        self.assertTrue(policy["block_restart_requires_fresh_authorization"])
        self.assertFalse(policy["partial_resume_within_block_allowed"])
        self.assertFalse(policy["retry_allowed"])
        self.assertEqual(
            policy["on_any_failure"],
            "preserve_block_and_require_fresh_block_authorization",
        )

    def test_policy_requires_the_source_snapshot(self) -> None:
        self.assertTrue(matrix._v12_execution_policy()["source_snapshot_required"])

    def test_scoring_requires_all_three_blocks(self) -> None:
        self.assertEqual(
            matrix._v12_next_artifact()["requires_complete_block_count"], 3
        )
        self.assertEqual(
            matrix._v12_next_artifact()["requires_complete_capture_count"], 432
        )


class LanePrecedentTests(unittest.TestCase):
    """Phase 96's precedent is referenced read-only, never rescored."""

    def test_precedent_is_not_reused_as_calibration(self) -> None:
        precedent = matrix._v12_lane_precedent()
        self.assertFalse(precedent["reused_as_calibration"])
        self.assertEqual(precedent["phase"], 96)

    def test_precedent_artifact_still_matches_phase_96(self) -> None:
        matrix._validate_v12_lane_precedent(matrix._v12_lane_precedent())

    def test_contention_probe_grants_no_authority(self) -> None:
        disposition = matrix._v12_contention_probe_disposition()
        self.assertFalse(disposition["measured_contention_bound_available"])
        self.assertFalse(disposition["dual_gpu_authority_from_probe"])
        matrix._validate_v12_contention_disposition(disposition)

    def test_consumed_v11_attempt_is_preserved_without_reuse(self) -> None:
        evidence = matrix._v12_consumed_attempt_evidence()
        self.assertEqual(evidence["complete_capture_count"], 48)
        self.assertFalse(evidence["score_emitted"])
        self.assertTrue(evidence["preserve_without_resume_or_reuse"])
        matrix._validate_v12_consumed_attempt(evidence)


if __name__ == "__main__":
    unittest.main()
