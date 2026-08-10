from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
SCRIPTS_DIR = PHASE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_p4_b0_r5cot_r8_diagnosis as diagnosis


class P4R5cotR8DiagnosisTests(unittest.TestCase):
    def test_exact_case_slices(self) -> None:
        authorization = {"run_contract": {"jobs": diagnosis.JOBS}}

        isolated = diagnosis._case_cohorts(authorization, "isolated-r8")
        transition = diagnosis._case_cohorts(authorization, "r5cot-to-r8")

        self.assertEqual([row[2] for row in isolated], ["r8-first"])
        self.assertEqual(
            [row[2] for row in transition], ["r5cot-last", "r8-first"]
        )
        self.assertEqual(
            isolated[0][1], tuple(f"R8-s0-p{index:03d}" for index in range(16))
        )
        self.assertEqual(
            transition[0][1],
            tuple(f"R5cot-s1-p{index:03d}" for index in range(24, 32)),
        )
        self.assertEqual(transition[1][1], isolated[0][1])
        self.assertEqual(transition[0][0]["generation"]["max_output_tokens"], 3072)
        self.assertEqual(isolated[0][0]["generation"]["max_output_tokens"], 2048)

    def test_classification_distinguishes_r8_from_history(self) -> None:
        expected = {"status": "observed_expected_r8_invariant_failure"}
        passed = {"status": "completed_without_invariant_failure"}

        both = diagnosis.classify_results(
            {"isolated-r8": expected, "r5cot-to-r8": expected}
        )
        transition_only = diagnosis.classify_results(
            {"isolated-r8": passed, "r5cot-to-r8": expected}
        )

        self.assertEqual(
            both["conclusion"],
            "r8_variable_width_prefill_evidence_bug_not_history_contamination",
        )
        self.assertEqual(
            transition_only["conclusion"],
            "r5cot_to_r8_transition_history_contamination",
        )

    def test_preserved_v10_boundary(self) -> None:
        output = PHASE_DIR / "data" / "p4" / "run_b0_value_screen_v9"
        manifest = json.loads((output / "capture_manifest.json").read_text())
        failure = json.loads((output / "failure.json").read_text())

        self.assertEqual(manifest["counts"]["complete_captures"], 72)
        self.assertEqual(manifest["counts"]["empty_placeholders"], 1)
        self.assertTrue(failure["disposition"]["v10_consumed"])
        self.assertFalse(failure["disposition"]["scoring_allowed"])

    def test_authorization_is_source_bound_and_create_only(self) -> None:
        authorization = json.loads(diagnosis.AUTHORIZATION_PATH.read_text())

        diagnosis.validate_authorization(
            authorization,
            authorization_path=diagnosis.AUTHORIZATION_PATH.resolve(),
            output_dir=diagnosis.OUTPUT_DIR.resolve(),
            require_output_absent=True,
        )


if __name__ == "__main__":
    unittest.main()
