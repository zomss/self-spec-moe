from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PHASE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PHASE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_p4_b0_r5cot_r8_diagnosis_v2 as diagnosis


class P4R5cotR8DiagnosisV2Tests(unittest.TestCase):
    def test_scheduler_snapshot_reads_cached_request_data(self) -> None:
        output = SimpleNamespace(
            total_num_scheduled_tokens=7,
            num_scheduled_tokens={"cached": 2, "new": 5},
            scheduled_new_reqs=[SimpleNamespace(req_id="new")],
            scheduled_cached_reqs=SimpleNamespace(req_ids=["cached"]),
            finished_req_ids={"finished"},
            koff_runtime=None,
        )

        snapshot = diagnosis.scheduler_snapshot(output)

        self.assertEqual(snapshot["new_request_ids"], ["new"])
        self.assertEqual(snapshot["cached_request_ids"], ["cached"])
        self.assertEqual(snapshot["finished_req_ids"], ["finished"])
        self.assertEqual(snapshot["total_num_scheduled_tokens"], 7)

    def test_v1_failure_is_consumed_and_bound(self) -> None:
        path = (
            PHASE_DIR
            / "data"
            / "p4"
            / "run_b0_r5cot_r8_diagnosis_v1"
            / "failure.json"
        )
        failure = json.loads(path.read_text())

        self.assertEqual(failure["status"], "consumed_inconclusive")
        self.assertTrue(failure["disposition"]["authorization_consumed"])
        self.assertFalse(failure["disposition"]["diagnostic_conclusion_allowed"])

    def test_v2_authorization_is_source_bound_and_create_only(self) -> None:
        diagnosis._configure_base()
        authorization = json.loads(diagnosis.AUTHORIZATION_PATH.read_text())

        diagnosis.base.validate_authorization(
            authorization,
            authorization_path=diagnosis.AUTHORIZATION_PATH.resolve(),
            output_dir=diagnosis.OUTPUT_DIR.resolve(),
            require_output_absent=True,
        )


if __name__ == "__main__":
    unittest.main()
