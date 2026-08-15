# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the w98-d2 trace -> accounting-row bridge.

The bridge is where an engine-side field-name mistake would turn into a
silently wrong acceptance number, so these pin the translation itself: the
clip derivation, the per-request u (not the batch summary), and refusal of a
stream that was not booted under the D2 scope.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from w98_accounting import W98AccountingError, close_interval  # noqa: E402
from w98d2_stream import rows_from_records, tau_profile  # noqa: E402


def _record(rows, armed=True):
    return {
        "counters": {"D_armed": len(rows) if armed else 0, "H_target_steps": len(rows)},
        "acceptance_rows": rows,
    }


def _row(rid="r0", accepted=3, committed=None, suffix=10, drafted=8):
    return {
        "request_id": rid,
        "accepted_draft_tokens": accepted,
        "drafted_tokens": drafted,
        "generated_suffix_len": suffix,
        "committed_tokens": accepted + 1 if committed is None else committed,
    }


class TranslationTests(unittest.TestCase):
    def test_unclipped_row_translates_field_for_field(self) -> None:
        (row,) = rows_from_records([_record([_row(accepted=5, suffix=42)])])
        self.assertEqual(row.request_id, "r0")
        self.assertEqual(row.suffix_len, 42)
        self.assertTrue(row.armed)
        self.assertEqual(row.proposed, 8)
        self.assertEqual(row.accepted, 5)
        self.assertEqual(row.clipped, 0)

    def test_clip_is_derived_from_committed(self) -> None:
        """clipped = accepted + 1 - committed, inverting row_emitted."""
        (row,) = rows_from_records([_record([_row(accepted=4, committed=3)])])
        self.assertEqual(row.clipped, 2)

    def test_negative_clip_is_refused(self) -> None:
        with self.assertRaises(W98AccountingError):
            rows_from_records([_record([_row(accepted=1, committed=9)])])

    def test_each_request_keeps_its_own_suffix(self) -> None:
        """The batch min/max cannot be inverted, so u must come per request."""
        rows = rows_from_records(
            [_record([_row("a", suffix=100), _row("b", suffix=900)])]
        )
        self.assertEqual([r.suffix_len for r in rows], [100, 900])

    def test_unarmed_step_carries_no_draft(self) -> None:
        rows = rows_from_records([_record([_row(accepted=0)], armed=False)])
        self.assertFalse(rows[0].armed)
        self.assertEqual((rows[0].proposed, rows[0].accepted), (0, 0))

    def test_a_non_d2_stream_is_refused(self) -> None:
        """A trace without acceptance_rows was booted under another scope."""
        with self.assertRaises(W98AccountingError):
            rows_from_records([{"counters": {"D_armed": 1, "H_target_steps": 1}}])


class EstimationTests(unittest.TestCase):
    def test_closure_holds_over_translated_rows(self) -> None:
        rows = rows_from_records(
            [_record([_row("a", accepted=8), _row("b", accepted=2)])]
        )
        result = close_interval(rows)
        self.assertEqual(
            result.e_emitted + result.c_clipped,
            result.a_accepted + result.h_steps,
        )

    def test_full_acceptance_gives_tau_kmax_plus_one(self) -> None:
        """8 accepted draft tokens plus the target's own token is tau = 9."""
        records = [_record([_row(f"r{i}", accepted=8, suffix=10)]) for i in range(40)]
        profile = tau_profile(rows_from_records(records), kmax=8, resamples=50)
        self.assertAlmostEqual(profile["buckets"]["0"]["tau"], 9.0, places=6)

    def test_rows_bin_by_their_own_suffix(self) -> None:
        records = [
            _record([_row(f"a{i}", accepted=8, suffix=10) for i in range(4)])
            for _ in range(10)
        ] + [
            _record([_row(f"b{i}", accepted=0, suffix=5000) for i in range(4)])
            for _ in range(10)
        ]
        profile = tau_profile(rows_from_records(records), kmax=8, resamples=50)
        self.assertAlmostEqual(profile["buckets"]["0"]["tau"], 9.0, places=6)
        self.assertAlmostEqual(profile["buckets"]["3"]["tau"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
