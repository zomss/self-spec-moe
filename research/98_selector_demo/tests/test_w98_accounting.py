# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic proofs for the G98-0 accounting and acceptance harness."""

import random

import pytest
from w98_accounting import (
    StepRow,
    W98AccountingError,
    bin_rows,
    bootstrap_paired_delta,
    bootstrap_tau,
    bucket_index,
    close_interval,
    tau_at_depth,
)

KMAX = 4
EDGES = [256, 1024, 3072]


def armed_row(rid, u, accepted, clipped=0, proposed=KMAX):
    return StepRow(rid, u, True, proposed, accepted, clipped)


def off_row(rid, u):
    return StepRow(rid, u, False, 0, 0)


def chain_stream(rid, p, n_steps, seed, u0=0):
    """Chain-censored stream with iid per-position accept probability p."""
    rng = random.Random(seed)
    rows, u = [], u0
    for _ in range(n_steps):
        accepted = 0
        for _ in range(KMAX):
            if rng.random() < p:
                accepted += 1
            else:
                break
        rows.append(armed_row(rid, u, accepted))
        u += accepted + 1
    return rows


class TestClosure:
    def test_exact_closure_identity(self):
        rows = [
            armed_row("r0", 0, 3),
            armed_row("r0", 4, 0),
            off_row("r0", 5),
            armed_row("r1", 0, 4, clipped=2),
        ]
        result = close_interval(rows)
        assert result.h_steps == 4
        assert result.d_armed == 3
        assert result.a_accepted == 7
        assert result.c_clipped == 2
        assert result.e_emitted + result.c_clipped == (
            result.a_accepted + result.h_steps
        )
        assert result.tau_eff == pytest.approx(9 / 4)

    def test_frontend_mismatch_rejected(self):
        rows = [armed_row("r0", 0, 2)]
        with pytest.raises(W98AccountingError, match="frontend"):
            close_interval(rows, emitted_total=99)

    def test_frontend_agreement_accepted(self):
        rows = [armed_row("r0", 0, 2), off_row("r0", 3)]
        assert close_interval(rows, emitted_total=4).e_emitted == 4

    @pytest.mark.parametrize(
        "row",
        [
            StepRow("r", 0, True, 0, 0),
            StepRow("r", 0, True, KMAX, KMAX + 1),
            StepRow("r", 0, False, 1, 0),
            StepRow("r", 0, False, 0, 1),
            StepRow("r", 0, True, KMAX, 1, clipped=3),
            StepRow("r", -1, True, KMAX, 1),
        ],
    )
    def test_invalid_rows_rejected(self, row):
        with pytest.raises(W98AccountingError):
            close_interval([row])


class TestBinning:
    def test_bucket_edges_are_right_open(self):
        assert bucket_index(EDGES, 0) == 0
        assert bucket_index(EDGES, 255) == 0
        assert bucket_index(EDGES, 256) == 1
        assert bucket_index(EDGES, 1023) == 1
        assert bucket_index(EDGES, 1024) == 2
        assert bucket_index(EDGES, 3072) == 3

    def test_rows_land_in_expected_buckets(self):
        rows = [
            armed_row("r0", 10, 2),
            armed_row("r0", 300, 1),
            armed_row("r0", 5000, 4),
        ]
        buckets = bin_rows(rows, KMAX, EDGES)
        assert sorted(buckets) == [0, 1, 3]
        assert all(b.armed_steps == 1 for b in buckets.values())

    def test_mixed_k_stream_rejected(self):
        rows = [armed_row("r0", 0, 1), armed_row("r0", 2, 1, proposed=2)]
        with pytest.raises(W98AccountingError, match="mixed-K"):
            bin_rows(rows, KMAX, EDGES)

    def test_clipped_rows_excluded_from_counters(self):
        rows = [armed_row("r0", 0, 4, clipped=1), armed_row("r0", 4, 2)]
        (bucket,) = bin_rows(rows, KMAX, EDGES).values()
        assert bucket.armed_steps == 1
        assert bucket.clipped_excluded == 1
        assert bucket.pos_accepted == [1, 1, 0, 0]

    def test_unarmed_rows_counted_outside_counters(self):
        rows = [off_row("r0", 0), armed_row("r0", 1, 3)]
        (bucket,) = bin_rows(rows, KMAX, EDGES).values()
        assert bucket.unarmed_steps == 1
        assert bucket.armed_steps == 1

    def test_unsorted_edges_rejected(self):
        with pytest.raises(W98AccountingError, match="ascending"):
            bin_rows([armed_row("r0", 0, 1)], KMAX, [1024, 256])


class TestTau:
    def test_full_acceptance_gives_k_plus_one(self):
        rows = [armed_row("r0", i * 5, KMAX) for i in range(10)]
        (bucket,) = bin_rows(rows, KMAX, EDGES).values()
        for k in range(1, KMAX + 1):
            assert tau_at_depth(bucket, k) == pytest.approx(k + 1)

    def test_zero_acceptance_gives_one(self):
        rows = [armed_row("r0", i, 0) for i in range(10)]
        (bucket,) = bin_rows(rows, KMAX, EDGES).values()
        assert tau_at_depth(bucket, KMAX) == pytest.approx(1.0)

    def test_known_chain_probability_recovered(self):
        p = 0.7
        rows = []
        for rid in range(64):
            rows.extend(chain_stream(f"r{rid}", p, 400, seed=rid))
        merged = bin_rows(rows, KMAX, [])
        bucket = merged[0]
        for k in range(1, KMAX + 1):
            expected = 1.0 + sum(p ** (i + 1) for i in range(k))
            assert tau_at_depth(bucket, k) == pytest.approx(expected, rel=0.02)

    def test_every_depth_from_one_stream_is_consistent(self):
        rows = chain_stream("r0", 0.5, 500, seed=7)
        (bucket,) = bin_rows(rows, KMAX, []).values()
        taus = [tau_at_depth(bucket, k) for k in range(1, KMAX + 1)]
        assert taus == sorted(taus)
        assert taus[-1] <= KMAX + 1

    def test_depth_outside_kmax_rejected(self):
        rows = [armed_row("r0", 0, 1)]
        (bucket,) = bin_rows(rows, KMAX, EDGES).values()
        with pytest.raises(W98AccountingError, match="depth"):
            tau_at_depth(bucket, KMAX + 1)


class TestBootstrap:
    def test_interval_contains_truth(self):
        p = 0.6
        rows = []
        for rid in range(32):
            rows.extend(chain_stream(f"r{rid}", p, 200, seed=100 + rid))
        result = bootstrap_tau(rows, KMAX, [], k=KMAX, n_boot=300, seed=1)
        truth = 1.0 + sum(p ** (i + 1) for i in range(KMAX))
        interval = result[0]
        assert interval.lo <= truth <= interval.hi
        assert interval.lo <= interval.point <= interval.hi

    def test_paired_delta_of_identical_streams_is_zero(self):
        rows = []
        for rid in range(8):
            rows.extend(chain_stream(f"r{rid}", 0.5, 50, seed=200 + rid))
        result = bootstrap_paired_delta(
            rows, list(rows), KMAX, [], k=KMAX, n_boot=100, seed=2
        )
        interval = result[0]
        assert interval.point == pytest.approx(0.0)
        assert interval.lo == pytest.approx(0.0)
        assert interval.hi == pytest.approx(0.0)

    def test_paired_delta_requires_matching_requests(self):
        rows_a = chain_stream("ra", 0.5, 10, seed=3)
        rows_b = chain_stream("rb", 0.5, 10, seed=4)
        with pytest.raises(W98AccountingError, match="identical request"):
            bootstrap_paired_delta(rows_a, rows_b, KMAX, [], k=KMAX)

    def test_paired_delta_detects_a_real_gap(self):
        rows_a, rows_b = [], []
        for rid in range(24):
            rows_a.extend(chain_stream(f"r{rid}", 0.8, 150, seed=300 + rid))
            rows_b.extend(chain_stream(f"r{rid}", 0.4, 150, seed=600 + rid))
        result = bootstrap_paired_delta(
            rows_a, rows_b, KMAX, [], k=KMAX, n_boot=300, seed=5
        )
        assert result[0].lo > 0.0

    def test_empty_stream_rejected(self):
        with pytest.raises(W98AccountingError, match="empty"):
            bootstrap_tau([], KMAX, [], k=1)
