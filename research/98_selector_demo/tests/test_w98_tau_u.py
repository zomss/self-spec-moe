# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Proofs for the u-resolved acceptance estimator."""

import math

import pytest
from w98_tau_u import (
    TauCurveError,
    bucket_bounds,
    pooled,
    scalar_from_profile,
    tau_effective,
)

EDGES = (256, 1024, 3072, 8192)
# The measured curves, from results_g98_longu.md.
SKIP8 = {"0": 3.642, "1": 3.701, "2": 3.986, "3": 4.106}
W512 = {"0": 4.553, "1": 4.397, "2": 4.228, "3": 4.266}


def test_bucket_bounds_are_half_open_and_cover_everything():
    bounds = bucket_bounds(EDGES)
    assert bounds[0] == (0.0, 256.0)
    assert bounds[-1] == (8192.0, math.inf)
    assert len(bounds) == len(EDGES) + 1


def test_constant_curve_returns_that_constant():
    flat = {str(i): 4.0 for i in range(5)}
    assert tau_effective(flat, EDGES, 5000) == pytest.approx(4.0)


def test_short_generation_uses_only_the_first_bucket():
    assert tau_effective(SKIP8, EDGES, 200) == pytest.approx(3.642)


def test_harmonic_mean_is_below_the_arithmetic_mean():
    """The whole point: slow stretches consume disproportionate steps."""
    weights = {"0": 256, "1": 768, "2": 2048, "3": 5120}
    harmonic = tau_effective(SKIP8, EDGES, 8192)
    arithmetic = pooled(SKIP8, weights)
    assert harmonic < arithmetic


def test_long_generation_moves_toward_the_late_buckets():
    """Deep skip improves with length, so tau_eff must rise with G."""
    short = tau_effective(SKIP8, EDGES, 640)
    long = tau_effective(SKIP8, EDGES, 8192)
    assert long > short
    assert short == pytest.approx(3.677, abs=5e-3)
    assert long == pytest.approx(4.019, abs=5e-3)


def test_windowed_curve_falls_with_length():
    """The window's acceptance decays, so tau_eff must fall with G."""
    assert tau_effective(W512, EDGES, 8192) < tau_effective(W512, EDGES, 640)


def test_the_two_curves_cross_ordering_is_preserved():
    """w512 beats skip8 at every length here; the estimator must not invert."""
    for gen in (256, 1024, 4096, 8192):
        assert tau_effective(W512, EDGES, gen) > tau_effective(SKIP8, EDGES, gen)


def test_gap_narrows_with_generation_length():
    """One rises and one falls, so the gap must close as G grows."""
    near = tau_effective(W512, EDGES, 256) / tau_effective(SKIP8, EDGES, 256)
    far = tau_effective(W512, EDGES, 8192) / tau_effective(SKIP8, EDGES, 8192)
    assert far < near


def test_unmeasured_tail_carries_the_last_bucket_forward():
    partial = {"0": 4.0, "1": 3.0}
    assert tau_effective(partial, EDGES, 20000) == pytest.approx(
        20000 / (256 / 4.0 + 768 / 3.0 + (20000 - 1024) / 3.0)
    )


def test_extrapolation_can_be_refused():
    with pytest.raises(TauCurveError, match="unmeasured"):
        tau_effective({"0": 4.0}, EDGES, 5000, extrapolate=False)


def test_empty_curve_is_an_error():
    with pytest.raises(TauCurveError, match="no usable bucket"):
        tau_effective({}, EDGES, 100)


def test_non_positive_generation_is_an_error():
    with pytest.raises(TauCurveError, match="must be positive"):
        tau_effective(SKIP8, EDGES, 0)


def test_scalar_from_profile_reads_depth_four():
    profile = {"buckets": {"0": {"armed_steps": 10, "pos_accepted": [10, 8, 6, 4, 2]}}}
    assert scalar_from_profile(profile, depth=4)["0"] == pytest.approx(1 + 28 / 10)
