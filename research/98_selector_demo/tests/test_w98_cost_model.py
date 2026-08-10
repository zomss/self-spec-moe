# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic proofs for the G98-0 factored cost model and elimination rule."""

import math
import random

import pytest
from w98_cost_model import (
    AffineFit,
    FactoredCostModel,
    LeverPoint,
    W98CostModelError,
    eliminate,
    required_acceptance,
    s_max,
    surviving_counts,
    tau_star,
)

KAPPA_W = 2.0e-9
KAPPA_KV = 0.5e-9
C0 = 1.0
N_LAYERS = 36


def true_cost(weight_bytes, kv_bytes, keep_frac, deviation=1.0):
    base = keep_frac * (weight_bytes * KAPPA_W + kv_bytes * KAPPA_KV + C0)
    return base * deviation


def single_lever_points(deviations=None):
    """Singles varying one axis at a time around a base configuration."""
    base_w, base_kv = 8.0e9, 1.0e9
    configs = [
        (base_w, base_kv, 1.0),
        (2.0e9, base_kv, 1.0),
        (base_w, 0.25e9, 1.0),
        (base_w, 2.0e9, 1.0),
        (base_w, 4.0e9, 1.0),
        (base_w, base_kv, 1.0 - 4 / N_LAYERS),
        (base_w, base_kv, 1.0 - 8 / N_LAYERS),
    ]
    deviations = deviations or [1.0] * len(configs)
    return [
        LeverPoint(w, kv, keep, true_cost(w, kv, keep, dev))
        for (w, kv, keep), dev in zip(configs, deviations)
    ]


class TestAffineFit:
    def test_exact_recovery(self):
        fit = AffineFit.fit([(q, 3.0 + 0.5 * q) for q in (1.0, 2.0, 4.0)])
        assert fit.alpha == pytest.approx(3.0)
        assert fit.beta == pytest.approx(0.5)
        assert fit.predict(10.0) == pytest.approx(8.0)
        assert fit.certified()

    def test_two_points_never_certified(self):
        fit = AffineFit.fit([(1.0, 2.0), (2.0, 3.0)])
        assert not fit.certified()

    def test_curved_data_fails_certification(self):
        points = [(q, 1.0 + q + 0.5 * q * q) for q in (1.0, 2.0, 3.0, 4.0)]
        assert not AffineFit.fit(points).certified(tol=0.05)

    def test_degenerate_q_rejected(self):
        with pytest.raises(W98CostModelError, match="degenerate"):
            AffineFit.fit([(1.0, 2.0), (1.0, 3.0)])

    def test_required_acceptance_is_p_over_t(self):
        t_fit = AffineFit.fit([(q, 2.0 + 0.1 * q) for q in (1.0, 2.0, 3.0)])
        p_fit = AffineFit.fit([(q, 3.0 + 0.2 * q) for q in (1.0, 2.0, 3.0)])
        q_state = 5.0
        expected = (3.0 + 0.2 * q_state) / (2.0 + 0.1 * q_state)
        assert required_acceptance(p_fit, t_fit, q_state) == pytest.approx(expected)


class TestFactoredModel:
    def test_exact_recovery_from_singles(self):
        model = FactoredCostModel.fit(single_lever_points())
        assert model.kappa_w == pytest.approx(KAPPA_W, rel=1e-9)
        assert model.kappa_kv == pytest.approx(KAPPA_KV, rel=1e-9)
        assert model.c0 == pytest.approx(C0, rel=1e-9)
        assert model.log_envelope == pytest.approx(0.0, abs=1e-12)

    def test_composed_triple_predicted_exactly(self):
        model = FactoredCostModel.fit(single_lever_points())
        w, kv, keep = 2.0e9, 0.25e9, 1.0 - 8 / N_LAYERS
        assert model.predict(w, kv, keep) == pytest.approx(
            true_cost(w, kv, keep), rel=1e-9
        )

    def test_interval_covers_bounded_deviation_composition(self):
        delta = 0.03
        rng = random.Random(11)
        deviations = [math.exp(rng.uniform(-delta, delta)) for _ in range(7)]
        model = FactoredCostModel.fit(single_lever_points(deviations))
        w, kv, keep = 2.0e9, 0.25e9, 1.0 - 8 / N_LAYERS
        truth = true_cost(w, kv, keep, math.exp(delta))
        lo, hi = model.predict_interval(w, kv, keep, inflation=2.0)
        assert lo <= truth <= hi

    def test_singular_design_rejected(self):
        base = LeverPoint(8.0e9, 1.0e9, 1.0, true_cost(8.0e9, 1.0e9, 1.0))
        points = [base, base, base]
        with pytest.raises(W98CostModelError, match="singular"):
            FactoredCostModel.fit(points)

    def test_invalid_points_rejected(self):
        with pytest.raises(W98CostModelError, match="invalid lever point"):
            FactoredCostModel.fit([LeverPoint(1.0, 1.0, 0.0, 1.0)] * 3)


class TestElimination:
    def test_s_max_formula(self):
        assert s_max(4, 2.0) == pytest.approx(2.5)

    def test_elimination_is_sound_against_planted_truth(self):
        """If eliminate() fires with a valid lower bound, no achievable
        acceptance can make the configuration pay."""
        rng = random.Random(23)
        fired = 0
        for _ in range(500):
            q_true = rng.uniform(0.2, 8.0)
            q_lo = q_true * rng.uniform(0.8, 1.0)
            k = rng.choice([2, 4])
            if eliminate(k, q_lo):
                fired += 1
                best_possible = (k + 1) / q_true
                assert best_possible < 1.0 + 0.015
        assert fired > 0

    def test_retains_when_uncertain(self):
        assert not eliminate(4, q_lo=4.0)
        assert eliminate(4, q_lo=5.1)

    def test_invalid_q_lo_rejected(self):
        with pytest.raises(W98CostModelError, match="positive"):
            eliminate(4, q_lo=0.0)


class TestCountLadder:
    def test_tau_star_identity(self):
        assert tau_star(4, 0.2, 1.1, 0.1) == pytest.approx(2.0)

    def test_survivors_form_a_suffix_under_monotone_cost(self):
        d_by_count = {0: 0.9, 4: 0.7, 8: 0.5, 16: 0.3}
        survivors = surviving_counts(
            d_by_count, k_depth=4, v_over_t=1.0, c_over_t=0.1, tau_bar=3.6
        )
        assert survivors == [8, 16]
        costs = sorted(d_by_count.items())
        kept = [k in survivors for k, _ in costs]
        assert kept == sorted(kept)

    def test_tau_bar_must_exceed_one(self):
        with pytest.raises(W98CostModelError, match="tau_bar"):
            surviving_counts({0: 0.5}, 4, 1.0, 0.1, tau_bar=1.0)
