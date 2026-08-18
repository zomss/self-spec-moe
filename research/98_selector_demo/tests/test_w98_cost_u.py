# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Proofs for the context-integrated draft cost."""

import pytest
from w98_cost_u import (
    WINDOW_SINKS,
    CostCurveError,
    draft_cost,
    flat_per_token,
    integrated_per_token,
    kv_positions,
    linear_verify,
)

EDGES = (256, 1024, 3072, 8192)
FIT = {
    "kappa_w": 2.0e-13,
    "kappa_kv": 1.1e-12,
    "c_layer": 2.0e-4,
    "f_win": 5.0e-5,
    "F": 3.7e-3,
}
KVB = 1.0e5  # bytes of KV per position
UNWINDOWED = {"keep_frac": 1.0, "window": "off", "weight_bytes": 1.389e10}
WINDOWED = {"keep_frac": 1.0, "window": 512, "weight_bytes": 1.389e10}
FLAT_TAU = {"0": 4.0, "1": 4.0, "2": 4.0, "3": 4.0}
VERIFY = linear_verify(3.0e-3, 1.0e-7)


def test_unwindowed_kv_tracks_the_whole_context():
    assert kv_positions("off", 12345) == 12345


def test_windowed_kv_saturates_at_window_plus_sinks():
    assert kv_positions(512, 100) == 100
    assert kv_positions(512, 50_000) == 512 + WINDOW_SINKS


def test_unwindowed_cost_grows_with_context_and_windowed_does_not():
    """Growth must equal the KV term exactly, not merely be positive."""
    near = draft_cost(FIT, UNWINDOWED, 500, KVB)
    far = draft_cost(FIT, UNWINDOWED, 20_000, KVB)
    expected = (20_000 - 500) * KVB * FIT["kappa_kv"] * UNWINDOWED["keep_frac"]
    assert far - near == pytest.approx(expected)
    w_near = draft_cost(FIT, WINDOWED, 500, KVB)
    w_far = draft_cost(FIT, WINDOWED, 20_000, KVB)
    assert w_far == pytest.approx(w_near, rel=0.05)


def test_integration_matches_flat_when_nothing_varies():
    """A window that saturates immediately and a flat tau must agree."""
    tiny = {"keep_frac": 1.0, "window": 8, "weight_bytes": 1.389e10}
    verify = linear_verify(3.0e-3, 0.0)
    integrated = integrated_per_token(
        FIT, tiny, FLAT_TAU, EDGES, 1000, 4000, verify, KVB
    )
    flat = flat_per_token(FIT, tiny, 4.0, 3000, verify, KVB)
    assert integrated["per_token_s"] == pytest.approx(flat["per_token_s"], rel=1e-3)


def test_flat_context_understates_an_unwindowed_long_generation():
    """Evaluating at the START context is optimistic; the KV read grows."""
    integrated = integrated_per_token(
        FIT, UNWINDOWED, FLAT_TAU, EDGES, 120, 16_000, VERIFY, KVB
    )
    at_prompt = flat_per_token(FIT, UNWINDOWED, 4.0, 120, VERIFY, KVB)
    assert integrated["per_token_s"] > at_prompt["per_token_s"]


def test_integration_favours_the_window_more_than_a_flat_model_does():
    """The point of the term: saturation only shows up when integrated."""
    gen, prompt = 16_000, 120
    unw = integrated_per_token(
        FIT, UNWINDOWED, FLAT_TAU, EDGES, prompt, gen, VERIFY, KVB
    )["per_token_s"]
    win = integrated_per_token(
        FIT, WINDOWED, FLAT_TAU, EDGES, prompt, gen, VERIFY, KVB
    )["per_token_s"]
    integrated_gain = unw / win
    flat_unw = flat_per_token(FIT, UNWINDOWED, 4.0, prompt, VERIFY, KVB)["per_token_s"]
    flat_win = flat_per_token(FIT, WINDOWED, 4.0, prompt, VERIFY, KVB)["per_token_s"]
    assert integrated_gain > flat_unw / flat_win


def test_mean_draft_cost_is_per_armed_step():
    out = integrated_per_token(FIT, WINDOWED, FLAT_TAU, EDGES, 120, 4000, VERIFY, KVB)
    assert out["armed_steps"] == pytest.approx(4000 / 4.0)
    assert out["mean_draft_cost_s"] > 0


def test_zero_generation_is_an_error():
    with pytest.raises(CostCurveError, match="must be positive"):
        integrated_per_token(FIT, UNWINDOWED, FLAT_TAU, EDGES, 120, 0, VERIFY, KVB)


def test_missing_fit_parameter_is_an_error():
    with pytest.raises(CostCurveError, match="missing"):
        draft_cost({"kappa_w": 1.0}, UNWINDOWED, 100, KVB)


# --- the batch drain ---

from w98_cost_u import integrated_with_drain, split_cost, survival  # noqa: E402


def test_survival_counts_requests_still_generating():
    assert survival([100, 200, 300], 0) == 3
    assert survival([100, 200, 300], 150) == 2
    assert survival([100, 200, 300], 400) == 0


def test_split_puts_weights_in_shared_and_kv_in_per_request():
    shared, per_req = split_cost(FIT, UNWINDOWED, 10_000, KVB, fit_batch=8)
    assert shared == pytest.approx(
        UNWINDOWED["weight_bytes"] * FIT["kappa_w"] + FIT["c_layer"] + FIT["F"]
    )
    assert per_req == pytest.approx(10_000 * KVB * FIT["kappa_kv"] / 8)


def test_windowed_per_request_cost_saturates_but_shared_gains_the_window_term():
    shared_w, per_w = split_cost(FIT, WINDOWED, 50_000, KVB, fit_batch=8)
    shared_u, per_u = split_cost(FIT, UNWINDOWED, 50_000, KVB, fit_batch=8)
    assert per_w < per_u / 50
    assert shared_w > shared_u  # the window costs a per-layer constant


def test_uniform_lengths_reproduce_the_constant_batch_result():
    """With no drain the term must not change the answer."""
    lengths = [4000.0] * 8
    verify_shared, verify_per = 10.0e-3, 1.2e-3
    drained = integrated_with_drain(
        FIT,
        UNWINDOWED,
        FLAT_TAU,
        EDGES,
        120,
        lengths,
        verify_shared,
        verify_per,
        KVB,
        fit_batch=8,
    )
    assert drained["mean_active_batch"] == pytest.approx(8.0)
    assert drained["per_token_s"] > 0


def test_drain_raises_per_token_cost_against_no_drain():
    """The measured effect: a decaying batch amortises shared work worse."""
    even = [4000.0] * 8
    ragged = [500.0, 1000.0, 1500.0, 2000.0, 4000.0, 6000.0, 8000.0, 10_000.0]
    kwargs = dict(
        verify_shared=10.0e-3,
        verify_per_request=1.2e-3,
        kv_bytes_per_token=KVB,
        fit_batch=8,
    )
    flat_run = integrated_with_drain(
        FIT, UNWINDOWED, FLAT_TAU, EDGES, 120, even, **kwargs
    )
    drained = integrated_with_drain(
        FIT, UNWINDOWED, FLAT_TAU, EDGES, 120, ragged, **kwargs
    )
    assert drained["mean_active_batch"] < 8.0
    assert drained["per_token_s"] > flat_run["per_token_s"]


def test_empty_lengths_is_an_error():
    with pytest.raises(CostCurveError, match="no generation lengths"):
        integrated_with_drain(
            FIT, UNWINDOWED, FLAT_TAU, EDGES, 120, [], 10e-3, 1e-3, KVB, fit_batch=8
        )


# --- window-aware attention ---

from w98_cost_u import draft_cost_windowed, kv_cost_factor  # noqa: E402


def test_gamma_one_recovers_the_linear_model():
    """The extension must contain the model it extends."""
    for positions in (128, 512, 17_000):
        assert kv_cost_factor(positions, 1.0, 14_000) == pytest.approx(1.0)
    linear = draft_cost(FIT, WINDOWED, 20_000, KVB)
    same = draft_cost_windowed(FIT, WINDOWED, 20_000, KVB, gamma=1.0)
    assert same == pytest.approx(linear)


def test_superlinearity_makes_short_working_sets_disproportionately_cheap():
    assert kv_cost_factor(528, 1.2, 14_000) < 1.0
    assert kv_cost_factor(17_000, 1.2, 14_000) > 1.0


def test_gamma_above_one_widens_the_window_advantage():
    """Exactly the residual it exists to explain."""
    ctx = 17_000

    def gap(gamma):
        unw = draft_cost_windowed(FIT, UNWINDOWED, ctx, KVB, gamma=gamma)
        win = draft_cost_windowed(FIT, WINDOWED, ctx, KVB, gamma=gamma)
        return unw / win

    assert gap(1.2) > gap(1.0)


def test_reference_anchors_the_parameterisation():
    """At p_ref the factor is 1 for any gamma, so gamma redistributes."""
    for gamma in (0.9, 1.0, 1.3):
        assert kv_cost_factor(14_000, gamma, 14_000) == pytest.approx(1.0)


def test_zero_positions_costs_nothing():
    assert kv_cost_factor(0, 1.2, 14_000) == 0.0


def test_bad_reference_is_an_error():
    with pytest.raises(CostCurveError, match="reference"):
        kv_cost_factor(100, 1.2, 0)
