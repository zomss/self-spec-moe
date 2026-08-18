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
