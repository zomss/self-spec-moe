# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Proofs for the fail-closed arming rule.

The tests that matter are the ones asserting the rule DECLINES: a rule that
silently arms on a missing envelope, or that keeps its threshold below the
registered arming rent, would reintroduce exactly the D3 defect it exists to
fix.
"""

import json
from pathlib import Path

import pytest
from w98_failclosed import (
    EPSILON_ARM,
    OFF_KEY,
    FailClosedError,
    apply_rule,
    commitment_digest,
    decide,
    envelopes_from_fits,
    firing_set,
    load_inputs,
    threshold,
)

DATA = Path(__file__).resolve().parent.parent / "data"
PREDICTIONS = DATA / "g98_e" / "d3_predictions.json"
FITS = DATA / "g98_c" / "d1p_fits.json"

ENVELOPES = {"R1": 0.02, "R4": 0.07, "R8": 0.005}


def predicted(margin_by_regime):
    """A two-cell map whose armed cell sits at the requested margin."""
    return {
        OFF_KEY: {r: 100.0 for r in margin_by_regime},
        "armed/cell": {r: 100.0 * m for r, m in margin_by_regime.items()},
    }


# --- the threshold ---


def test_threshold_takes_the_larger_of_rent_and_envelope():
    assert threshold("R4", ENVELOPES) == pytest.approx(0.07)
    assert threshold("R8", ENVELOPES) == pytest.approx(EPSILON_ARM)


def test_threshold_never_drops_below_the_registered_arming_rent():
    """A tight fit must not license a win the phase already called noise."""
    for regime in ENVELOPES:
        assert threshold(regime, ENVELOPES) >= EPSILON_ARM


def test_missing_envelope_is_an_error_not_a_pass():
    """An unresolved regime is a reason to decline, not to guess."""
    with pytest.raises(FailClosedError, match="no fitted envelope"):
        threshold("R5cot", ENVELOPES)


def test_envelopes_skip_unfitted_regimes():
    fits = {
        "R1": {"fitted": True, "envelope": 0.03},
        "R9": {"fitted": False, "reason": "no usable points"},
    }
    assert envelopes_from_fits(fits) == {"R1": 0.03}


def test_envelopes_require_at_least_one_fit():
    with pytest.raises(FailClosedError, match="no fitted regime"):
        envelopes_from_fits({"R1": {"fitted": False}})


# --- the decision ---


def test_arms_when_the_margin_clears_the_envelope():
    d = decide(predicted({"R4": 1.30}), "R4", ENVELOPES)
    assert d["armed"] is True and d["choice"] == "armed/cell"
    assert d["rule_fired"] is False


def test_declines_when_the_margin_sits_inside_the_envelope():
    """The D3 defect in miniature."""
    d = decide(predicted({"R4": 1.026}), "R4", ENVELOPES)
    assert d["armed"] is False and d["choice"] == OFF_KEY
    assert d["rule_fired"] is True
    assert d["best_armed"] == "armed/cell"


def test_boundary_is_strict():
    """Just inside the threshold declines; clearly outside it arms.

    Stated as a pair either side of the threshold rather than an equality at
    it: `1 + t` does not round-trip exactly in binary floating point, so an
    exact-boundary assertion would test the float representation instead of
    the rule.
    """
    limit = threshold("R4", ENVELOPES)
    just_under = decide(predicted({"R4": 1.0 + limit - 1e-6}), "R4", ENVELOPES)
    just_over = decide(predicted({"R4": 1.0 + limit + 1e-3}), "R4", ENVELOPES)
    assert just_under["armed"] is False
    assert just_over["armed"] is True


def test_a_losing_candidate_is_declined():
    d = decide(predicted({"R4": 0.76}), "R4", ENVELOPES)
    assert d["armed"] is False and d["margin"] < 1.0


def test_map_without_off_is_an_error():
    with pytest.raises(FailClosedError, match="no 'off' cell"):
        decide({"armed/cell": {"R4": 120.0}}, "R4", ENVELOPES)


def test_map_without_an_armed_candidate_is_an_error():
    with pytest.raises(FailClosedError, match="no armed candidate"):
        decide({OFF_KEY: {"R4": 100.0}}, "R4", ENVELOPES)


def test_non_positive_off_rate_is_an_error():
    bad = {OFF_KEY: {"R4": 0.0}, "armed/cell": {"R4": 120.0}}
    with pytest.raises(FailClosedError, match="non-positive"):
        decide(bad, "R4", ENVELOPES)


# --- the commitment digest ---


def test_digest_is_stable_and_sensitive():
    a = apply_rule(predicted({"R4": 1.30, "R1": 1.50}), ENVELOPES, ("R1", "R4"))
    b = apply_rule(predicted({"R4": 1.30, "R1": 1.50}), ENVELOPES, ("R1", "R4"))
    c = apply_rule(predicted({"R4": 1.02, "R1": 1.50}), ENVELOPES, ("R1", "R4"))
    assert commitment_digest(a) == commitment_digest(b)
    assert commitment_digest(a) != commitment_digest(c)


# --- against the real committed artifacts ---


@pytest.mark.skipif(
    not (PREDICTIONS.is_file() and FITS.is_file()),
    reason="scored D3/D1' artifacts absent",
)
def test_rule_fires_at_r4_alone_on_the_committed_map():
    """The registered prediction, checked against the frozen inputs.

    Both inputs predate the rule: the prediction map is committed behind D3's
    barrier and the envelopes come from the closed G98-C fit.
    """
    predictions, envelopes = load_inputs(PREDICTIONS, FITS)
    decisions = apply_rule(predictions, envelopes)
    assert firing_set(decisions) == ["R4"]
    assert decisions["R4"]["choice"] == OFF_KEY
    assert decisions["R4"]["margin"] == pytest.approx(1.026, abs=5e-3)
    for regime in ("R1", "R5", "R5cot", "R6", "R8"):
        assert decisions[regime]["armed"] is True


@pytest.mark.skipif(
    not (PREDICTIONS.is_file() and FITS.is_file()),
    reason="scored D3/D1' artifacts absent",
)
def test_every_armed_regime_clears_its_threshold_with_room():
    """Not a knife edge: the rule's non-firing must not be luck."""
    predictions, envelopes = load_inputs(PREDICTIONS, FITS)
    for regime, d in apply_rule(predictions, envelopes).items():
        if d["armed"]:
            assert (d["margin"] - 1.0) > 3.0 * d["threshold"], regime


@pytest.mark.skipif(
    not (DATA / "g98_f" / "commitment.json").is_file(),
    reason="commitment not yet issued",
)
def test_committed_digest_still_matches_the_inputs():
    """The barrier: the rule may not be retuned after the commitment."""
    committed = json.loads((DATA / "g98_f" / "commitment.json").read_text())
    predictions, envelopes = load_inputs(PREDICTIONS, FITS)
    decisions = apply_rule(predictions, envelopes)
    assert commitment_digest(decisions) == committed["digest"]
    assert firing_set(decisions) == committed["firing_set"]
