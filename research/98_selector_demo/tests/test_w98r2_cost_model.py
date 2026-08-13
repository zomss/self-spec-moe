# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic proofs for the W98-R2 corrected cost model.

The important tests are the ones that would have caught Round 1's faults:
`F` estimated outside `keep_frac`, and a weight column that is not collinear
with the quant indicator.
"""

import json
import math
from pathlib import Path

import pytest
from w98_cost_model import combined_envelope
from w98r2_cost_model import (
    DRAFT_LAYERS,
    KV_BYTES_PER_TOKEN,
    KV_SINKS,
    N_PARAMS,
    W_LAYER_BYTES,
    R2CostModel,
    R2LeverPoint,
    W98CostModelError,
    design_strength,
    keep_frac,
    kv_bytes,
)

PREREG2 = Path(__file__).resolve().parent.parent / "data" / "prereg2"
TRUE = {
    "kappa_w": 2.0e-12,
    "kappa_kv": 1.1e-12,
    "c_layer": 2.0e-4,
    "f_win": 5.0e-5,
    "f_fixed": 3.66e-3,
}
CONTEXT = 431.0


def make_point(quant, window, skip_count, context=CONTEXT, d_measured=None):
    point = R2LeverPoint(
        weight_layer_bytes=W_LAYER_BYTES[quant],
        kv_bytes=kv_bytes(window, context),
        keep_frac=keep_frac(skip_count),
        windowed=window != "off",
        d_measured=d_measured if d_measured is not None else 1.0,
    )
    if d_measured is not None:
        return point
    row = point.design_row()
    truth = (
        row[0] * TRUE["kappa_w"]
        + row[1] * TRUE["kappa_kv"]
        + row[2] * TRUE["c_layer"]
        + row[3] * TRUE["f_win"]
        + row[4] * TRUE["f_fixed"]
    )
    return R2LeverPoint(
        point.weight_layer_bytes,
        point.kv_bytes,
        point.keep_frac,
        point.windowed,
        truth,
    )


def frozen_fit_set(context=CONTEXT):
    cells = json.loads((PREREG2 / "w98r2_fit.json").read_text())["fit"]
    return [
        make_point(c["quant"], c["window"], c["skip_count"], context) for c in cells
    ]


def frozen_heldout_set(context=CONTEXT):
    cells = json.loads((PREREG2 / "w98r2_heldout.json").read_text())["heldout"]
    return [
        make_point(c["quant"], c["window"], c["skip_count"], context) for c in cells
    ]


# --- the model recovers what it should ---


def test_recovers_every_parameter_from_the_frozen_fit_set():
    """The registered 15 points identify all five parameters exactly."""
    model = R2CostModel.fit(frozen_fit_set())
    for name, expected in TRUE.items():
        assert getattr(model, name) == pytest.approx(expected, rel=1e-6), name
    assert model.log_envelope < 1e-9


def test_predicts_held_out_composed_cells_exactly_when_the_model_is_true():
    """No composed cell is special: error there is specification, not extrapolation."""
    model = R2CostModel.fit(frozen_fit_set())
    for point in frozen_heldout_set():
        assert model.predict(point) == pytest.approx(point.d_measured, rel=1e-6)


def test_fit_needs_enough_points():
    with pytest.raises(W98CostModelError, match="at least"):
        R2CostModel.fit(frozen_fit_set()[: N_PARAMS - 1])


def test_singular_design_is_refused():
    """A fit set that never varies an axis must fail loudly, not silently."""
    flat = [make_point("target-matching", "off", 0) for _ in range(N_PARAMS + 2)]
    with pytest.raises(W98CostModelError, match="singular"):
        R2CostModel.fit(flat)


def test_rejects_invalid_points():
    bad = list(frozen_fit_set())
    bad[0] = R2LeverPoint(1.0, 1.0, 0.5, False, -1.0)
    with pytest.raises(W98CostModelError, match="invalid lever point"):
        R2CostModel.fit(bad)


# --- the two Round-1 faults this model exists to fix ---


def test_f_is_outside_keep_frac():
    """F must not scale with skipped layers -- the Round-1 fault.

    Two configurations differing only in skip must differ by exactly the scaled
    body terms, leaving F untouched.
    """
    model = R2CostModel.fit(frozen_fit_set())
    full = make_point("target-matching", "off", 0)
    half = make_point("target-matching", "off", 16)
    body_full = model.predict(full) - model.f_fixed
    body_half = model.predict(half) - model.f_fixed
    assert body_half / body_full == pytest.approx(half.keep_frac / full.keep_frac)


def test_f_is_an_irreducible_floor():
    """No lever setting can drive predicted cost below F."""
    model = R2CostModel.fit(frozen_fit_set())
    cheapest = make_point("w4a16-quantized", 128, DRAFT_LAYERS - 1)
    assert model.predict(cheapest) > model.f_fixed
    assert 0.0 < model.floor_fraction(cheapest) < 1.0
    # The floor binds harder as the levers bite.
    unlevered = make_point("target-matching", "off", 0)
    assert model.floor_fraction(cheapest) > model.floor_fraction(unlevered)


def test_weight_column_is_body_only_and_not_collinear_with_quant():
    """Round 1's weight column was an affine function of the quant indicator.

    Body bytes must not be reconstructible as `a + b*q`, which is what made
    kappa_w a fitted constant rather than a bytes coefficient. With only two
    quant arms any two values are trivially affine in q, so the real check is
    that body bytes exclude the non-body remainder that F now carries.
    """
    whole_checkpoint = {"target-matching": 16.4e9, "w4a16-quantized": 6.1e9}
    for arm, body in W_LAYER_BYTES.items():
        assert body < whole_checkpoint[arm]
    # The excluded remainder is the same order in both arms, because lm_head
    # and the embeddings stay bf16 either way -- which is why F is
    # quant-independent by construction.
    remainder = {a: whole_checkpoint[a] - W_LAYER_BYTES[a] for a in W_LAYER_BYTES}
    lo, hi = sorted(remainder.values())
    assert hi / lo < 1.2, remainder


def test_f_is_shared_across_quant_arms():
    """One F is fitted for both arms; a quant-specific F is not representable."""
    model = R2CostModel.fit(frozen_fit_set())
    bf16 = make_point("target-matching", "off", 0)
    quant = make_point("w4a16-quantized", "off", 0)
    assert model.predict(bf16) - model.predict(quant) == pytest.approx(
        bf16.keep_frac
        * model.kappa_w
        * (W_LAYER_BYTES["target-matching"] - W_LAYER_BYTES["w4a16-quantized"])
    )


# --- KV and keep semantics ---


def test_window_is_inactive_below_the_context():
    """A window larger than the context cannot save anything."""
    assert kv_bytes(1024, 200.0) == 200.0 * KV_BYTES_PER_TOKEN
    assert kv_bytes(128, 14357.0) == (128 + KV_SINKS) * KV_BYTES_PER_TOKEN


def test_window_off_reads_the_whole_context():
    assert kv_bytes("off", 14357.0) == 14357.0 * KV_BYTES_PER_TOKEN


def test_sinks_are_counted():
    assert kv_bytes(256, 9999.0) == (256 + KV_SINKS) * KV_BYTES_PER_TOKEN


def test_kv_bytes_rejects_nonsense():
    with pytest.raises(W98CostModelError):
        kv_bytes(256, 0.0)
    with pytest.raises(W98CostModelError, match="invalid window"):
        kv_bytes(-8, 100.0)


def test_keep_frac_matches_the_registered_levels():
    assert keep_frac(0) == 1.0
    assert keep_frac(4) == pytest.approx(0.888889, abs=1e-6)
    assert keep_frac(8) == pytest.approx(0.777778, abs=1e-6)
    assert keep_frac(16) == pytest.approx(0.555556, abs=1e-6)
    with pytest.raises(W98CostModelError):
        keep_frac(DRAFT_LAYERS)


# --- envelope, shared with amendment 1 ---


def test_interval_uses_the_amendment_1_envelope():
    model = R2CostModel.fit(frozen_fit_set())
    point = frozen_heldout_set()[0]
    sigma = 0.01
    lo, hi = model.predict_interval(point, sigma)
    half = combined_envelope(model.log_envelope, sigma)
    assert lo == pytest.approx(model.predict(point) * math.exp(-half))
    assert hi == pytest.approx(model.predict(point) * math.exp(half))


def test_interval_widens_with_sigma_repro():
    model = R2CostModel.fit(frozen_fit_set())
    point = frozen_heldout_set()[0]
    narrow = model.predict_interval(point, 0.001)
    wide = model.predict_interval(point, 0.05)
    assert wide[0] < narrow[0] and wide[1] > narrow[1]


def test_sigma_repro_has_no_default():
    """Omitting the measurement term is the defect amendment 1 exists to fix."""
    model = R2CostModel.fit(frozen_fit_set())
    with pytest.raises(TypeError):
        model.predict_interval(frozen_heldout_set()[0])


# --- conditioning: the registered reason the fit set grew ---


def _design(cells, context=CONTEXT):
    return [make_point(q, w, k, context) for q, w, k in cells]


CURRENT = [("target-matching", w, 0) for w in ("off", 128, 256, 512, 1024)] + [
    ("target-matching", "off", 4),
    ("target-matching", "off", 8),
    ("w4a16-quantized", "off", 0),
]
CROSSES = [
    ("target-matching", 256, 4),
    ("target-matching", 256, 8),
    ("w4a16-quantized", "off", 8),
    ("w4a16-quantized", 256, 0),
]
SKIP16 = [("target-matching", "off", 16), ("target-matching", 256, 16)]


def test_skip16_buys_more_than_the_crosses():
    """The registered justification for the lattice growing."""
    base = design_strength(_design(CURRENT))
    with_crosses = design_strength(_design(CURRENT + CROSSES))
    with_skip16 = design_strength(_design(CURRENT + CROSSES + SKIP16))
    assert with_crosses > base
    # Adding two skip16 points must beat adding four crosses.
    assert (with_skip16 - with_crosses) > (with_crosses - base)


def test_frozen_fit_set_is_the_strongest_registered_design():
    full = design_strength(frozen_fit_set())
    assert full > design_strength(_design(CURRENT + CROSSES + SKIP16))


def test_design_strength_is_zero_for_a_degenerate_design():
    assert design_strength([]) == 0.0


def test_frozen_fit_and_heldout_do_not_overlap():
    """Registered at freeze time; re-checked here so it cannot rot."""
    fit = {
        (c["quant"], str(c["window"]), c["skip_count"])
        for c in json.loads((PREREG2 / "w98r2_fit.json").read_text())["fit"]
    }
    heldout = {
        (c["quant"], str(c["window"]), c["skip_count"])
        for c in json.loads((PREREG2 / "w98r2_heldout.json").read_text())["heldout"]
    }
    assert len(fit) == 15
    assert len(heldout) == 8
    assert not (fit & heldout)
