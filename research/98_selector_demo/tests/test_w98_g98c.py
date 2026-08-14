# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Proofs for the W98-R2 (G98-C) campaign runner.

The tests that matter are the ones covering the falsifiability barrier and the
authorization binding: a campaign that can commit predictions after seeing the
held-out measurements, or run under a modified model, produces a number that
looks exactly like a real result.

Nothing here boots an engine.
"""

import json

import pytest
import run_w98_g98c_round2 as g98c
from w98r2_cost_model import W_LAYER_BYTES, R2LeverPoint, keep_frac, kv_bytes

TRUE = {
    "kappa_w": 2.0e-12,
    "kappa_kv": 1.1e-12,
    "c_layer": 2.0e-4,
    "f_win": 5.0e-5,
    "f_fixed": 3.66e-3,
}
REGIMES = {"R1": 431.0, "R8": 398.0, "R6": 393.0}


def synth_cost(cfg, context):
    point = R2LeverPoint(
        W_LAYER_BYTES[cfg["quant"]],
        kv_bytes(cfg["window"], context),
        keep_frac(cfg["skip_count"]),
        cfg["window"] != "off",
        1.0,
    )
    row = point.design_row()
    return (
        row[0] * TRUE["kappa_w"]
        + row[1] * TRUE["kappa_kv"]
        + row[2] * TRUE["c_layer"]
        + row[3] * TRUE["f_win"]
        + row[4] * TRUE["f_fixed"]
    )


def write_cell(directory, cfg, jitter=0.0):
    directory.mkdir(parents=True, exist_ok=True)
    observations = {
        regime: {
            "draft_chain_s": synth_cost(cfg, context) * (1.0 + jitter),
            "context_tokens": context,
            "usable": True,
            "armed_step_count": 200,
        }
        for regime, context in REGIMES.items()
    }
    path = directory / f"{g98c._config_key(cfg).replace('/', '_')}.json"
    path.write_text(
        json.dumps({"config": cfg, "observations": observations}, sort_keys=True)
    )
    return path


# Deterministic misfit applied to the fit cells. Without it the synthetic fit is
# EXACT, so fit_term is 0 while sigma_repro is positive, and every regime is
# correctly reported NOT RESOLVABLE -- see the dedicated test below. Real
# campaigns always carry some misfit; this makes the fixture representative.
FIT_MISFIT = 0.01


def build_campaign(root, with_heldout=False, fit_misfit=FIT_MISFIT):
    """A complete synthetic campaign directory, generated from the true model."""
    for stage in (g98c.ANCHOR_PRE_DIR, g98c.ANCHOR_POST_DIR):
        for repeat in range(g98c.ANCHOR_REPEATS):
            for anchor in g98c.ANCHORS:
                # A little spread so sigma_repro is positive and finite.
                write_cell(
                    root / stage / f"r{repeat}",
                    anchor,
                    jitter=1e-4 * (repeat - 1),
                )
    for index, cfg in enumerate(g98c.fit_profiles()):
        write_cell(
            root / g98c.FIT_DIR, cfg, jitter=fit_misfit * (1 if index % 2 else -1)
        )
    if with_heldout:
        for cfg in g98c.heldout_profiles():
            write_cell(root / g98c.HELDOUT_DIR, cfg)
    return root


# --- the frozen lattice ---


def test_lattice_check_passes_on_the_frozen_artifacts():
    check = g98c._lattice_check()
    assert check["fit_count"] == g98c.FIT_COUNT
    assert check["heldout_count"] == g98c.HELDOUT_COUNT
    assert check["disjoint"] is True
    assert check["every_heldout_varies_two_axes"] is True


def test_every_heldout_cell_varies_at_least_two_axes():
    """Round 1's split failed this and a cell had to be dropped after the fact."""
    for cell in g98c.heldout_profiles():
        varied = sum(
            (
                cell["quant"] != "target-matching",
                cell["window"] != "off",
                cell["skip_count"] != 0,
            )
        )
        assert varied >= 2, cell


def test_fit_and_heldout_are_disjoint():
    fit = {g98c._triple_key(c) for c in g98c.fit_profiles()}
    held = {g98c._triple_key(c) for c in g98c.heldout_profiles()}
    assert not (fit & held)


def test_every_cell_has_a_registered_skip_set():
    for cell in g98c.fit_profiles() + g98c.heldout_profiles():
        assert cell["skip_count"] in g98c.r1.SKIP_SETS


def test_anchors_are_fit_cells_and_not_held_out():
    """Amendment 1 section 3 requires anchors to come from the fit set."""
    fit = {g98c._triple_key(c) for c in g98c.fit_profiles()}
    held = {g98c._triple_key(c) for c in g98c.heldout_profiles()}
    for anchor in g98c.ANCHORS:
        key = g98c._triple_key(anchor)
        assert key in fit and key not in held


# --- authorization ---


def test_expected_authorization_hashes_every_source():
    package = g98c.expected_authorization()
    for name in (
        "round1_runner",
        "round1_cost_model",
        "round2_cost_model",
        "round2_runner",
        "host_load_gate",
        "koff_runtime",
        "prereg_doc",
        "prereg_fit",
        "prereg_heldout",
        "prereg_matrix",
    ):
        assert len(package["source_artifacts"][name]["sha256"]) == 64, name


def test_authorization_round_trips():
    g98c.validate_authorization(g98c.expected_authorization())


def test_authorization_rejects_a_changed_source_artifact():
    package = g98c.expected_authorization()
    package["source_artifacts"]["round2_cost_model"]["sha256"] = "0" * 64
    with pytest.raises(g98c.G98CError, match="changed since"):
        g98c.validate_authorization(package)


def test_authorization_rejects_a_missing_source_artifact():
    package = g98c.expected_authorization()
    del package["source_artifacts"]["host_load_gate"]
    with pytest.raises(g98c.G98CError, match="omits source artifact"):
        g98c.validate_authorization(package)


def test_authorization_rejects_a_foreign_package():
    package = g98c.expected_authorization()
    package["package_id"] = "some-other-package"
    with pytest.raises(g98c.G98CError, match="is not"):
        g98c.validate_authorization(package)


def test_authorization_rejects_a_retuned_model_or_runtime():
    for field in ("runtime", "model", "lattice", "stages", "d1_prime"):
        package = g98c.expected_authorization()
        package[field] = {"tampered": True}
        with pytest.raises(g98c.G98CError, match="does not match"):
            g98c.validate_authorization(package)


def test_authorization_must_enforce_the_host_load_gate():
    package = g98c.expected_authorization()
    package["host_load_gate"]["enforced"] = False
    with pytest.raises(g98c.G98CError, match="host-load gate"):
        g98c.validate_authorization(package)


def test_issue_authorization_is_create_only(tmp_path):
    path = tmp_path / "auth.json"
    g98c.issue_authorization(path)
    assert path.is_file()
    with pytest.raises(g98c.G98CError, match="already exists"):
        g98c.issue_authorization(path)


# --- the falsifiability barrier ---


def test_commit_refuses_after_heldout_measurements_exist(tmp_path):
    """The whole point of the barrier."""
    root = build_campaign(tmp_path / "run", with_heldout=True)
    fitted = g98c.fit_and_predict(root)
    with pytest.raises(g98c.G98CError, match="after held-out measurements exist"):
        g98c.commit_predictions(root, fitted["predictions"])


def test_commit_is_immutable(tmp_path):
    root = build_campaign(tmp_path / "run")
    fitted = g98c.fit_and_predict(root)
    g98c.commit_predictions(root, fitted["predictions"])
    with pytest.raises(g98c.G98CError, match="immutable"):
        g98c.commit_predictions(root, fitted["predictions"])


def test_commit_must_cover_exactly_the_registered_heldout_set(tmp_path):
    root = build_campaign(tmp_path / "run")
    fitted = g98c.fit_and_predict(root)
    partial = dict(list(fitted["predictions"].items())[:-1])
    with pytest.raises(g98c.G98CError, match="exactly the registered"):
        g98c.commit_predictions(root, partial)


def test_reveal_refuses_without_a_commitment(tmp_path):
    root = build_campaign(tmp_path / "run", with_heldout=True)
    with pytest.raises(g98c.G98CError, match="no committed predictions"):
        g98c.require_committed_predictions(root)


def test_tampered_commitment_is_detected(tmp_path):
    root = build_campaign(tmp_path / "run")
    fitted = g98c.fit_and_predict(root)
    g98c.commit_predictions(root, fitted["predictions"])
    path = root / g98c.PREDICTIONS_NAME
    record = json.loads(path.read_text())
    key = next(iter(record["predictions"]))
    regime = next(iter(record["predictions"][key]))
    record["predictions"][key][regime]["hi"] *= 10.0
    path.write_text(json.dumps(record))
    with pytest.raises(g98c.G98CError, match="modified after commitment"):
        g98c.require_committed_predictions(root)


def test_commitment_must_claim_pre_reveal(tmp_path):
    root = build_campaign(tmp_path / "run")
    fitted = g98c.fit_and_predict(root)
    g98c.commit_predictions(root, fitted["predictions"])
    path = root / g98c.PREDICTIONS_NAME
    record = json.loads(path.read_text())
    record["committed_before_reveal"] = False
    path.write_text(json.dumps(record))
    with pytest.raises(g98c.G98CError, match="pre-reveal"):
        g98c.require_committed_predictions(root)


# --- fit, predict, score ---


def test_fit_requires_the_whole_fit_set(tmp_path):
    root = build_campaign(tmp_path / "run")
    next(iter((root / g98c.FIT_DIR).glob("*.json"))).unlink()
    with pytest.raises(g98c.G98CError, match="expected 15 fit cells"):
        g98c.fit_and_predict(root)


def test_fit_recovers_the_true_parameters(tmp_path):
    """With no misfit the five parameters come back exactly."""
    root = build_campaign(tmp_path / "run", fit_misfit=0.0)
    fitted = g98c.fit_and_predict(root)
    assert set(fitted["fitted_regimes"]) == set(REGIMES)
    for regime in REGIMES:
        fit = fitted["fits"][regime]
        assert fit["fitted"] is True
        assert fit["kappa_w"] == pytest.approx(TRUE["kappa_w"], rel=1e-5)
        assert fit["F"] == pytest.approx(TRUE["f_fixed"], rel=1e-5)
        assert fit["points"] == g98c.FIT_COUNT
        assert 0.0 < fit["floor_fraction_unlevered"] < 1.0


def test_a_perfect_fit_is_not_resolvable(tmp_path):
    """fit_term 0 with positive sigma_repro means noise dominates.

    Counter-intuitive but correct, and worth pinning: a model that fits its own
    fit set exactly has no residual to resolve against, so the section-5 rule
    reports NOT RESOLVABLE rather than claiming a perfect result. A future
    change that let this pass as `resolvable` would manufacture coverage.
    """
    root = build_campaign(tmp_path / "run", fit_misfit=0.0)
    fitted = g98c.fit_and_predict(root)
    for regime in REGIMES:
        fit = fitted["fits"][regime]
        assert fit["fit_term"] < 1e-9
        assert fit["sigma_repro"] > 0.0
        assert fit["resolvable"] is False


def test_end_to_end_scores_full_coverage_when_the_model_is_true(tmp_path):
    """Barrier respected, and a correct model covers every held-out cell."""
    root = build_campaign(tmp_path / "run")
    fitted = g98c.fit_and_predict(root)
    (root / "d1p_fits.json").write_text(json.dumps({"fits": fitted["fits"]}))
    g98c.commit_predictions(root, fitted["predictions"])
    for cfg in g98c.heldout_profiles():
        write_cell(root / g98c.HELDOUT_DIR, cfg)
    result = g98c.score_reveal(root)
    assert result["total"] > 0
    assert result["covered"] == result["total"]
    assert result["coverage"] == 1.0


def test_score_reports_not_resolvable_rather_than_dropping(tmp_path):
    """Section-5 rule: name the regime, do not count it and do not hide it."""
    root = build_campaign(tmp_path / "run")
    fitted = g98c.fit_and_predict(root)
    fits = dict(fitted["fits"])
    fits["R1"] = {**fits["R1"], "resolvable": False}
    (root / "d1p_fits.json").write_text(json.dumps({"fits": fits}))
    g98c.commit_predictions(root, fitted["predictions"])
    for cfg in g98c.heldout_profiles():
        write_cell(root / g98c.HELDOUT_DIR, cfg)
    result = g98c.score_reveal(root)
    assert result["not_resolvable"]
    assert all(entry.endswith("@R1") for entry in result["not_resolvable"])
    assert "R1" not in result["resolvable_regimes"]
    assert all(row["regime"] != "R1" for row in result["rows"])


def test_score_requires_the_whole_heldout_set(tmp_path):
    root = build_campaign(tmp_path / "run")
    fitted = g98c.fit_and_predict(root)
    (root / "d1p_fits.json").write_text(json.dumps({"fits": fitted["fits"]}))
    g98c.commit_predictions(root, fitted["predictions"])
    for cfg in g98c.heldout_profiles()[:-1]:
        write_cell(root / g98c.HELDOUT_DIR, cfg)
    with pytest.raises(g98c.G98CError, match="expected 8 held-out cells"):
        g98c.score_reveal(root)


# --- the clamp diagnostic ---


def test_cheap_regime_spread_is_recorded():
    observations = {
        "R1": {"draft_chain_s": 0.030},
        "R8": {"draft_chain_s": 0.032},
        "R6": {"draft_chain_s": 0.034},
    }
    assert g98c._cheap_regime_spread(observations) == pytest.approx(0.004)


def test_cheap_regime_spread_is_none_when_a_regime_is_missing():
    assert g98c._cheap_regime_spread({"R1": {"draft_chain_s": 0.03}}) is None


# --- the v6 resume and telemetry fixes ---


def test_attempt_numbering_continues_past_stale_traces(tmp_path):
    """An aborted run's traces are audit trail; a resume must not collide."""
    key = "target-matching_woff_skip0"
    for stale in ("attempt1", "attempt3", "attemptX"):
        (tmp_path / f"{key}.{stale}.jsonl").write_text("", encoding="utf-8")
    assert g98c._max_attempt_index(tmp_path, key, ".jsonl") == 3
    assert g98c._max_attempt_index(tmp_path, "some_other_key", ".jsonl") == 0


def test_attempt_numbering_also_respects_surviving_logs(tmp_path):
    """A deleted trace whose log survives must still advance the numbering."""
    key = "target-matching_woff_skip0"
    (tmp_path / f"{key}.attempt3.log").write_text("", encoding="utf-8")
    assert g98c._max_attempt_index(tmp_path, key, ".jsonl") == 0
    assert g98c._max_attempt_index(tmp_path, key, ".log") == 3


def test_authorization_pins_telemetry_on_and_persisted_per_attempt():
    package = g98c.expected_authorization()
    assert package["gpu_telemetry"] == {
        "enabled": True,
        "persisted_per_attempt": True,
    }
