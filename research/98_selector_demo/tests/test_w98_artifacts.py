# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Proofs for artifact identity and integrity.

The regression tests at the top replay the two collisions that actually
happened: OFF against the unlevered armed cell in the dense grid, and the two
quant arms of one lever point in the MoE probe. Both were silent, and both
cost a measurement.
"""

import json

import pytest
from w98_artifacts import (
    ArtifactError,
    audit,
    cell_key,
    claim,
    envelope,
    plan,
    read,
    slug,
)

OFF = {"action": "off", "quant": "target-matching", "window": "off", "skip_count": 0}
UNLEVERED_ARMED = {
    "action": "armed",
    "quant": "target-matching",
    "window": "off",
    "skip_count": 0,
}
QUANT_ARM = {
    "action": "armed",
    "quant": "w4a16-quantized",
    "window": "off",
    "skip_count": 0,
}


# --- the two collisions that happened ---


def test_off_and_unlevered_armed_are_different_identities():
    """The dense grid's bug: identical levers, different measurement."""
    assert cell_key(OFF) != cell_key(UNLEVERED_ARMED)
    assert slug(OFF) != slug(UNLEVERED_ARMED)


def test_quant_arms_are_different_identities():
    """The MoE probe's bug: the quant axis was left out of the name."""
    assert cell_key(UNLEVERED_ARMED) != cell_key(QUANT_ARM)
    assert slug(UNLEVERED_ARMED) != slug(QUANT_ARM)


def test_plan_accepts_the_real_grid_shape():
    """The three cells that used to collide now plan cleanly."""
    assert len(plan([OFF, UNLEVERED_ARMED, QUANT_ARM])) == 3


def test_plan_tolerates_the_same_cell_listed_twice():
    assert len(plan([UNLEVERED_ARMED, dict(UNLEVERED_ARMED)])) == 1


def test_plan_refuses_a_genuine_collision():
    """The guard that would have failed the grid before it booted anything.

    Constructed from separator ambiguity, which is the hazard that survives
    once every axis is in the name: two DIFFERENT identities can still render
    to one stem if a value contains the separator.
    """
    a = {**UNLEVERED_ARMED, "quant": "w4a16/x"}
    b = {**UNLEVERED_ARMED, "quant": "w4a16_x"}
    assert cell_key(a) != cell_key(b)
    assert slug(a) == slug(b)
    with pytest.raises(ArtifactError, match="collision"):
        plan([a, b])


# --- identity lives in the content ---


def test_unknown_axis_is_refused_rather_than_ignored():
    """An axis nobody named is how the next collision would arrive."""
    with pytest.raises(ArtifactError, match="does not know how to name"):
        cell_key({**UNLEVERED_ARMED, "kv_dtype": "fp8"})


def test_bookkeeping_keys_do_not_change_identity():
    assert cell_key({**UNLEVERED_ARMED, "_repeat": 2}) == cell_key(UNLEVERED_ARMED)
    assert slug(UNLEVERED_ARMED, repeat=2).endswith("__r2")


def test_read_rejects_a_record_whose_identity_contradicts_its_config(tmp_path):
    """Exactly the corruption a rename-by-filename repair produces."""
    path = tmp_path / "off.json"
    bad = envelope(UNLEVERED_ARMED, "w98_test", {"observations": {}})
    bad["cell"] = "off"  # mislabelled, as the blind rename did
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ArtifactError, match="does not match its own config"):
        read(path)


def test_round_trip_is_self_describing(tmp_path):
    path = tmp_path / slug(QUANT_ARM)
    path.write_text(
        json.dumps(envelope(QUANT_ARM, "w98_test", {"observations": {"R1": 1}})),
        encoding="utf-8",
    )
    assert read(path)["cell"] == cell_key(QUANT_ARM)


# --- resume must verify, not assume ---


def test_claim_is_false_when_nothing_is_there(tmp_path):
    assert claim(tmp_path / "absent.json", OFF) is False


def test_claim_is_true_for_the_same_configuration(tmp_path):
    path = tmp_path / "x.json"
    path.write_text(json.dumps(envelope(OFF, "w98_test", {})), encoding="utf-8")
    assert claim(path, OFF) is True


def test_claim_refuses_to_skip_a_boot_on_someone_elses_record(tmp_path):
    """The silent skip that dropped a cell from the scored grid."""
    path = tmp_path / "x.json"
    path.write_text(json.dumps(envelope(OFF, "w98_test", {})), encoding="utf-8")
    with pytest.raises(ArtifactError, match="refusing to skip"):
        claim(path, UNLEVERED_ARMED)


# --- coverage is counted from records, never from the plan ---


def _write(directory, cfg):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{slug(cfg)}.json"
    path.write_text(json.dumps(envelope(cfg, "w98_test", {})), encoding="utf-8")
    return path


def test_audit_passes_on_complete_coverage(tmp_path):
    cells = [OFF, UNLEVERED_ARMED, QUANT_ARM]
    for rd in ("r0", "r1"):
        for cfg in cells:
            _write(tmp_path / rd, cfg)
    report = audit([tmp_path / "r0", tmp_path / "r1"], cells, replicates=2)
    assert report["ok"] is True
    assert report["records"] == 6 and report["observed_cells"] == 3


def test_audit_catches_the_dropped_cell(tmp_path):
    """The grid reported 31 cells while holding 30; this is that check."""
    cells = [OFF, UNLEVERED_ARMED, QUANT_ARM]
    for cfg in (OFF, QUANT_ARM):
        _write(tmp_path / "r0", cfg)
    report = audit([tmp_path / "r0"], cells, replicates=1)
    assert report["ok"] is False
    assert report["missing"] == [cell_key(UNLEVERED_ARMED)]


def test_audit_catches_an_uneven_replicate_count(tmp_path):
    cells = [OFF, QUANT_ARM]
    _write(tmp_path / "r0", OFF)
    _write(tmp_path / "r0", QUANT_ARM)
    _write(tmp_path / "r1", OFF)
    report = audit([tmp_path / "r0", tmp_path / "r1"], cells, replicates=2)
    assert report["ok"] is False
    assert report["wrong_replicate_count"] == {cell_key(QUANT_ARM): 1}


def test_audit_flags_a_name_that_contradicts_its_content(tmp_path):
    directory = tmp_path / "r0"
    directory.mkdir(parents=True)
    (directory / "off.json").write_text(
        json.dumps(envelope(QUANT_ARM, "w98_test", {})), encoding="utf-8"
    )
    report = audit([directory], [QUANT_ARM], replicates=1)
    assert report["ok"] is False
    assert any("name says" in m for m in report["identity_mismatches"])


def test_envelope_refuses_a_payload_that_would_overwrite_identity():
    """Caught in practice: a probe's content-cell key clobbered `cell`."""
    with pytest.raises(ArtifactError, match="carry the record's identity"):
        envelope(OFF, "w98_test", {"cell": "LO", "observations": {}})
    with pytest.raises(ArtifactError, match="carry the record's identity"):
        envelope(OFF, "w98_test", {"config": {"something": "else"}})
