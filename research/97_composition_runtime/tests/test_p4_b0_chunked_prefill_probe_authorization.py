"""CPU regressions for the one-shot chunked-prefill GPU probe package."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from jsonschema import Draft202012Validator

PHASE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PHASE_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_p4_b0_chunked_prefill_probe import (  # noqa: E402
    AUTHORIZATION_PATH,
    OUTPUT_DIR,
    REPO_ROOT,
    P4ChunkedPrefillProbeError,
    _probe_cells,
    _validate_probe_evidence,
    execute_parent,
    main,
)
from validate_p4_b0_chunked_prefill_probe_authorization_v5 import (  # noqa: E402
    P4ChunkedPrefillAuthorizationError,
    validate_authorization,
)


def _authorization() -> dict:
    return json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))


def _complete_probe_evidence():
    expected_request_ids = tuple(
        tuple(f"{regime_id}-s0-p{index:03d}" for index in range(8))
        for regime_id in ("R4", "R5", "R5cot")
    )
    histories = []
    events = []
    for request_ids in expected_request_ids:
        histories.append(
            {
                "state": "complete",
                "action_id": "target-matching-k4",
                "max_num_batched_tokens": 8192,
                "max_num_scheduled_tokens": 8160,
                "pure_first_measured_decode": True,
                "unmeasured_prefill_tokens_per_request": 1,
                "measured_decode_tokens_per_request": 1,
                "abort_reason": None,
                "committed_by_request": dict.fromkeys(request_ids, 1),
            }
        )
        scheduler_order = request_ids[-1:] + request_ids[:-1]
        events.append(
            {
                "pure_decode": True,
                "score_eligible": True,
                "quality": {
                    "preemptions": 0,
                    "recomputed_tokens": 0,
                    "invalid_spec_tokens": 0,
                },
                "request_steps": [
                    {
                        "request_id": f"{request_id}-{index + 1:08x}",
                        "committed_tokens": 1,
                    }
                    for index, request_id in enumerate(scheduler_order)
                ],
            }
        )
    return histories, events, expected_request_ids


def test_probe_evidence_canonicalizes_randomized_internal_request_ids():
    histories, events, expected_request_ids = _complete_probe_evidence()

    normalized = _validate_probe_evidence(
        histories,
        events,
        expected_request_ids,
    )

    assert len(normalized) == 3
    for event, request_ids in zip(normalized, expected_request_ids, strict=True):
        observed = {row["request_id"] for row in event["request_steps"]}
        assert observed == set(request_ids)
    assert events[0]["request_steps"][0]["request_id"].endswith("-00000001")


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("malformed", "neither frozen"),
        ("foreign", "neither frozen"),
        ("collision", "collide after canonicalization"),
    ],
)
def test_probe_evidence_rejects_invalid_internal_request_ids(case, message):
    histories, events, expected_request_ids = _complete_probe_evidence()
    if case == "malformed":
        events[0]["request_steps"][0]["request_id"] = "R4-s0-p007-NOT-HEX"
    elif case == "foreign":
        events[0]["request_steps"][0]["request_id"] = "foreign-00000000"
    else:
        events[0]["request_steps"][0]["request_id"] = "R4-s0-p000-ffffffff"

    with pytest.raises(P4ChunkedPrefillProbeError, match=message):
        _validate_probe_evidence(histories, events, expected_request_ids)


def test_v5_schema_accepts_the_checked_authorization():
    authorization = _authorization()
    schema_path = (
        REPO_ROOT / authorization["source_artifacts"]["authorization_schema"]["path"]
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(authorization)


def test_checked_authorization_grants_only_the_non_scored_gpu4_probe():
    result = validate_authorization(_authorization())

    assert result["status"] == "pass"
    assert result["gpu_authority_granted"] is True
    assert result["gpu_executed"] is False
    assert result["scoring_authorized"] is False
    assert result["v10_authorized"] is False
    assert result["output_absent"] is True
    assert result["v4_failure_bound"] is True
    assert result["v4_complete_cohort_histories"] == 3
    assert result["configured_max_num_batched_tokens"] == 8192
    assert result["effective_max_num_scheduled_tokens"] == 8160
    assert result["pure_decode_draft_step0_query_width"] == 1
    assert result["prefill_query_width_phase_repaired"] is True
    assert result["probe_result_request_id_canonicalization_repaired"] is True
    assert result["randomized_internal_request_ids_preserved"] is True


@pytest.mark.parametrize(
    "field",
    ["v10_value_screen", "p4a_engineering", "action_admission"],
)
def test_authorization_rejects_broader_authority(field: str):
    authorization = copy.deepcopy(_authorization())
    authorization["authorizations"][field] = True

    with pytest.raises(P4ChunkedPrefillAuthorizationError, match="beyond one"):
        validate_authorization(authorization)


@pytest.mark.parametrize(
    "field",
    [
        "retry_allowed",
        "resume_allowed",
        "reuse_v1_output_allowed",
        "reuse_v2_output_allowed",
        "reuse_v3_output_allowed",
        "reuse_v4_output_allowed",
        "reuse_v9_captures_allowed",
        "fallback_gpu_allowed",
        "score_output",
    ],
)
def test_authorization_rejects_retry_fallback_or_scoring(field: str):
    authorization = copy.deepcopy(_authorization())
    authorization["execution_policy"][field] = True

    with pytest.raises(P4ChunkedPrefillAuthorizationError, match="execution policy"):
        validate_authorization(authorization)


def test_authorization_rejects_full_prefill_budget():
    authorization = copy.deepcopy(_authorization())
    authorization["run_contract"]["engine"]["max_num_batched_tokens"] = 114688

    with pytest.raises(P4ChunkedPrefillAuthorizationError, match="bounded 8192"):
        validate_authorization(authorization)


def test_authorization_rejects_effective_budget_gate_drift():
    authorization = copy.deepcopy(_authorization())
    authorization["run_contract"]["pass_gates"][
        "effective_max_num_scheduled_tokens"
    ] = 8192

    with pytest.raises(P4ChunkedPrefillAuthorizationError, match="separate"):
        validate_authorization(authorization)


def test_authorization_rejects_query_width_phase_gate_drift():
    authorization = copy.deepcopy(_authorization())
    authorization["run_contract"]["pass_gates"][
        "pure_decode_draft_step0_query_width"
    ] = 4081

    with pytest.raises(P4ChunkedPrefillAuthorizationError, match="query phases"):
        validate_authorization(authorization)


def test_authorization_rejects_request_id_gate_drift():
    authorization = copy.deepcopy(_authorization())
    authorization["run_contract"]["pass_gates"][
        "canonical_probe_result_request_ids"
    ] = False

    with pytest.raises(P4ChunkedPrefillAuthorizationError, match="canonical IDs"):
        validate_authorization(authorization)


def test_authorization_rejects_source_drift():
    authorization = copy.deepcopy(_authorization())
    authorization["source_artifacts"]["scheduler"]["sha256"] = "0" * 64

    with pytest.raises(P4ChunkedPrefillAuthorizationError, match="scheduler"):
        validate_authorization(authorization)


def test_authorization_rejects_another_path():
    with pytest.raises(P4ChunkedPrefillAuthorizationError, match="reviewed"):
        validate_authorization(
            _authorization(),
            authorization_path=AUTHORIZATION_PATH.with_name("another.json"),
        )


def test_probe_cells_select_exact_seed0_ingress_members():
    manifest = json.loads(
        (PHASE_DIR / "data" / "p4" / "p4_b0_prompt_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    cells = _probe_cells(manifest)

    assert [cell[0]["capture_id"].rsplit("-", 1)[-1] for cell in cells] == [
        "R4",
        "R5",
        "R5cot",
    ]
    assert all(len(request_ids) == 8 for _, request_ids in cells)
    assert all(request_ids[0].endswith("p000") for _, request_ids in cells)


def test_registered_relative_main_normalizes_paths_before_dispatch(monkeypatch):
    authorization = _authorization()
    registered_argv = authorization["run_contract"]["invocation"]["argv"]
    assert registered_argv[0] == ".venv/bin/python"

    monkeypatch.chdir(REPO_ROOT)
    with (
        mock.patch.object(sys, "argv", registered_argv[1:]),
        mock.patch("run_p4_b0_chunked_prefill_probe.execute_parent") as dispatch,
    ):
        assert main() == 0

    dispatch.assert_called_once_with(
        authorization,
        AUTHORIZATION_PATH.resolve(),
        OUTPUT_DIR.resolve(),
    )


def test_parent_dispatches_exactly_one_child_without_scoring():
    authorization = _authorization()
    data_dir = PHASE_DIR / "data" / "p4"
    calls = []
    with tempfile.TemporaryDirectory(dir=data_dir) as directory:
        output_dir = Path(directory) / "probe-output"

        def launch(argv, **kwargs):
            calls.append((argv, kwargs))
            (output_dir / "probe_result.json").write_text("{}\n", encoding="utf-8")
            return mock.Mock(returncode=0)

        with (
            mock.patch(
                "run_p4_b0_chunked_prefill_probe._child_environment",
                return_value={"CUDA_VISIBLE_DEVICES": "4"},
            ),
            mock.patch("run_p4_b0_value_screen._preflight_native_sampler"),
            mock.patch("run_p4_b0_value_screen._preflight_inprocess_engine_core"),
            mock.patch(
                "run_p4_b0_value_screen._preflight_gpu4_identity_and_idle",
                return_value={
                    "gpu_uuid": "GPU-c9d19019-5065-2353-80a9-f1797eb19d51",
                    "active_compute_processes": 0,
                },
            ),
            mock.patch(
                "run_p4_b0_chunked_prefill_probe.subprocess.run",
                side_effect=launch,
            ),
        ):
            execute_parent(
                authorization,
                AUTHORIZATION_PATH.resolve(),
                output_dir.resolve(),
            )

        assert len(calls) == 1
        argv, kwargs = calls[0]
        assert "--child" in argv
        assert "score" not in " ".join(argv).lower()
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "4"
        assert (output_dir / "preparation.json").is_file()
