#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""G98-B Round 1: single-lever profiles, factored fit, held-out reveal.

D1 claims the factored cost model eliminates *soundly*. That claim is only
testable if the predictions for the held-out composed set were fixed before
those configurations were measured, so this gate is built around a
**commitment barrier**:

    B1 singles (GPU)  ->  B2 fit + commit (CPU)  ||  B3 reveal (GPU)  ->  B4 score

B3 refuses to run unless B2's predictions exist and are hash-registered, and
refuses if any held-out measurement already exists. B4 refuses to score unless
the predictions it compares against are the committed ones. The barrier is the
gate's reason for existing; without it D1 is unfalsifiable.

Round 1 reads the PASSIVE step trace (VLLM_SELF_SPEC_KOFF_TRACE ->
build_live_step_record), which is scope-aware, not the P4 capture recorder.
The capture path still asserts target/draft weight equality in two places and
would refuse a quantized boot; that is Round 2's problem, not this gate's.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "research/97_composition_runtime/scripts"))
sys.path.insert(0, str(PHASE_DIR / "scripts"))

import run_p4_b0_value_screen as matrix  # noqa: E402

PACKAGE_ID = "w98-g98b-round1-authorization-v6"
AUTHORIZATION_PATH = "research/98_selector_demo/data/w98_g98b_authorization_v6.json"
OUTPUT_PATH = "research/98_selector_demo/data/g98_b_v6"
V1_FAILURE_PATH = "research/98_selector_demo/data/g98_b/failure.json"
PREREG_MATRIX = "research/98_selector_demo/data/prereg/w98_prereg_matrix.json"
PREREG_HELDOUT = "research/98_selector_demo/data/prereg/w98_d1_heldout.json"
PROMPT_MANIFEST = "research/98_selector_demo/data/prereg/w98_prompt_manifest.json"
G98A_RESULT = "research/98_selector_demo/data/g98_a_v4/g98a_result.json"
PREDICTIONS_NAME = "d1_predictions.json"
SINGLES_DIR = "singles"
HELDOUT_DIR = "heldout"
# Amendment 1 section 3: repeat boots of two fit-set anchors, bracketing the
# campaign, supply the measured reproducibility term. PRE runs before the
# singles, POST after the held-out reveal, so drift over the session is
# captured rather than assumed away.
ANCHOR_PRE_DIR = "anchors_pre"
ANCHOR_POST_DIR = "anchors_post"
ANCHORS = [
    {"quant": "target-matching", "window": "off", "skip_count": 0},
    {"quant": "w4a16-quantized", "window": "off", "skip_count": 0},
]
ANCHOR_REPEATS = 3
EPSILON_ARM = 0.015
# Measured at G98-A: a resident W4A16 draft leaves far less shared-KV headroom
# than the target-matching path, so Round 1's shapes are sized against the
# quantized figure rather than the comfortable one.
QUANTIZED_KV_BLOCKS = 22190
PHASE_97_KV_FLOOR = 21682
QUANT_CKPT = {
    "target-matching": (
        "/data/smcho/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/"
        "b968826d9c46dd6066d109eabc6255188de91218"
    ),
    "w4a16-quantized": "/data/smcho/ckpts/Qwen3-8B-W4A16-INT4",
}
# Draft weight bytes per step, from the checkpoints G98-A actually booted.
WEIGHT_BYTES = {"target-matching": 16.4e9, "w4a16-quantized": 6.1e9}
DRAFT_LAYERS = 36
# Qwen3-8B GQA: 8 kv heads x 128 head dim x 2 (K and V) x 2 bytes, per layer.
KV_BYTES_PER_TOKEN = DRAFT_LAYERS * 8 * 128 * 2 * 2
WINDOW_SINKS = 16
# The sets are NESTED (skip4 subset skip8 subset skip16) so that increasing the
# skip count removes a superset of layers and keep_frac stays a clean scalar.
# A non-nested set would confound "how many layers" with "which layers", which
# X1 showed costs up to 5% on its own.
SKIP_SETS = {
    0: "",
    4: "2,4,7,16",
    8: "2,4,7,11,16,20,25,30",
    16: "1,2,4,6,7,9,11,13,16,18,20,23,25,27,30,33",
}
# Sampling depth. The factored model is STATE-INDEXED, so each regime is
# measured at its own registered batch rather than a fixed prompt count --
# otherwise R1 (batch 1) and R6 (batch 32) would both be fit at the wrong
# state. Token depth sets the stability of the mean armed step time, which
# sets the fitted envelope width and therefore D1's coverage.
# Token depth is DERIVED from the step target, not chosen alongside it. Under
# K=4 an armed step commits up to K+1 tokens, so a token budget set
# independently of the step threshold silently caps the achievable step count:
# 256 tokens yields ~51 armed steps, which a 64-step floor then rejects.
TARGET_ARMED_STEPS = 128
COMMITTED_TOKENS_PER_ARMED_STEP = 5
MEASURE_TOKENS = TARGET_ARMED_STEPS * COMMITTED_TOKENS_PER_ARMED_STEP
MIN_ARMED_STEPS = TARGET_ARMED_STEPS // 2
# the profiler drops this many samples before reporting a mean
PROFILER_WARMUP = 20
# batch 1..32 -> K=4, so the chain actually arms and a draft cost exists.
DYNAMIC_K_SCHEDULE = [[1, 32, 4]]


class G98BError(RuntimeError):
    """Raised when Round 1 cannot prove its authority or its barrier."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G98BError(message)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict), f"{path} is not a JSON object")
    return payload


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def single_lever_profiles() -> list[dict[str, Any]]:
    """Return the single-lever configurations the factored model fits from."""
    prereg = _load_json(matrix._repository_path(PREREG_MATRIX))
    lattice = prereg["lattice"]
    base_quant = lattice["quant"][0]
    singles = [{"quant": base_quant, "window": "off", "skip_count": 0}]
    singles += [
        {"quant": base_quant, "window": w, "skip_count": 0}
        for w in lattice["window"]
        if w != "off"
    ]
    singles += [
        {"quant": base_quant, "window": "off", "skip_count": s}
        for s in lattice["skip_counts"]
        if s
    ]
    singles += [{"quant": lattice["quant"][1], "window": "off", "skip_count": 0}]
    return singles


def composed_heldout() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the frozen held-out set into genuinely composed and invalid cells.

    D1 claims a model fit from SINGLE-LEVER profiles covers COMPOSED
    configurations. One frozen cell -- target-matching/woff/skip8 -- varies a
    single axis, so it is not composed and is also in the fit set; predicting
    it would be predicting a point the model was fitted on. The frozen split is
    NOT rewritten, because it was registered before data. The invalid cell is
    excluded from D1 with its reason recorded.

    Returns:
        ``(composed, invalid)`` from the registered held-out set.
    """
    heldout = _load_json(matrix._repository_path(PREREG_HELDOUT))["heldout"]
    composed, invalid = [], []
    for triple in heldout:
        varied = sum(
            (
                triple["quant"] != "target-matching",
                triple["window"] != "off",
                triple["skip_count"] != 0,
            )
        )
        (composed if varied >= 2 else invalid).append(triple)
    return composed, invalid


def expected_authorization() -> dict[str, Any]:
    """Return the exact package this gate will execute under."""
    heldout = _load_json(matrix._repository_path(PREREG_HELDOUT))
    g98a = _load_json(matrix._repository_path(G98A_RESULT))
    return {
        "schema_version": 1,
        "package_id": PACKAGE_ID,
        "gate": "G98-B",
        "status": "authorized_round1_profiles_and_heldout_reveal",
        "source_artifacts": {
            role: matrix._file_reference(path)
            for role, path in {
                "prereg_matrix": PREREG_MATRIX,
                "prereg_heldout": PREREG_HELDOUT,
                "prompt_manifest": PROMPT_MANIFEST,
                "g98a_result": G98A_RESULT,
                "round1_runner": (
                    "research/98_selector_demo/scripts/run_w98_g98b_round1.py"
                ),
                "round1_tests": ("research/98_selector_demo/tests/test_w98_g98b.py"),
                "cost_model": ("research/98_selector_demo/scripts/w98_cost_model.py"),
                "koff_runtime": "vllm/v1/spec_decode/koff_runtime.py",
            }.items()
        },
        "prerequisite": {
            "gate": "G98-A",
            "passed": g98a["passed"] == g98a["total"],
            "quantized_draft_with_shared_kv_verified": g98a[
                "quantized_draft_with_shared_kv_verified"
            ],
        },
        "consumed_v1": {
            "failure": matrix._file_reference(V1_FAILURE_PATH),
            "stage_reached": "B2 fit; refused before commit",
            "predictions_committed": False,
            "heldout_measured": 0,
            "cause": "trace reader read exclusions/elapsed_s, not "
            "exclusion_reasons/counters.decode_time_s",
            "barrier_held": True,
            "retracted_second_cause": "the draft armed normally; 365/405 steps "
            "were target-matching-k4",
        },
        "commitment_barrier": {
            "why": (
                "D1 claims sound elimination, which is testable only if the "
                "held-out predictions were fixed before those configurations "
                "were measured"
            ),
            "predictions_committed_before_reveal": True,
            "reveal_refuses_without_committed_predictions": True,
            "reveal_refuses_if_heldout_measurements_exist": True,
            "score_refuses_uncommitted_predictions": True,
        },
        "stages": [
            {
                "id": "B1",
                "name": "singles",
                "gpu": True,
                "boots": len(single_lever_profiles()),
            },
            {"id": "B2", "name": "fit_and_commit", "gpu": False, "boots": 0},
            {
                "id": "B3",
                "name": "heldout_reveal",
                "gpu": True,
                "boots": len(heldout["heldout"]),
            },
            {"id": "B4", "name": "score", "gpu": False, "boots": 0},
        ],
        "amendment_1_envelope": {
            "approved": "2026-08-13",
            "rule": "sqrt(fit_term^2 + (Z*sigma_repro)^2) * INFLATION",
            "z_coverage": 2.0,
            "inflation": 2.0,
            "sigma_repro_source": "3 repeats of 2 fit-set anchors, bracketing",
            "validity_condition": (
                "Z*sigma_repro >= fit_term -> regime is NOT RESOLVABLE, not "
                "covered; a noisier campaign must not buy an easier D1"
            ),
        },
        "runtime": {
            "class": "piecewise",
            "wholechain": False,
            "fullcg": False,
            "w4a16_kernel": "MarlinLinearKernel",
            "why": (
                "Option A (X15): piecewise carries no window>0 constraint, so "
                "skip and quant are sampled on the same runtime as the rest and "
                "the fit is identifiable. Marlin makes it resolvable -- it cut "
                "sigma_repro 3-10x vs Machete. Price: 1.21x slower than "
                "whole-chain at batch 1, ~1% at batch >= 32"
            ),
        },
        "measurement": {
            "d_measured": "draft_chain",
            "source": "VLLM_SELF_SPEC_PROFILE region timings",
            "why": (
                "the step trace carries only whole-step time, 61-81% of which "
                "is the target verify -- a constant the model would then scale "
                "by keep_frac, which produced a one-directional underprediction "
                "tracking skip depth (1.05x at skip4, 1.44x at skip8)"
            ),
            "verify_recorded_as_control": True,
            "profiler_warmup": PROFILER_WARMUP,
        },
        "d1": {
            "epsilon_arm": EPSILON_ARM,
            "elimination_rule": "(K+1)/q_lo < 1 + epsilon_arm",
            "false_elimination_budget": 0,
            "heldout_count": heldout["heldout_count"],
            "composed_scored": 7,
            "excluded_invalid": 1,
            "exclusion_reason": (
                "target-matching/woff/skip8 varies a single axis, so it is not "
                "composed and is also a fit point; the frozen split is not "
                "rewritten"
            ),
            "fit_uses_single_lever_profiles_only": True,
            "on_coverage_failure": "local non-pruning mask, not global failure",
        },
        "w14d_fallback": {
            "phase_96_w14d_scored_surface_present": False,
            "verified_on": "2026-08-11",
            "consequence": (
                "affected strata run Round 1 in measure-everything mode and D1 "
                "is reported as NOT EXERCISED there, per the preregistration"
            ),
        },
        "trace_path": {
            "uses": "passive koff step trace (VLLM_SELF_SPEC_KOFF_TRACE)",
            "scope_aware": True,
            "uses_p4_capture_recorder": False,
            "note": (
                "the P4 capture path still asserts target/draft weight equality "
                "at koff_runtime.py:1927 and :2503 and would refuse a quantized "
                "boot; Round 2 must resolve that, Round 1 does not touch it"
            ),
        },
        "sampling": {
            "prompts_per_regime": "the regime's registered batch",
            "state_indexed": True,
            "target_armed_steps": TARGET_ARMED_STEPS,
            "measure_tokens": MEASURE_TOKENS,
            "tokens_derived_from_step_target": True,
            "min_armed_steps_for_a_usable_mean": MIN_ARMED_STEPS,
            "rationale": (
                "the factored model is state-indexed, so each regime is "
                "measured at its own batch; token depth sets the stability of "
                "the mean, which sets the envelope width and hence coverage"
            ),
        },
        "resource_note": {
            "quantized_shared_kv_blocks": QUANTIZED_KV_BLOCKS,
            "phase_97_floor": PHASE_97_KV_FLOOR,
            "headroom_fraction": round(QUANTIZED_KV_BLOCKS / PHASE_97_KV_FLOOR - 1, 4),
            "shapes_sized_against": "quantized",
        },
        "execution_policy": {
            "lane": matrix.lane_for_block(1),
            "boot_scope": "w98-lattice",
            "create_new_output_required": True,
            "retry_allowed": False,
            "on_any_failure": "preserve_and_require_fresh_authorization",
        },
        "authorizations": {
            "gpu_measurement": True,
            "round1_execution": True,
            "round1_scoring": True,
            "round2_execution": False,
            "round2_scoring": False,
            "d3_claim": False,
            "production_value_claim": False,
        },
        "next_artifact": {
            "kind": "w98_round1_result",
            "may_authorize_round2": False,
        },
    }


def validate_authorization(package: Mapping[str, Any]) -> None:
    """Refuse anything but the exact registered Round-1 gate.

    Args:
        package: The parsed G98-B authorization.

    Raises:
        G98BError: If any registered field or source hash drifted.
    """
    expected = expected_authorization()
    _require(set(package) == set(expected), "G98-B authorization fields drifted")
    for key, value in expected.items():
        _require(package[key] == value, f"G98-B authorization drifted at {key}")
    _require(
        package["prerequisite"]["passed"],
        "G98-B requires a passing G98-A",
    )
    _require(
        package["prerequisite"]["quantized_draft_with_shared_kv_verified"],
        "G98-B requires the verified quantized-draft assumption",
    )
    prereg = _load_json(matrix._repository_path(PREREG_MATRIX))
    _require(
        prereg["status"] == "frozen_before_any_scored_data_manifest_committed",
        "G98-B requires the completed preregistration",
    )
    _require(
        prereg["tolerances"]["epsilon_arm"] == EPSILON_ARM,
        "G98-B epsilon_arm differs from the preregistration",
    )


def commit_predictions(
    output_dir: Path, predictions: Mapping[str, Any]
) -> dict[str, Any]:
    """Freeze the held-out predictions before any held-out boot runs.

    Args:
        output_dir: The create-only Round-1 output directory.
        predictions: One entry per held-out triple.

    Returns:
        The committed record, including its own digest.

    Raises:
        G98BError: If predictions already exist, or any held-out measurement
            has already been taken.
    """
    output_dir = output_dir.resolve()
    path = output_dir / PREDICTIONS_NAME
    _require(
        not path.exists(),
        "held-out predictions are already committed and are immutable",
    )
    heldout_dir = output_dir / HELDOUT_DIR
    existing = sorted(heldout_dir.glob("*.json")) if heldout_dir.is_dir() else []
    _require(
        not existing,
        "refusing to commit predictions after held-out measurements exist: "
        f"{[p.name for p in existing]}",
    )
    registered = _load_json(matrix._repository_path(PREREG_HELDOUT))["heldout"]
    keys = {_triple_key(t) for t in registered}
    _require(
        set(predictions) == keys,
        "predictions must cover exactly the registered held-out set",
    )
    record = {
        "schema_version": 1,
        "record_type": "w98_d1_committed_predictions",
        "package_id": PACKAGE_ID,
        "committed_before_reveal": True,
        "heldout_count": len(keys),
        "predictions": dict(predictions),
    }
    record["digest"] = _digest(record["predictions"])
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def require_committed_predictions(output_dir: Path) -> dict[str, Any]:
    """Refuse the reveal unless predictions were committed first.

    Args:
        output_dir: The Round-1 output directory.

    Returns:
        The committed predictions record.

    Raises:
        G98BError: If predictions are missing or their digest drifted.
    """
    path = output_dir.resolve() / PREDICTIONS_NAME
    _require(
        path.is_file(),
        "held-out reveal refused: no committed predictions. D1 is "
        "unfalsifiable without them.",
    )
    record = _load_json(path)
    _require(
        record.get("committed_before_reveal") is True,
        "committed predictions do not claim pre-reveal commitment",
    )
    _require(
        record.get("digest") == _digest(record.get("predictions", {})),
        "committed predictions were modified after commitment",
    )
    return record


def _triple_key(triple: Mapping[str, Any]) -> str:
    return f"{triple['quant']}/w{triple['window']}/skip{triple['skip_count']}"


def _config_key(cfg: Mapping[str, Any]) -> str:
    return f"{cfg['quant']}/w{cfg['window']}/skip{cfg['skip_count']}"


def lever_geometry(cfg: Mapping[str, Any], context_tokens: int) -> dict[str, float]:
    """Return the model's inputs for one configuration at one state.

    Args:
        cfg: A lattice configuration.
        context_tokens: The state's context length, which sets the KV read when
            the window is off.

    Returns:
        ``weight_bytes``, ``kv_bytes`` and ``keep_frac`` for the factored model.
    """
    window = cfg["window"]
    kv_tokens = (
        context_tokens
        if window == "off"
        else min(context_tokens, int(window) + WINDOW_SINKS)
    )
    return {
        "weight_bytes": WEIGHT_BYTES[cfg["quant"]],
        "kv_bytes": float(kv_tokens * KV_BYTES_PER_TOKEN),
        "keep_frac": 1.0 - cfg["skip_count"] / DRAFT_LAYERS,
    }


def boot_environment(cfg: Mapping[str, Any], trace_path: Path) -> dict[str, str]:
    """Build the lane-pinned, trace-enabled environment for one boot."""
    base = _load_json(matrix._repository_path(matrix.BASE_AUTHORIZATION_PATH))[
        "run_contract"
    ]["environment"]
    lane = matrix.lane_for_block(1)
    env = {k: str(v) for k, v in base.items()}
    env.pop(matrix.DEVICE_PIN_ENV, None)
    if env.get("VLLM_SELF_SPEC_DRAFT_PARTIAL_REPLICA") == "0":
        env["VLLM_SELF_SPEC_DRAFT_PARTIAL_REPLICA"] = ""
    env.update(
        {
            matrix.DEVICE_PIN_ENV: str(lane["physical_gpu_index"]),
            matrix.CACHE_ROOT_ENV: lane["cache_root"],
            matrix.V1_MULTIPROCESSING_ENV: matrix.V1_MULTIPROCESSING_VALUE,
            matrix.NATIVE_SAMPLER_ENV: matrix.NATIVE_SAMPLER_VALUE,
            "VLLM_SELF_SPEC_BOOT_SCOPE": "w98-lattice",
            "VLLM_SELF_SPEC_SHARED_KV": "1",
            "VLLM_SELF_SPEC_SHARE_WEIGHTS": (
                "1" if cfg["quant"] == "target-matching" else "0"
            ),
            "VLLM_SELF_SPEC_DRAFT_KV_WINDOW": (
                "0" if cfg["window"] == "off" else str(cfg["window"])
            ),
            "VLLM_SELF_SPEC_DRAFT_KV_SINKS": (
                "0" if cfg["window"] == "off" else str(WINDOW_SINKS)
            ),
            "VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS": SKIP_SETS[cfg["skip_count"]],
            "VLLM_SELF_SPEC_KOFF_TRACE": str(trace_path),
            # Option A (X15): piecewise + Marlin. Pinned here rather than left
            # to the operator so the scored runtime is part of the record.
            "VLLM_SELF_SPEC_DRAFT_WHOLECHAIN": "0",
            "VLLM_SELF_SPEC_DRAFT_FULLCG": "0",
            "VLLM_DISABLED_KERNELS": "MacheteLinearKernel",
            # draft_chain is the draft-step cost the factored model wants. The
            # step trace only carries whole-step time, 61-81% of which is the
            # target verify -- a constant the model would then scale by keep_frac.
            "VLLM_SELF_SPEC_PROFILE": "1",
        }
    )
    return env


def _prompts_for(regime_id: str, limit: int) -> list[list[int]]:
    bundle = matrix._repository_path(
        "research/98_selector_demo/data/prereg/w98_prompt_tokens.jsonl.gz"
    )
    rows = []
    with gzip.open(bundle, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["regime_id"] == regime_id:
                rows.append(row["token_ids"])
            if len(rows) >= limit:
                break
    _require(bool(rows), f"no prompts for regime {regime_id}")
    return rows


def measure_config(cfg: Mapping[str, Any], trace_path: Path) -> dict[str, Any]:
    """Boot one configuration and record its armed step costs per regime.

    ``d_measured`` is the mean armed engine-step time. The target verify cost
    is identical across configurations, so its constant offset is absorbed by
    the model's ``c0`` term and differences across configurations reflect draft
    cost. Any consistent unit satisfies the model; this one is consistent by
    construction because every configuration runs the same prompts.

    Args:
        cfg: The configuration to boot.
        trace_path: Where the passive koff step trace is written.

    Returns:
        Per-regime mean armed step time and context length.
    """
    from vllm import LLMEngine, SamplingParams

    manifest = _load_json(matrix._repository_path(PROMPT_MANIFEST))
    regimes = {r["regime_id"]: r for r in manifest["prompt_plan"]["regimes"]}
    from vllm.v1.spec_decode.self_spec_profiler import get_profiler

    profiler = get_profiler()
    _require(profiler.enabled, "the self-spec profiler is not enabled")
    engine = LLMEngine.from_engine_args(_engine_args(cfg))
    observations: dict[str, Any] = {}
    try:
        for regime_id, spec in regimes.items():
            prompts = _prompts_for(regime_id, spec["batch"])
            before = _trace_len(trace_path)
            profiler.reset()
            for index, tokens in enumerate(prompts):
                engine.add_request(
                    f"{regime_id}-{index}",
                    {"prompt_token_ids": tokens},
                    SamplingParams(
                        temperature=0.0, max_tokens=MEASURE_TOKENS, ignore_eos=True
                    ),
                )
            while engine.has_unfinished_requests():
                engine.step()
            steps = _armed_steps(trace_path, before)
            summary = profiler.summary(warmup=PROFILER_WARMUP)
            chain = summary.get("draft_chain", {})
            verify = summary.get("verify", {})
            usable = (
                len(steps) >= MIN_ARMED_STEPS
                and bool(chain.get("n"))
                and chain.get("mean_ms") is not None
            )
            observations[regime_id] = {
                "armed_step_count": len(steps),
                "usable": usable,
                # d_measured: the DRAFT chain alone, not the whole engine step.
                "draft_chain_s": (
                    chain["mean_ms"] / 1000.0 if chain.get("mean_ms") else None
                ),
                "draft_chain_samples": chain.get("n", 0),
                # Recorded as an independent check: no lever touches the
                # target, so verify should be roughly constant across the
                # single-lever configurations at a fixed regime.
                "verify_s": (
                    verify["mean_ms"] / 1000.0 if verify.get("mean_ms") else None
                ),
                "mean_armed_step_s": statistics.mean(steps) if steps else None,
                "stdev_armed_step_s": (
                    statistics.stdev(steps) if len(steps) > 1 else None
                ),
                "prompt_tokens": int(statistics.mean(len(t) for t in prompts)),
                # The draft's KV grows during decode, so the mean KV length
                # over the measured steps is prompt + generated/2. Using the
                # prompt alone left every window above the context, which
                # collapsed the window axis and made the design singular.
                "context_tokens": int(
                    statistics.mean(len(t) for t in prompts) + MEASURE_TOKENS / 2
                ),
                "batch": spec["batch"],
            }
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()
    return observations


def _trace_len(trace_path: Path) -> int:
    if not trace_path.is_file():
        return 0
    with trace_path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _armed_steps(trace_path: Path, skip: int) -> list[float]:
    """Return elapsed times for armed pure-decode steps written after ``skip``."""
    if not trace_path.is_file():
        return []
    out = []
    with trace_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < skip:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("record_type") != "koff_engine_step":
                continue
            # The record names these `exclusion_reasons` and nests timing under
            # counters; reading top-level `exclusions`/`elapsed_s` silently
            # collected nothing at all.
            if row.get("exclusion_reasons"):
                continue
            counters = row.get("counters") or {}
            if not counters.get("D_armed"):
                continue
            elapsed = counters.get("decode_time_s")
            if isinstance(elapsed, (int, float)) and elapsed > 0:
                out.append(float(elapsed))
    return out


def _engine_args(cfg: Mapping[str, Any]):
    from vllm import EngineArgs

    base = _load_json(matrix._repository_path(matrix.BASE_AUTHORIZATION_PATH))[
        "run_contract"
    ]["engine"]
    return EngineArgs(
        model=QUANT_CKPT["target-matching"],
        speculative_config={
            "method": "draft_model",
            "model": QUANT_CKPT[cfg["quant"]],
            "num_speculative_tokens": 4,
            # Without the per-batch schedule the K/OFF policy never selects K4,
            # so every step is OFF and Round 1 measures no draft cost at all.
            "num_speculative_tokens_per_batch_size": DYNAMIC_K_SCHEDULE,
            "draft_tensor_parallel_size": 1,
        },
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=base["max_model_len"],
        max_num_batched_tokens=matrix.SERVING_MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=base["max_num_seqs"],
        enable_chunked_prefill=True,
        gpu_memory_utilization=matrix.SERVING_GPU_MEMORY_UTILIZATION,
        enable_prefix_caching=False,
        async_scheduling=False,
        enforce_eager=False,
        enable_flashinfer_autotune=False,
        seed=0,
        disable_log_stats=True,
        generation_config="vllm",
    )


def _run_anchor_stage(stage_dir: Path, traces: Path, authorization_path: Path) -> None:
    """Boot each anchor ANCHOR_REPEATS times (amendment 1 section 3)."""
    for repeat in range(ANCHOR_REPEATS):
        for anchor in ANCHORS:
            tagged = dict(anchor)
            tagged["_repeat"] = repeat
            _run_stage_boots(
                stage_dir / f"r{repeat}",
                [anchor],
                traces / stage_dir.name / f"r{repeat}",
                authorization_path,
            )


def measured_sigma_repro(output_dir: Path) -> dict[str, float]:
    """Log-scale boot-to-boot stdev per regime, max across anchors.

    Amendment 1 section 3: anchors are fit-set cells (never held out), repeats
    bracket the campaign, and the maximum across anchors is taken rather than
    the mean.
    """
    per_anchor: dict[str, dict[str, list[float]]] = {}
    for stage in (ANCHOR_PRE_DIR, ANCHOR_POST_DIR):
        for path in sorted((output_dir / stage).rglob("*.json")):
            record = _load_json(path)
            key = _config_key(record["config"])
            for regime, obs in record["observations"].items():
                value = obs.get("draft_chain_s")
                if value:
                    per_anchor.setdefault(key, {}).setdefault(regime, []).append(value)
    _require(bool(per_anchor), "no anchor measurements found")
    sigma: dict[str, float] = {}
    for byreg in per_anchor.values():
        for regime, values in byreg.items():
            if len(values) < 2:
                continue
            logs = [math.log(v) for v in values]
            sigma[regime] = max(sigma.get(regime, 0.0), statistics.stdev(logs))
    _require(bool(sigma), "anchors produced no regime with >= 2 repeats")
    return sigma


def fit_and_predict(output_dir: Path) -> dict[str, Any]:
    """Fit the factored model per regime from singles and predict the held-out.

    The fit consumes ONLY single-lever measurements. Predictions carry the
    inflated symmetric log envelope the harness defines for composed use.
    """
    from w98_cost_model import FactoredCostModel, LeverPoint

    singles = {
        p.stem: _load_json(p) for p in sorted((output_dir / SINGLES_DIR).glob("*.json"))
    }
    _require(
        len(singles) == len(single_lever_profiles()),
        f"expected {len(single_lever_profiles())} single profiles, got {len(singles)}",
    )
    heldout = _load_json(matrix._repository_path(PREREG_HELDOUT))["heldout"]
    sigma_repro = measured_sigma_repro(output_dir)
    predictions: dict[str, Any] = {}
    fits: dict[str, Any] = {}
    regimes = sorted({r for row in singles.values() for r in row["observations"]})
    for regime in regimes:
        points, context = [], None
        for row in singles.values():
            obs = row["observations"].get(regime)
            if not obs or not obs.get("draft_chain_s"):
                continue
            context = obs["context_tokens"]
            geom = lever_geometry(row["config"], obs["context_tokens"])
            points.append(
                LeverPoint(
                    weight_bytes=geom["weight_bytes"],
                    kv_bytes=geom["kv_bytes"],
                    keep_frac=geom["keep_frac"],
                    d_measured=obs["draft_chain_s"],
                )
            )
        if len(points) < 3 or context is None:
            fits[regime] = {"fitted": False, "reason": "fewer than three points"}
            continue
        try:
            model = FactoredCostModel.fit(points)
        except Exception as exc:  # noqa: BLE001 - a regime may be unidentifiable
            # Per the preregistration a local failure is a local mask, not a
            # global one: this regime contributes no prediction and is reported
            # as unfitted, while the others proceed.
            fits[regime] = {"fitted": False, "reason": str(exc)}
            continue
        from w98_cost_model import combined_envelope, is_resolvable

        sigma = sigma_repro.get(regime)
        if sigma is None:
            fits[regime] = {"fitted": False, "reason": "no anchor reproducibility"}
            continue
        fits[regime] = {
            "fitted": True,
            "kappa_w": model.kappa_w,
            "kappa_kv": model.kappa_kv,
            "c0": model.c0,
            # Amendment 1: both terms recorded so a reader can see which binds.
            "fit_term": model.log_envelope,
            "sigma_repro": sigma,
            "z_sigma": 2.0 * sigma,
            "envelope": combined_envelope(model.log_envelope, sigma),
            "resolvable": is_resolvable(model.log_envelope, sigma),
            "points": len(points),
        }
        for triple in heldout:
            geom = lever_geometry(triple, context)
            lo, hi = model.predict_interval(
                geom["weight_bytes"],
                geom["kv_bytes"],
                geom["keep_frac"],
                sigma_repro=sigma,
            )
            predictions.setdefault(_triple_key(triple), {})[regime] = {
                "lo": lo,
                "hi": hi,
                "point": model.predict(
                    geom["weight_bytes"], geom["kv_bytes"], geom["keep_frac"]
                ),
            }
    fitted = [r for r, v in fits.items() if v.get("fitted")]
    _require(
        bool(fitted),
        "no regime produced an identifiable fit; predictions cannot be "
        f"committed. Reasons: { {r: v.get('reason') for r, v in fits.items()} }",
    )
    return {"fits": fits, "predictions": predictions, "fitted_regimes": fitted}


def score_reveal(output_dir: Path) -> dict[str, Any]:
    """Compare the committed predictions against the revealed measurements."""
    committed = require_committed_predictions(output_dir)
    revealed = {
        p.stem: _load_json(p) for p in sorted((output_dir / HELDOUT_DIR).glob("*.json"))
    }
    covered = total = 0
    rows = []
    for row in revealed.values():
        key = _config_key(row["config"])
        predicted = committed["predictions"].get(key, {})
        for regime, obs in row["observations"].items():
            band = predicted.get(regime)
            if not band or not obs.get("draft_chain_s"):
                continue
            total += 1
            inside = band["lo"] <= obs["draft_chain_s"] <= band["hi"]
            covered += inside
            rows.append(
                {
                    "cell": f"{key}/{regime}",
                    "measured": obs["draft_chain_s"],
                    "lo": band["lo"],
                    "hi": band["hi"],
                    "covered": inside,
                }
            )
    composed, invalid = composed_heldout()
    fits = _load_json(output_dir / "d1_fits.json")
    unresolvable = sorted(
        r for r, v in fits.items() if v.get("fitted") and not v.get("resolvable")
    )
    scored = [c for c in rows if c["cell"].rsplit("/", 1)[1] not in unresolvable]
    return {
        "schema_version": 1,
        "record_type": "w98_round1_result",
        # Amendment 1 section 5: a regime whose measurement floor dominates its
        # fit residual is NOT RESOLVABLE, not covered. Reported separately so a
        # noisier campaign cannot buy an easier D1.
        "amendment_1": {
            "z_coverage": 2.0,
            "inflation": 2.0,
            "unresolvable_regimes": unresolvable,
            "coverage_resolvable_only": {
                "covered": sum(1 for c in scored if c["covered"]),
                "total": len(scored),
            },
        },
        "heldout_scope": {
            "registered": len(composed) + len(invalid),
            "composed_scored": len(composed),
            "excluded_invalid": [_triple_key(t) for t in invalid],
            "exclusion_reason": (
                "varies a single axis, so it is not a composed configuration "
                "and is also a fit point; the frozen split is not rewritten"
            ),
        },
        "package_id": PACKAGE_ID,
        "scored": True,
        "predictions_committed_before_reveal": True,
        "coverage": {"covered": covered, "total": total},
        "cells": rows,
        "d1_exercised": False,
        "d1_note": (
            "W14/D has no scored surface, so per the preregistration D1 is "
            "reported as NOT EXERCISED; this run reports coverage only"
        ),
        "may_authorize_round2": False,
    }


def parse_args() -> argparse.Namespace:
    """Parse the Round-1 interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--measure-config", help=argparse.SUPPRESS)
    parser.add_argument("--stage-dir", help=argparse.SUPPRESS)
    parser.add_argument("--trace", help=argparse.SUPPRESS)
    return parser.parse_args()


def _run_stage_boots(
    stage_dir: Path, configs, trace_root: Path, authorization_path: Path
) -> None:
    """Boot each configuration in its own child, pinned to the reserved lane.

    Traces are namespaced by stage: a held-out cell and a single-lever profile
    can carry the same key, and a shared trace root would have one boot read
    the other's steps.
    """
    stage_dir.mkdir(parents=True, exist_ok=True)
    trace_root = trace_root / stage_dir.name
    trace_root.mkdir(parents=True, exist_ok=True)
    lane = matrix.lane_for_block(1)
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = sorted(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        key = _config_key(cfg).replace("/", "_")
        target = stage_dir / f"{key}.json"
        if target.exists():
            continue
        trace = trace_root / f"{key}.jsonl"
        _require(
            not trace.exists(),
            f"trace {trace} already exists; a boot must write its own trace",
        )
        env = matrix._boot_child_environment(boot_environment(cfg, trace))
        log = stage_dir / f"{key}.log"
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--authorization",
                    str(authorization_path),
                    "--output-dir",
                    str(stage_dir.parent),
                    "--measure-config",
                    json.dumps(cfg),
                    "--stage-dir",
                    str(stage_dir),
                    "--trace",
                    str(trace),
                ],
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                preexec_fn=lambda: os.sched_setaffinity(0, affinity),
            )
        _require(
            completed.returncode == 0 and target.exists(),
            f"boot {key} failed; output preserved at {log}",
        )


def execute(authorization_path: Path, output_dir: Path) -> int:
    """Run B1, commit at B2, reveal at B3 and score at B4, in that order."""
    output_dir = output_dir.resolve()
    authorization_path = authorization_path.resolve()
    validate_authorization(_load_json(authorization_path))
    base_env = matrix._boot_child_environment({})
    matrix._preflight_native_sampler(base_env)
    matrix._preflight_inprocess_engine_core(base_env)
    lane = matrix.lane_for_block(1)
    matrix._preflight_gpu_identity_and_idle(
        lane["physical_gpu_index"], lane["physical_gpu_uuid"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    traces = output_dir / "traces"
    traces.mkdir(exist_ok=True)

    # B0 -- PRE anchors. Amendment 1 section 3 requires the reproducibility
    # repeats to bracket the campaign; back-to-back repeats understate drift.
    _run_anchor_stage(output_dir / ANCHOR_PRE_DIR, traces, authorization_path)

    # B1 -- singles only. The held-out set must not be touched before B2.
    _run_stage_boots(
        output_dir / SINGLES_DIR, single_lever_profiles(), traces, authorization_path
    )

    # B2 -- fit and COMMIT before any held-out boot exists. commit_predictions
    # refuses if the reveal directory already holds measurements.
    fitted = fit_and_predict(output_dir)
    (output_dir / "d1_fits.json").write_text(
        json.dumps(fitted["fits"], indent=2, sort_keys=True) + "\n"
    )
    commit_predictions(output_dir, fitted["predictions"])

    # B3 -- reveal, gated on the commitment.
    require_committed_predictions(output_dir)
    composed, invalid = composed_heldout()
    _run_stage_boots(output_dir / HELDOUT_DIR, composed, traces, authorization_path)

    # B3b -- POST anchors, closing the bracket.
    _run_anchor_stage(output_dir / ANCHOR_POST_DIR, traces, authorization_path)

    # B4 -- score against the committed predictions.
    result = score_reveal(output_dir)
    (output_dir / "round1_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result["coverage"], indent=2, sort_keys=True))
    return 0


def main() -> int:
    """Validate the gate, then run its four stages in order."""
    args = parse_args()
    validate_authorization(_load_json(args.authorization.resolve()))
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "mode": "validate_only",
                    "gpu_executed": False,
                    "singles": len(single_lever_profiles()),
                    "heldout": len(
                        _load_json(matrix._repository_path(PREREG_HELDOUT))["heldout"]
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.measure_config is not None:
        cfg = json.loads(args.measure_config)
        record = {
            "schema_version": 1,
            "record_type": "w98_round1_measurement",
            "config": cfg,
            "observations": measure_config(cfg, Path(args.trace)),
        }
        key = _config_key(cfg).replace("/", "_")
        Path(args.stage_dir).joinpath(f"{key}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
        return 0
    return execute(args.authorization, args.output_dir)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (G98BError, matrix.P4RunnerError) as exc:
        print(f"G98-B refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
