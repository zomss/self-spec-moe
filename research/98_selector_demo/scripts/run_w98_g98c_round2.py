# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W98-R2 (G98-C) campaign runner — the corrected cost model, measured.

Round 2 exists because Round 1's model under-predicted quantized composed cells
by 4-23%, one-directionally, with R5 and R5cot NOT RESOLVABLE. Two
specification faults were identified and are fixed in `w98r2_cost_model`; this
runner measures the lattice that identifies them.

Relationship to Round 1
-----------------------

The Round-1 runner is hash-bound as `round1_runner` in the v6 authorization and
its campaign is scored and closed, so it is not edited. It IS imported: the
boot environment, engine arguments, prompt selection, armed-step extraction and
per-regime measurement all come from it unchanged. That is deliberate rather
than lazy -- the preregistration requires Round-2 cost to be measured on
byte-identical prompts so the two rounds are directly comparable, and
re-implementing the measurement path would put that at risk for no benefit.
G98-C hash-binds both runners, so neither can drift underneath the other.

What this runner adds over Round 1
----------------------------------

1. The five-parameter corrected model, fitted per regime from the 15 registered
   fit cells (`w98r2_cost_model`).
2. The per-boot host-load gate. X17 showed a contaminated boot does not read as
   noise -- it CLAMPS the draft chain to a host floor that hides the GPU work,
   inflating cheap configurations toward a common value. That is precisely the
   "composition costs more than predicted" signature Round 2 is trying to
   explain, so an ungated campaign could confirm its own hypothesis. Rejected
   attempts are retained in the record.
3. Two per-boot diagnostics that are recorded but do NOT gate: runqueue wait,
   and the spread across the cheap regimes. X18 produced one boot that passed
   the startup gate and was still clamped; X19 failed to reproduce it in four
   attempts. Neither instrument is calibrated well enough to gate on -- the
   runqueue reading has only ever been ~0, and cheap-regime spread varies 3x
   across cells for legitimate reasons -- but recording them makes such a boot
   auditable after the fact instead of invisible.

Order of operations is the falsifiability barrier and is enforced, not merely
documented: anchors, then the 15 fit cells, then predictions are COMMITTED with
a digest, and only then are the 8 held-out cells booted, followed by closing
anchors and scoring. `commit_predictions` refuses to run if any held-out
measurement already exists, and `score_reveal` refuses to run without a
committed, digest-matching prediction record.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402
import w98_host_load as hostload  # noqa: E402
from w98r2_cost_model import (  # noqa: E402
    COMPOSED_INFLATION,
    W_LAYER_BYTES,
    Z_COVERAGE,
    R2CostModel,
    R2LeverPoint,
    combined_envelope,
    design_strength,
    is_resolvable,
    keep_frac,
    kv_bytes,
)

matrix = r1.matrix

PACKAGE_ID = "w98-g98c-round2-authorization-v1"
AUTHORIZATION_PATH = "research/98_selector_demo/data/w98_g98c_authorization_v1.json"
OUTPUT_PATH = "research/98_selector_demo/data/g98_c"
PREREG_DOC = "research/98_selector_demo/w98r2_prereg.md"
PREREG_MATRIX = "research/98_selector_demo/data/prereg2/w98r2_matrix.json"
PREREG_FIT = "research/98_selector_demo/data/prereg2/w98r2_fit.json"
PREREG_HELDOUT = "research/98_selector_demo/data/prereg2/w98r2_heldout.json"
G98B_RESULT = "research/98_selector_demo/data/g98_b_v6/round1_result.json"
PREDICTIONS_NAME = "d1p_predictions.json"
FIT_DIR = "fit"
HELDOUT_DIR = "heldout"
ANCHOR_PRE_DIR = r1.ANCHOR_PRE_DIR
ANCHOR_POST_DIR = r1.ANCHOR_POST_DIR
# Amendment 1 section 3 requires anchors to be fit-set cells. Both of Round 1's
# anchors are in the Round-2 fit set and neither is held out, so they carry over
# unchanged and sigma_repro stays comparable across rounds.
ANCHORS = r1.ANCHORS
ANCHOR_REPEATS = r1.ANCHOR_REPEATS
EPSILON_ARM = r1.EPSILON_ARM
FIT_COUNT = 15
HELDOUT_COUNT = 8
# Cheap regimes, recorded as a clamp diagnostic. On a quiet box these separate
# by batch; a clamped boot compresses them toward a common floor.
CHEAP_REGIMES = ("R1", "R8", "R6")


class G98CError(RuntimeError):
    """Raised when Round 2 cannot prove its authority or its barrier."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G98CError(message)


def _load_json(path: Path) -> dict[str, Any]:
    return r1._load_json(path)


def _digest(payload: Any) -> str:
    return r1._digest(payload)


def _triple_key(triple: Mapping[str, Any]) -> str:
    return r1._triple_key(triple)


def _config_key(cfg: Mapping[str, Any]) -> str:
    return r1._config_key(cfg)


def fit_profiles() -> list[dict[str, Any]]:
    """The 15 registered fit cells, read from the frozen artifact."""
    cells = _load_json(matrix._repository_path(PREREG_FIT))["fit"]
    _require(len(cells) == FIT_COUNT, f"fit set is not {FIT_COUNT} cells")
    return list(cells)


def heldout_profiles() -> list[dict[str, Any]]:
    """The 8 registered held-out cells, read from the frozen artifact."""
    cells = _load_json(matrix._repository_path(PREREG_HELDOUT))["heldout"]
    _require(len(cells) == HELDOUT_COUNT, f"held-out set is not {HELDOUT_COUNT} cells")
    return list(cells)


def _lattice_check() -> dict[str, Any]:
    """Re-verify at run time what was machine-checked at freeze time.

    Round 1's split failed this: a cell was held out while varying a single
    axis AND sitting in the fit set, and had to be excluded after the fact.
    """
    fit = fit_profiles()
    heldout = heldout_profiles()
    fit_keys = {_triple_key(c) for c in fit}
    heldout_keys = {_triple_key(c) for c in heldout}
    _require(len(fit_keys) == FIT_COUNT, "fit set contains duplicates")
    _require(len(heldout_keys) == HELDOUT_COUNT, "held-out set contains duplicates")
    _require(
        not (fit_keys & heldout_keys),
        f"held-out cells appear in the fit set: {sorted(fit_keys & heldout_keys)}",
    )
    for cell in heldout:
        varied = sum(
            (
                cell["quant"] != "target-matching",
                cell["window"] != "off",
                cell["skip_count"] != 0,
            )
        )
        _require(
            varied >= 2,
            f"held-out cell {_triple_key(cell)} varies fewer than two axes",
        )
    matrix_doc = _load_json(matrix._repository_path(PREREG_MATRIX))
    for cell in fit + heldout:
        _require(
            cell["quant"] in matrix_doc["quant"]
            and cell["window"] in matrix_doc["window"]
            and cell["skip_count"] in matrix_doc["skip_counts"],
            f"cell {_triple_key(cell)} is outside the registered matrix",
        )
        _require(
            cell["skip_count"] in r1.SKIP_SETS,
            f"cell {_triple_key(cell)} has no registered skip set",
        )
    return {
        "fit_count": len(fit_keys),
        "heldout_count": len(heldout_keys),
        "disjoint": True,
        "every_heldout_varies_two_axes": True,
    }


def lever_point(
    cfg: Mapping[str, Any], context_tokens: float, d_measured: float
) -> R2LeverPoint:
    """Build the model's design row for one configuration at one regime."""
    return R2LeverPoint(
        weight_layer_bytes=W_LAYER_BYTES[cfg["quant"]],
        kv_bytes=kv_bytes(cfg["window"], context_tokens),
        keep_frac=keep_frac(cfg["skip_count"]),
        windowed=cfg["window"] != "off",
        d_measured=d_measured,
    )


def expected_authorization() -> dict[str, Any]:
    """The authorization package this runner will accept, and nothing else."""
    sources = {
        "prereg_doc": PREREG_DOC,
        "prereg_matrix": PREREG_MATRIX,
        "prereg_fit": PREREG_FIT,
        "prereg_heldout": PREREG_HELDOUT,
        "round1_runner": "research/98_selector_demo/scripts/run_w98_g98b_round1.py",
        "round1_cost_model": "research/98_selector_demo/scripts/w98_cost_model.py",
        "round2_cost_model": "research/98_selector_demo/scripts/w98r2_cost_model.py",
        "round2_runner": "research/98_selector_demo/scripts/run_w98_g98c_round2.py",
        "host_load_gate": "research/98_selector_demo/scripts/w98_host_load.py",
        "koff_runtime": "vllm/v1/spec_decode/koff_runtime.py",
        "g98b_result": G98B_RESULT,
    }
    source_artifacts = {}
    for name, rel in sources.items():
        path = matrix._repository_path(rel)
        _require(path.is_file(), f"source artifact missing: {rel}")
        source_artifacts[name] = {
            "path": rel,
            "sha256": r1.hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {
        "schema_version": 1,
        "package_id": PACKAGE_ID,
        "gate": "G98-C",
        "prerequisite": {
            "gate": "G98-B",
            "result": G98B_RESULT,
            "status": "scored and closed; its runner and model are not edited",
        },
        "source_artifacts": source_artifacts,
        "runtime": {
            # Preregistration section 6. Pinned here so the scored runtime is
            # part of the record rather than an operator's choice.
            "draft": "piecewise",
            "VLLM_SELF_SPEC_DRAFT_WHOLECHAIN": "0",
            "VLLM_SELF_SPEC_DRAFT_FULLCG": "0",
            "kernel": "Marlin",
            "VLLM_DISABLED_KERNELS": "MacheteLinearKernel",
            "boot_scope": "w98-lattice",
            "registered_cost": (
                "1.21x slower than whole-chain at batch 1, ~1% at batch >= 32"
            ),
        },
        "host_load_gate": {
            "enforced": True,
            "quiet_max_compile_s": hostload.QUIET_MAX_COMPILE_S,
            "quiet_max_capture_s": hostload.QUIET_MAX_CAPTURE_S,
            "max_attempts": hostload.DEFAULT_MAX_ATTEMPTS,
            "rejected_attempts_retained": True,
            "diagnostics_recorded_not_gated": ["runqueue_wait", "cheap_regime_spread"],
        },
        "model": {
            "form": (
                "D = keep*(W_layer*kappa_w + KV*kappa_kv + c_layer + f_win*[w>0]) + F"
            ),
            "parameters": ["kappa_w", "kappa_kv", "c_layer", "f_win", "F"],
            "F_outside_keep_frac": True,
            "F_quant_independent": True,
            "weight_column": "body bytes only",
            "W_layer_bytes": dict(W_LAYER_BYTES),
        },
        "sampling": {
            "measure_tokens": r1.MEASURE_TOKENS,
            "target_armed_steps": r1.TARGET_ARMED_STEPS,
            "min_armed_steps": r1.MIN_ARMED_STEPS,
            "profiler_warmup": r1.PROFILER_WARMUP,
            "content_seeds": [2, 3],
            "seed_note": "cost uses seeds 2,3 so Round-1 and Round-2 are comparable",
        },
        "lattice": _lattice_check(),
        "stages": [
            {"id": "C0", "action": "anchors_pre", "repeats": ANCHOR_REPEATS},
            {"id": "C1", "action": "fit_cells", "count": FIT_COUNT},
            {"id": "C2", "action": "commit_predictions", "barrier": True},
            {"id": "C3", "action": "heldout_cells", "count": HELDOUT_COUNT},
            {"id": "C4", "action": "anchors_post", "repeats": ANCHOR_REPEATS},
            {"id": "C5", "action": "score_reveal"},
        ],
        "d1_prime": {
            "claim": (
                "intervals under amendment 1 cover the held-out composed "
                "configurations with zero false eliminations"
            ),
            "envelope": "sqrt(fit_term^2 + (Z*sigma_repro)^2) * inflation",
            "z_coverage": Z_COVERAGE,
            "composed_inflation": COMPOSED_INFLATION,
            "epsilon_arm": EPSILON_ARM,
            "not_resolvable_rule": (
                "where Z*sigma_repro >= fit_term the regime is reported NOT "
                "RESOLVABLE, not covered"
            ),
            "registered_in_advance": (
                "Round 1 found R5 and R5cot not resolvable on piecewise+Marlin; "
                "if they are again, D1' is reported over the resolvable regimes "
                "and the others are named, not dropped"
            ),
        },
        "execution_policy": {
            "output_dir": OUTPUT_PATH,
            "create_only": True,
            "lane": "lane-a",
            "resumable": "a completed cell is never re-measured",
        },
        "status": "issued",
    }


def validate_authorization(package: Mapping[str, Any]) -> None:
    """Refuse to run under anything but the exact issued package.

    Every source artifact is re-hashed. A change to the model, either runner,
    the gate, the preregistration or the frozen lattice invalidates the
    authorization rather than silently altering what gets measured.
    """
    expected = expected_authorization()
    _require(
        package.get("package_id") == PACKAGE_ID,
        f"authorization is not {PACKAGE_ID}",
    )
    for name, want in expected["source_artifacts"].items():
        got = (package.get("source_artifacts") or {}).get(name)
        _require(got is not None, f"authorization omits source artifact {name}")
        _require(
            got.get("sha256") == want["sha256"],
            f"source artifact {name} changed since the authorization was issued",
        )
    for field in ("runtime", "model", "lattice", "stages", "d1_prime"):
        _require(
            package.get(field) == expected[field],
            f"authorization field {field!r} does not match this runner",
        )
    _require(
        (package.get("host_load_gate") or {}).get("enforced") is True,
        "authorization does not enforce the host-load gate",
    )


# --- measurement -----------------------------------------------------------


def boot_environment(cfg: Mapping[str, Any], trace_path: Path) -> dict[str, str]:
    """The Round-1 boot environment, unchanged.

    Reused rather than restated so the two rounds cannot drift apart in a way
    that would break the registered comparability of their cost numbers.
    """
    return r1.boot_environment(cfg, trace_path)


def measure_config(cfg: Mapping[str, Any], trace_path: Path) -> dict[str, Any]:
    """The Round-1 per-regime measurement, unchanged."""
    return r1.measure_config(cfg, trace_path)


def _cheap_regime_spread(observations: Mapping[str, Any]) -> float | None:
    """Max-minus-min draft chain over the cheap regimes, in seconds.

    Recorded, not gated. On a quiet box these separate by batch; a clamped boot
    compresses them. It flagged X18's bad boot, but it varies threefold across
    cells for legitimate reasons and there is one clamped example to calibrate
    against, which is not enough to threshold.
    """
    values = [
        observations[r]["draft_chain_s"]
        for r in CHEAP_REGIMES
        if observations.get(r, {}).get("draft_chain_s")
    ]
    if len(values) < len(CHEAP_REGIMES):
        return None
    return max(values) - min(values)


def _run_stage_boots(
    stage_dir: Path,
    configs: Sequence[Mapping[str, Any]],
    trace_root: Path,
    authorization_path: Path,
) -> None:
    """Boot each configuration under the host-load gate, pinned to the lane.

    Traces are namespaced by stage: a held-out cell and a fit cell can carry
    the same key, and a shared trace root would have one boot read the other's
    steps.
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

        def boot(attempt: int, cfg=cfg, key=key, target=target) -> str:
            trace = trace_root / f"{key}.attempt{attempt}.jsonl"
            log = stage_dir / f"{key}.attempt{attempt}.log"
            _require(
                not trace.exists(),
                f"trace {trace} already exists; a boot must write its own trace",
            )
            env = matrix._boot_child_environment(boot_environment(cfg, trace))
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
                        json.dumps(dict(cfg)),
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
                f"boot {key} attempt {attempt} failed; output preserved at {log}",
            )
            return log.read_text(encoding="utf-8", errors="replace")

        def discard(label: str, attempt: int, signature, target=target) -> None:
            """Drop a contaminated attempt's measurement, keeping its log.

            The log stays so the rejection is auditable; the measurement goes so
            the retry cannot be mistaken for a completed cell.
            """
            with contextlib.suppress(OSError):
                target.unlink()
            print(f"[g98c] rejected {label} attempt {attempt}: {signature.reason}")

        signature, attempts = hostload.guarded_boot(key, boot, on_reject=discard)
        record = _load_json(target)
        record["host_load"] = signature.as_record()
        record["host_load_attempts"] = attempts
        record["cheap_regime_spread_s"] = _cheap_regime_spread(record["observations"])
        target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def _run_anchor_stage(stage_dir: Path, traces: Path, authorization_path: Path) -> None:
    """Boot each anchor ANCHOR_REPEATS times (amendment 1 section 3)."""
    for repeat in range(ANCHOR_REPEATS):
        for anchor in ANCHORS:
            _run_stage_boots(
                stage_dir / f"r{repeat}",
                [anchor],
                traces / stage_dir.name / f"r{repeat}",
                authorization_path,
            )


def measured_sigma_repro(output_dir: Path) -> dict[str, float]:
    """Log-scale boot-to-boot stdev per regime, max across anchors."""
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
    for by_regime in per_anchor.values():
        for regime, values in by_regime.items():
            if len(values) < 2:
                continue
            logs = [math.log(v) for v in values]
            sigma[regime] = max(sigma.get(regime, 0.0), statistics.stdev(logs))
    _require(bool(sigma), "anchors produced no regime with >= 2 repeats")
    return sigma


# --- fit, commit, reveal, score -------------------------------------------


def fit_and_predict(output_dir: Path) -> dict[str, Any]:
    """Fit per regime from the 15 fit cells and predict the held-out set.

    A regime that cannot be fitted is a LOCAL mask, not a global failure: it
    contributes no prediction and is reported unfitted while the others proceed.
    """
    fit_rows = {
        p.stem: _load_json(p) for p in sorted((output_dir / FIT_DIR).glob("*.json"))
    }
    _require(
        len(fit_rows) == FIT_COUNT,
        f"expected {FIT_COUNT} fit cells, got {len(fit_rows)}",
    )
    heldout = heldout_profiles()
    sigma_repro = measured_sigma_repro(output_dir)
    predictions: dict[str, Any] = {}
    fits: dict[str, Any] = {}
    regimes = sorted({r for row in fit_rows.values() for r in row["observations"]})
    for regime in regimes:
        points, context = [], None
        for row in fit_rows.values():
            obs = row["observations"].get(regime)
            if not obs or not obs.get("draft_chain_s"):
                continue
            context = obs["context_tokens"]
            points.append(
                lever_point(row["config"], obs["context_tokens"], obs["draft_chain_s"])
            )
        sigma = sigma_repro.get(regime)
        if context is None or sigma is None:
            fits[regime] = {
                "fitted": False,
                "reason": "no usable points" if context is None else "no anchor sigma",
            }
            continue
        try:
            model = R2CostModel.fit(points)
        except Exception as exc:  # noqa: BLE001 - a regime may be unidentifiable
            fits[regime] = {"fitted": False, "reason": str(exc)}
            continue
        unlevered = lever_point(
            {"quant": "target-matching", "window": "off", "skip_count": 0}, context, 1.0
        )
        fits[regime] = {
            "fitted": True,
            "kappa_w": model.kappa_w,
            "kappa_kv": model.kappa_kv,
            "c_layer": model.c_layer,
            "f_win": model.f_win,
            "F": model.f_fixed,
            # The registered "irreducible floor" claim, checked rather than
            # asserted: F as a share of the unlevered draft chain.
            "floor_fraction_unlevered": model.floor_fraction(unlevered),
            "fit_term": model.log_envelope,
            "sigma_repro": sigma,
            "z_sigma": Z_COVERAGE * sigma,
            "envelope": combined_envelope(model.log_envelope, sigma),
            "resolvable": is_resolvable(model.log_envelope, sigma),
            "points": len(points),
            "design_strength": design_strength(points),
        }
        for triple in heldout:
            point = lever_point(triple, context, 1.0)
            lo, hi = model.predict_interval(point, sigma_repro=sigma)
            predictions.setdefault(_triple_key(triple), {})[regime] = {
                "lo": lo,
                "hi": hi,
                "point": model.predict(point),
            }
    fitted = [r for r, v in fits.items() if v.get("fitted")]
    _require(
        bool(fitted),
        "no regime produced an identifiable fit; predictions cannot be committed. "
        f"Reasons: { {r: v.get('reason') for r, v in fits.items()} }",
    )
    return {"fits": fits, "predictions": predictions, "fitted_regimes": fitted}


def commit_predictions(
    output_dir: Path, predictions: Mapping[str, Any]
) -> dict[str, Any]:
    """Freeze the held-out predictions before any held-out boot runs."""
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
    keys = {_triple_key(t) for t in heldout_profiles()}
    _require(
        set(predictions) == keys,
        "predictions must cover exactly the registered held-out set",
    )
    record = {
        "schema_version": 1,
        "record_type": "w98r2_d1p_committed_predictions",
        "package_id": PACKAGE_ID,
        "committed_before_reveal": True,
        "heldout_count": len(keys),
        "predictions": dict(predictions),
    }
    record["digest"] = _digest(record["predictions"])
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def require_committed_predictions(output_dir: Path) -> dict[str, Any]:
    """Refuse the reveal unless predictions were committed first."""
    path = output_dir.resolve() / PREDICTIONS_NAME
    _require(
        path.is_file(),
        "held-out reveal refused: no committed predictions. D1' is "
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


def score_reveal(output_dir: Path) -> dict[str, Any]:
    """Compare the committed predictions against the revealed measurements."""
    committed = require_committed_predictions(output_dir)
    revealed = {
        p.stem: _load_json(p) for p in sorted((output_dir / HELDOUT_DIR).glob("*.json"))
    }
    _require(
        len(revealed) == HELDOUT_COUNT,
        f"expected {HELDOUT_COUNT} held-out cells, got {len(revealed)}",
    )
    fits = _load_json(output_dir / "d1p_fits.json")["fits"]
    covered = total = 0
    not_resolvable: list[str] = []
    rows = []
    for row in revealed.values():
        key = _triple_key(row["config"])
        predicted = committed["predictions"].get(key, {})
        for regime, obs in row["observations"].items():
            measured = obs.get("draft_chain_s")
            interval = predicted.get(regime)
            if measured is None or interval is None:
                continue
            if not fits.get(regime, {}).get("resolvable", False):
                # Section-5 rule: report NOT RESOLVABLE, do not count as
                # covered and do not quietly drop it either.
                not_resolvable.append(f"{key}@{regime}")
                continue
            total += 1
            inside = interval["lo"] <= measured <= interval["hi"]
            covered += int(inside)
            rows.append(
                {
                    "cell": key,
                    "regime": regime,
                    "measured_s": measured,
                    "lo": interval["lo"],
                    "hi": interval["hi"],
                    "covered": inside,
                    "rel_error": measured / interval["point"] - 1.0,
                }
            )
    return {
        "schema_version": 1,
        "record_type": "w98r2_d1p_result",
        "package_id": PACKAGE_ID,
        "covered": covered,
        "total": total,
        "coverage": (covered / total) if total else None,
        "not_resolvable": sorted(set(not_resolvable)),
        "resolvable_regimes": sorted(r for r, v in fits.items() if v.get("resolvable")),
        "rows": rows,
    }


# --- entry points ----------------------------------------------------------


def issue_authorization(path: Path) -> dict[str, Any]:
    """Write the authorization package. Create-only, never overwritten."""
    path = path.resolve()
    _require(
        not path.exists(),
        f"authorization {path} already exists; issuing again would rebind hashes",
    )
    package = expected_authorization()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
    return package


def execute(authorization_path: Path, output_dir: Path) -> int:
    """Run C0 through C5, in that order, with the barrier enforced at C2."""
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

    _run_anchor_stage(output_dir / ANCHOR_PRE_DIR, traces, authorization_path)
    _run_stage_boots(output_dir / FIT_DIR, fit_profiles(), traces, authorization_path)
    fitted = fit_and_predict(output_dir)
    (output_dir / "d1p_fits.json").write_text(
        json.dumps(
            {"schema_version": 1, "record_type": "w98r2_d1p_fits", **fitted},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    commit_predictions(output_dir, fitted["predictions"])
    _run_stage_boots(
        output_dir / HELDOUT_DIR, heldout_profiles(), traces, authorization_path
    )
    _run_anchor_stage(output_dir / ANCHOR_POST_DIR, traces, authorization_path)
    result = score_reveal(output_dir)
    (output_dir / "round2_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps({k: result[k] for k in ("covered", "total", "coverage")}, indent=2)
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--issue-authorization", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--measure-config", help=argparse.SUPPRESS)
    parser.add_argument("--stage-dir", help=argparse.SUPPRESS)
    parser.add_argument("--trace", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    authorization = (
        args.authorization or matrix._repository_path(AUTHORIZATION_PATH)
    ).resolve()
    output_dir = (args.output_dir or matrix._repository_path(OUTPUT_PATH)).resolve()
    if args.measure_config:
        # Child: measure one configuration and write it into the stage dir.
        cfg = json.loads(args.measure_config)
        validate_authorization(_load_json(authorization))
        stage_dir = Path(args.stage_dir).resolve()
        stage_dir.mkdir(parents=True, exist_ok=True)
        observations = measure_config(cfg, Path(args.trace).resolve())
        (stage_dir / f"{_config_key(cfg).replace('/', '_')}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "record_type": "w98r2_cell",
                    "package_id": PACKAGE_ID,
                    "config": cfg,
                    "observations": observations,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return 0
    if args.issue_authorization:
        issue_authorization(authorization)
        print(json.dumps({"issued": str(authorization)}, indent=2))
        return 0
    if args.dry_run:
        package = expected_authorization()
        print(
            json.dumps(
                {
                    "package_id": package["package_id"],
                    "lattice": package["lattice"],
                    "stages": [s["id"] for s in package["stages"]],
                    "source_artifacts": sorted(package["source_artifacts"]),
                    "boots": (
                        FIT_COUNT + HELDOUT_COUNT + 2 * ANCHOR_REPEATS * len(ANCHORS)
                    ),
                },
                indent=2,
            )
        )
        return 0
    return execute(authorization, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
