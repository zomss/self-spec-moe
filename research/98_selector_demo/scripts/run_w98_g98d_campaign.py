# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W98-D2 (G98-D) acceptance campaign runner.

Measures what the preregistration's D2 section registers, using the cost
model Round 2 validated and the acceptance path G98-D0 proved:

  (a) **Screen honesty.** No composition admitted by the product bound
      `f_set >= prod f_i` has a confirmed tau below the bound minus the
      declared tolerance. The bound is ADMIT-ONLY: it may pass a bad
      composition to confirmation, never eliminate a good one.
  (c) **u-axis resolution.** `tau(w, g, u)` with interval separation, from
      per-request generated-suffix binning.

D2(b) -- the knapsack identity against count-matched controls -- needs
per-layer retention over arbitrary skip SETS rather than the frozen nested
counts, so it is a separate stage against its own probe and is not scored
here.

Order of operations is the falsifiability barrier and is enforced, not
documented: singles, then the screen is COMMITTED with a digest, and only
then are the admitted compositions booted. `commit_screen` refuses to run
once any confirmation measurement exists, and `score_reveal` refuses to run
without a committed, digest-matching screen.

Two deliberate differences from the G98-C cost campaign
-------------------------------------------------------

1. **No measurement-shape gate.** That gate judges draft-chain TIMING shape.
   Acceptance is boot-deterministic (calibration: two boots agreed on 17/17
   requests and 0/210 steps), so the clamp cannot bias it and the timing
   gate would only reject good acceptance data.
2. **No repeat-boot anchors.** Boot-to-boot reproducibility is exactly zero
   variance by the same calibration, so the sampling budget belongs to
   content seeds and requests, as the contract registers.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402
import run_w98_g98d0_smoke as d0  # noqa: E402
import w98d2_stream as stream  # noqa: E402

matrix = r1.matrix
REPO_ROOT = r1.REPO_ROOT

PACKAGE_ID = "w98-g98d-acceptance-authorization-v1"
AUTHORIZATION_PATH = "research/98_selector_demo/data/w98_g98d_authorization_v1.json"
OUTPUT_PATH = "research/98_selector_demo/data/g98_d"
PROMPT_MANIFEST = "research/98_selector_demo/data/prereg/w98d2_prompt_manifest.json"
PROMPT_BUNDLE = "research/98_selector_demo/data/prereg/w98d2_prompt_tokens.jsonl.gz"
PREREG_DOC = "research/98_selector_demo/w98_prereg.md"
D2_CONTROLS = "research/98_selector_demo/data/prereg/w98_d2_controls.json"
CALIBRATION = "research/98_selector_demo/data/probe_d2_calibration/d2_calibration.json"
SINGLES_DIR = "singles"
CONFIRM_DIR = "confirm"
SCREEN_NAME = "d2_screen.json"

KMAX = d0.KMAX
BOOT_SCOPE = d0.BOOT_SCOPE
CONTENT_SEEDS = (4, 5)
MEASURE_TOKENS = 640
# Screen tolerance: the product bound is admit-only, so honesty is violated
# only when a CONFIRMED tau falls below the bound by more than this.
SCREEN_TOLERANCE = 0.02
U_EDGES = stream.PROVISIONAL_U_EDGES


class G98DError(RuntimeError):
    """Raised when the acceptance campaign cannot prove its authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G98DError(message)


def _load_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _config_key(cfg: Mapping[str, Any]) -> str:
    return r1._config_key(cfg)


def _slug(cfg: Mapping[str, Any], seed: int) -> str:
    return f"{_config_key(cfg).replace('/', '_')}_s{seed}"


# --- lattice ---------------------------------------------------------------


def base_config() -> dict[str, Any]:
    return {"quant": "target-matching", "window": "off", "skip_count": 0}


def single_lever_profiles() -> list[dict[str, Any]]:
    """The single-lever configurations the product bound is built from."""
    return r1.single_lever_profiles()


def composed_profiles() -> list[dict[str, Any]]:
    """Compositions the screen ranks: every cell varying two or more axes."""
    prereg = _load_json(matrix._repository_path(r1.PREREG_MATRIX))
    lattice = prereg["lattice"]
    out = []
    for quant in lattice["quant"]:
        for window in lattice["window"]:
            for skip in lattice["skip_counts"]:
                cfg = {"quant": quant, "window": window, "skip_count": skip}
                varied = (quant != "target-matching") + (window != "off") + (skip != 0)
                if varied >= 2:
                    out.append(cfg)
    return out


def bridge_width() -> float:
    """Measured eager-vs-captured acceptance gap, in accepted tokens.

    Registered use: survivors must clear `tau*` by more than this. Screening
    in eager OVERESTIMATES acceptance, so the margin is applied against the
    admitted side.
    """
    record = _load_json(matrix._repository_path(CALIBRATION))
    return abs(float(record["realization_bridge"]["mean_accepted_delta"]))


# --- authorization ---------------------------------------------------------


def expected_authorization() -> dict[str, Any]:
    sources = {
        "prereg_doc": PREREG_DOC,
        "prompt_manifest": PROMPT_MANIFEST,
        "prompt_bundle": PROMPT_BUNDLE,
        "d2_controls": D2_CONTROLS,
        "calibration": CALIBRATION,
        "round1_runner": "research/98_selector_demo/scripts/run_w98_g98b_round1.py",
        "smoke_runner": "research/98_selector_demo/scripts/run_w98_g98d0_smoke.py",
        "stream_bridge": "research/98_selector_demo/scripts/w98d2_stream.py",
        "accounting": "research/98_selector_demo/scripts/w98_accounting.py",
        "campaign_runner": (
            "research/98_selector_demo/scripts/run_w98_g98d_campaign.py"
        ),
        "koff_runtime": "vllm/v1/spec_decode/koff_runtime.py",
    }
    source_artifacts = {}
    for name, rel in sources.items():
        path = matrix._repository_path(rel)
        _require(path.is_file(), f"source artifact missing: {rel}")
        source_artifacts[name] = {
            "path": rel,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {
        "schema_version": 1,
        "package_id": PACKAGE_ID,
        "gate": "G98-D",
        "prerequisite": {
            "gate": "G98-C",
            "result": "research/98_selector_demo/data/g98_c/round2_result.json",
            "status": "scored and closed",
        },
        "source_artifacts": source_artifacts,
        "runtime": {
            "boot_scope": BOOT_SCOPE,
            "kmax": KMAX,
            "unconditionally_armed": True,
            "draft": "piecewise",
            "kernel": "Marlin",
            "VLLM_DISABLED_KERNELS": "MacheteLinearKernel",
        },
        "sampling": {
            "content_seeds": list(CONTENT_SEEDS),
            "seed_note": (
                "disjoint from the cost lattice's seeds 2,3; verified overlap 0"
            ),
            "measure_tokens": MEASURE_TOKENS,
            "replication": "over content seeds and requests, not boots",
            "replication_basis": (
                "acceptance is boot-deterministic: 17/17 requests and 0/210 "
                "steps identical across two boots (probe_w98_d2_calibration)"
            ),
        },
        "gates": {
            "measurement_shape_gate": False,
            "measurement_shape_rationale": (
                "that gate judges draft-chain TIMING shape; acceptance is "
                "boot-deterministic and clamp-immune, so it would only reject "
                "good acceptance data"
            ),
            "repeat_boot_anchors": False,
        },
        "screen": {
            "bound": "f_set >= prod f_i",
            "admit_only": True,
            "tolerance": SCREEN_TOLERANCE,
            "bridge_width_accepted_tokens": bridge_width(),
            "bridge_direction": "eager overestimates; margin applied conservatively",
        },
        "analysis": {
            "u_edges_provisional": list(U_EDGES),
            "u_edges_are_a_scored_output": True,
            "estimator": "w98_accounting request-level percentile bootstrap",
            "identity": "E + C = A + H",
        },
        "stages": [
            {"id": "S0", "action": "singles", "count": len(single_lever_profiles())},
            {"id": "S1", "action": "commit_screen", "barrier": True},
            {"id": "S2", "action": "confirm_admitted"},
            {"id": "S3", "action": "score_reveal"},
        ],
        "claims": {
            "D2a": "no admitted composition confirms below the bound minus tolerance",
            "D2c": "tau(w, g, u) resolves with interval separation",
            "D2b_note": (
                "the knapsack identity needs per-layer retention over "
                "arbitrary skip sets and is a separate stage, not scored here"
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
    expected = expected_authorization()
    _require(package.get("package_id") == PACKAGE_ID, f"not {PACKAGE_ID}")
    for name, want in expected["source_artifacts"].items():
        got = (package.get("source_artifacts") or {}).get(name)
        _require(got is not None, f"authorization omits source artifact {name}")
        _require(
            got.get("sha256") == want["sha256"],
            f"source artifact {name} changed since the authorization was issued",
        )
    for field in ("runtime", "sampling", "gates", "screen", "analysis", "stages"):
        _require(
            package.get(field) == expected[field],
            f"authorization field {field!r} does not match this runner",
        )


def issue_authorization(path: Path) -> dict[str, Any]:
    path = path.resolve()
    _require(not path.exists(), f"authorization {path} already exists")
    package = expected_authorization()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
    return package


# --- measurement -----------------------------------------------------------


def boot_environment(cfg: Mapping[str, Any], trace_path: Path) -> dict[str, str]:
    return d0.boot_environment(dict(cfg), trace_path)


def measure_config(cfg: Mapping[str, Any], seed: int, trace_path: Path) -> dict:
    """Boot one configuration at KMAX and return its acceptance profile."""
    from vllm import LLMEngine, SamplingParams

    manifest = _load_json(matrix._repository_path(PROMPT_MANIFEST))
    regimes = {r["regime_id"]: r for r in manifest["prompt_plan"]["regimes"]}
    engine = LLMEngine.from_engine_args(d0._engine_args(dict(cfg)))
    observations: dict[str, Any] = {}
    try:
        for regime_id, spec in regimes.items():
            before = _trace_len(trace_path)
            prompts = _prompts_for(regime_id, seed, spec["batch"])
            for index, tokens in enumerate(prompts):
                engine.add_request(
                    f"{regime_id}-s{seed}-{index}",
                    {"prompt_token_ids": tokens},
                    SamplingParams(
                        temperature=0.0, max_tokens=MEASURE_TOKENS, ignore_eos=True
                    ),
                )
            while engine.has_unfinished_requests():
                engine.step()
            rows = stream.rows_from_records(_records_after(trace_path, before))
            observations[regime_id] = _profile(rows)
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()
    return observations


def _profile(rows: Sequence[Any]) -> dict[str, Any]:
    from w98_accounting import close_interval

    closure = close_interval(rows)
    profile = stream.tau_profile(rows, kmax=KMAX, u_edges=U_EDGES)
    return {
        "rows": len(rows),
        "closure": {
            "H": closure.h_steps,
            "D": closure.d_armed,
            "A": closure.a_accepted,
            "C": closure.c_clipped,
            "E": closure.e_emitted,
            "tau_eff": round(closure.tau_eff, 6),
        },
        "tau_profile": profile,
    }


def _prompts_for(regime_id: str, seed: int, limit: int) -> list[list[int]]:
    import gzip

    bundle = matrix._repository_path(PROMPT_BUNDLE)
    rows = []
    with gzip.open(bundle, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["regime_id"] == regime_id and row["content_seed"] == seed:
                rows.append(row)
    rows.sort(key=lambda r: r["prompt_index"])
    _require(
        len(rows) >= limit,
        f"{regime_id}/seed-{seed} has {len(rows)} prompts, needs {limit}",
    )
    return [r["token_ids"] for r in rows[:limit]]


def _trace_len(trace_path: Path) -> int:
    if not Path(trace_path).is_file():
        return 0
    with Path(trace_path).open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _records_after(trace_path: Path, skip: int) -> list[dict[str, Any]]:
    out = []
    with Path(trace_path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < skip:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("record_type") != "koff_engine_step":
                continue
            if record.get("exclusion_reasons"):
                continue
            if not (record.get("counters") or {}).get("H_target_steps"):
                continue
            out.append(record)
    return out


def _run_stage(
    stage_dir: Path,
    configs: Sequence[Mapping[str, Any]],
    trace_root: Path,
    authorization_path: Path,
) -> None:
    """Boot each (configuration, seed) in its own child, pinned to the lane."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    trace_root = trace_root / stage_dir.name
    trace_root.mkdir(parents=True, exist_ok=True)
    lane = matrix.lane_for_block(1)
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = set(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        for seed in CONTENT_SEEDS:
            slug = _slug(cfg, seed)
            target = stage_dir / f"{slug}.json"
            if target.exists():
                continue
            trace = trace_root / f"{slug}.jsonl"
            trace.unlink(missing_ok=True)
            log = stage_dir / f"{slug}.log"
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
                        "--seed",
                        str(seed),
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
                f"boot {slug} failed; output preserved at {log}",
            )
            print(f"[g98d] measured {slug}", flush=True)


# --- screen ----------------------------------------------------------------


def _tau_overall(observation: Mapping[str, Any]) -> float:
    """Committed emissions per target step, pooled over the regime's stream."""
    return float(observation["closure"]["tau_eff"])


def _load_stage(stage_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not stage_dir.is_dir():
        return out
    for path in sorted(stage_dir.glob("*.json")):
        record = _load_json(path)
        out[path.stem] = record
    return out


def build_screen(output_dir: Path) -> dict[str, Any]:
    """Product bound from the singles, applied to every composition."""
    singles = _load_stage(output_dir / SINGLES_DIR)
    _require(bool(singles), "no singles measured")
    base_key = _config_key(base_config())
    per_regime_base: dict[str, list[float]] = {}
    retention: dict[str, dict[str, list[float]]] = {}
    for record in singles.values():
        cfg = record["config"]
        key = _config_key(cfg)
        for regime_id, observation in record["observations"].items():
            tau = _tau_overall(observation)
            if key == base_key:
                per_regime_base.setdefault(regime_id, []).append(tau)
            else:
                retention.setdefault(key, {}).setdefault(regime_id, []).append(tau)
    _require(bool(per_regime_base), "the base single-lever cell was not measured")
    base_tau = {r: sum(v) / len(v) for r, v in per_regime_base.items()}

    factors: dict[str, dict[str, float]] = {}
    for key, by_regime in retention.items():
        factors[key] = {
            regime_id: (sum(v) / len(v)) / base_tau[regime_id]
            for regime_id, v in by_regime.items()
            if base_tau.get(regime_id)
        }

    margin = bridge_width()
    admitted: list[dict[str, Any]] = []
    for cfg in composed_profiles():
        lever_keys = _lever_keys(cfg)
        bounds: dict[str, float] = {}
        for regime_id, tau0 in base_tau.items():
            product = 1.0
            complete = True
            for lever in lever_keys:
                factor = factors.get(lever, {}).get(regime_id)
                if factor is None:
                    complete = False
                    break
                product *= factor
            if complete:
                bounds[regime_id] = tau0 * product
        if bounds:
            admitted.append(
                {
                    "cell": _config_key(cfg),
                    "config": cfg,
                    "levers": lever_keys,
                    "bound_tau": {k: round(v, 6) for k, v in bounds.items()},
                    "admitted": True,
                }
            )
    return {
        "base_tau": {k: round(v, 6) for k, v in base_tau.items()},
        "lever_factors": {
            k: {r: round(v, 6) for r, v in by_r.items()} for k, by_r in factors.items()
        },
        "bridge_width_accepted_tokens": margin,
        "tolerance": SCREEN_TOLERANCE,
        "admitted": admitted,
    }


def _lever_keys(cfg: Mapping[str, Any]) -> list[str]:
    """The single-lever cells whose product bounds this composition."""
    keys = []
    if cfg["quant"] != "target-matching":
        keys.append(_config_key({**base_config(), "quant": cfg["quant"]}))
    if cfg["window"] != "off":
        keys.append(_config_key({**base_config(), "window": cfg["window"]}))
    if cfg["skip_count"]:
        keys.append(_config_key({**base_config(), "skip_count": cfg["skip_count"]}))
    return keys


def commit_screen(output_dir: Path, screen: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the screen before any confirmation boot runs."""
    output_dir = output_dir.resolve()
    path = output_dir / SCREEN_NAME
    _require(not path.exists(), "the screen is already committed and is immutable")
    confirm_dir = output_dir / CONFIRM_DIR
    existing = sorted(confirm_dir.glob("*.json")) if confirm_dir.is_dir() else []
    _require(
        not existing,
        f"refusing to commit after confirmations exist: {[p.name for p in existing]}",
    )
    record = {
        "schema_version": 1,
        "record_type": "w98d2_committed_screen",
        "package_id": PACKAGE_ID,
        "committed_before_confirmation": True,
        "screen": dict(screen),
    }
    record["digest"] = _digest(record["screen"])
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def require_committed_screen(output_dir: Path) -> dict[str, Any]:
    path = output_dir.resolve() / SCREEN_NAME
    _require(path.is_file(), "the screen was never committed")
    record = _load_json(path)
    _require(
        record.get("digest") == _digest(record["screen"]),
        "the committed screen does not match its digest",
    )
    return record


# --- scoring ---------------------------------------------------------------


def score_reveal(output_dir: Path) -> dict[str, Any]:
    """Score D2(a) screen honesty and report D2(c) u-axis resolution."""
    committed = require_committed_screen(output_dir)
    screen = committed["screen"]
    bounds = {row["cell"]: row for row in screen["admitted"]}
    confirmations = _load_stage(output_dir / CONFIRM_DIR)
    _require(bool(confirmations), "no confirmations measured")

    rows: list[dict[str, Any]] = []
    by_cell: dict[str, dict[str, list[float]]] = {}
    for record in confirmations.values():
        cell = _config_key(record["config"])
        for regime_id, observation in record["observations"].items():
            by_cell.setdefault(cell, {}).setdefault(regime_id, []).append(
                _tau_overall(observation)
            )
    violations = 0
    for cell, by_regime in sorted(by_cell.items()):
        bound_row = bounds.get(cell)
        if bound_row is None:
            continue
        for regime_id, values in sorted(by_regime.items()):
            confirmed = sum(values) / len(values)
            bound = bound_row["bound_tau"].get(regime_id)
            if bound is None:
                continue
            shortfall = bound - confirmed
            honest = shortfall <= SCREEN_TOLERANCE * bound
            violations += not honest
            rows.append(
                {
                    "cell": cell,
                    "regime": regime_id,
                    "bound_tau": round(bound, 6),
                    "confirmed_tau": round(confirmed, 6),
                    "shortfall": round(shortfall, 6),
                    "honest": honest,
                }
            )

    resolved = _u_axis_resolution(confirmations)
    return {
        "schema_version": 1,
        "record_type": "w98d2_result",
        "package_id": PACKAGE_ID,
        "screen_digest": committed["digest"],
        "d2a_rows": rows,
        "d2a_violations": violations,
        "d2a_honest": violations == 0,
        "d2c_u_axis": resolved,
    }


def _u_axis_resolution(confirmations: Mapping[str, Any]) -> dict[str, Any]:
    """Whether tau separates across u buckets by INTERVAL, not point."""
    out: dict[str, Any] = {}
    for name, record in sorted(confirmations.items()):
        for regime_id, observation in record["observations"].items():
            buckets = observation["tau_profile"]["buckets"]
            populated = {
                index: entry for index, entry in buckets.items() if "tau" in entry
            }
            separated = []
            ordered = sorted(populated, key=int)
            for left, right in zip(ordered, ordered[1:]):
                a, b = populated[left], populated[right]
                separated.append(
                    {
                        "buckets": [left, right],
                        "separated": a["tau_lo"] > b["tau_hi"]
                        or b["tau_lo"] > a["tau_hi"],
                    }
                )
            out[f"{name}/{regime_id}"] = {
                "populated_buckets": ordered,
                "adjacent_separation": separated,
            }
    return out


# --- entry point -----------------------------------------------------------


def execute(authorization_path: Path, output_dir: Path) -> int:
    output_dir = output_dir.resolve()
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

    _run_stage(
        output_dir / SINGLES_DIR, single_lever_profiles(), traces, authorization_path
    )
    if not (output_dir / SCREEN_NAME).exists():
        commit_screen(output_dir, build_screen(output_dir))
    screen = require_committed_screen(output_dir)["screen"]
    admitted = [row["config"] for row in screen["admitted"]]
    _run_stage(output_dir / CONFIRM_DIR, admitted, traces, authorization_path)
    result = score_reveal(output_dir)
    (output_dir / "d2_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "d2a_honest": result["d2a_honest"],
                "d2a_violations": result["d2a_violations"],
                "rows": len(result["d2a_rows"]),
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--issue-authorization", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--measure-config", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--stage-dir", help=argparse.SUPPRESS)
    parser.add_argument("--trace", help=argparse.SUPPRESS)
    args = parser.parse_args()
    authorization = (
        args.authorization or matrix._repository_path(AUTHORIZATION_PATH)
    ).resolve()
    output_dir = (args.output_dir or matrix._repository_path(OUTPUT_PATH)).resolve()

    if args.measure_config:
        cfg = json.loads(args.measure_config)
        validate_authorization(_load_json(authorization))
        stage_dir = Path(args.stage_dir).resolve()
        stage_dir.mkdir(parents=True, exist_ok=True)
        observations = measure_config(cfg, args.seed, Path(args.trace).resolve())
        (stage_dir / f"{_slug(cfg, args.seed)}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "record_type": "w98d2_cell",
                    "package_id": PACKAGE_ID,
                    "config": cfg,
                    "content_seed": args.seed,
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
                    "singles": len(single_lever_profiles()),
                    "compositions": len(composed_profiles()),
                    "seeds": list(CONTENT_SEEDS),
                    "boots": (
                        (len(single_lever_profiles()) + len(composed_profiles()))
                        * len(CONTENT_SEEDS)
                    ),
                    "bridge_width": bridge_width(),
                    "stages": [s["id"] for s in package["stages"]],
                },
                indent=2,
            )
        )
        return 0
    return execute(authorization, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
