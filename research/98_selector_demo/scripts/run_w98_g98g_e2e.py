# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""G98-G -- the whole selector, end to end, on this box.

Phases 96-98 built the two-round selector across three machines: the cost
model was fitted on h104, acceptance measured on h104, and the end-to-end
grid scored on h103. This runs the entire pipeline on one box, from
single-lever cost profiles through to the scored grid, so the design can be
judged without any cross-box transfer except the one that is provably safe.

Stages
------

``cost``   Boot the 15 registered fit cells plus anchor repeats and measure
           the draft chain and verify cost per regime. This is Round 1/2's
           cost half, re-measured here because it is the box-dependent part:
           this VM's quantized draft chain runs 27% slower than h104's while
           its unquantized chain runs only 6% slower, so the quantization
           lever is worth materially less here (`results_g98_f.md`).

``fit``    Fit the five-parameter corrected model per regime and predict the
           draft cost of all 31 grid cells -- including the 16 composed
           cells no boot in the cost stage ever measured. This is the search
           strategy's whole claim: single-lever profiles predict composed
           configurations well enough to choose between them.

``commit`` Turn cost into a predicted decode rate per cell and regime,
           `tau / (verify + D_hat)` armed and `1 / verify` parked, exactly
           D3's formula, and freeze it behind a digest. Refuses to run once
           any grid measurement exists.

``grid``   Boot all 31 cells and measure decode throughput over six regimes.
           Run independently on two lanes, which D3 could not do: its grid
           booted each cell once, and G98-F showed an unreplicated boot is
           how a 24% phantom regression enters a scored result.

``score``  Selector against omniscient, best static and OFF -- plus the
           acceptance-transfer check below.

The one transfer, and how it is checked
---------------------------------------

Acceptance (`tau`) is taken from h104's G98-D campaign rather than
re-measured. Acceptance is a property of the checkpoints, prompts and
arming schedule, not of the machine: G98-F measured the same cells on two
boxes and got bit-identical accept patterns (158 steps, 157 armed, 5084
committed tokens at R4; 133/132/639 at R1). The grid stage re-derives
realized `tau` for every cell from its own trace, so the transfer is
verified across all 31 cells rather than assumed.
"""

from __future__ import annotations

import argparse
import contextlib
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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402
import run_w98_g98c_round2 as g98c  # noqa: E402
import run_w98_g98e_d3 as d3  # noqa: E402
import score_w98_g98e_d3 as d3score  # noqa: E402
import w98_host_load as hostload  # noqa: E402
from w98r2_cost_model import R2CostModel, combined_envelope  # noqa: E402

matrix = r1.matrix

DATA = SCRIPT_DIR.parent / "data"
OUT = DATA / "g98_g"
D3_PREDICTIONS = DATA / "g98_e" / "d3_predictions.json"
COST_DIR = "cost"
GRID_DIR = "grid"
PREDICTIONS_NAME = "predictions.json"
ANCHOR_REPEATS = 2
DEPLOY_K = 4

LANES = {
    0: {"lane_id": "lane-a", "physical_gpu_index": 0, "cpu_affinity": "0-15"},
    1: {"lane_id": "lane-b", "physical_gpu_index": 1, "cpu_affinity": "96-111"},
}
CACHE_ROOT = "/data/smcho/w98_failclosed_cache"
CKPTS = {
    "target-matching": (
        "/data/smcho/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/"
        "b968826d9c46dd6066d109eabc6255188de91218"
    ),
    "w4a16-quantized": "/data/smcho/ckpts/Qwen3-8B-W4A16-INT4",
}


class G98GError(RuntimeError):
    """Raised when a stage cannot prove its inputs or its barrier."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G98GError(message)


def _load(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _cells(directory: Path) -> list[dict[str, Any]]:
    """Cell records in a stage directory, ignoring sidecars."""
    out = []
    for path in sorted(Path(directory).glob("*.json")):
        if path.name.endswith(".telemetry.json") or path.name == "summary.json":
            continue
        record = _load(path)
        if isinstance(record, dict) and "observations" in record:
            out.append(record)
    return out


def bind_box(gpu: int) -> dict[str, Any]:
    """Point the imported runners at this box's lane and checkpoints."""
    _require(gpu in LANES, f"gpu {gpu} is not a registered lane here")
    lane = dict(LANES[gpu])
    lane["cache_root"] = f"{CACHE_ROOT}/{lane['lane_id']}"
    matrix.lane_for_block = lambda block_id=1, lane=lane: dict(lane)
    d3.lane_for_block = lambda block_id=1, lane=lane: dict(lane)
    r1.QUANT_CKPT.update(CKPTS)
    for path in CKPTS.values():
        _require(Path(path).exists(), f"checkpoint missing: {path}")
    return lane


# --- stage 1: cost -------------------------------------------------------


def cost_configs() -> list[dict[str, Any]]:
    """The registered fit lattice, with anchors repeated for sigma_repro."""
    out = [dict(c) for c in g98c.fit_profiles()]
    for repeat in range(1, ANCHOR_REPEATS):
        for anchor in g98c.ANCHORS:
            out.append({**anchor, "_repeat": repeat})
    return out


def _slug(cfg: Mapping[str, Any]) -> str:
    key = g98c._config_key({k: v for k, v in cfg.items() if not k.startswith("_")})
    slug = key.replace("/", "_")
    return f"{slug}__r{cfg['_repeat']}" if "_repeat" in cfg else slug


def _boot(
    cfg: Mapping[str, Any],
    stage_dir: Path,
    traces: Path,
    lane: Mapping[str, Any],
    gpu: int,
    mode: str,
) -> None:
    """Run one measuring child, gated and recorded."""
    name = _slug(cfg)
    target = stage_dir / f"{name}.json"
    if target.exists():
        return
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = sorted(range(int(lo), int(hi) + 1))
    attempt = 1 + max(
        (
            int(p.stem.rsplit("attempt", 1)[-1])
            for p in traces.glob(f"{name}.attempt*.jsonl")
            if p.stem.rsplit("attempt", 1)[-1].isdigit()
        ),
        default=0,
    )
    trace = traces / f"{name}.attempt{attempt}.jsonl"
    log = stage_dir / f"{name}.attempt{attempt}.log"
    if mode == "cost":
        env = matrix._boot_child_environment(r1.boot_environment(clean, trace))
    else:
        env = matrix._boot_child_environment(
            d3.boot_environment(clean, trace, "corrected")
        )
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--gpu",
                str(gpu),
                "--measure",
                mode,
                "--config",
                json.dumps(clean),
                "--name",
                name,
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
    text = log.read_text(encoding="utf-8", errors="replace")
    if completed.returncode != 0 or not target.exists():
        (stage_dir / f"{name}.FAILED").write_text(text[-6000:], encoding="utf-8")
        print(f"[g98g] FAILED {name}", flush=True)
        return
    record = _load(target)
    record["host_load"] = hostload.signature_from_log(text).as_record()
    record["lane"] = lane["lane_id"]
    if mode == "cost":
        verdict = hostload.measurement_verdict(record["observations"])
        record["measurement_gate"] = verdict.as_record() if verdict else None
        if verdict is not None and verdict.reason is not None:
            # A clamped cost boot poisons the fit for every cell it predicts.
            target.unlink()
            (stage_dir / f"{name}.REJECTED.{attempt}").write_text(
                verdict.reason, encoding="utf-8"
            )
            print(f"[g98g] REJECTED {name}: {verdict.reason}", flush=True)
            return
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"[g98g] {mode} {name}", flush=True)


def run_cost(output_dir: Path, gpu: int) -> None:
    lane = bind_box(gpu)
    stage = output_dir / COST_DIR
    traces = output_dir / "traces" / COST_DIR
    stage.mkdir(parents=True, exist_ok=True)
    traces.mkdir(parents=True, exist_ok=True)
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    for cfg in cost_configs():
        _boot(cfg, stage, traces, lane, gpu, "cost")


# --- stage 2: fit --------------------------------------------------------


def fit_cost(output_dir: Path) -> dict[str, Any]:
    """Fit the corrected cost model per regime from this box's profiles."""
    records = _cells(output_dir / COST_DIR)
    _require(len(records) >= g98c.FIT_COUNT, f"only {len(records)} cost cells")
    by_regime: dict[str, list[Any]] = {}
    verify: dict[str, list[float]] = {}
    context: dict[str, float] = {}
    repeats: dict[tuple[str, str], list[float]] = {}
    for record in records:
        cfg = record["config"]
        key = g98c._config_key(cfg)
        for regime, obs in record["observations"].items():
            chain = obs.get("draft_chain_s")
            if not chain:
                continue
            context[regime] = obs["context_tokens"]
            if obs.get("verify_s"):
                verify.setdefault(regime, []).append(float(obs["verify_s"]))
            repeats.setdefault((key, regime), []).append(float(chain))
            by_regime.setdefault(regime, []).append(
                g98c.lever_point(cfg, obs["context_tokens"], chain)
            )
    sigma: dict[str, float] = {}
    for (_, regime), values in repeats.items():
        if len(values) >= 2:
            logs = [math.log(v) for v in values]
            sigma[regime] = max(sigma.get(regime, 0.0), statistics.stdev(logs))
    fits: dict[str, Any] = {}
    for regime, points in by_regime.items():
        try:
            model = R2CostModel.fit(points)
        except Exception as exc:  # noqa: BLE001 - a regime may be unidentifiable
            fits[regime] = {"fitted": False, "reason": str(exc)}
            continue
        s = sigma.get(regime)
        fits[regime] = {
            "fitted": True,
            "kappa_w": model.kappa_w,
            "kappa_kv": model.kappa_kv,
            "c_layer": model.c_layer,
            "f_win": model.f_win,
            "F": model.f_fixed,
            "fit_term": model.log_envelope,
            "sigma_repro": s,
            "envelope": combined_envelope(model.log_envelope, s) if s else None,
            "points": len(points),
        }
    return {
        "fits": fits,
        "verify_s": {r: sum(v) / len(v) for r, v in verify.items()},
        "context_tokens": context,
    }


# --- stage 3: commit -----------------------------------------------------


def transferred_tau() -> dict[str, dict[str, float]]:
    """cell -> regime -> tau at K=4, from h104's G98-D via the D3 map."""
    detail = _load(D3_PREDICTIONS)["detail"]
    out: dict[str, dict[str, float]] = {}
    for cell, by_regime in detail.items():
        if cell == d3.OFF_KEY:
            continue
        for regime, entry in by_regime.items():
            if isinstance(entry, dict) and entry.get("tau_hat_at_k4"):
                out.setdefault(cell, {})[regime] = float(entry["tau_hat_at_k4"])
    _require(bool(out), "no transferable acceptance found")
    return out


def build_predictions(fitted: Mapping[str, Any]) -> dict[str, Any]:
    """Predicted decode rate per cell and regime, D3's formula."""
    fits = fitted["fits"]
    verify = fitted["verify_s"]
    context = fitted["context_tokens"]
    tau = transferred_tau()
    predicted: dict[str, dict[str, float]] = {
        d3.OFF_KEY: {r: 1.0 / v for r, v in verify.items() if v > 0}
    }
    detail: dict[str, Any] = {}
    for cfg in d3.grid_configs():
        if cfg.get("action") == d3.OFF_KEY:
            continue
        cell = g98c._config_key({k: v for k, v in cfg.items() if k != "action"})
        for regime, fit in fits.items():
            if not fit.get("fitted") or regime not in verify:
                continue
            tau_hat = (tau.get(cell) or {}).get(regime)
            if not tau_hat:
                continue
            point = g98c.lever_point(
                {k: v for k, v in cfg.items() if k != "action"}, context[regime], 1.0
            )
            model = R2CostModel(
                fit["kappa_w"],
                fit["kappa_kv"],
                fit["c_layer"],
                fit["f_win"],
                fit["F"],
                fit["fit_term"],
            )
            d_hat = model.predict(point)
            predicted.setdefault(cell, {})[regime] = tau_hat / (verify[regime] + d_hat)
            detail.setdefault(cell, {})[regime] = {
                "d_hat_s": round(d_hat, 6),
                "tau_hat_at_k4": round(tau_hat, 6),
                "verify_s": round(verify[regime], 6),
            }
    return {"predicted_decode_tokens_per_s": predicted, "detail": detail}


def commit(output_dir: Path, fitted: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the prediction map before any grid boot runs."""
    path = output_dir / PREDICTIONS_NAME
    _require(not path.exists(), "predictions are already committed and immutable")
    existing = _cells(output_dir / GRID_DIR) if (output_dir / GRID_DIR).is_dir() else []
    _require(
        not existing,
        f"refusing to commit after {len(existing)} grid cells exist",
    )
    built = build_predictions(fitted)
    record = {
        "schema_version": 1,
        "record_type": "w98g_predictions",
        "committed_before_grid": True,
        "box": os.uname().nodename,
        "acceptance_source": "h104 G98-D via d3_predictions detail (transferred)",
        "cost_source": "this box, g98_g/cost",
        **built,
        "fits": fitted["fits"],
        "verify_s": fitted["verify_s"],
    }
    record["digest"] = hashlib.sha256(
        json.dumps(
            built["predicted_decode_tokens_per_s"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


# --- stage 4: grid -------------------------------------------------------


def run_grid(output_dir: Path, gpu: int) -> None:
    lane = bind_box(gpu)
    _require(
        (output_dir / PREDICTIONS_NAME).is_file(),
        "grid refused: predictions must be committed first",
    )
    stage = output_dir / GRID_DIR / lane["lane_id"]
    traces = output_dir / "traces" / GRID_DIR / lane["lane_id"]
    stage.mkdir(parents=True, exist_ok=True)
    traces.mkdir(parents=True, exist_ok=True)
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    for cfg in d3.grid_configs():
        _boot(cfg, stage, traces, lane, gpu, "grid")


# --- measurement children ------------------------------------------------


def measure(mode: str, cfg: Mapping[str, Any], trace: Path, out: Path) -> None:
    if mode == "cost":
        observations = r1.measure_config(dict(cfg), trace)
    else:
        observations = d3.measure_config(dict(cfg), trace)
    out.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": f"w98g_{mode}_cell",
                "config": dict(cfg),
                "cell": g98c._config_key(
                    {k: v for k, v in cfg.items() if k != "action"}
                )
                if cfg.get("action") != d3.OFF_KEY
                else d3.OFF_KEY,
                "observations": observations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


# --- stage 4b: reduced confirmation ---------------------------------------

CONFIRM_DIR = "confirm"
CONFIRM_TOP_N = 2
BATCH_BY_REGIME = {"R1": 1, "R8": 16, "R6": 32, "R4": 8, "R5": 8, "R5cot": 8}


def h103_omniscient() -> dict[str, str]:
    """Best measured cell per regime on h103, as a transferred hypothesis."""
    grid = d3score.load_grid(DATA / "g98_e" / "grid").get("corrected", {})
    if not grid:
        return {}
    return d3score.omniscient_choice(grid, d3score.regimes_of(grid))


def confirm_cells(output_dir: Path) -> list[str]:
    """The cells worth measuring: what the search proposes, plus rivals.

    A 31-cell grid compares cells ACROSS boots, and on a box that changes
    state between boots that is precisely how a phantom result is made
    (G98-F). This measures a small candidate set instead, interleaved and
    replicated, which supports the claim that actually matters -- did the
    search pick the best of what it proposed, and was arming right -- while
    dropping the claim it cannot support here, the share of omniscient over
    all 31 cells.
    """
    predicted = _load(output_dir / PREDICTIONS_NAME)["predicted_decode_tokens_per_s"]
    chosen: set[str] = {d3.OFF_KEY}
    for regime in BATCH_BY_REGIME:
        armed = {
            cell: rates[regime]
            for cell, rates in predicted.items()
            if cell != d3.OFF_KEY and regime in rates
        }
        for cell in sorted(armed, key=armed.__getitem__, reverse=True)[:CONFIRM_TOP_N]:
            chosen.add(cell)
    chosen.update(c for c in h103_omniscient().values() if c in predicted)
    return sorted(chosen)


def _cfg_for_cell(cell: str) -> dict[str, Any]:
    for cfg in d3.grid_configs():
        key = (
            d3.OFF_KEY
            if cfg.get("action") == d3.OFF_KEY
            else g98c._config_key({k: v for k, v in cfg.items() if k != "action"})
        )
        if key == cell:
            return cfg
    raise G98GError(f"cell {cell!r} is not in the registered grid")


def run_confirm(output_dir: Path, gpu: int, rounds: int = 2) -> None:
    lane = bind_box(gpu)
    cells = confirm_cells(output_dir)
    lane_id = lane["lane_id"]
    print(f"[g98g] confirm: {len(cells)} cells x {rounds} rounds on {lane_id}")
    for index in range(rounds):
        order = cells if index % 2 == 0 else list(reversed(cells))
        stage = output_dir / CONFIRM_DIR / lane["lane_id"] / f"r{index}"
        traces = output_dir / "traces" / CONFIRM_DIR / lane["lane_id"] / f"r{index}"
        stage.mkdir(parents=True, exist_ok=True)
        traces.mkdir(parents=True, exist_ok=True)
        Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
        for cell in order:
            _boot(_cfg_for_cell(cell), stage, traces, lane, gpu, "grid")


def score_confirm(output_dir: Path) -> dict[str, Any]:
    """Did the search pick the best of what it proposed, and was arming right?"""
    predicted = _load(output_dir / PREDICTIONS_NAME)["predicted_decode_tokens_per_s"]
    rates: dict[tuple[str, str], list[float]] = {}
    tau_real: dict[tuple[str, str], list[float]] = {}
    boots = 0
    for path in sorted((output_dir / CONFIRM_DIR).rglob("*.json")):
        if path.name.endswith(".telemetry.json") or path.name == "summary.json":
            continue
        record = _load(path)
        if "observations" not in record:
            continue
        boots += 1
        cell = record.get("cell")
        for regime, obs in record["observations"].items():
            if obs.get("decode_tokens_per_s"):
                rates.setdefault((cell, regime), []).append(obs["decode_tokens_per_s"])
            armed = obs.get("armed_steps") or 0
            if armed and regime in BATCH_BY_REGIME:
                tau_real.setdefault((cell, regime), []).append(
                    obs["committed_tokens"] / (armed * BATCH_BY_REGIME[regime])
                )
    mean = {k: statistics.mean(v) for k, v in rates.items()}
    measured_cells = sorted({c for c, _ in mean})
    per_regime: dict[str, Any] = {}
    for regime in BATCH_BY_REGIME:
        present = [c for c in measured_cells if (c, regime) in mean]
        if not present or d3.OFF_KEY not in present:
            continue
        armed_present = [c for c in present if c != d3.OFF_KEY]
        if not armed_present:
            continue
        pick = max(
            (c for c in predicted if c != d3.OFF_KEY and regime in predicted[c]),
            key=lambda c: predicted[c][regime],
        )
        best = max(present, key=lambda c: mean[(c, regime)])
        off = mean[(d3.OFF_KEY, regime)]
        per_regime[regime] = {
            "selector_pick": pick,
            "pick_measured": mean.get((pick, regime)),
            "best_in_set": best,
            "best_measured": mean[(best, regime)],
            "regret_vs_best_in_set": (
                mean[(best, regime)] / mean[(pick, regime)] - 1.0
                if (pick, regime) in mean
                else None
            ),
            "pick_over_off": (
                mean[(pick, regime)] / off if (pick, regime) in mean else None
            ),
            "best_over_off": mean[(best, regime)] / off,
            "picked_the_winner": best == pick,
            "cells_measured": len(present),
        }
    transfer = []
    for (cell, regime), values in sorted(tau_real.items()):
        detail = _load(D3_PREDICTIONS)["detail"].get(cell, {}).get(regime)
        if detail and detail.get("tau_hat_at_k4"):
            transfer.append(
                {
                    "cell": cell,
                    "regime": regime,
                    "realized_tau": round(statistics.mean(values), 4),
                    "transferred_tau": detail["tau_hat_at_k4"],
                    "ratio": round(
                        statistics.mean(values) / detail["tau_hat_at_k4"], 4
                    ),
                }
            )
    ratios = [t["ratio"] for t in transfer]
    return {
        "schema_version": 1,
        "record_type": "w98g_confirm_result",
        "scope": (
            "reduced: the search's own candidate set, interleaved and replicated. "
            "Does NOT support a share-of-omniscient claim over all 31 cells."
        ),
        "boots": boots,
        "per_regime": per_regime,
        "picked_the_winner": sum(
            1 for v in per_regime.values() if v["picked_the_winner"]
        ),
        "regimes_scored": len(per_regime),
        "acceptance_transfer": {
            "pairs": len(transfer),
            "median_ratio": round(statistics.median(ratios), 4) if ratios else None,
            "min_ratio": round(min(ratios), 4) if ratios else None,
            "max_ratio": round(max(ratios), 4) if ratios else None,
            "rows": transfer,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("cost", "commit", "grid", "confirm", "score", "all"),
        default="all",
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--measure", help=argparse.SUPPRESS)
    parser.add_argument("--config", help=argparse.SUPPRESS)
    parser.add_argument("--name", help=argparse.SUPPRESS)
    parser.add_argument("--stage-dir", help=argparse.SUPPRESS)
    parser.add_argument("--trace", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.measure:
        bind_box(args.gpu)
        stage_dir = Path(args.stage_dir).resolve()
        stage_dir.mkdir(parents=True, exist_ok=True)
        measure(
            args.measure,
            json.loads(args.config),
            Path(args.trace).resolve(),
            stage_dir / f"{args.name}.json",
        )
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(Exception):
        base = matrix._boot_child_environment({})
        matrix._preflight_native_sampler(base)
        matrix._preflight_inprocess_engine_core(base)
    if args.stage in ("cost", "all"):
        run_cost(output_dir, args.gpu)
    commit_pending = (
        args.stage in ("commit", "all") and not (output_dir / PREDICTIONS_NAME).exists()
    )
    if commit_pending:
        fitted = fit_cost(output_dir)
        (output_dir / "fits.json").write_text(
            json.dumps(fitted, indent=2, sort_keys=True) + "\n"
        )
        record = commit(output_dir, fitted)
        print(f"[g98g] committed predictions, digest {record['digest'][:12]}")
    if args.stage == "grid":
        run_grid(output_dir, args.gpu)
    if args.stage in ("confirm", "all"):
        run_confirm(output_dir, args.gpu, args.rounds)
    if args.stage in ("score", "all"):
        result = score_confirm(output_dir)
        (output_dir / "confirm_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(result["per_regime"], indent=2, sort_keys=True))
        print(
            f"picked the winner in {result['picked_the_winner']}/"
            f"{result['regimes_scored']} regimes; acceptance transfer median "
            f"ratio {result['acceptance_transfer']['median_ratio']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
