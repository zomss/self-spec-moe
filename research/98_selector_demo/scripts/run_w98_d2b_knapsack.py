# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""D2(b) — the knapsack identity against count-matched controls.

Registered claim: at each surviving count `k`, the knapsack set beats the
count-matched random-set and worst-set controls on confirmed tau, per
regime, outside the tie-set noise.

Order of operations, with the barrier enforced:

    derive sets from the retention probe  ->  COMMIT them with a digest
    ->  confirm each set on the held-back seed  ->  score

`commit_sets` refuses once any confirmation exists, so the sets cannot be
adjusted to the measurement that judges them. The controls' seeds were
frozen long before any of this, in `data/prereg/w98_d2_controls.json`.

Circularity is designed out per Phase 84, which retracted a lever whose gate
came from its own evaluation set: retention was measured on content seed 4
and every confirmation here runs on seed 5.

The frozen nested `skip4/8/16` sets are carried as an extra REFERENCE arm.
They are not one of the registered controls -- they are what the cost
campaigns actually booted, so the comparison says whether the phase's own
choice was leaving anything on the table.
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

import probe_w98_d2_layer_retention as probe  # noqa: E402
import run_w98_g98b_round1 as r1  # noqa: E402
import run_w98_g98d0_smoke as d0  # noqa: E402
import w98d2_stream as stream  # noqa: E402
from w98_knapsack import (  # noqa: E402
    random_k,
    top_k_by_retention,
    worst_k_by_retention,
)

matrix = r1.matrix
OUTPUT_PATH = "research/98_selector_demo/data/g98_d2b"
RETENTION = "research/98_selector_demo/data/probe_d2_retention/layer_retention.json"
CONTROLS = "research/98_selector_demo/data/prereg/w98_d2_controls.json"
SETS_NAME = "d2b_sets.json"
CONFIRM_DIR = "confirm"
COUNTS = (4, 8, 16)
CONFIRM_SEED = probe.CONFIRM_SEED
REGIME = probe.PROBE_REGIME
MEASURE_TOKENS = probe.MEASURE_TOKENS


class D2BError(RuntimeError):
    """Raised when the knapsack identity cannot be proven."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise D2BError(message)


def _load(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def retention_vector() -> list[float]:
    record = _load(matrix._repository_path(RETENTION))
    values = record["retention"]
    _require(
        len(values) == record["layers"],
        f"retention covers {len(values)} of {record['layers']} layers",
    )
    return [float(values[str(i)]) for i in range(record["layers"])]


def derive_sets() -> dict[str, Any]:
    """Knapsack, worst and count-matched random sets at each count."""
    r = retention_vector()
    controls = _load(matrix._repository_path(CONTROLS))
    seeds = list(controls["random_set_control_seeds"])
    arms: dict[str, Any] = {}
    for count in COUNTS:
        # Uniform per-layer cost, so the knapsack reduces to top-k by
        # retention; the DP exists for heterogeneous costs and is not needed
        # here. Keeping the reduction explicit rather than implied.
        arms[f"knapsack_k{count}"] = sorted(top_k_by_retention(r, count))
        arms[f"worst_k{count}"] = sorted(worst_k_by_retention(r, count))
        for seed in seeds:
            arms[f"random_k{count}_s{seed}"] = sorted(random_k(len(r), count, seed))
        arms[f"frozen_k{count}"] = sorted(
            int(x) for x in r1.SKIP_SETS[count].split(",") if x.strip()
        )
    return {"counts": list(COUNTS), "control_seeds": seeds, "sets": arms}


def commit_sets(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    path = output_dir / SETS_NAME
    _require(not path.exists(), "the D2(b) sets are already committed")
    confirm = output_dir / CONFIRM_DIR
    existing = sorted(confirm.glob("*.json")) if confirm.is_dir() else []
    _require(
        not existing,
        f"refusing to commit after confirmations exist: {[p.name for p in existing]}",
    )
    derived = derive_sets()
    record = {
        "schema_version": 1,
        "record_type": "w98_d2b_committed_sets",
        "committed_before_confirmation": True,
        "retention_seed": probe.RETENTION_SEED,
        "confirmation_seed": CONFIRM_SEED,
        "regime": REGIME,
        "kmax": d0.KMAX,
        **derived,
    }
    record["digest"] = _digest(record["sets"])
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def require_committed(output_dir: Path) -> dict[str, Any]:
    path = output_dir.resolve() / SETS_NAME
    _require(path.is_file(), "the D2(b) sets were never committed")
    record = _load(path)
    _require(record.get("digest") == _digest(record["sets"]), "sets fail their digest")
    return record


def measure(skip_layers: str, trace_path: Path) -> dict[str, Any]:
    from vllm import LLMEngine, SamplingParams

    cfg = {"quant": "w4a16-quantized", "window": "off", "skip_count": 0}
    manifest = _load(matrix._repository_path(probe.PROMPT_MANIFEST))
    regimes = {r["regime_id"]: r for r in manifest["prompt_plan"]["regimes"]}
    batch = regimes[REGIME]["batch"]
    engine = LLMEngine.from_engine_args(d0._engine_args(cfg))
    try:
        for index, tokens in enumerate(probe._prompts(REGIME, CONFIRM_SEED, batch)):
            engine.add_request(
                f"{REGIME}-c{index}",
                {"prompt_token_ids": tokens},
                SamplingParams(
                    temperature=0.0, max_tokens=MEASURE_TOKENS, ignore_eos=True
                ),
            )
        while engine.has_unfinished_requests():
            engine.step()
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()
    rows = stream.rows_from_trace(trace_path)
    armed = [row for row in rows if row.armed and not row.clipped]
    _require(bool(armed), "no scored armed rows")
    accepted = sum(row.accepted for row in armed)
    return {
        "skip_layers": skip_layers,
        "armed_rows": len(armed),
        "tau": round(1.0 + accepted / len(armed), 6),
    }


def _confirm(output_dir: Path, sets: Mapping[str, Sequence[int]]) -> None:
    stage = output_dir / CONFIRM_DIR
    stage.mkdir(parents=True, exist_ok=True)
    traces = output_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    lane = matrix.lane_for_block(1)
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = set(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    for name, layers in sorted(sets.items()):
        target = stage / f"{name}.json"
        if target.exists():
            continue
        skip = ",".join(str(x) for x in layers)
        trace = traces / f"{name}.jsonl"
        trace.unlink(missing_ok=True)
        log = stage / f"{name}.log"
        env = matrix._boot_child_environment(probe.boot_environment(skip, trace))
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output-dir",
                    str(output_dir),
                    "--arm",
                    name,
                    "--skip-layers",
                    skip,
                    "--trace",
                    str(trace),
                ],
                cwd=r1.REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                preexec_fn=lambda: os.sched_setaffinity(0, affinity),
            )
        _require(
            completed.returncode == 0 and target.exists(),
            f"arm {name} failed; see {log}",
        )
        print(f"[d2b] {name}", flush=True)


def score(output_dir: Path) -> dict[str, Any]:
    committed = require_committed(output_dir)
    stage = output_dir / CONFIRM_DIR
    tau = {p.stem: float(_load(p)["tau"]) for p in sorted(stage.glob("*.json"))}
    rows = []
    wins = losses = 0
    for count in committed["counts"]:
        kn = tau.get(f"knapsack_k{count}")
        if kn is None:
            continue
        controls = {
            name: value
            for name, value in tau.items()
            if name.startswith((f"random_k{count}_", f"worst_k{count}"))
        }
        for name, value in sorted(controls.items()):
            beat = kn > value
            wins += beat
            losses += not beat
            rows.append(
                {
                    "count": count,
                    "control": name,
                    "knapsack_tau": round(kn, 6),
                    "control_tau": round(value, 6),
                    "delta": round(kn - value, 6),
                    "knapsack_wins": beat,
                }
            )
        frozen = tau.get(f"frozen_k{count}")
        if frozen is not None:
            rows.append(
                {
                    "count": count,
                    "control": f"frozen_k{count}",
                    "knapsack_tau": round(kn, 6),
                    "control_tau": round(frozen, 6),
                    "delta": round(kn - frozen, 6),
                    "knapsack_wins": kn > frozen,
                    "reference_not_a_registered_control": True,
                }
            )
    return {
        "schema_version": 1,
        "record_type": "w98_d2b_result",
        "sets_digest": committed["digest"],
        "regime": REGIME,
        "confirmation_seed": CONFIRM_SEED,
        "registered_controls_beaten": wins,
        "registered_controls_lost": losses,
        "knapsack_beats_every_registered_control": losses == 0,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--arm", help=argparse.SUPPRESS)
    parser.add_argument("--skip-layers", help=argparse.SUPPRESS)
    parser.add_argument("--trace", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output_dir = (args.output_dir or matrix._repository_path(OUTPUT_PATH)).resolve()

    if args.arm:
        result = measure(args.skip_layers or "", Path(args.trace).resolve())
        stage = output_dir / CONFIRM_DIR
        stage.mkdir(parents=True, exist_ok=True)
        (stage / f"{args.arm}.json").write_text(
            json.dumps({"arm": args.arm, **result}, indent=2, sort_keys=True) + "\n"
        )
        return 0
    if args.dry_run:
        print(json.dumps(derive_sets(), indent=2, sort_keys=True))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    if not (output_dir / SETS_NAME).exists():
        commit_sets(output_dir)
    committed = require_committed(output_dir)
    matrix._preflight_gpu_identity_and_idle(
        matrix.lane_for_block(1)["physical_gpu_index"],
        matrix.lane_for_block(1)["physical_gpu_uuid"],
    )
    _confirm(output_dir, committed["sets"])
    result = score(output_dir)
    (output_dir / "d2b_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "registered_controls_beaten",
                    "registered_controls_lost",
                    "knapsack_beats_every_registered_control",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
