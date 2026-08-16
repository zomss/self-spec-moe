#!/usr/bin/env python3
"""G98-F -- the skip lever re-measured on knapsack-chosen layer sets.

D2(b) carried a non-registered reference arm showing that the frozen nested
skip-8 set every cost campaign booted reaches tau 5.327 where a
knapsack-chosen set of the SAME COUNT reaches 6.423: +20.6% acceptance at
identical cost. `results_lever_mechanics.md` estimated that this moves skip-8
from beating no-speculation in 1 of 6 regimes to 4 of 6 -- which would make
"window and skip are worthless alone" false, and would mean the composition
premium the phase reports is measured against a handicapped baseline.

That estimate is what this campaign replaces with measurement.

## The circularity this design exists to avoid

The knapsack sets were CHOSEN from per-layer retention measured on **R8, content
seed 4**, and confirmed on seed 5. Scoring them on seeds 4-5 would be scoring
a selection rule on its own selection data -- the defect Phase 84 retracted a
lever for. This campaign therefore evaluates on the **cost lattice's seeds
2-3** (`w98_prompt_tokens.jsonl.gz`, the content D3 itself ran on), which is
disjoint from the selection data and directly comparable to D3.

It also means the transfer question is live: a set chosen on R8 has no
guarantee at R1, R4, R5, R5cot or R6, and five of the six regimes here are
out-of-domain in that second sense too.

## Design

Six cells, `target-matching` with no window, so the measurement is
attributable to the skip lever and nothing else:

    off, skip0, skip4_frozen, skip4_knapsack, skip8_frozen, skip8_knapsack

`off` supplies the denominator, `skip0` the unmodified-draft reference. Frozen
and knapsack are measured in the SAME campaign on the same box, so the
comparison never crosses a session boundary. Both instrument arms run, per
amendment 2.

## Registered predictions

Committed with a digest before the first boot; see `--commit`.

* **F1** knapsack-8 beats frozen-8 in acceptance at R8 (in-domain; the weak
  claim, and a failure here would indicate a protocol error rather than a
  transfer failure).
* **F2** knapsack-8 beats frozen-8 in acceptance in **>= 4 of 6** regimes.
  This is the transfer claim and the point of the campaign.
* **F3** draft-chain cost differs by **< 2%** between frozen and knapsack at
  the same count -- the sets remove the same NUMBER of layers, and every layer
  of Qwen3-8B dense has identical dimensions, so cost should not move.
* **F4** with knapsack sets, skip-8 beats OFF in **>= 3 of 6** regimes
  (`results_lever_mechanics.md` estimated 4).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE = SCRIPT_DIR.parent
REPO = PHASE.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402
import run_w98_g98e_d3 as d3  # noqa: E402  (rebinds the lane to this host)

matrix = r1.matrix

REGIMES = ("R1", "R4", "R5", "R5cot", "R6", "R8")

# From D2(b)'s committed set file (digest f9474093...), which was itself
# committed before any confirmation boot.
SETS = {
    "frozen_k4": "2,4,7,16",
    "knapsack_k4": "2,4,9,16",
    "frozen_k8": "2,4,7,11,16,20,25,30",
    "knapsack_k8": "2,4,6,7,8,9,10,16",
}

CELLS: list[dict[str, Any]] = [
    {"name": "off", "armed": False, "skip_count": 0, "layers": ""},
    {"name": "skip0", "armed": True, "skip_count": 0, "layers": ""},
    {"name": "skip4_frozen", "armed": True, "skip_count": 4,
     "layers": SETS["frozen_k4"]},
    {"name": "skip4_knapsack", "armed": True, "skip_count": 4,
     "layers": SETS["knapsack_k4"]},
    {"name": "skip8_frozen", "armed": True, "skip_count": 8,
     "layers": SETS["frozen_k8"]},
    {"name": "skip8_knapsack", "armed": True, "skip_count": 8,
     "layers": SETS["knapsack_k8"]},
]

ARMS = ("corrected", "legacy")


class G98FError(RuntimeError):
    """Raised when the skip-set campaign cannot prove its authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G98FError(message)


def _base_cfg(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "quant": "target-matching",
        "window": "off",
        "skip_count": cell["skip_count"],
    }


def _decode_rate(trace: Path, skip: int) -> dict[str, Any]:
    """Committed tokens per second of DECODE time, and acceptance."""
    committed = 0
    seconds = 0.0
    steps = armed = 0
    with Path(trace).open(encoding="utf-8") as handle:
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
            counters = record.get("counters") or {}
            if not counters.get("H_target_steps"):
                continue
            elapsed = counters.get("decode_time_s")
            if not isinstance(elapsed, (int, float)) or elapsed <= 0:
                continue
            committed += int(counters.get("E_committed", 0))
            seconds += float(elapsed)
            steps += 1
            armed += bool(counters.get("D_armed"))
    _require(steps > 0 and seconds > 0, "no scored decode steps in the interval")
    return {
        "committed_tokens": committed,
        "decode_seconds": round(seconds, 6),
        "decode_tokens_per_s": round(committed / seconds, 6),
        "steps": steps,
        "armed_steps": armed,
        "committed_per_step": round(committed / steps, 6),
    }


def measure(trace: Path, out_json: Path, cell_name: str, arm: str) -> None:
    from vllm import LLMEngine, SamplingParams
    from vllm.v1.spec_decode.self_spec_profiler import get_profiler

    cell = next(c for c in CELLS if c["name"] == cell_name)
    manifest = r1._load_json(matrix._repository_path(r1.PROMPT_MANIFEST))
    regimes = {r["regime_id"]: r for r in manifest["prompt_plan"]["regimes"]}
    profiler = get_profiler()
    cfg = _base_cfg(cell)
    engine_cfg = dict(cfg)
    engine_cfg["action"] = "armed" if cell["armed"] else "off"
    engine = LLMEngine.from_engine_args(d3._engine_args(engine_cfg))
    observations: dict[str, Any] = {}
    try:
        for regime_id in REGIMES:
            spec = regimes[regime_id]
            prompts = r1._prompts_for(regime_id, spec["batch"])
            before = r1._trace_len(trace)
            profiler.reset()
            for index, tokens in enumerate(prompts):
                engine.add_request(
                    f"{regime_id}-{index}",
                    {"prompt_token_ids": tokens},
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=r1.MEASURE_TOKENS,
                        ignore_eos=True,
                    ),
                )
            while engine.has_unfinished_requests():
                engine.step()
            summary = profiler.summary(warmup=r1.PROFILER_WARMUP)
            chain = summary.get("draft_chain", {})
            observations[regime_id] = {
                **_decode_rate(trace, before),
                "draft_chain_ms": chain.get("mean_ms"),
                "batch": spec["batch"],
            }
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()
    out_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98f_skipset_cell",
                "cell": cell_name,
                "arm": arm,
                "skip_layers": cell["layers"],
                "skip_count": cell["skip_count"],
                "armed": cell["armed"],
                "content_seeds": [2, 3],
                "seed_note": (
                    "the cost lattice's seeds; DISJOINT from the seed-4/5 "
                    "retention data the knapsack sets were chosen from"
                ),
                "observations": observations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def commit_predictions(output_dir: Path) -> Path:
    """Write the barrier BEFORE any boot; refuse to overwrite it."""
    target = output_dir / "f_predictions.json"
    _require(not target.exists(), f"predictions already committed at {target}")
    payload = {
        "record_type": "w98f_predictions",
        "schema_version": 1,
        "sets": SETS,
        "content_seeds": [2, 3],
        "selection_provenance": (
            "knapsack sets chosen from R8 seed-4 per-layer retention, "
            "confirmed on seed 5 (D2b, digest f9474093...); evaluation here "
            "uses seeds 2-3, so the sets are scored out of sample"
        ),
        "predictions": {
            "F1": "knapsack_k8 acceptance > frozen_k8 acceptance at R8",
            "F2": "knapsack_k8 acceptance > frozen_k8 in >= 4 of 6 regimes",
            "F3": "draft_chain differs < 2% between frozen and knapsack "
                  "at the same count",
            "F4": "skip8_knapsack decode rate > off in >= 3 of 6 regimes",
        },
    }
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(body.encode()).hexdigest()
    payload["digest"] = digest
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"committed predictions, digest {digest[:16]}")
    return target


def run_all(output_dir: Path) -> None:
    lane = d3.lane_for_block(1)
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = sorted(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    traces = output_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    _require(
        (output_dir / "f_predictions.json").exists(),
        "commit predictions before measuring (--commit)",
    )
    for arm in ARMS:
        for cell in CELLS:
            name = f"{cell['name']}__{arm}"
            target = output_dir / f"{name}.json"
            if target.exists():
                continue
            trace = traces / f"{name}.jsonl"
            _require(not trace.exists(), f"trace {trace} already exists")
            cfg = _base_cfg(cell)
            env = matrix._boot_child_environment(
                d3.boot_environment({**cfg, "action": "armed"}, trace, arm)
            )
            # The lattice gate validates the skip COUNT, not which layers, so
            # swapping membership is inside the booted contract.
            env["VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS"] = cell["layers"]
            print(f"[{name}] layers={cell['layers'] or '(none)'} ...", flush=True)
            log = output_dir / f"{name}.log"
            with log.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--measure",
                        "--cell",
                        cell["name"],
                        "--arm",
                        arm,
                        "--trace",
                        str(trace),
                        "--out",
                        str(target),
                    ],
                    env=env,
                    cwd=str(REPO),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    preexec_fn=lambda: os.sched_setaffinity(0, affinity),
                )
            if completed.returncode != 0 or not target.exists():
                raise G98FError(f"{name} failed ({completed.returncode}); see {log}")
            print(f"[g98f] ok {name}", flush=True)


def score(output_dir: Path) -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        for cell in CELLS:
            path = output_dir / f"{cell['name']}__{arm}.json"
            if path.exists():
                cells.setdefault(arm, {})[cell["name"]] = json.loads(
                    path.read_text()
                )

    results: dict[str, Any] = {"arms": {}}
    for arm, by_cell in cells.items():
        if "off" not in by_cell:
            continue
        rows: dict[str, Any] = {}
        for regime in REGIMES:
            off_rate = by_cell["off"]["observations"][regime]["decode_tokens_per_s"]
            entry: dict[str, Any] = {"off_tokens_per_s": off_rate}
            for name, record in by_cell.items():
                if name == "off":
                    continue
                obs = record["observations"][regime]
                entry[name] = {
                    "tokens_per_s": obs["decode_tokens_per_s"],
                    "speedup_vs_off": round(
                        obs["decode_tokens_per_s"] / off_rate, 6
                    ),
                    "committed_per_step": obs["committed_per_step"],
                    "draft_chain_ms": obs["draft_chain_ms"],
                }
            rows[regime] = entry
        results["arms"][arm] = {"per_regime": rows, "verdicts": _verdicts(rows)}
    return results


def _verdicts(rows: dict[str, Any]) -> dict[str, Any]:
    def have(name: str) -> bool:
        return all(name in rows[r] for r in rows)

    out: dict[str, Any] = {}
    if have("skip8_frozen") and have("skip8_knapsack"):
        better = [
            r
            for r in rows
            if rows[r]["skip8_knapsack"]["committed_per_step"]
            > rows[r]["skip8_frozen"]["committed_per_step"]
        ]
        out["F1_r8_in_domain"] = "R8" in better
        out["F2_regimes_knapsack_better"] = sorted(better)
        out["F2_pass"] = len(better) >= 4
        deltas = {
            r: round(
                rows[r]["skip8_knapsack"]["draft_chain_ms"]
                / rows[r]["skip8_frozen"]["draft_chain_ms"]
                - 1,
                6,
            )
            for r in rows
            if rows[r]["skip8_frozen"]["draft_chain_ms"]
        }
        out["F3_cost_delta"] = deltas
        out["F3_pass"] = all(abs(v) < 0.02 for v in deltas.values())
        wins = [r for r in rows if rows[r]["skip8_knapsack"]["speedup_vs_off"] > 1.0]
        out["F4_regimes_beating_off"] = sorted(wins)
        out["F4_pass"] = len(wins) >= 3
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--cell")
    parser.add_argument("--arm")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=PHASE / "data/g98_f"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.measure:
        measure(args.trace, args.out, args.cell, args.arm)
        return 0
    if args.commit:
        commit_predictions(args.output_dir)
        return 0
    if not args.score:
        run_all(args.output_dir)
    result = score(args.output_dir)
    (args.output_dir / "f_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    for arm, payload in result["arms"].items():
        print(f"\n=== {arm} ===")
        print("regime   off t/s   frozen8 (tau, x)      knapsack8 (tau, x)")
        for regime, row in payload["per_regime"].items():
            f = row.get("skip8_frozen")
            k = row.get("skip8_knapsack")
            if not (f and k):
                continue
            print(
                f"{regime:6s} {row['off_tokens_per_s']:8.1f}   "
                f"{f['committed_per_step']:5.3f}  {f['speedup_vs_off']:.3f}      "
                f"{k['committed_per_step']:5.3f}  {k['speedup_vs_off']:.3f}"
            )
        print(json.dumps(payload["verdicts"], indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
