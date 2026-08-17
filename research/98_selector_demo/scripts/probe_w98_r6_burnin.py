# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""G98-G/R6 -- can a short burn-in fix the acceptance transfer?

G98-G's selector missed at R6 by 12%, picking `w4a16/w256/skip8` where
`w4a16/woff/skip4` won. The error was localised and it is not the cost
model: `D_hat` under-predicts every R6 cell by 17.7-21.2%, a 3.5-point
spread that barely moves a ranking. Substituting realized acceptance for
transferred acceptance puts the measured winner on top immediately.

The transfer error is systematic in skip depth:

    skip0   +3.4%, +6.3%      optimism of transferred tau over realized
    skip4   +5.0% .. +9.6%
    skip8  +16.4% .. +17.1%

The deeper the skip, the worse acceptance generalises to unseen content --
so the selector over-rated the deep-skip cell and picked it. Two candidate
corrections were rejected before this one:

* **A confidence bound from seed spread.** G98-D's own seed-4-vs-5 spread
  does rise with skip depth (median 0.59 / 0.85 / 1.20%), so the direction
  is right, but it is an order of magnitude too small to close a 16% gap. An
  LCB built from it would not change the pick.
* **Re-measuring acceptance on this box.** Acceptance is deterministic given
  checkpoints and prompts -- G98-F measured bit-identical accept patterns
  across boxes -- so re-measuring seeds 4-5 here reproduces h104's numbers
  exactly and changes nothing. The gap is content generalisation
  (seeds 4-5 -> 2-3), not a box effect.

What remains is the design's own answer: acceptance is the half that must be
MEASURED, and the deployed selector measures it online. This prices that
directly -- boot each R6 candidate for a short burn-in on the deployment
content, read tau from it, and re-rank. If a 64-token burn-in recovers the
right pick, the online layer buys the missing 12% for about a tenth of the
evaluation's cost.

The burn-in is a genuinely different measurement from the one being
corrected: 64 generated tokens against the scored run's 640, so it sees only
the earliest part of the generation and must EXTRAPOLATE across the u-axis
that D2(c) showed acceptance varies along.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402
import run_w98_g98e_d3 as d3  # noqa: E402
import run_w98_g98g_e2e as g98g  # noqa: E402

matrix = r1.matrix

REGIME = "R6"
REGIMES = tuple(os.environ.get("W98_BURNIN_REGIMES", "R6").split(","))
BATCH_BY_REGIME = {"R1": 1, "R8": 16, "R6": 32, "R4": 8, "R5": 8, "R5cot": 8}
BATCH = 32
# Acceptance is inferred from an integer step count against a FIXED token
# budget, so tau resolves only to about tau/steps -- 5.3% at 64 tokens, where
# five candidates collided on one value and the ranking was noise. Swept via
# the environment to find the shortest burn-in that actually separates them.
BURNIN_TOKENS = int(os.environ.get("W98_BURNIN_TOKENS", "64"))
G98G = SCRIPT_DIR.parent / "data" / "g98_g"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def candidates() -> list[str]:
    """The armed cells G98-G considered at R6."""
    return [c for c in g98g.confirm_cells(G98G) if c != d3.OFF_KEY]


def measure(cell: str, trace: Path, out: Path) -> None:
    """Boot one cell and read acceptance from a short generation."""
    from vllm import LLMEngine, SamplingParams

    cfg = g98g._cfg_for_cell(cell)
    manifest = r1._load_json(matrix._repository_path(d3.PROMPT_MANIFEST))
    specs = {r["regime_id"]: r for r in manifest["prompt_plan"]["regimes"]}
    engine = LLMEngine.from_engine_args(d3._engine_args(cfg))
    per_regime = {}
    try:
        for regime in REGIMES:
            before = d3._trace_len(trace)
            for index, tokens in enumerate(
                r1._prompts_for(regime, specs[regime]["batch"])
            ):
                engine.add_request(
                    f"{regime}-{index}",
                    {"prompt_token_ids": tokens},
                    SamplingParams(
                        temperature=0.0, max_tokens=BURNIN_TOKENS, ignore_eos=True
                    ),
                )
            while engine.has_unfinished_requests():
                engine.step()
            per_regime[regime] = d3._decode_rate(trace, before)
        observed = per_regime[REGIMES[0]]
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()
    armed = observed["armed_steps"]
    _require(armed > 0, f"{cell}: no armed steps in the burn-in")
    out.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_r6_burnin",
                "cell": cell,
                "burnin_tokens": BURNIN_TOKENS,
                "observed": observed,
                "per_regime": per_regime,
                "tau_burnin": observed["committed_tokens"] / (armed * BATCH),
                "tau_burnin_by_regime": {
                    r: o["committed_tokens"] / (o["armed_steps"] * BATCH_BY_REGIME[r])
                    for r, o in per_regime.items()
                    if o.get("armed_steps")
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_all(output_dir: Path, gpu: int) -> None:
    lane = g98g.bind_box(gpu)
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = sorted(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    traces = output_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    for cell in candidates():
        name = cell.replace("/", "_")
        target = output_dir / f"{name}.json"
        if target.exists():
            continue
        trace = traces / f"{name}.jsonl"
        if trace.exists():
            trace.unlink()
        env = matrix._boot_child_environment(
            d3.boot_environment(g98g._cfg_for_cell(cell), trace, "corrected")
        )
        log = output_dir / f"{name}.log"
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output-dir",
                    str(output_dir),
                    "--gpu",
                    str(gpu),
                    "--cell",
                    cell,
                    "--trace",
                    str(trace),
                ],
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                preexec_fn=lambda: os.sched_setaffinity(0, affinity),
            )
        if completed.returncode != 0 or not target.exists():
            print(f"[r6burn] FAILED {cell}", flush=True)
            continue
        record = r1._load_json(target)
        print(f"[r6burn] {cell} tau_burnin={record['tau_burnin']:.3f}", flush=True)


def summarise(output_dir: Path) -> dict[str, Any]:
    """Rank by burn-in acceptance and check it against the scored run."""
    pred = r1._load_json(G98G / "predictions.json")
    detail = pred["detail"]
    verify = pred["verify_s"][REGIME]
    scored = r1._load_json(G98G / "confirm_result.json")["per_regime"][REGIME]
    measured: dict[str, list[float]] = {}
    for path in sorted((G98G / "confirm").rglob("*.json")):
        if path.name.endswith(".telemetry.json"):
            continue
        record = r1._load_json(path)
        obs = record.get("observations", {}).get(REGIME)
        if obs and record.get("cell") != d3.OFF_KEY:
            measured.setdefault(record["cell"], []).append(obs["decode_tokens_per_s"])
    rows = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        record = r1._load_json(path)
        cell = record["cell"]
        d_hat = detail[cell][REGIME]["d_hat_s"]
        rows.append(
            {
                "cell": cell,
                "tau_transferred": detail[cell][REGIME]["tau_hat_at_k4"],
                "tau_burnin": record["tau_burnin"],
                "rate_with_transferred": (
                    detail[cell][REGIME]["tau_hat_at_k4"] / (verify + d_hat)
                ),
                "rate_with_burnin": record["tau_burnin"] / (verify + d_hat),
                "measured_640": (
                    statistics.mean(measured[cell]) if cell in measured else None
                ),
            }
        )
    scored_rows = [r for r in rows if r["measured_640"]]
    truth = max(scored_rows, key=lambda r: r["measured_640"])["cell"]
    old = max(rows, key=lambda r: r["rate_with_transferred"])["cell"]
    new = max(rows, key=lambda r: r["rate_with_burnin"])["cell"]
    by_cell = {r["cell"]: r for r in rows}
    return {
        "schema_version": 1,
        "record_type": "w98_r6_burnin_result",
        "burnin_tokens": BURNIN_TOKENS,
        "measured_winner": truth,
        "pick_with_transferred_tau": old,
        "pick_with_burnin_tau": new,
        "burnin_fixes_the_pick": new == truth and old != truth,
        "regret_before": scored["regret_vs_best_in_set"],
        "regret_after": (
            max(r["measured_640"] for r in scored_rows) / by_cell[new]["measured_640"]
            - 1.0
            if by_cell.get(new, {}).get("measured_640")
            else None
        ),
        "rows": sorted(rows, key=lambda r: -r["rate_with_burnin"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--summarise-only", action="store_true")
    parser.add_argument("--cell", help=argparse.SUPPRESS)
    parser.add_argument("--trace", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.cell:
        g98g.bind_box(args.gpu)
        output_dir.mkdir(parents=True, exist_ok=True)
        measure(
            args.cell,
            Path(args.trace).resolve(),
            output_dir / f"{args.cell.replace('/', '_')}.json",
        )
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarise_only:
        run_all(output_dir, args.gpu)
    summary = summarise(output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
