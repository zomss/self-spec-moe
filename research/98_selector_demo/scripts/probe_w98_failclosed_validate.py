# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""G98-F -- out-of-sample check of the fail-closed rule's premise.

`score_w98_failclosed.py` applies the rule to the D3 grid and recovers the
R4 regression, but that grid is the same data the rule was written after
seeing: in-sample by construction. This measures the rule's two empirical
premises again, on different hardware, with fresh boots.

The premises, and why both are needed
-------------------------------------

1. **At R4 the armed pick loses to OFF.** If this does not reproduce, the
   rule fixes a measurement artifact rather than a defect.
2. **At R1 the armed pick still beats OFF, comfortably.** This is the guard
   that matters. A rule that parks is trivially "safe"; what makes it a fix
   rather than a retreat is that it declines *only* where the margin is
   invisible. R1's predicted margin is 1.335 against a 0.025 threshold --
   the rule must not fire there, and the measurement must agree.

Design
------

Interleaved, with the arm order reversed between rounds (A B C / C B A), so
a drifting box cannot manufacture the result -- the design lesson this phase
paid a week to learn. One boot measures both regimes, so the R4 and R1
comparisons share a boot and cannot disagree about the box state.

This box is the KVM guest whose host-side clamp gated the original campaign.
That biases in a KNOWN DIRECTION: the clamp adds a fixed per-step host cost,
and armed steps carry more host work than parked ones, so a clamped boot
penalises the armed arm harder. It therefore inflates premise 1 and works
AGAINST premise 2 -- which is exactly why premise 2 is the load-bearing one.
Startup signatures are recorded per boot so the state is on the record.

Nothing here edits a hash-bound artifact. The Round-1 module pins h104's
lane and checkpoint paths and is hashed by three closed authorizations, so
this rebinds both at import, the way the D3 runner rebinds the lane.
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
import w98_host_load as hostload  # noqa: E402

matrix = r1.matrix

# This box, not h104. GPU 0 is lane-a and GPU 1 lane-b, matching the CPU
# split every probe in this phase has used here.
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

REGIMES = ("R4", "R1")
# R4's selector pick, OFF, and R1's selector pick. The rule fires on the
# first and must not fire on the third.
CELLS = {
    "r4pick": {
        "action": "armed",
        "quant": "w4a16-quantized",
        "window": "off",
        "skip_count": 0,
    },
    "off": {
        "action": d3.OFF_KEY,
        "quant": "target-matching",
        "window": "off",
        "skip_count": 0,
    },
    "r1pick": {
        "action": "armed",
        "quant": "w4a16-quantized",
        "window": "off",
        "skip_count": 4,
    },
}
ROUNDS = (("r0", ("r4pick", "off", "r1pick")), ("r1", ("r1pick", "off", "r4pick")))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _bind_box(gpu: int) -> dict[str, Any]:
    """Point the imported runners at this box's lane and checkpoints."""
    _require(gpu in LANES, f"gpu {gpu} is not a registered lane here")
    lane = dict(LANES[gpu])
    lane["cache_root"] = f"{CACHE_ROOT}/{lane['lane_id']}"
    matrix.lane_for_block = lambda block_id=1, lane=lane: dict(lane)
    d3.lane_for_block = lambda block_id=1, lane=lane: dict(lane)
    r1.QUANT_CKPT.update(CKPTS)
    for path in CKPTS.values():
        _require(Path(path).exists(), f"checkpoint missing on this box: {path}")
    return lane


def measure(cell_key: str, trace: Path, out_json: Path) -> None:
    """Boot one cell and measure the two regimes."""
    from vllm import LLMEngine, SamplingParams

    cfg = CELLS[cell_key]
    manifest = r1._load_json(matrix._repository_path(d3.PROMPT_MANIFEST))
    specs = {r["regime_id"]: r for r in manifest["prompt_plan"]["regimes"]}
    engine = LLMEngine.from_engine_args(d3._engine_args(cfg))
    observations: dict[str, Any] = {}
    try:
        for regime_id in REGIMES:
            spec = specs[regime_id]
            before = d3._trace_len(trace)
            for index, tokens in enumerate(r1._prompts_for(regime_id, spec["batch"])):
                engine.add_request(
                    f"{regime_id}-{index}",
                    {"prompt_token_ids": tokens},
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=d3.MEASURE_TOKENS,
                        ignore_eos=True,
                    ),
                )
            while engine.has_unfinished_requests():
                engine.step()
            observations[regime_id] = d3._decode_rate(trace, before)
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()
    out_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_failclosed_validation_boot",
                "cell": cell_key,
                "config": cfg,
                "observations": observations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_all(output_dir: Path, gpu: int) -> None:
    lane = _bind_box(gpu)
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = sorted(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    traces = output_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    for round_id, order in ROUNDS:
        for cell_key in order:
            name = f"{round_id}_{cell_key}"
            target = output_dir / f"{name}.json"
            if target.exists():
                continue
            trace = traces / f"{name}.jsonl"
            _require(not trace.exists(), f"trace {trace} already exists")
            env = matrix._boot_child_environment(
                d3.boot_environment(CELLS[cell_key], trace, "corrected")
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
                        cell_key,
                        "--name",
                        name,
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
                (output_dir / f"{name}.FAILED").write_text(
                    f"returncode={completed.returncode}\n\n{text[-6000:]}",
                    encoding="utf-8",
                )
                print(f"[g98f] FAILED {name}", flush=True)
                continue
            record = r1._load_json(target)
            record["host_load"] = hostload.signature_from_log(text).as_record()
            record["gpu"] = gpu
            target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            rates = {
                r: round(o["decode_tokens_per_s"], 1)
                for r, o in record["observations"].items()
            }
            print(f"[g98f] {name} {rates}", flush=True)


def summarise(output_dir: Path) -> dict[str, Any]:
    rates: dict[str, dict[str, list[float]]] = {}
    boots: dict[str, Any] = {}
    for round_id, order in ROUNDS:
        for cell_key in order:
            path = output_dir / f"{round_id}_{cell_key}.json"
            if not path.is_file():
                boots[f"{round_id}_{cell_key}"] = {"status": "FAILED"}
                continue
            record = r1._load_json(path)
            per_regime = {
                r: round(o["decode_tokens_per_s"], 2)
                for r, o in record["observations"].items()
            }
            for regime, value in per_regime.items():
                rates.setdefault(regime, {}).setdefault(cell_key, []).append(value)
            boots[f"{round_id}_{cell_key}"] = {
                "status": "ok",
                "decode_tokens_per_s": per_regime,
                "host_load": (record.get("host_load") or {}).get("verdict"),
            }

    def mean(regime: str, cell: str) -> float | None:
        values = rates.get(regime, {}).get(cell)
        return statistics.mean(values) if values else None

    r4_armed, r4_off = mean("R4", "r4pick"), mean("R4", "off")
    r1_armed, r1_off = mean("R1", "r1pick"), mean("R1", "off")
    premise1 = premise2 = None
    if r4_armed and r4_off:
        premise1 = r4_armed < r4_off
    if r1_armed and r1_off:
        premise2 = r1_armed > r1_off * 1.05
    verdict = "INCOMPLETE"
    if premise1 is not None and premise2 is not None:
        if premise1 and premise2:
            verdict = "RULE_VALIDATED"
        elif not premise1:
            verdict = "PREMISE1_REFUTED_ARMED_WINS_AT_R4"
        else:
            verdict = "PREMISE2_REFUTED_ARMED_LOSES_AT_R1_TOO"
    return {
        "schema_version": 1,
        "record_type": "w98_failclosed_validation",
        "boots": boots,
        "R4": {
            "armed": r4_armed,
            "off": r4_off,
            "armed_over_off": (r4_armed / r4_off) if r4_armed and r4_off else None,
            "rule_says": "park",
        },
        "R1": {
            "armed": r1_armed,
            "off": r1_off,
            "armed_over_off": (r1_armed / r1_off) if r1_armed and r1_off else None,
            "rule_says": "arm",
        },
        "premise_1_r4_armed_loses": premise1,
        "premise_2_r1_armed_wins": premise2,
        "verdict": verdict,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--summarise-only", action="store_true")
    parser.add_argument("--cell", help=argparse.SUPPRESS)
    parser.add_argument("--name", help=argparse.SUPPRESS)
    parser.add_argument("--trace", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.cell:
        _bind_box(args.gpu)
        measure(args.cell, Path(args.trace).resolve(), output_dir / f"{args.name}.json")
        return 0
    if not args.summarise_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        run_all(output_dir, args.gpu)
    summary = summarise(output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
