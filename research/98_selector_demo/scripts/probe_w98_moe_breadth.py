# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""G98-H -- does the two-round design transfer to a sparse architecture?

The phase's largest stated limitation is that everything is one model on one
hardware family: Qwen3-8B dense. Its central claim, though, is architectural
rather than model-specific -- cost is offline-predictable, acceptance is not
-- and C1 found dense and sparse behave OPPOSITELY along the batch axis
(dense wins at low batch, sparse only marginally at high batch, because a
sparse draft step is dispatch-bound rather than bandwidth-bound). So the
design deserves a test on a sparse target, and this box has one cached.

Target: Qwen3-30B-A3B, 48 layers, 128 experts, same tokenizer as Qwen3-8B
(vocab 151936) -- which is why the frozen prompt bundle transfers verbatim
and no re-tokenisation is needed. DeepSeek-V2-Lite and Llama-3.1 are also on
this box but use different tokenisers, so their prompts would have to be
rebuilt before they could be compared to anything.

**Shared weights only.** The MoE target is ~60 GB in bf16 and its W4A16
draft is another 16 GB, which does not fit beside it on one 80 GB device.
The quant axis is therefore dropped and only window x skip are swept. That
has a modelling consequence worth stating rather than hiding: with a single
weight version the `keep * W_layer` column is collinear with `keep`, so the
five-parameter model is not identifiable and a FOUR-parameter reduction is
fitted instead (`kappa_kv`, `c_layer`, `f_win`, `F`).

What this can and cannot answer
-------------------------------

CAN: whether the cost model's *structure* survives a sparse architecture --
fit it on single-lever cells, predict composed cells never measured, and
compare. That is D1' in miniature on a new architecture.

CANNOT: anything about the quantization lever on MoE, or a selector claim
against an omniscient baseline over a full lattice. The skip SETS are also
inherited from the 36-layer dense model rather than re-derived for 48 layers,
so they are a valid perturbation but not a tuned one.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
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

MOE_TARGET = (
    "/data/smcho/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/"
    "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
)
# Lower than the dense campaign's: a 60 GB target leaves little headroom, and
# an OOM at capture wastes a four-minute boot.
MOE_GPU_UTIL = 0.92
REGIMES = tuple(os.environ.get("W98_MOE_REGIMES", "R1,R6").split(","))
# Single-lever cells first (they identify the model), then composed cells the
# fit never saw. Held-out cells vary two axes at once, as D1' requires.
FIT_CELLS = [
    {"quant": "target-matching", "window": "off", "skip_count": 0},
    {"quant": "target-matching", "window": 256, "skip_count": 0},
    {"quant": "target-matching", "window": 1024, "skip_count": 0},
    {"quant": "target-matching", "window": "off", "skip_count": 4},
    {"quant": "target-matching", "window": "off", "skip_count": 8},
]
HELDOUT_CELLS = [
    {"quant": "target-matching", "window": 256, "skip_count": 4},
    {"quant": "target-matching", "window": 1024, "skip_count": 8},
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _slug(cfg: dict[str, Any]) -> str:
    window = cfg["window"]
    return f"moe_w{window}_skip{cfg['skip_count']}"


def _engine_args(cfg: dict[str, Any]):
    """Round-1 engine arguments, retargeted at the MoE and shared weights."""
    args = r1._engine_args(cfg)
    args.model = MOE_TARGET
    args.speculative_config = {
        **args.speculative_config,
        "model": MOE_TARGET,
        "num_speculative_tokens": 4,
    }
    args.gpu_memory_utilization = MOE_GPU_UTIL
    return args


def measure(cfg: dict[str, Any], trace: Path, out: Path) -> None:
    from vllm import LLMEngine, SamplingParams

    manifest = r1._load_json(matrix._repository_path(d3.PROMPT_MANIFEST))
    specs = {r["regime_id"]: r for r in manifest["prompt_plan"]["regimes"]}
    engine = LLMEngine.from_engine_args(_engine_args(cfg))
    observations: dict[str, Any] = {}
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
                        temperature=0.0, max_tokens=d3.MEASURE_TOKENS, ignore_eos=True
                    ),
                )
            while engine.has_unfinished_requests():
                engine.step()
            observations[regime] = d3._decode_rate(trace, before)
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()
    out.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_moe_breadth_cell",
                "target": "Qwen3-30B-A3B",
                "config": cfg,
                "cell": _slug(cfg),
                "observations": observations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_all(output_dir: Path, gpu: int, cells: list[dict[str, Any]]) -> None:
    lane = g98g.bind_box(gpu)
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = sorted(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    traces = output_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    for cfg in cells:
        name = _slug(cfg)
        target = output_dir / f"{name}.json"
        if target.exists():
            continue
        trace = traces / f"{name}.jsonl"
        if trace.exists():
            trace.unlink()
        env = matrix._boot_child_environment(r1.boot_environment(cfg, trace))
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
                    json.dumps(cfg),
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
            text = log.read_text(encoding="utf-8", errors="replace")
            (output_dir / f"{name}.FAILED").write_text(text[-4000:], encoding="utf-8")
            print(f"[moe] FAILED {name}", flush=True)
            continue
        record = r1._load_json(target)
        rates = {
            r: round(o["decode_tokens_per_s"], 1)
            for r, o in record["observations"].items()
        }
        print(f"[moe] {name} {rates}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="one cell only")
    parser.add_argument("--cell", help=argparse.SUPPRESS)
    parser.add_argument("--trace", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.cell:
        g98g.bind_box(args.gpu)
        cfg = json.loads(args.cell)
        output_dir.mkdir(parents=True, exist_ok=True)
        measure(cfg, Path(args.trace).resolve(), output_dir / f"{_slug(cfg)}.json")
        return 0
    _require(Path(MOE_TARGET).exists(), f"MoE target missing: {MOE_TARGET}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = FIT_CELLS[:1] if args.smoke else FIT_CELLS + HELDOUT_CELLS
    run_all(output_dir, args.gpu, cells)
    records = [
        r1._load_json(p)
        for p in sorted(output_dir.glob("moe_*.json"))
        if p.name != "summary.json"
    ]
    summary = {
        "schema_version": 1,
        "record_type": "w98_moe_breadth",
        "target": "Qwen3-30B-A3B",
        "cells_measured": len(records),
        "rates": {
            r["cell"]: {
                k: round(v["decode_tokens_per_s"], 2)
                for k, v in r["observations"].items()
            }
            for r in records
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
