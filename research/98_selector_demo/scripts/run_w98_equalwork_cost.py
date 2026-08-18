# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equal-work cost calibration on the refined cell's context regime.

`results_refined_lo.md` section 10 tried to refit the cost coefficients from
the natural-EOS throughput runs and could not, for two reasons that were in
the data rather than the arithmetic:

* `f_win` and `kappa_kv` correlated at -0.985, because the window varied at
  ONE keep, so nothing separated the window constant from the KV term;
* each arm ran a different realized workload -- 104K to 159K tokens, a 1.52x
  spread -- because natural EOS plus T=0 divergence gives every arm its own
  generation lengths, so the regressors encoded workload as well as
  configuration.

This run fixes both, and fixes them with the protocol rather than the model.

**Equal work.** `ignore_eos` with a fixed token budget, so every arm emits
exactly the same tokens over exactly the same contexts. That is Phase 98's
calibration instrument, which Phase 100 retires from SCORING while keeping
for MEASUREMENT -- and section 10 is the demonstration of why the boundary
exists.

**Cost measured, not inverted.** The draft chain is read straight from the
profiler instead of being backed out of throughput through an acceptance
model. Acceptance then drops out of the calibration entirely, which removes
the largest remaining confound: no tau curve, no ladder parking, no
inversion.

**Window varied at three keeps.** 3 windows x 3 skips breaks the collinearity
that made the previous fit degenerate: `f_win` is now identified by arms that
share a window while differing in keep, and `kappa_kv` by arms that share a
keep while differing in window.

The profiler's own syncs cost ~2.4 ms/step (X25) and inflate every arm's
draft chain equally, so they land in the fitted floor `F` and are named here
rather than silently carried.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
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
import w98_artifacts as artifacts  # noqa: E402

matrix = r1.matrix

PROMPTS = (
    SCRIPT_DIR.parent.parent / "100_baselines/data/registration/w100_prompts.jsonl.gz"
)
CELL = os.environ.get("W98_EW_CELL", "LO")
BATCH = int(os.environ.get("W98_EW_BATCH", "8"))
GEN = int(os.environ.get("W98_EW_TOKENS", "8192"))
MAX_MODEL_LEN = 40_960
WINDOWS = ("off", 256, 1024)
SKIPS = (0, 4, 8)
KEEP = {0: 1.0, 4: 32 / 36, 8: 28 / 36}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def arms() -> list[dict[str, Any]]:
    """Window x keep, plus w512/skip4 for continuity with the earlier sweep.

    On cells other than LO this trims to four arms spanning the levers: the
    question there is whether LO's coefficients TRANSFER, which four points
    answer, not whether a second lattice can be fitted.
    """
    if CELL != "LO":
        return [
            {
                "action": "armed",
                "quant": "target-matching",
                "window": w,
                "skip_count": s,
            }
            for w, s in (("off", 0), ("off", 4), (256, 4), (1024, 4))
        ]
    out = [
        {"action": "armed", "quant": "target-matching", "window": w, "skip_count": s}
        for w in WINDOWS
        for s in SKIPS
    ]
    out.append(
        {"action": "armed", "quant": "target-matching", "window": 512, "skip_count": 4}
    )
    return out


def prompts(limit: int) -> list[list[int]]:
    out = []
    with gzip.open(PROMPTS, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("cell") == CELL:
                out.append(row["token_ids"])
            if len(out) >= limit:
                break
    _require(len(out) >= limit, f"not enough {CELL} prompts")
    return out


def measure(cfg: dict[str, Any], trace: Path, out: Path) -> None:
    """Boot one arm and read its draft chain straight from the profiler."""
    from vllm import LLMEngine, SamplingParams
    from vllm.v1.spec_decode.self_spec_profiler import get_profiler

    profiler = get_profiler()
    _require(profiler.enabled, "the self-spec profiler is not enabled")
    args = d3._engine_args(cfg)
    args.max_model_len = MAX_MODEL_LEN
    engine = LLMEngine.from_engine_args(args)
    try:
        profiler.reset()
        for index, tokens in enumerate(prompts(BATCH)):
            engine.add_request(
                f"{CELL}-{index}",
                {"prompt_token_ids": tokens},
                # EQUAL WORK: every arm emits exactly GEN tokens per request
                # over exactly the same contexts.
                SamplingParams(temperature=0.0, max_tokens=GEN, ignore_eos=True),
            )
        steps = 0
        while engine.has_unfinished_requests():
            engine.step()
            steps += 1
        summary = profiler.summary(warmup=r1.PROFILER_WARMUP)
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()
    chain = summary.get("draft_chain") or {}
    verify = summary.get("verify") or {}
    _require(bool(chain.get("mean_ms")), "no draft chain samples")
    out.write_text(
        json.dumps(
            artifacts.envelope(
                cfg,
                "w98_equalwork_cost",
                {
                    "content_cell": CELL,
                    "batch": BATCH,
                    "gen_tokens": GEN,
                    "prompt_tokens": len(prompts(BATCH)[0]),
                    "engine_steps": steps,
                    "draft_chain_ms": chain["mean_ms"],
                    "draft_chain_samples": chain.get("count"),
                    "verify_ms": verify.get("mean_ms"),
                    "keep_frac": KEEP[cfg["skip_count"]],
                },
            ),
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
    planned = arms()
    artifacts.plan(planned)
    for cfg in planned:
        name = artifacts.slug(cfg)
        target = output_dir / f"{name}.json"
        if artifacts.claim(target, cfg):
            continue
        trace = traces / f"{name}.jsonl"
        if trace.exists():
            trace.unlink()
        env = matrix._boot_child_environment(d3.boot_environment(cfg, trace, "legacy"))
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
                    "--config",
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
            print(f"[ew] FAILED {name}", flush=True)
            continue
        record = artifacts.read(target)
        print(
            f"[ew] {name:34s} draft {record['draft_chain_ms']:7.3f} ms  "
            f"steps {record['engine_steps']}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", help=argparse.SUPPRESS)
    parser.add_argument("--trace", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.config:
        g98g.bind_box(args.gpu)
        cfg = json.loads(args.config)
        output_dir.mkdir(parents=True, exist_ok=True)
        measure(
            cfg, Path(args.trace).resolve(), output_dir / f"{artifacts.slug(cfg)}.json"
        )
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    run_all(output_dir, args.gpu)
    records = [
        artifacts.read(p)
        for p in sorted(output_dir.glob("*.json"))
        if p.name != "summary.json"
    ]
    summary = {
        "schema_version": 1,
        "record_type": "w98_equalwork_cost_summary",
        "content_cell": CELL,
        "batch": BATCH,
        "gen_tokens": GEN,
        "equal_work": True,
        "arms": {
            r["cell"]: {
                "draft_chain_ms": r["draft_chain_ms"],
                "verify_ms": r["verify_ms"],
                "keep_frac": r["keep_frac"],
                "engine_steps": r["engine_steps"],
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
