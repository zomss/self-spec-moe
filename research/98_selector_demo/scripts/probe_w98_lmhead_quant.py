# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X16 -- what does quantizing the draft's lm_head buy, and what does it cost?

DIAGNOSTIC, NOT SCORED.

The draft's `lm_head` is a 1.245 GB bf16 full-vocab GEMM run once per chain
step: 4.98 GB/step, 31% of the draft's weight traffic, measured at 412.8 us/call
and 90% of bandwidth-bound. It is also the `F` term that breaks the cost model,
since the model multiplies ALL weight bytes by keep_frac while lm_head does not
scale with skipped layers at all.

The standard recipe excludes it (`ignore=["lm_head"]`) because for a STANDALONE
model its output is the product. A speculative draft is different: every draft
token is verified by the target, so a degraded draft logit is rejected rather
than emitted. Quantizing it cannot change correctness -- only ACCEPTANCE, which
is the thing to measure.

So this probe reports BOTH halves and refuses to treat either alone as the
answer:

  * speed    -- draft_chain and whole-step time
  * quality  -- accepted tokens per armed step, from the same trace

A step-time win bought with an acceptance collapse is a loss. The two arms run
the identical lattice cell and the identical prompts; only the draft checkpoint
differs.

Arms are interleaved (baseline, lmhead, baseline, lmhead, ...) so session drift
hits both equally -- the lesson from X5, where consecutive repeats understated
drift by 50x.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402

matrix = r1.matrix

BASELINE_CKPT = "/data/smcho/ckpts/Qwen3-8B-W4A16-INT4"
LMHEAD_CKPT = "/data/smcho/ckpts/Qwen3-8B-W4A16-INT4-lmhead"
ARMS = {"baseline": BASELINE_CKPT, "lmhead": LMHEAD_CKPT}

# The cell where the residual is worst, plus the quant single for reference.
CELLS = [
    {"quant": "w4a16-quantized", "window": 128, "skip_count": 8},
    {"quant": "w4a16-quantized", "window": "off", "skip_count": 0},
]
# Restricted to the SHORT-context regimes. That is not only a time budget: the
# non-per-layer cost F this probe targets is +3.66 ms at R1 and +3.51 at R8 but
# fits NEGATIVE at R4/R5 (-1.03, -2.77), so the short regimes are exactly where
# an lm_head effect should be visible. The long regimes cost most of the wall
# time (14k-token prefill x batch 8) and would show the effect least.
PROBE_REGIMES = ["R1", "R8"]
REPEATS = 2
ORDER = [
    (rep, ci, arm) for rep in range(REPEATS) for ci in range(len(CELLS)) for arm in ARMS
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _measure_regimes(cfg: dict[str, Any], trace: Path) -> dict[str, Any]:
    """r1.measure_config restricted to PROBE_REGIMES.

    Mirrors the scored measurement exactly -- same profiler, same warmup, same
    draft_chain quantity -- and differs only in which regimes it visits.
    """
    from vllm import LLMEngine, SamplingParams
    from vllm.v1.spec_decode.self_spec_profiler import get_profiler

    manifest = r1._load_json(matrix._repository_path(r1.PROMPT_MANIFEST))
    regimes = {r["regime_id"]: r for r in manifest["prompt_plan"]["regimes"]}
    profiler = get_profiler()
    _require(profiler.enabled, "the self-spec profiler is not enabled")
    engine = LLMEngine.from_engine_args(r1._engine_args(cfg))
    out: dict[str, Any] = {}
    try:
        for regime_id in PROBE_REGIMES:
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
            steps = r1._armed_steps(trace, before)
            summary = profiler.summary(warmup=r1.PROFILER_WARMUP)
            chain = summary.get("draft_chain", {})
            out[regime_id] = {
                "armed_step_count": len(steps),
                "draft_chain_s": (
                    chain["mean_ms"] / 1000.0 if chain.get("mean_ms") else None
                ),
                "draft_chain_samples": chain.get("n", 0),
                "mean_armed_step_s": statistics.mean(steps) if steps else None,
                "stdev_armed_step_s": (
                    statistics.stdev(steps) if len(steps) > 1 else None
                ),
                "batch": spec["batch"],
            }
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()
    return out


def measure(cfg: dict[str, Any], ckpt: str, trace: Path, out_json: Path) -> None:
    """Round-1 measurement plus the acceptance rate from the same trace.

    The draft checkpoint comes from the module-level QUANT_CKPT map, not an
    environment variable, so the arm is selected by overriding that map in the
    child before measuring.
    """
    r1.QUANT_CKPT["w4a16-quantized"] = ckpt
    observations = _measure_regimes(cfg, trace)
    # Acceptance, from the SAME trace the timing came from. The step record
    # carries A_accepted (draft tokens accepted), D_armed (armed draft steps)
    # and E_committed (tokens committed). Acceptance per armed step is the
    # quantity that decides whether a step-time win is real: at K=4 a step
    # commits 1 (all rejected) to 5 (all accepted) tokens.
    accepted: dict[str, Any] = {}
    if trace.is_file():
        a = d = e = 0
        armed_steps = 0
        with trace.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                c = row.get("counters") or {}
                if not c.get("D_armed"):
                    continue
                armed_steps += 1
                a += int(c.get("A_accepted", 0))
                d += int(c.get("D_armed", 0))
                e += int(c.get("E_committed", 0))
        if armed_steps:
            accepted = {
                "armed_steps": armed_steps,
                "A_accepted": a,
                "D_armed": d,
                "E_committed": e,
                "accepted_per_armed_step": a / d if d else None,
                "committed_per_armed_step": e / d if d else None,
            }
    out_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_lmhead_quant",
                "config": cfg,
                "observations": observations,
                "acceptance": accepted,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_all(output_dir: Path) -> None:
    lane = matrix.lane_for_block(1)
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    traces = output_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    for rep, ci, arm in ORDER:
        cfg = CELLS[ci]
        name = f"r{rep}_{r1._config_key(cfg).replace('/', '_')}_{arm}"
        target = output_dir / f"{name}.json"
        if target.exists():
            continue
        trace = traces / f"{name}.jsonl"
        _require(not trace.exists(), f"trace {trace} already exists")
        # Only the draft checkpoint differs between arms; it is passed to the
        # child in the arm payload, not the environment.
        env = matrix._boot_child_environment(r1.boot_environment(cfg, trace))
        log = output_dir / f"{name}.log"
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output-dir",
                    str(output_dir),
                    "--arm",
                    json.dumps({"name": name, "cfg": cfg, "ckpt": ARMS[arm]}),
                    "--trace",
                    str(trace),
                ],
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        if completed.returncode != 0 or not target.exists():
            (output_dir / f"{name}.FAILED").write_text(
                f"returncode={completed.returncode}\n", encoding="utf-8"
            )
            continue
        text = log.read_text(encoding="utf-8", errors="replace")
        record = r1._load_json(target)
        record["arm"] = arm
        record["draft_ckpt"] = ARMS[arm]
        # Verify the intended checkpoint actually loaded.
        # The engine logs the speculative model path; confirm the intended
        # checkpoint actually loaded rather than assuming the override took.
        record["ckpt_confirmed"] = ARMS[arm] in text
        record["kernel_resolved"] = next(
            (
                tok
                for line in text.splitlines()
                if "for CompressedTensorsWNA16" in line
                for tok in line.split()
                if tok.endswith("LinearKernel")
            ),
            None,
        )
        target.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", help=argparse.SUPPRESS)
    parser.add_argument("--trace", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.arm:
        arm = json.loads(args.arm)
        measure(
            arm["cfg"],
            arm["ckpt"],
            Path(args.trace).resolve(),
            output_dir / f"{arm['name']}.json",
        )
        return 0
    with contextlib.suppress(Exception):
        base_env = matrix._boot_child_environment({})
        matrix._preflight_native_sampler(base_env)
        matrix._preflight_inprocess_engine_core(base_env)
    lane = matrix.lane_for_block(1)
    matrix._preflight_gpu_identity_and_idle(
        lane["physical_gpu_index"], lane["physical_gpu_uuid"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_all(output_dir)
    print(json.dumps({"arms": len(ORDER)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
