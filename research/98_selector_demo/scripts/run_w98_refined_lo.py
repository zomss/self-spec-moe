# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refined-grid evaluation of the LO cell, and the test of tau(u).

Phase 100's protocol, applied to the arms whose acceptance curves
`results_g98_longu.md` measured: LO content from the registered bundle,
**natural EOS** (no `ignore_eos`), the registered 32K cap, throughput as
output tokens over wall seconds.

The question it answers is not "how fast is our selector" but **whether
acceptance has to be u-resolved to predict the LO cell**. Two predictions are
made from the same cost input and compared against the same measurement:

* `scalar`  -- tau pooled below 1K, which is what the selector consumed;
* `tau_eff` -- the token-weighted harmonic mean over the generation actually
  produced (`w98_tau_u`).

Because every arm here is distribution-preserving and T=0, all arms must emit
**identical tokens** and stop at the same EOS. That is the protocol's free
correctness gate and it is enforced: a per-request output digest is recorded
and any cross-arm divergence fails the run rather than being averaged into a
rate.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
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
# The registered Phase-100 grid. LO is the default because it is the cell
# this file was written for; the other three are selectable so the arming
# decision can be measured where Campaign 1 says speculation loses.
CAPS = {"LI": 2_048, "LO": 32_768, "LIO": 4_096, "SS": 1_024}
CELL = os.environ.get("W98_LO_CELL", "LO")
if CELL not in CAPS:
    raise SystemExit(f"unknown cell {CELL!r}; expected one of {sorted(CAPS)}")
CAP = CAPS[CELL]
# Phase 100 re-pins the geometry for exactly this reason: with the
# default 20480 a long LO generation runs into the context limit, the
# scheduler shortens the draft near it, and the K/OFF registry rejects
# the resulting unregistered width ("permits only K=0 or K=4, got K=2").
MAX_MODEL_LEN = 40_960
BATCH = int(os.environ.get("W98_LO_BATCH", "8"))
# Requests SUBMITTED, against BATCH concurrent. Phase 100 registers
# `n = 4 x max batch` ("submitted at once; continuous batching drains the
# set"); this runner has always submitted exactly BATCH, so every measurement
# in phases 98-101 ran an unbackfilled batch that drained 8 -> 1 with nothing
# to replace finished requests. E3 measured what that costs: occupancy swings
# +-30% with the prompt draw and throughput follows it, putting a 49-point
# band on a single-draw arm comparison. With n > BATCH the queue backfills and
# occupancy stays pinned until the queue empties, which is also the shape a
# real rollout has.
N_REQUESTS = int(os.environ.get("W98_LO_N", "0")) or BATCH
# Fixed-length mode: generation budget in tokens, EOS ignored. Zero keeps the
# scored protocol's natural EOS.
FIXED_TOKENS = int(os.environ.get("W98_LO_FIXED", "0"))
REPLICATE = os.environ.get("W98_LO_REPLICATE") == "1"
# The arms whose tau(u) curves are measured, plus the parked reference.
ARMS: dict[str, dict[str, Any]] = {
    # Plain vLLM: no speculative_config and no self-spec environment at all,
    # which is Campaign 1's denominator. `off` is NOT this -- it is our
    # runtime with the K schedule pinned to 0, so the gap between them is the
    # engine overhead our arms must also pay. Section 29 measured against
    # `off` and had to borrow Campaign 1's stock-to-off factor from another
    # box; this arm removes that borrowing.
    "stock": {
        "action": "stock",
        "quant": "target-matching",
        "window": "off",
        "skip_count": 0,
    },
    "off": {
        "action": "off",
        "quant": "target-matching",
        "window": "off",
        "skip_count": 0,
    },
    "base": {
        "action": "armed",
        "quant": "target-matching",
        "window": "off",
        "skip_count": 0,
    },
    "skip4": {
        "action": "armed",
        "quant": "target-matching",
        "window": "off",
        "skip_count": 4,
    },
    "skip8": {
        "action": "armed",
        "quant": "target-matching",
        "window": "off",
        "skip_count": 8,
    },
    "w512skip4": {
        "action": "armed",
        "quant": "target-matching",
        "window": 512,
        "skip_count": 4,
    },
}

# Window sweep at fixed skip, to IDENTIFY the superlinearity of the KV term.
# One residual cannot fix a curve; three more window sizes traced against the
# unwindowed arm can, because they vary KV positions with everything else
# held constant.
SWEEP: dict[str, dict[str, Any]] = {
    f"w{w}skip4": {
        "action": "armed",
        "quant": "target-matching",
        "window": w,
        "skip_count": 4,
    }
    for w in (128, 256, 1024)
}
# The quantized family: the same eight configurations under the OTHER draft
# weight version. Sections 2-13 scored `target-matching` exclusively while
# every selector pick in the 31-cell grid was `w4a16`, so the refined grid had
# never measured the arm family the selector actually chooses -- and the
# quant-axis calibration then showed the two families do not share a cost
# surface. Like for like: same arms, same protocol, same cell, and its own OFF
# so the two families' denominators can be checked against each other rather
# than assumed equal.
if os.environ.get("W98_LO_QUANT") == "1":
    ARMS = {
        name: {**cfg, "quant": "w4a16-quantized"}
        for name, cfg in {**ARMS, **SWEEP}.items()
    }
elif os.environ.get("W98_LO_SWEEP") == "1":
    ARMS = dict(SWEEP)
# A batch sweep needs a few arms at many batches rather than many arms at one,
# because what it separates is batch-SHARED cost from batch-proportional cost:
# the shared part falls as 1/B per token and the per-request part does not.
if os.environ.get("W98_LO_LATTICE") == "1":
    # The FULL registered lattice, both weight versions: window x skip over
    # every combination, not the five the refined grid sampled. Sections 39-43
    # scored the selector against a "best static" chosen from four arms of
    # which exactly one carried a window, so `w1024/skip4` won by being the
    # only windowed arm rather than by beating a field. The claim that no
    # static policy is best cannot be made on that set, and the untested
    # region -- windowed arms at deep skip -- is where D2(a) measured
    # composition to be CONSTRUCTIVE (interaction ratio up to 1.670): a window
    # that already discards the context removes the very information layer
    # skipping would have degraded, so their losses overlap rather than
    # compound.
    ARMS = {
        f"{q[0]}_{'woff' if w == 'off' else 'w' + str(w)}_skip{s}": {
            "action": "armed",
            "quant": q,
            "window": w,
            "skip_count": s,
        }
        for q in ("target-matching", "w4a16-quantized")
        for w in ("off", 128, 256, 512, 1024)
        for s in (0, 4, 8, 16)
    }
    ARMS["off"] = {
        "action": "off",
        "quant": "w4a16-quantized",
        "window": "off",
        "skip_count": 0,
    }
    ARMS["stock"] = {
        "action": "stock",
        "quant": "target-matching",
        "window": "off",
        "skip_count": 0,
    }
if os.environ.get("W98_LO_ARMS"):
    wanted = os.environ["W98_LO_ARMS"].split(",")
    missing = [name for name in wanted if name not in ARMS]
    if missing:
        raise SystemExit(f"unknown arms requested: {missing}")
    ARMS = {name: ARMS[name] for name in wanted}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def prompts(limit: int) -> list[list[int]]:
    """LO prompts, or one prompt replicated to fill the batch.

    Batch and content are otherwise the same variable in this design -- batch
    B means the first B prompts -- so "acceptance rises with batch" and
    "prompts 3-16 are more predictable than prompts 1-2" are indistinguishable.
    Replicating one prompt holds content exactly fixed while batch varies,
    which is what separates a numerical batch effect from a content effect.
    Prefix caching is off, so replicas share no state.
    """
    if REPLICATE:
        return [_load_prompts(1)[0] for _ in range(limit)]
    return _load_prompts(limit)


def _load_prompts(limit: int) -> list[list[int]]:
    """Prompts for this cell, optionally from a disjoint offset.

    `W98_LO_PROMPT_OFFSET` selects a different slice of the frozen bundle.
    Under natural EOS each arm stops each request at a slightly different
    place (section 21: batch-composition numerics flip near-tie argmaxes), so
    the ACTIVE batch over a run -- and therefore tokens per step -- depends on
    which prompts were drawn. Replicating boots cannot detect that, because a
    replicate reuses the same prompts; only a disjoint slice can.
    """
    offset = int(os.environ.get("W98_LO_PROMPT_OFFSET", "0"))
    out = []
    with gzip.open(PROMPTS, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("cell") == CELL:
                out.append(row["token_ids"])
            if len(out) >= limit + offset:
                break
    _require(
        len(out) >= limit + offset,
        f"not enough {CELL} prompts for offset {offset} + {limit}",
    )
    return out[offset : offset + limit]


def measure(cfg: dict[str, Any], trace: Path, out: Path) -> None:
    """One arm on the LO cell under natural EOS."""
    from vllm import LLMEngine, SamplingParams

    if cfg.get("action") == "stock":
        from vllm import EngineArgs

        args = EngineArgs(
            model=r1._engine_args({"quant": "target-matching"}).model,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=BATCH,  # concurrency; N_REQUESTS may exceed it
            max_num_batched_tokens=8192,
            enable_chunked_prefill=True,
            gpu_memory_utilization=0.90,
            enable_prefix_caching=False,
            enforce_eager=False,
            seed=0,
            disable_log_stats=True,
        )
    else:
        args = d3._engine_args(cfg)
        args.max_model_len = MAX_MODEL_LEN
    # The registered geometry pins max_num_seqs at 32. A batch above that is
    # silently CLAMPED rather than refused, so the run would report a batch it
    # never had -- the same class of defect as the filename collision
    # `w98_artifacts` exists to prevent, and invisible in the record because
    # every arm would be clamped identically.
    # Concurrency is BATCH; anything beyond it queues and backfills. The
    # earlier `max()` guarded against a silent clamp when BATCH exceeded the
    # registered 32, and still does, but it must not raise concurrency to N.
    args.max_num_seqs = BATCH
    if os.environ.get("W98_LO_K"):
        # Draft depth as a swept dimension. K was frozen at 4 for every scored
        # run of this phase; the measured acceptance profiles put the optimum
        # at 2 for fourteen of twenty-eight (cell, arm) pairs. Needs the
        # w98-ksweep boot scope, which is the only registry admitting K=1,2.
        k = int(os.environ["W98_LO_K"])
        args.speculative_config = {
            **args.speculative_config,
            "num_speculative_tokens": max(k, 1),
            "num_speculative_tokens_per_batch_size": [[1, 64, k]],
        }
    if os.environ.get("W98_LO_ASYNC") == "1":
        # Async scheduling moves draft-token bookkeeping into the worker, so
        # `_copy_draft_token_ids_to_cpu` takes its early return and the
        # per-step host round trip disappears (`gpu_model_runner.py:5078`).
        # Only P4 same-event capture forbids it, and the scored boots set no
        # capture paths, so `capture_enabled` is False here.
        args.async_scheduling = True
    engine = LLMEngine.from_engine_args(args)
    outputs: dict[str, list[int]] = {}
    try:
        for index, tokens in enumerate(prompts(N_REQUESTS)):
            engine.add_request(
                f"{CELL}-{index}",
                {"prompt_token_ids": tokens},
                # Natural EOS: the cap is a safety net, not a budget.
                # Under W98_LO_FIXED the cap becomes the budget and EOS is
                # ignored, which is how a batch sweep separates batch from
                # context. At T=0 the arms still do not emit identical tokens
                # -- batch-composition numerics flip near-tie argmaxes, so
                # every armed arm diverges from the parked one on 8 of 8
                # requests -- and under natural EOS that divergence moves
                # where each request STOPS. Generation length then varies by
                # arm and by batch, and it drives the drain term, which is
                # first order. Fixing the length makes the workload identical
                # by construction so a batch effect can be read as one.
                SamplingParams(
                    temperature=0.0,
                    max_tokens=FIXED_TOKENS or CAP,
                    ignore_eos=bool(FIXED_TOKENS),
                ),
            )
        started = time.perf_counter()
        while engine.has_unfinished_requests():
            for request_output in engine.step():
                if request_output.finished:
                    outputs[request_output.request_id] = list(
                        request_output.outputs[0].token_ids
                    )
        wall = time.perf_counter() - started
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()
    total = sum(len(v) for v in outputs.values())
    _require(total > 0, "no output tokens")
    digests = {
        rid: hashlib.sha256(
            json.dumps(toks, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        for rid, toks in sorted(outputs.items())
    }
    out.write_text(
        json.dumps(
            artifacts.envelope(
                cfg,
                "w98_refined_lo",
                {
                    "content_cell": CELL,
                    "batch": BATCH,
                    "cap": CAP,
                    "fixed_tokens": FIXED_TOKENS,
                    "replicated_prompt": REPLICATE,
                    "stopping_rule": "fixed" if FIXED_TOKENS else "natural_eos",
                    "wall_s": round(wall, 3),
                    "total_out_tokens": total,
                    "tokens_per_s": round(total / wall, 3),
                    "out_tokens_by_request": {
                        k: len(v) for k, v in sorted(outputs.items())
                    },
                    "output_digests": digests,
                    "cap_hits": sum(1 for v in outputs.values() if len(v) >= CAP),
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
    artifacts.plan(list(ARMS.values()))
    for name, cfg in ARMS.items():
        target = output_dir / f"{artifacts.slug(cfg)}.json"
        if artifacts.claim(target, cfg):
            continue
        trace = traces / f"{artifacts.slug(cfg)}.jsonl"
        if trace.exists():
            trace.unlink()
        # Scored runs should carry no instrument. The region profiler's inner
        # syncs cost +2.4 ms/step (amendment 2) and the koff trace writes a
        # JSON line per step, and BOTH are paid by our runtime and by neither
        # stock nor a deployment -- so leaving them on charges every armed arm
        # and `off` a cost the baseline never pays. Phase 100's campaign
        # runner turns both off for exactly this reason.
        # Stock is tested FIRST. It has no speculative config, so any
        # `VLLM_SELF_SPEC_*` left in its environment makes the boot-scope
        # validator refuse it ("minimal-B0 requires speculative method
        # 'draft_model', got None"). Ordering this after the instrument
        # branch silently dropped the stock arm from every point of a
        # campaign that set `W98_LO_NOINSTRUMENT`.
        if cfg.get("action") == "stock":
            base = d3.boot_environment(cfg, trace, "corrected")
            env = matrix._boot_child_environment(
                {k: v for k, v in base.items() if not k.startswith("VLLM_SELF_SPEC")}
            )
        if os.environ.get("W98_LO_SKIP_LAYERS"):
            # Override the layer IDENTITY at a fixed count. The registered
            # sets are NESTED (skip4 subset skip8 subset skip16) so that
            # `keep_frac` is a scalar the cost model can carry; D2(b) measured
            # a knapsack-chosen set beating that nested set at k=8 on its own
            # protocol, and this is what tests whether the nesting costs
            # throughput.
            env_override = os.environ["W98_LO_SKIP_LAYERS"]
        else:
            env_override = None
        if os.environ.get("W98_LO_K"):
            # The K sweep needs its own registry; `boot_environment` pins the
            # scored scope, which admits only K=4 and would refuse the boot.
            base = d3.boot_environment(cfg, trace, "corrected")
            base["VLLM_SELF_SPEC_BOOT_SCOPE"] = "w98-ksweep"
            base["VLLM_SELF_SPEC_PROFILE"] = "0"
            base.pop("VLLM_SELF_SPEC_KOFF_TRACE", None)
            env = matrix._boot_child_environment(base)
        elif os.environ.get("W98_LO_NOINSTRUMENT") == "1":
            base = d3.boot_environment(cfg, trace, "corrected")
            base["VLLM_SELF_SPEC_PROFILE"] = "0"
            base.pop("VLLM_SELF_SPEC_KOFF_TRACE", None)
            env = matrix._boot_child_environment(base)
        else:
            env = matrix._boot_child_environment(
                d3.boot_environment(cfg, trace, "corrected")
            )
        if env_override:
            env["VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS"] = env_override
        log = output_dir / f"{artifacts.slug(cfg)}.log"
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
            (output_dir / f"{artifacts.slug(cfg)}.FAILED").write_text(
                text[-4000:], encoding="utf-8"
            )
            print(f"[lo] FAILED {name}", flush=True)
            continue
        rec = artifacts.read(target)
        print(
            f"[lo] {name:10s} {rec['tokens_per_s']:8.1f} tok/s  "
            f"{rec['total_out_tokens']:7d} tokens  caps {rec['cap_hits']}",
            flush=True,
        )


def divergence_gate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Every distribution-preserving arm must emit identical tokens at T=0."""
    reference = None
    mismatches = []
    for record in records:
        digests = record["output_digests"]
        if reference is None:
            reference = (record["cell"], digests)
            continue
        for rid, digest in digests.items():
            if reference[1].get(rid) != digest:
                mismatches.append(
                    f"{record['cell']} diverges from {reference[0]} on {rid}"
                )
    return {
        "reference": reference[0] if reference else None,
        "mismatches": mismatches,
        "ok": not mismatches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--summarise-only", action="store_true")
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
    if not args.summarise_only:
        run_all(output_dir, args.gpu)
    records = [
        artifacts.read(p)
        for p in sorted(output_dir.glob("*.json"))
        if p.name != "summary.json"
    ]
    summary = {
        "schema_version": 1,
        "record_type": "w98_refined_lo_summary",
        "content_cell": CELL,
        "batch": BATCH,
        "divergence_gate": divergence_gate(records),
        "arms": {
            r["cell"]: {
                "tokens_per_s": r["tokens_per_s"],
                "total_out_tokens": r["total_out_tokens"],
                "wall_s": r["wall_s"],
                "cap_hits": r["cap_hits"],
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
