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
CELL = "LO"
CAP = 32_768  # the registered Phase-100 cap for LO
# Phase 100 re-pins the geometry for exactly this reason: with the
# default 20480 a long LO generation runs into the context limit, the
# scheduler shortens the draft near it, and the K/OFF registry rejects
# the resulting unregistered width ("permits only K=0 or K=4, got K=2").
MAX_MODEL_LEN = 40_960
BATCH = int(os.environ.get("W98_LO_BATCH", "8"))
# The arms whose tau(u) curves are measured, plus the parked reference.
ARMS: dict[str, dict[str, Any]] = {
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
if os.environ.get("W98_LO_SWEEP") == "1":
    ARMS = dict(SWEEP)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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
    """One arm on the LO cell under natural EOS."""
    from vllm import LLMEngine, SamplingParams

    args = d3._engine_args(cfg)
    args.max_model_len = MAX_MODEL_LEN
    engine = LLMEngine.from_engine_args(args)
    outputs: dict[str, list[int]] = {}
    try:
        for index, tokens in enumerate(prompts(BATCH)):
            engine.add_request(
                f"{CELL}-{index}",
                {"prompt_token_ids": tokens},
                # Natural EOS: the cap is a safety net, not a budget.
                SamplingParams(temperature=0.0, max_tokens=CAP),
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
        env = matrix._boot_child_environment(
            d3.boot_environment(cfg, trace, "corrected")
        )
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
