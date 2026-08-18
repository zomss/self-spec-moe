# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B -- acceptance as a function of generated-suffix length, out to LO lengths.

The selector's acceptance input is a scalar per (cell, regime), pooled from
G98-D. That campaign generated 640 tokens, so with u-edges [256, 1024, 3072]
it populated ONLY buckets 0 and 1. Its existing data already shows the axis
is real and lever-dependent:

    skip0   tau 4.421 -> 4.420   across u<256 -> 256-1024   (flat)
    skip4   tau 4.229 -> 4.222                              (flat)
    skip8   tau 3.548 -> 3.775                              (+6.4%)

The refined evaluation's LO cell generates 5-25K tokens. Every step past 1K
of generation -- where the overwhelming majority of LO tokens live -- is
unmeasured, and the one lever that visibly moves with u is the one whose
acceptance was already the hardest to transfer. Extrapolating a flat scalar
5-25x beyond its calibration is the largest unquantified assumption the
selector would carry into that grid.

This measures it: LO content, long generation, u-edges extended so buckets 2
and 3 are populated for the first time.

Instrument, not a scored number. `ignore_eos` is used deliberately -- the
Phase-100 protocol retires it from SCORED runs while keeping it for
calibration, and here it is what guarantees every request reaches the long-u
buckets instead of stopping early and leaving them empty.
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
import run_w98_g98d0_smoke as d0  # noqa: E402
import run_w98_g98g_e2e as g98g  # noqa: E402
import w98_artifacts as artifacts  # noqa: E402
import w98d2_stream as stream  # noqa: E402

matrix = r1.matrix

# Extends the registered edges past 1024 so the LO regime's steps land in
# buckets of their own instead of all collapsing into the last one.
U_EDGES = (256, 1024, 3072, 8192)
GEN_TOKENS = int(os.environ.get("W98_LONGU_TOKENS", "8192"))
BATCH = int(os.environ.get("W98_LONGU_BATCH", "8"))
PROMPTS = (
    SCRIPT_DIR.parent.parent / "100_baselines/data/registration/w100_prompts.jsonl.gz"
)
CELL = os.environ.get("W98_LONGU_CELL", "LO")
# The three cells that define the u-curve, plus one windowed arm: the window
# bounds what the draft can see, so its acceptance has its own reason to move
# with generation length.
CONFIGS = [
    {"quant": "target-matching", "window": "off", "skip_count": 0},
    {"quant": "target-matching", "window": "off", "skip_count": 4},
    {"quant": "target-matching", "window": "off", "skip_count": 8},
    {"quant": "target-matching", "window": 512, "skip_count": 4},
]
if os.environ.get("W98_LONGU_WINDOWS") == "1":
    # The window sweep's acceptance curves. The LO throughput sweep showed
    # cost is NOT monotone in window size -- w128 and w256 measure slower
    # than w512 -- which no KV-bytes model produces. Acceptance is the only
    # candidate, and it was measured for w512 alone.
    CONFIGS = [
        {"quant": "target-matching", "window": w, "skip_count": 4}
        for w in (128, 256, 1024)
    ]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def prompts(limit: int) -> list[list[int]]:
    """Long-output cell prompts from the registered Phase-100 bundle."""
    _require(PROMPTS.is_file(), f"prompt bundle missing: {PROMPTS}")
    out = []
    with gzip.open(PROMPTS, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("cell") == CELL:
                out.append(row["token_ids"])
            if len(out) >= limit:
                break
    _require(bool(out), f"no prompts for cell {CELL}")
    return out


def measure(cfg: dict[str, Any], trace: Path, out: Path) -> None:
    from vllm import LLMEngine, SamplingParams

    engine = LLMEngine.from_engine_args(d0._engine_args(dict(cfg)))
    try:
        before = _trace_len(trace)
        for index, tokens in enumerate(prompts(BATCH)):
            engine.add_request(
                f"{CELL}-{index}",
                {"prompt_token_ids": tokens},
                SamplingParams(temperature=0.0, max_tokens=GEN_TOKENS, ignore_eos=True),
            )
        while engine.has_unfinished_requests():
            engine.step()
        rows = stream.rows_from_records(_records_after(trace, before))
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()
    profile = stream.tau_profile(rows, kmax=d0.KMAX, u_edges=U_EDGES)
    out.write_text(
        json.dumps(
            artifacts.envelope(
                cfg,
                "w98_long_u",
                {
                    "content_cell": CELL,
                    "u_edges": list(U_EDGES),
                    "gen_tokens": GEN_TOKENS,
                    "batch": BATCH,
                    "rows": len(rows),
                    "tau_profile": profile,
                },
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _trace_len(path: Path) -> int:
    if not Path(path).is_file():
        return 0
    with Path(path).open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _records_after(path: Path, before: int) -> list[dict[str, Any]]:
    """Decode-step records only, filtered exactly as the D2 campaign does.

    Acceptance rows are emitted only when a step actually ran target decode
    work, so prefill-only and excluded steps carry none; admitting them makes
    the stream parser reject the whole run.
    """
    out = []
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < before:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("record_type") != "koff_engine_step":
                continue
            if record.get("exclusion_reasons"):
                continue
            if not (record.get("counters") or {}).get("H_target_steps"):
                continue
            out.append(record)
    return out


def run_all(output_dir: Path, gpu: int) -> None:
    lane = g98g.bind_box(gpu)
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = sorted(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    traces = output_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    artifacts.plan(CONFIGS)
    for cfg in CONFIGS:
        name = artifacts.slug(cfg)
        target = output_dir / f"{name}.json"
        if artifacts.claim(target, cfg):
            continue
        trace = traces / f"{name}.jsonl"
        if trace.exists():
            trace.unlink()
        env = matrix._boot_child_environment(d0.boot_environment(dict(cfg), trace))
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
            print(f"[longu] FAILED {name}", flush=True)
            continue
        print(f"[longu] {name}", flush=True)


def summarise(output_dir: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        record = artifacts.read(path)
        buckets = record["tau_profile"]["buckets"]
        taus = {}
        for name, entry in sorted(buckets.items()):
            armed = int(entry.get("armed_steps", 0))
            if not armed:
                continue
            pos = entry.get("pos_accepted") or []
            taus[name] = {
                "tau_k4": round(1.0 + sum(pos[:4]) / armed, 4),
                "armed_steps": armed,
            }
        rows.append(
            {
                "content_cell": record["content_cell"],
                "config": record["config"],
                "tau": taus,
            }
        )
    return {
        "schema_version": 1,
        "record_type": "w98_long_u_summary",
        "u_edges": list(U_EDGES),
        "content_cell": CELL,
        "rows": rows,
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
    summary = summarise(output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
