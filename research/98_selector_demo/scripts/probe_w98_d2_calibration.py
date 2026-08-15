# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""D2 calibration -- two assumptions the campaign design rests on. NON-SCORED.

Both are registered in `w98_prereg.md` and neither has ever been measured:

  * **Boot determinism.** The D2 measurement contract replicates over content
    seeds, NOT over boots, "since acceptance is near-deterministic under
    greedy at fixed batch and realization". If that is false the sampling
    plan is wrong: boot-to-boot variance would have to enter the interval,
    and 2,000-4,000 armed steps per cell would be sizing against the wrong
    noise source. Arm: the same configuration booted twice, captured
    realization, compared per position and per request.
  * **The realization bridge.** Screening may run eager while finalists are
    confirmed under the deployed (captured) realization, and survivors must
    clear `tau*` by more than the bridge width -- so the width has to be
    measured before it can gate anything. Arm: the same configuration under
    `enforce_eager=True` against the captured baseline.

Runs on the quantized path: target-matching is degenerate for this purpose
(a draft whose weights are the target's accepts everything, so it cannot
show a difference between realizations or boots -- see G98-D0).

    numactl --physcpubind=56-63 --membind=1 \
      .venv/bin/python research/98_selector_demo/scripts/probe_w98_d2_calibration.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402
import run_w98_g98d0_smoke as d0  # noqa: E402

matrix = r1.matrix
OUTPUT_PATH = "research/98_selector_demo/data/probe_d2_calibration"
CFG = {"quant": "w4a16-quantized", "window": "off", "skip_count": 0}
ARMS = ("captured_a", "captured_b", "eager")


def _engine_args(cfg: dict[str, Any], eager: bool):
    args = d0._engine_args(cfg)
    args.enforce_eager = eager
    return args


def _acceptance_profile(trace_path: Path) -> dict[str, Any]:
    """Per-position and per-request acceptance from one boot's trace."""
    armed, priming, violations = d0._decode_records(trace_path)
    reached = [0] * d0.KMAX
    accepted_hist = [0] * d0.KMAX
    per_request: dict[str, list[int]] = {}
    total_accepted = rows = 0
    for record in armed:
        for row in record.get("acceptance_rows") or []:
            accepted = row["accepted_draft_tokens"]
            for position in range(d0.KMAX):
                if position <= accepted:
                    reached[position] += 1
                if position < accepted:
                    accepted_hist[position] += 1
            # Request ids carry a per-boot suffix; key on the stable prefix
            # so the same logical request lines up across boots.
            key = row["request_id"].rsplit("-", 1)[0]
            per_request.setdefault(key, []).append(accepted)
            total_accepted += accepted
            rows += 1
    return {
        "armed_steps": len(armed),
        "priming_steps": priming,
        "unarmed_violations": violations,
        "rows": rows,
        "mean_accepted": round(total_accepted / max(rows, 1), 6),
        "per_position_reached": reached,
        "per_position_accepted": accepted_hist,
        "per_position_conditional": [
            round(a / r, 6) if r else None for a, r in zip(accepted_hist, reached)
        ],
        "per_request_sequences": {k: v for k, v in sorted(per_request.items())},
    }


def measure(arm: str, trace_path: Path) -> dict[str, Any]:
    from vllm import LLMEngine, SamplingParams

    manifest = r1._load_json(matrix._repository_path(r1.PROMPT_MANIFEST))
    regimes = {r["regime_id"]: r for r in manifest["prompt_plan"]["regimes"]}
    engine = LLMEngine.from_engine_args(_engine_args(CFG, eager=arm == "eager"))
    try:
        for regime_id in d0.SMOKE_REGIMES:
            prompts = r1._prompts_for(regime_id, regimes[regime_id]["batch"])
            for index, tokens in enumerate(prompts):
                engine.add_request(
                    f"{regime_id}-{index}",
                    {"prompt_token_ids": tokens},
                    SamplingParams(
                        temperature=0.0, max_tokens=d0.SMOKE_TOKENS, ignore_eos=True
                    ),
                )
            while engine.has_unfinished_requests():
                engine.step()
    finally:
        import contextlib

        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()
    return _acceptance_profile(trace_path)


def _compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Exact and aggregate agreement between two acceptance profiles."""
    shared = sorted(
        set(left["per_request_sequences"]) & set(right["per_request_sequences"])
    )
    identical = sum(
        left["per_request_sequences"][k] == right["per_request_sequences"][k]
        for k in shared
    )
    steps_compared = mismatched = 0
    for key in shared:
        a = left["per_request_sequences"][key]
        b = right["per_request_sequences"][key]
        for x, y in zip(a, b):
            steps_compared += 1
            mismatched += x != y
    return {
        "requests_compared": len(shared),
        "requests_bit_identical": identical,
        "steps_compared": steps_compared,
        "steps_mismatched": mismatched,
        "mean_accepted_delta": round(right["mean_accepted"] - left["mean_accepted"], 6),
        "per_position_conditional_delta": [
            None if a is None or b is None else round(b - a, 6)
            for a, b in zip(
                left["per_position_conditional"], right["per_position_conditional"]
            )
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--arm", help=argparse.SUPPRESS)
    parser.add_argument("--trace", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output_dir = (args.output_dir or matrix._repository_path(OUTPUT_PATH)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.arm:
        profile = measure(args.arm, Path(args.trace).resolve())
        (output_dir / f"{args.arm}.json").write_text(
            json.dumps(profile, indent=2, sort_keys=True) + "\n"
        )
        return 0

    lane = matrix.lane_for_block(1)
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = set(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    traces = output_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)

    profiles: dict[str, Any] = {}
    for arm in ARMS:
        trace = traces / f"{arm}.jsonl"
        trace.unlink(missing_ok=True)
        env = matrix._boot_child_environment(d0.boot_environment(CFG, trace))
        log = output_dir / f"{arm}.log"
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output-dir",
                    str(output_dir),
                    "--arm",
                    arm,
                    "--trace",
                    str(trace),
                ],
                cwd=r1.REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                preexec_fn=lambda: os.sched_setaffinity(0, affinity),
            )
        if completed.returncode != 0:
            raise RuntimeError(f"arm {arm} failed; see {log}")
        profiles[arm] = r1._load_json(output_dir / f"{arm}.json")
        print(
            f"[d2cal] {arm}: mean_accepted="
            f"{profiles[arm]['mean_accepted']} rows={profiles[arm]['rows']}",
            flush=True,
        )

    record = {
        "schema_version": 1,
        "record_type": "w98_d2_calibration",
        "scored": False,
        "config": CFG,
        "kmax": d0.KMAX,
        "regimes": list(d0.SMOKE_REGIMES),
        "boot_determinism": _compare(profiles["captured_a"], profiles["captured_b"]),
        "realization_bridge": _compare(profiles["captured_a"], profiles["eager"]),
        "profiles": {
            arm: {k: v for k, v in p.items() if k != "per_request_sequences"}
            for arm, p in profiles.items()
        },
    }
    (output_dir / "d2_calibration.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "boot_determinism": record["boot_determinism"],
                "realization_bridge": record["realization_bridge"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
