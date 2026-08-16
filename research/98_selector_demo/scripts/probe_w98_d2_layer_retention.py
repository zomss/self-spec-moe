# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""D2(b) step 1 — per-layer acceptance retention by leave-one-out. NON-SCORED.

The knapsack, and the worst-set control it is compared against, both consume
a per-layer retention vector `r`. Nothing in the preregistration says how to
obtain it and no artifact carries it, so this measures it directly:

    r_i = tau(skip only layer i) / tau(skip nothing)

both at KMAX=8, on the same position-resolved stream the rest of D2 uses.
That needs a ONE-layer skip, which the frozen counts {0,4,8,16} forbid; skip
count 1 is admitted for this probe under `w98-d2` only, additively, leaving
the cost campaigns' scope untouched (decision recorded 2026-08-16 in
`design_g98d_d2_campaign.md`).

**This probe claims nothing.** It is an input to the knapsack in the same
way Round 1's single-lever profiles are inputs to a fit whose predictions
carry the claim. D2(b)'s claim stays what was registered: at each surviving
count, the knapsack set beats its count-matched controls on confirmed tau.

**Content seed 4 only.** The confirmation runs on seed 5, so the sets are
never chosen using the content that later judges them — Phase 84 retracted a
lever whose gate was derived from its own evaluation set, and the rule it
produced is that profiled artifacts are built on disjoint data.
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
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402
import run_w98_g98d0_smoke as d0  # noqa: E402

# Importing the D3 runner rebinds matrix.lane_for_block to the HOST-keyed
# lane, so this probe runs on whichever registered box it is launched from
# rather than the one the round-1 module hardcodes.
import run_w98_g98e_d3 as d3  # noqa: E402,F401
import w98d2_stream as stream  # noqa: E402

matrix = r1.matrix
OUTPUT_PATH = "research/98_selector_demo/data/probe_d2_retention"
PROMPT_MANIFEST = "research/98_selector_demo/data/prereg/w98d2_prompt_manifest.json"
PROMPT_BUNDLE = "research/98_selector_demo/data/prereg/w98d2_prompt_tokens.jsonl.gz"
RETENTION_SEED = 4
CONFIRM_SEED = 5
DRAFT_LAYERS = r1.DRAFT_LAYERS
# One regime. Retention is a per-layer property of the draft; measuring it
# on every regime would multiply 37 boots by six for a vector the knapsack
# consumes once. R8 is chosen for batch 16 -- enough concurrent requests for
# a tight per-layer estimate without the long-context regimes' boot cost.
PROBE_REGIME = "R8"
MEASURE_TOKENS = 256


class RetentionError(RuntimeError):
    """Raised when the retention probe cannot be trusted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RetentionError(message)


def _load_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _prompts(regime_id: str, seed: int, limit: int) -> list[list[int]]:
    import gzip

    rows = []
    with gzip.open(
        matrix._repository_path(PROMPT_BUNDLE), "rt", encoding="utf-8"
    ) as handle:
        for line in handle:
            row = json.loads(line)
            if row["regime_id"] == regime_id and row["content_seed"] == seed:
                rows.append(row)
    rows.sort(key=lambda r: r["prompt_index"])
    _require(len(rows) >= limit, f"{regime_id}/seed-{seed}: {len(rows)} < {limit}")
    return [r["token_ids"] for r in rows[:limit]]


def boot_environment(skip_layers: str, trace_path: Path) -> dict[str, str]:
    cfg = {"quant": "w4a16-quantized", "window": "off", "skip_count": 0}
    env = d0.boot_environment(cfg, trace_path)
    env["VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS"] = skip_layers
    return env


def measure(skip_layers: str, trace_path: Path) -> dict[str, Any]:
    """Boot one skip set at KMAX and return its acceptance."""
    from vllm import LLMEngine, SamplingParams

    cfg = {"quant": "w4a16-quantized", "window": "off", "skip_count": 0}
    manifest = _load_json(matrix._repository_path(PROMPT_MANIFEST))
    regimes = {r["regime_id"]: r for r in manifest["prompt_plan"]["regimes"]}
    batch = regimes[PROBE_REGIME]["batch"]
    engine = LLMEngine.from_engine_args(d0._engine_args(cfg))
    try:
        for index, tokens in enumerate(_prompts(PROBE_REGIME, RETENTION_SEED, batch)):
            engine.add_request(
                f"{PROBE_REGIME}-r{index}",
                {"prompt_token_ids": tokens},
                SamplingParams(
                    temperature=0.0, max_tokens=MEASURE_TOKENS, ignore_eos=True
                ),
            )
        while engine.has_unfinished_requests():
            engine.step()
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()

    rows = stream.rows_from_trace(trace_path)
    armed = [r for r in rows if r.armed and not r.clipped]
    _require(bool(armed), "no scored armed rows")
    accepted = sum(r.accepted for r in armed)
    return {
        "skip_layers": skip_layers,
        "armed_rows": len(armed),
        "tau": round(1.0 + accepted / len(armed), 6),
        "mean_accepted": round(accepted / len(armed), 6),
    }


def _run(output_dir: Path) -> None:
    lane = matrix.lane_for_block(1)
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = set(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    traces = output_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    arms = [("reference", "")] + [
        (f"layer{i:02d}", str(i)) for i in range(DRAFT_LAYERS)
    ]
    for name, skip in arms:
        target = output_dir / f"{name}.json"
        if target.exists():
            continue
        trace = traces / f"{name}.jsonl"
        trace.unlink(missing_ok=True)
        log = output_dir / f"{name}.log"
        env = matrix._boot_child_environment(boot_environment(skip, trace))
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output-dir",
                    str(output_dir),
                    "--arm",
                    name,
                    "--skip-layers",
                    skip,
                    "--trace",
                    str(trace),
                ],
                cwd=r1.REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                preexec_fn=lambda: os.sched_setaffinity(0, affinity),
            )
        _require(
            completed.returncode == 0 and target.exists(),
            f"arm {name} failed; see {log}",
        )
        print(f"[retention] {name}", flush=True)


def summarize(output_dir: Path) -> dict[str, Any]:
    reference = _load_json(output_dir / "reference.json")
    tau0 = float(reference["tau"])
    _require(tau0 > 1.0, "reference tau shows no acceptance to retain")
    retention: dict[str, float] = {}
    for index in range(DRAFT_LAYERS):
        path = output_dir / f"layer{index:02d}.json"
        if not path.is_file():
            continue
        tau_i = float(_load_json(path)["tau"])
        # Retention of ACCEPTANCE, not of tau: tau carries the target's own
        # guaranteed token, which no skipped layer can cost.
        retention[str(index)] = round((tau_i - 1.0) / (tau0 - 1.0), 6)
    record = {
        "schema_version": 1,
        "record_type": "w98_d2_layer_retention",
        "scored": False,
        "regime": PROBE_REGIME,
        "content_seed": RETENTION_SEED,
        "confirmation_seed_reserved": CONFIRM_SEED,
        "kmax": d0.KMAX,
        "reference_tau": round(tau0, 6),
        "layers": DRAFT_LAYERS,
        "measured_layers": len(retention),
        "retention": retention,
    }
    (output_dir / "layer_retention.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--arm", help=argparse.SUPPRESS)
    parser.add_argument("--skip-layers", help=argparse.SUPPRESS)
    parser.add_argument("--trace", help=argparse.SUPPRESS)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    output_dir = (args.output_dir or matrix._repository_path(OUTPUT_PATH)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.arm:
        result = measure(args.skip_layers or "", Path(args.trace).resolve())
        (output_dir / f"{args.arm}.json").write_text(
            json.dumps({"arm": args.arm, **result}, indent=2, sort_keys=True) + "\n"
        )
        return 0
    if not args.summarize_only:
        matrix._preflight_gpu_identity_and_idle(
            matrix.lane_for_block(1)["physical_gpu_index"],
            matrix.lane_for_block(1)["physical_gpu_uuid"],
        )
        _run(output_dir)
    record = summarize(output_dir)
    ordered = sorted(record["retention"].items(), key=lambda kv: kv[1])
    print(
        json.dumps(
            {
                "reference_tau": record["reference_tau"],
                "measured_layers": record["measured_layers"],
                "most_important_layers": ordered[:5],
                "least_important_layers": ordered[-5:],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
