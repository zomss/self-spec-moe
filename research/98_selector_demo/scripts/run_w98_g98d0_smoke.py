# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""G98-D0 -- the acceptance-path smoke gate. NON-SCORED.

The G98-A analogue for D2. `w98-d2` (KMAX=8 arming, per-request acceptance
rows) is unit-tested on CPU but has never booted, and the preregistration's
own risk register predicted that the capture-path assertions would fire on
Round-2-style streams. This proves the path end to end before any scored
acceptance number is taken:

  1. the boot contract admits w98-d2 and the engine boots on both quant paths;
  2. K=8 actually arms -- the trace's armed steps carry k=8, not 4;
  3. `acceptance_rows` are written, one per decode request, each carrying its
     OWN generated-suffix length (the batch min/max cannot be inverted);
  4. accepted counts respect the prefix property and the K bound;
  5. the accounting identity E + C = A + H closes on real steps.

Deliberately uses the EXISTING seeds 2-3 prompt bundle: a smoke gate is not
scored, so the seed-disjointness the D2 contract requires for acceptance
does not apply here, and waiting for the seeds 4-5 freeze would leave the
engine path unproven. No scored artifact is written -- only a smoke record.

Run (h104):
    CUDA_VISIBLE_DEVICES set by the lane; pin to the lane CPUs.
    .venv/bin/python research/98_selector_demo/scripts/run_w98_g98d0_smoke.py
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

from vllm.v1.spec_decode.koff_runtime import KMAX8_ACTION_ID  # noqa: E402

matrix = r1.matrix
REPO_ROOT = r1.REPO_ROOT
OUTPUT_PATH = "research/98_selector_demo/data/g98_d0"
BOOT_SCOPE = "w98-d2"
KMAX = 8
# One armed K=8 schedule across the batch sizes the smoke regimes use.
D2_K_SCHEDULE = [[1, 32, KMAX]]
SMOKE_REGIMES = ("R1", "R8")
SMOKE_TOKENS = 96
QUANT_PATHS = ("target-matching", "w4a16-quantized")


class G98D0Error(RuntimeError):
    """Raised when the acceptance path cannot be proven."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G98D0Error(message)


def boot_environment(cfg: dict[str, Any], trace_path: Path) -> dict[str, str]:
    """Round-1's boot environment with the D2 scope substituted."""
    env = r1.boot_environment(cfg, trace_path)
    env["VLLM_SELF_SPEC_BOOT_SCOPE"] = BOOT_SCOPE
    return env


def _engine_args(cfg: dict[str, Any]):
    """Round-1's engine arguments at KMAX, unconditionally armed."""
    args = r1._engine_args(cfg)
    args.speculative_config = {
        **args.speculative_config,
        "num_speculative_tokens": KMAX,
        "num_speculative_tokens_per_batch_size": D2_K_SCHEDULE,
    }
    return args


def _decode_records(trace_path: Path) -> tuple[list[dict[str, Any]], int, int]:
    """Return (armed steps, priming steps, UNARMED steps that violate).

    The unarmed count is the runtime half of the "unconditionally armed"
    contract. The boot check cannot express it -- the scheduler receives a
    dense batch-size -> K lookup that is zero-filled outside its configured
    ranges -- so the guarantee is proven here, against what the engine
    actually did.

    One unarmed class is STRUCTURAL and permitted: the first decode step of a
    cohort, at generated suffix 1, where the chain has no previous token to
    draft from. Measured on the first D2 boot: exactly one such step per
    regime, immediately followed by armed steps. Those steps carry no draft
    and no acceptance, so they contribute nothing to the acceptance sample
    -- but an unarmed step at suffix > 1 would mean the stream really did
    disarm, and that is a contract violation.
    """
    armed: list[dict[str, Any]] = []
    priming = violations = 0
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("record_type") != "koff_engine_step":
                continue
            if row.get("exclusion_reasons"):
                continue
            counters = row.get("counters") or {}
            if not counters.get("H_target_steps"):
                continue
            if counters.get("D_armed"):
                armed.append(row)
                continue
            suffix_max = (row.get("engine") or {}).get("generated_suffix_max")
            if suffix_max == 1:
                priming += 1
            else:
                violations += 1
    return armed, priming, violations


def smoke_one(cfg: dict[str, Any], trace_path: Path) -> dict[str, Any]:
    """Boot one configuration at KMAX and audit its trace."""
    from vllm import LLMEngine, SamplingParams

    manifest = r1._load_json(matrix._repository_path(r1.PROMPT_MANIFEST))
    regimes = {r["regime_id"]: r for r in manifest["prompt_plan"]["regimes"]}
    engine = LLMEngine.from_engine_args(_engine_args(cfg))
    try:
        for regime_id in SMOKE_REGIMES:
            prompts = r1._prompts_for(regime_id, regimes[regime_id]["batch"])
            for index, tokens in enumerate(prompts):
                engine.add_request(
                    f"{regime_id}-{index}",
                    {"prompt_token_ids": tokens},
                    SamplingParams(
                        temperature=0.0, max_tokens=SMOKE_TOKENS, ignore_eos=True
                    ),
                )
            while engine.has_unfinished_requests():
                engine.step()
    finally:
        import contextlib

        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()

    armed, priming, violations = _decode_records(trace_path)
    _require(bool(armed), "no armed pure-decode steps were traced")
    _require(
        violations == 0,
        f"{violations} clean decode steps ran UNARMED past the priming step; "
        "w98-d2 requires an unconditionally armed stream",
    )

    # The passive step record identifies the action, not K directly (a
    # top-level "k" belongs to the P4 capture record, which D2 does not use).
    actions = {row.get("verified_action_id") for row in armed}
    _require(
        actions == {KMAX8_ACTION_ID},
        f"armed steps verified {sorted(actions)}, expected {KMAX8_ACTION_ID!r}",
    )

    rows_seen = accepted_total = drafted_total = 0
    suffix_spreads = 0
    per_position_reached = [0] * KMAX
    per_position_accepted = [0] * KMAX
    for record in armed:
        rows = record.get("acceptance_rows")
        _require(rows is not None, "an armed w98-d2 step carries no acceptance_rows")
        counters = record["counters"]
        _require(
            len(rows) == counters["H_target_steps"],
            "acceptance_rows do not cover every decode request",
        )
        _require(
            sum(row["accepted_draft_tokens"] for row in rows) == counters["A_accepted"],
            "per-request accepted counts disagree with the step aggregate",
        )
        _require(
            counters["E_committed"] + counters["C_clipped"]
            == counters["A_accepted"] + counters["H_target_steps"],
            "accounting identity E + C = A + H does not close",
        )
        suffixes = [row["generated_suffix_len"] for row in rows]
        suffix_spreads += min(suffixes) != max(suffixes)
        engine_block = record["engine"]
        _require(
            min(suffixes) == engine_block["generated_suffix_min"]
            and max(suffixes) == engine_block["generated_suffix_max"],
            "per-request suffix lengths disagree with the batch summary",
        )
        for row in rows:
            accepted = row["accepted_draft_tokens"]
            _require(
                0 <= accepted <= KMAX,
                f"accepted={accepted} outside the K={KMAX} bound",
            )
            _require(row["drafted_tokens"] == KMAX, "an armed row drafted != KMAX")
            # Acceptance is a prefix: positions 1..accepted were accepted and
            # accepted+1 was rejected. This is what makes per-position
            # counters unnecessary.
            for position in range(KMAX):
                if position <= accepted:
                    per_position_reached[position] += 1
                if position < accepted:
                    per_position_accepted[position] += 1
            rows_seen += 1
            accepted_total += accepted
            drafted_total += KMAX

    return {
        "config": cfg,
        "armed_steps": len(armed),
        "priming_steps_unarmed": priming,
        "unarmed_violations": violations,
        "acceptance_rows": rows_seen,
        "action_observed": sorted(actions),
        "mean_accepted_per_request": round(accepted_total / max(rows_seen, 1), 4),
        "acceptance_rate_over_drafted": round(
            accepted_total / max(drafted_total, 1), 4
        ),
        "steps_with_suffix_spread": suffix_spreads,
        "per_position_reached": per_position_reached,
        "per_position_accepted": per_position_accepted,
        "per_position_conditional": [
            round(a / r, 4) if r else None
            for a, r in zip(per_position_accepted, per_position_reached)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--measure-config", help=argparse.SUPPRESS)
    parser.add_argument("--trace", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output_dir = (args.output_dir or matrix._repository_path(OUTPUT_PATH)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.measure_config:
        cfg = json.loads(args.measure_config)
        result = smoke_one(cfg, Path(args.trace).resolve())
        key = r1._config_key(cfg).replace("/", "_")
        (output_dir / f"{key}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "record_type": "w98_g98d0_smoke",
                    "scored": False,
                    "boot_scope": BOOT_SCOPE,
                    "kmax": KMAX,
                    **result,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return 0

    lane = matrix.lane_for_block(1)
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = set(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    matrix._preflight_gpu_identity_and_idle(
        lane["physical_gpu_index"], lane["physical_gpu_uuid"]
    )
    traces = output_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)

    results = []
    for quant in QUANT_PATHS:
        cfg = {"quant": quant, "window": "off", "skip_count": 0}
        key = r1._config_key(cfg).replace("/", "_")
        trace = traces / f"{key}.jsonl"
        trace.unlink(missing_ok=True)
        log = output_dir / f"{key}.log"
        env = matrix._boot_child_environment(boot_environment(cfg, trace))
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output-dir",
                    str(output_dir),
                    "--measure-config",
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
        target = output_dir / f"{key}.json"
        _require(
            completed.returncode == 0 and target.exists(),
            f"smoke boot {key} failed; output preserved at {log}",
        )
        results.append(r1._load_json(target))
        print(f"[g98d0] {key} OK", flush=True)

    summary = {
        "schema_version": 1,
        "record_type": "w98_g98d0_smoke_summary",
        "scored": False,
        "boot_scope": BOOT_SCOPE,
        "kmax": KMAX,
        "note": (
            "non-scored engine-path proof on the seeds 2-3 bundle; the scored "
            "D2 campaign uses the disjoint seeds 4-5 freeze"
        ),
        "boots": results,
    }
    (output_dir / "g98d0_smoke.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                r["config"]["quant"]: {
                    "armed_steps": r["armed_steps"],
                    "rows": r["acceptance_rows"],
                    "action": r["action_observed"],
                    "mean_accepted": r["mean_accepted_per_request"],
                    "suffix_spread_steps": r["steps_with_suffix_spread"],
                    "per_position_conditional": r["per_position_conditional"],
                }
                for r in results
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
