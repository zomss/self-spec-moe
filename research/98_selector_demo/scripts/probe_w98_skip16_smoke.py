# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X17 -- smoke the skip16 cut point before the W98-R2 campaign commits to it.

DIAGNOSTIC, NOT SCORED. Nothing here is hash-bound and nothing writes into a
frozen artifact.

Round 2 extends the skip axis to 16 layers to widen the support the cost model
is fitted over: Round 1 measured keep_frac only at 1.00 / 0.889 / 0.778, and the
residual it could not resolve lives at the low-keep_frac end. skip16 puts a
point at 0.556. That is the whole reason the lattice grew, so the cut point had
better boot, arm, and still produce accepted tokens.

Three things have never been exercised at this width:

  1. `W98_SKIP_COUNTS` was extended to include 16, so the runtime now ACCEPTS a
     16-layer cut. Accepting is not the same as capturing one.
  2. Both capture-path assertions are now gated on `boot_scope`, which means a
     divergent draft weight version no longer raises under `w98-lattice`. That
     gate has never actually been taken at a width Round 1 did not run.
  3. Acceptance at 0.556 keep_frac is unmeasured. Step time falls roughly
     linearly on the Round-1 ladder (R1 woff: 30.36 / 27.30 / 24.43 ms at
     skip 0/4/8), but a draft that is fast and never accepted is worthless --
     the target rejects it and the chain cost is pure loss. A cheap cell that
     collapses acceptance must be caught HERE, not 3 hours into the campaign.

So the probe reports speed and quality together and refuses to call either one
the answer on its own.

Cells (3 boots, ~2 min each, all six regimes):

  * target-matching / window off / skip16 -- the clean skip axis, directly
    comparable to the Round-1 woff skip ladder.
  * target-matching / window 256 / skip16 -- skip composed with window.
  * w4a16-quantized / window 256 / skip16 -- the triple composition, which is
    where Round 1's residual was worst and where Round 2 most needs support.

All six regimes run, in the manifest's order, over the identical prompt set the
scored campaign uses -- so `draft_chain_s` drops straight onto the Round-1 table
and pooled acceptance is comparable across cells without segmenting the trace.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402

matrix = r1.matrix

SKIP_COUNT = 16
CELLS = [
    {"quant": "target-matching", "window": "off", "skip_count": SKIP_COUNT},
    {"quant": "target-matching", "window": 256, "skip_count": SKIP_COUNT},
    {"quant": "w4a16-quantized", "window": 256, "skip_count": SKIP_COUNT},
]
# Round-1 v6 traces for the same prompt set, used as the acceptance reference.
# Pooled over all regimes, which is comparable because every cell runs the same
# prompts in the same order.
REFERENCE_TRACES = {
    "skip0": "target-matching_woff_skip0.jsonl",
    "skip4": "target-matching_woff_skip4.jsonl",
    "skip8": "target-matching_woff_skip8.jsonl",
    "skip0_quant": "w4a16-quantized_woff_skip0.jsonl",
}
V6_TRACE_DIR = "research/98_selector_demo/data/g98_b_v6/traces/singles"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def acceptance_from_trace(trace: Path) -> dict[str, Any]:
    """Pool the acceptance counters over every armed step in a trace.

    The counters are per-REQUEST, not per-token: `koff_runtime` builds them as
    ``D_armed = sum(row["draft_armed"])`` over the step's request rows, so
    ``D_armed`` counts armed REQUESTS while ``A_accepted`` counts accepted draft
    TOKENS. The acceptance quantity is therefore ``A / D`` -- accepted draft
    tokens per armed request, bounded above by K -- and dividing A by D as if
    both were token counts silently reports a rate above 1.

    ``E_committed / H_target_steps`` is tokens committed per request-step, which
    is what throughput sees: at K=4 a request commits 1 (everything rejected) to
    5 (everything accepted).

    Args:
        trace: Path to a koff step trace in JSON Lines form.

    Returns:
        Pooled acceptance counters, or an empty mapping if no step armed.
    """
    if not trace.is_file():
        return {}
    a = d = e = h = 0
    armed_steps = 0
    with trace.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("record_type") != "koff_engine_step":
                continue
            counters = row.get("counters") or {}
            if not counters.get("D_armed"):
                continue
            armed_steps += 1
            a += int(counters.get("A_accepted", 0))
            d += int(counters.get("D_armed", 0))
            e += int(counters.get("E_committed", 0))
            h += int(counters.get("H_target_steps", 0))
    if not armed_steps:
        return {}
    return {
        "armed_steps": armed_steps,
        "A_accepted": a,
        "D_armed_requests": d,
        "E_committed": e,
        "H_target_steps": h,
        "accepted_tokens_per_armed_request": a / d if d else None,
        "committed_tokens_per_request_step": e / h if h else None,
    }


def measure(cfg: dict[str, Any], trace: Path, out_json: Path) -> None:
    """Run the scored Round-1 measurement, then add acceptance from the trace."""
    observations = r1.measure_config(cfg, trace)
    out_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_skip16_smoke",
                "config": cfg,
                "skip_layers": r1.SKIP_SETS[cfg["skip_count"]],
                "keep_frac": (r1.DRAFT_LAYERS - cfg["skip_count"]) / r1.DRAFT_LAYERS,
                "observations": observations,
                "acceptance": acceptance_from_trace(trace),
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
    for cfg in CELLS:
        name = r1._config_key(cfg).replace("/", "_")
        target = output_dir / f"{name}.json"
        if target.exists():
            continue
        trace = traces / f"{name}.jsonl"
        _require(not trace.exists(), f"trace {trace} already exists")
        env = matrix._boot_child_environment(r1.boot_environment(cfg, trace))
        # The gate under test: the boot must accept a 16-layer cut, and the
        # skip set must reach the engine intact rather than being silently
        # dropped to a shorter one.
        _require(
            env["VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS"] == r1.SKIP_SETS[SKIP_COUNT],
            f"skip set for {name} did not reach the boot environment",
        )
        log = output_dir / f"{name}.log"
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output-dir",
                    str(output_dir),
                    "--cell",
                    json.dumps({"name": name, "cfg": cfg}),
                    "--trace",
                    str(trace),
                ],
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        text = log.read_text(encoding="utf-8", errors="replace")
        if completed.returncode != 0 or not target.exists():
            # A wedge or a refused cut point is the finding, so keep enough of
            # the log to name which one it was rather than just a returncode.
            (output_dir / f"{name}.FAILED").write_text(
                f"returncode={completed.returncode}\n\n" + text[-8000:],
                encoding="utf-8",
            )
            print(f"[skip16] FAILED {name} rc={completed.returncode}", flush=True)
            continue
        record = r1._load_json(target)
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
        # Proof the gated assertion was reached and did not fire.
        record["koff_runtime_error"] = "KOffRuntimeError" in text
        target.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[skip16] ok {name}", flush=True)


def summarise(output_dir: Path) -> dict[str, Any]:
    """Put skip16 on the Round-1 ladder and next to the acceptance reference."""
    reference = {
        key: acceptance_from_trace(matrix._repository_path(V6_TRACE_DIR) / name)
        for key, name in REFERENCE_TRACES.items()
    }
    cells: dict[str, Any] = {}
    for cfg in CELLS:
        name = r1._config_key(cfg).replace("/", "_")
        path = output_dir / f"{name}.json"
        if not path.is_file():
            cells[name] = {"status": "FAILED"}
            continue
        record = r1._load_json(path)
        cells[name] = {
            "status": "ok",
            "keep_frac": record["keep_frac"],
            "kernel_resolved": record.get("kernel_resolved"),
            "koff_runtime_error": record.get("koff_runtime_error"),
            "draft_chain_ms": {
                regime: (
                    round(obs["draft_chain_s"] * 1000.0, 2)
                    if obs.get("draft_chain_s")
                    else None
                )
                for regime, obs in record["observations"].items()
            },
            "usable_regimes": sum(
                1 for obs in record["observations"].values() if obs.get("usable")
            ),
            "acceptance": record.get("acceptance"),
        }
    return {"cells": cells, "reference_acceptance": reference}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summarise-only", action="store_true")
    parser.add_argument("--cell", help=argparse.SUPPRESS)
    parser.add_argument("--trace", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.cell:
        cell = json.loads(args.cell)
        measure(
            cell["cfg"],
            Path(args.trace).resolve(),
            output_dir / f"{cell['name']}.json",
        )
        return 0
    if not args.summarise_only:
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
    summary = summarise(output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
