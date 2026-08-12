# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X15 -- is Option A resolvable? Measure sigma_repro on piecewise + Marlin.

DIAGNOSTIC, NOT SCORED. This decides a design choice; it scores nothing.

The open decision is whether to re-measure the lattice on piecewise (Option A,
frozen prereg stands, no lattice blocker) or re-parameterise for the whole-chain
class (Option B, a new preregistration). X14 made A attractive: with Marlin,
piecewise trails whole-chain by only 1.21x at batch 1 and ~1% at batch >= 32,
and it carries no `window > 0` constraint, so the blocker -- skip and quant
being sampled only at `woff` -- disappears.

What could still sink A is amendment 1 section 5: if `Z * sigma_repro > fit_term`
the regime is NOT RESOLVABLE, and Round 1's piecewise campaign failed that test
(Z*sigma 0.0337 against fit terms 0.0068-0.0300).

But that sigma was measured with **Machete** on the host-bound path. Marlin cut
the batch-1 bubble from 9.51 ms to 3.12 ms and the per-step stdev to 0.09 ms, so
reproducibility should be far better. This measures it.

Method: the two `woff` fit-set anchors, three repeats each, INTERLEAVED so the
repeats span time rather than running back to back -- X6's consecutive pair gave
CV 0.03% where X5's interleaved repeats gave 1.70% on the same box. Each boot
runs `measure_config` from the Round-1 harness unmodified, so the quantity is
exactly the `draft_chain` that D1 scores, measured the same way.

This is a FEASIBILITY estimate, not the operative `sigma_repro`. Amendment 1
section 3 requires the repeats to bracket the campaign; there is no campaign yet.
The operative value must be re-measured around the real run and may differ.

Decision rule, fixed before the data, per regime:
  2 * sigma_repro <  fit_term  -> Option A is resolvable; proceed on the frozen
                                  lattice with no new preregistration
  2 * sigma_repro >= fit_term  -> Option A is not resolvable; Option B is forced,
                                  and we know why
"""

from __future__ import annotations

import argparse
import json
import math
import os
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

# Fit-set singles, both `woff`, so both run the piecewise class under test.
ANCHORS = {
    "baseline": {"quant": "target-matching", "window": "off", "skip_count": 0},
    "quant": {"quant": "w4a16-quantized", "window": "off", "skip_count": 0},
}
REPEATS = 3
# Interleaved: baseline, quant, baseline, quant, ... so repeats span the session.
ORDER = [(rep, name) for rep in range(REPEATS) for name in ANCHORS]

# Round 1's fit envelopes per regime, for the section-5 comparison.
ROUND1_FIT_TERM = {
    "R1": 0.0300,
    "R4": 0.0068,
    "R5": 0.0095,
    "R5cot": 0.0074,
    "R6": 0.0208,
    "R8": 0.0262,
}
Z = 2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_all(output_dir: Path) -> None:
    lane = matrix.lane_for_block(1)
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = sorted(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    traces = output_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    for rep, name in ORDER:
        tag = f"r{rep}_{name}"
        target = output_dir / f"{tag}.json"
        if target.exists():
            continue
        cfg = ANCHORS[name]
        trace = traces / f"{tag}.jsonl"
        _require(not trace.exists(), f"trace {trace} already exists")
        env = r1.boot_environment(cfg, trace)
        # Option A's runtime: piecewise, explicitly, plus Marlin.
        env["VLLM_SELF_SPEC_DRAFT_WHOLECHAIN"] = "0"
        env["VLLM_SELF_SPEC_DRAFT_FULLCG"] = "0"
        env["VLLM_DISABLED_KERNELS"] = "MacheteLinearKernel"
        env = matrix._boot_child_environment(env)
        log = output_dir / f"{tag}.log"
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output-dir",
                    str(output_dir),
                    "--arm",
                    json.dumps({"tag": tag, "cfg": cfg}),
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
            (output_dir / f"{tag}.FAILED").write_text(
                f"returncode={completed.returncode}\n", encoding="utf-8"
            )
            continue
        text = log.read_text(encoding="utf-8", errors="replace")
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
        target.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def summarise(output_dir: Path) -> dict[str, Any]:
    """Per-regime log-scale stdev across repeats; sigma_repro is the max anchor."""
    per_anchor: dict[str, dict[str, list[float]]] = {}
    for rep, name in ORDER:
        path = output_dir / f"r{rep}_{name}.json"
        if not path.exists():
            continue
        record = r1._load_json(path)
        for regime, obs in record["observations"].items():
            value = obs.get("draft_chain_s")
            if value:
                per_anchor.setdefault(name, {}).setdefault(regime, []).append(value)

    regimes = sorted({r for a in per_anchor.values() for r in a})
    out: dict[str, Any] = {"per_anchor": {}, "sigma_repro": {}, "verdict": {}}
    for name, byreg in per_anchor.items():
        out["per_anchor"][name] = {
            regime: {
                "n": len(vals),
                "values_ms": [v * 1000 for v in vals],
                "cv_pct": (statistics.stdev(vals) / statistics.mean(vals) * 100)
                if len(vals) > 1
                else None,
                "sigma_log": statistics.stdev([math.log(v) for v in vals])
                if len(vals) > 1
                else None,
            }
            for regime, vals in byreg.items()
        }
    for regime in regimes:
        sigmas = [
            out["per_anchor"][n][regime]["sigma_log"]
            for n in out["per_anchor"]
            if out["per_anchor"][n].get(regime, {}).get("sigma_log") is not None
        ]
        if not sigmas:
            continue
        sigma = max(sigmas)  # amendment section 3: max across anchors
        fit = ROUND1_FIT_TERM.get(regime)
        out["sigma_repro"][regime] = sigma
        out["verdict"][regime] = {
            "sigma_repro": sigma,
            "z_sigma": Z * sigma,
            "round1_fit_term": fit,
            "resolvable": (Z * sigma < fit) if fit else None,
        }
    return out


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
        observations = r1.measure_config(arm["cfg"], Path(args.trace).resolve())
        (output_dir / f"{arm['tag']}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "record_type": "w98_sigma_repro",
                    "config": arm["cfg"],
                    "observations": observations,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0
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
    (output_dir / "sigma_repro_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary.get("verdict", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
