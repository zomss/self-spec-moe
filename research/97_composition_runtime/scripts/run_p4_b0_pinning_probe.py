# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Non-scored probe: does CPU reservation remove the V13 episode noise?

The V13 screen completed all 432 captures but the frozen scorer refused at the
2% cross-boot certification in 3 of 36 cells. The lane assignment was refuted
as the cause (median per-block offsets +0.099%/+0.208%/-0.133%, and the two
blocks that disagreed most sat on the same GPU). The evidence pointed at CPU
dispatch contention instead: the PIECEWISE draft chain is host-launch-bound
(~98 launches and ~16 ms host per step against ~5.3 ms GPU), the lanes claimed
all 192 cores, and the unpinned Lean server floated across every one of them.

This probe reruns only the four cells that exceeded 5% within-cell spread,
under the reserved-core assignment, with the Lean server deliberately left
running so the mitigation is tested against the real disturbance.

It is non-scored. It grants no value-screen, scoring, or admission authority.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import run_p4_b0_value_screen as matrix

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
PROBE_PACKAGE_ID = "p4-b0-pinning-probe-authorization-v1"
PROBE_AUTHORIZATION_PATH = (
    "research/97_composition_runtime/data/p4/p4_b0_pinning_probe_authorization_v1.json"
)
PROBE_OUTPUT_PATH = "research/97_composition_runtime/data/p4/run_b0_pinning_probe_v1"
SOURCE_RUN_PATH = "research/97_composition_runtime/data/p4/run_b0_value_screen_v12"
REFUSAL_PATH = f"{SOURCE_RUN_PATH}/certification_refusal.json"
PROBE_BLOCK_ID = 1
# The four cells whose within-cell spread exceeded 5% in V13, with the spread
# each one showed. All four sit in block 1; none are in the `off` action.
TARGET_CELLS = (
    {"action_id": "target-matching-k4", "regime_id": "R6", "content_seed": 0},
    {"action_id": "target-matching-k4", "regime_id": "R8", "content_seed": 1},
    {
        "action_id": "target-matching-w512-masked-k4",
        "regime_id": "R1",
        "content_seed": 0,
    },
    {
        "action_id": "target-matching-w512-masked-k4",
        "regime_id": "R5cot",
        "content_seed": 1,
    },
)
V13_WITHIN_CELL_SPREAD = {
    ("target-matching-k4", "R6", 0): 0.2464,
    ("target-matching-k4", "R8", 1): 0.1807,
    ("target-matching-w512-masked-k4", "R1", 0): 0.1300,
    ("target-matching-w512-masked-k4", "R5cot", 1): 0.0687,
}
V13_BLOCK1_REJECTED_ROUNDS = 6
PROBE_SOURCE_PATHS = {
    "matrix_runner": (
        "research/97_composition_runtime/scripts/run_p4_b0_value_screen.py"
    ),
    "probe_runner": (
        "research/97_composition_runtime/scripts/run_p4_b0_pinning_probe.py"
    ),
    "probe_tests": (
        "research/97_composition_runtime/tests/test_p4_b0_pinning_probe.py"
    ),
    "adapter": "research/97_composition_runtime/scripts/adapt_p4_b0_same_event.py",
    "koff_runtime": "vllm/v1/spec_decode/koff_runtime.py",
    "draft_proposer": "vllm/v1/spec_decode/llm_base_proposer.py",
    "source_run_refusal": REFUSAL_PATH,
}


class PinningProbeError(RuntimeError):
    """Raised when the probe cannot prove its authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PinningProbeError(message)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict), f"{path} is not a JSON object")
    return payload


def expected_authorization() -> dict[str, Any]:
    """Return the exact package this probe will execute under."""
    lane = matrix.lane_for_block(PROBE_BLOCK_ID)
    return {
        "schema_version": 1,
        "package_id": PROBE_PACKAGE_ID,
        "status": "authorized_non_scored_pinning_probe_only",
        "source_artifacts": {
            role: matrix._file_reference(path)
            for role, path in PROBE_SOURCE_PATHS.items()
        },
        "hypothesis": {
            "claim": "cpu_dispatch_contention_not_gpu_lane_effect",
            "lane_effect_refuted_by": "median_block_offsets_within_0.21_percent",
            "mechanism": "piecewise_chain_is_host_launch_bound",
            "v13_lanes_claimed_all_cpus": True,
            "lean_server_left_running": True,
        },
        "target_cells": list(TARGET_CELLS),
        "comparison": {
            "source_run": SOURCE_RUN_PATH,
            "v13_within_cell_spread": {
                f"{a}/{r}/s{s}": v for (a, r, s), v in V13_WITHIN_CELL_SPREAD.items()
            },
            "v13_block1_rejected_rounds": V13_BLOCK1_REJECTED_ROUNDS,
            "source_run_remeasured": False,
        },
        "execution_policy": {
            "lane": lane,
            "block_id": PROBE_BLOCK_ID,
            "physical_boot_count": 2,
            "capture_count": 16,
            "rounds_per_cell": len(matrix.ROUNDS),
            "create_new_output_required": True,
            "retry_allowed": False,
            "on_any_failure": "preserve_and_require_fresh_authorization",
        },
        "authorizations": {
            "gpu_measurement": True,
            "pinning_probe_execution": True,
            "value_screen_scoring": False,
            "value_screen_execution": False,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "next_artifact": {
            "kind": "p4_b0_pinning_probe_result",
            "scored": False,
            "may_authorize_rescreen": False,
        },
    }


def validate_authorization(package: Mapping[str, Any]) -> None:
    """Refuse anything but the exact registered non-scored probe.

    Args:
        package: The parsed probe authorization.

    Raises:
        PinningProbeError: If any registered field or source hash drifted.
    """
    expected = expected_authorization()
    _require(set(package) == set(expected), "probe authorization fields drifted")
    for key, value in expected.items():
        _require(package[key] == value, f"probe authorization drifted at {key}")
    refusal = _load_json(matrix._repository_path(REFUSAL_PATH))
    _require(
        refusal.get("record_type") == "p4_b0_value_screen_certification_refusal"
        and refusal.get("scored") is False,
        "probe source run is not the preserved unscored refusal",
    )
    _require(
        refusal["episode_noise"]["rounds_below_95pct_floor_by_block"]["1"]
        == V13_BLOCK1_REJECTED_ROUNDS,
        "probe comparison baseline differs from the preserved record",
    )


def build_probe_specs(output_dir: Path) -> list[dict[str, Any]]:
    """Build the two boot specs, each carrying only its target cells.

    Args:
        output_dir: The create-only probe output directory.

    Returns:
        The filtered boot specs, in registered action order.
    """
    v13 = _load_json(matrix._repository_path(matrix.V13_AUTHORIZATION_PATH))
    authorization = matrix.resolve_authorization_package(
        v13, require_output_absent=False
    )
    specs = matrix.build_boot_specs(authorization, output_dir)
    wanted = {(c["action_id"], c["regime_id"], c["content_seed"]) for c in TARGET_CELLS}
    probe_specs = []
    for spec in specs:
        if spec["boot_block_id"] != PROBE_BLOCK_ID:
            continue
        cells = [
            cell
            for cell in spec["plan"]["cells"]
            if (
                spec["action_id"],
                cell["matrix"]["regime_id"],
                cell["matrix"]["content_seed"],
            )
            in wanted
        ]
        if not cells:
            continue
        spec["plan"]["cells"] = cells
        spec["expected_capture_count"] = len(cells)
        probe_specs.append(spec)
    _require(
        len(probe_specs) == 2,
        f"probe did not resolve two boots (got {len(probe_specs)})",
    )
    _require(
        sum(s["expected_capture_count"] for s in probe_specs) == 16,
        "probe cells do not close to 16 captures",
    )
    return probe_specs


def prepare(output_dir: Path, specs: Sequence[Mapping[str, Any]]) -> None:
    """Write the probe package without touching a GPU."""
    _require(not output_dir.exists(), f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "plans").mkdir()
    (output_dir / "captures").mkdir()
    (output_dir / "boot_specs").mkdir()
    for spec in specs:
        plan = dict(spec)["plan"]
        Path(spec["plan_path"]).write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        Path(spec["capture_dir"]).mkdir(parents=True, exist_ok=True)
        body = {k: v for k, v in spec.items() if k != "plan"}
        (output_dir / "boot_specs" / f"{spec['boot_id']}.json").write_text(
            json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def run_child(spec_path: Path) -> None:
    """Run one probe boot's filtered cells in a single engine process."""
    spec = _load_json(spec_path)
    plan = _load_json(Path(spec["plan_path"]))
    from vllm import LLMEngine
    from vllm.v1.engine.core_client import InprocClient

    manifest, prompt_rows = matrix._load_prompt_rows()
    engine = LLMEngine.from_engine_args(matrix._engine_args(spec))
    _require(
        isinstance(engine.engine_core, InprocClient),
        "probe child did not construct InprocClient",
    )
    try:
        for cell in plan["cells"]:
            prompt_ids = cell["generation"]["prompt_record_ids"]
            batch = cell["generation"]["batch"]
            for start in range(0, len(prompt_ids), batch):
                matrix._run_prompt_chunk(
                    engine, cell, prompt_ids[start : start + batch], prompt_rows
                )
    finally:
        engine.engine_core.shutdown()
    matrix._write_boot_observation(spec)
    captures = sorted(Path(spec["capture_dir"]).glob("*.json"))
    complete = [p for p in captures if p.stat().st_size > 0]
    _require(
        len(complete) == spec["expected_capture_count"],
        f"probe boot emitted {len(complete)} captures, "
        f"expected {spec['expected_capture_count']}",
    )


def summarize(output_dir: Path) -> dict[str, Any]:
    """Compare the probe's spread and rejections against the V13 record."""
    import statistics

    from adapt_p4_b0_same_event import adapt_capture

    rounds = [
        adapt_capture(_load_json(path))
        for path in sorted(output_dir.glob("captures/*/*.json"))
    ]
    cells: dict[tuple[str, str, int], list[float]] = {}
    for row in rounds:
        key = (
            row["matrix"]["action_id"],
            row["matrix"]["regime_id"],
            row["matrix"]["content_seed"],
        )
        cells.setdefault(key, []).append(row["estimands"]["decode_rate_req"])
    comparison = []
    rejected = 0
    for key, rates in sorted(cells.items()):
        spread = (max(rates) - min(rates)) / statistics.median(rates)
        reference = sorted(rates)[-2] if len(rates) > 1 else rates[0]
        below = sum(1 for r in rates if r < reference * 0.95)
        rejected += below
        comparison.append(
            {
                "cell": f"{key[0]}/{key[1]}/s{key[2]}",
                "rounds": len(rates),
                "v13_spread_fraction": V13_WITHIN_CELL_SPREAD.get(key),
                "probe_spread_fraction": round(spread, 6),
                "probe_rounds_below_95pct_floor": below,
                "improved": spread < V13_WITHIN_CELL_SPREAD.get(key, 1.0),
            }
        )
    return {
        "schema_version": 1,
        "record_type": "p4_b0_pinning_probe_result",
        "package_id": PROBE_PACKAGE_ID,
        "scored": False,
        "grants_rescreen_authority": False,
        "lane": matrix.lane_for_block(PROBE_BLOCK_ID),
        "capture_count": len(rounds),
        "cells": comparison,
        "probe_rejected_rounds": rejected,
        "v13_block1_rejected_rounds": V13_BLOCK1_REJECTED_ROUNDS,
        "cells_improved": sum(1 for c in comparison if c["improved"]),
        "cells_total": len(comparison),
    }


def execute(authorization_path: Path, output_dir: Path) -> int:
    """Preflight, run both probe boots on the reserved lane, then summarize."""
    validate_authorization(_load_json(authorization_path))
    lane = matrix.lane_for_block(PROBE_BLOCK_ID)
    base_env = matrix._boot_child_environment({})
    matrix._preflight_native_sampler(base_env)
    matrix._preflight_inprocess_engine_core(base_env)
    matrix._preflight_gpu_identity_and_idle(
        lane["physical_gpu_index"], lane["physical_gpu_uuid"]
    )
    specs = build_probe_specs(output_dir)
    prepare(output_dir, specs)
    affinity = sorted(
        range(
            int(lane["cpu_affinity"].split("-")[0]),
            int(lane["cpu_affinity"].split("-")[1]) + 1,
        )
    )
    for spec in specs:
        spec_path = output_dir / "boot_specs" / f"{spec['boot_id']}.json"
        child_env = matrix._boot_child_environment(spec["environment"])
        log_path = output_dir / f"{spec['boot_id']}.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--authorization",
                    str(authorization_path.resolve()),
                    "--output-dir",
                    str(output_dir.resolve()),
                    "--child-spec",
                    str(spec_path.resolve()),
                ],
                cwd=REPO_ROOT,
                env=child_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                preexec_fn=lambda: os.sched_setaffinity(0, affinity),
            )
        _require(
            completed.returncode == 0,
            f"probe boot {spec['boot_id']} failed; output preserved",
        )
    result = summarize(output_dir)
    (output_dir / "probe_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    """Parse the probe interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--child-spec", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    """Run one authorized non-scored pinning probe."""
    args = parse_args()
    if args.child_spec is not None:
        run_child(args.child_spec.resolve())
        return 0
    return execute(args.authorization.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PinningProbeError, matrix.P4RunnerError) as exc:
        print(f"P4 pinning probe refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
