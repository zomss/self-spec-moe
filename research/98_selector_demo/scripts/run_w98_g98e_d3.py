# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W98-D3 (G98-E) end-to-end selector value.

Measures ONE grid -- `rate(regime, configuration)` in decode currency under
equal work -- from which every quantity D3 registers is an aggregation over
a workload mix:

    selector     sum_R v_R * rate(R, selector_choice(R))
    omniscient   sum_R v_R * max_C rate(R, C)
    static C     sum_R v_R * rate(R, C)          one C for every regime

Because the grid is MIX-INDEPENDENT it is measured before the mix is
applied, which makes it structurally impossible to choose the mix after
seeing which configuration wins where. Scoring lives in
`score_w98_g98e_d3.py`.

Runtime is the REGISTERED DEPLOYMENT runtime, not the acceptance stream:
`w98-lattice` scope with the live K/OFF ladder over K in {0, 4}. OFF is a
first-class cell (a K=0 schedule, the registered `live-b0-forced-off`
realization) because it is the baseline D3 must beat. KMAX=8 belongs to
G98-D and is not used here.

Currency is DECODE-ONLY: committed tokens divided by summed decode step
time, taken from the koff trace's own `decode_time_s`. Phase 96's I1
retracted a claim to mixing decode and wall currencies (prefill is 46.6% of
wall at R5), so wall clock is never used. Equal work is enforced with
`ignore_eos` and a fixed token budget, because under natural EOS the
speculative and autoregressive arms drain asymmetrically -- the defect that
moved one MoE cell from 1.288 to 0.954 when Phase 96 re-measured it.

Amendment 2's dual arm: the whole grid runs under BOTH instruments, the
corrected one and the legacy one
(`VLLM_SELF_SPEC_PROFILE_LEGACY_SYNCS=1`). That matters more here than
anywhere else in the phase -- the inner-sync cost is paid only by ARMED
configurations, and OFF is one of the baselines being scored.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import socket
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402

matrix = r1.matrix
REPO_ROOT = r1.REPO_ROOT

PACKAGE_ID = "w98-g98e-d3-authorization-v1"
AUTHORIZATION_PATH = "research/98_selector_demo/data/w98_g98e_authorization_v1.json"
OUTPUT_PATH = "research/98_selector_demo/data/g98_e"
PROMPT_MANIFEST = r1.PROMPT_MANIFEST
PREREG_DOC = "research/98_selector_demo/w98_prereg.md"
GRID_DIR = "grid"
ARMS = ("corrected", "legacy")
MEASURE_TOKENS = r1.MEASURE_TOKENS
CONTENT_SEED = 2
OFF_KEY = "off"
K_ARMED = 4

# Lane per box. /h/v-sukmincho is ONE NFS tree shared by these hosts, so the
# code is identical on both and only the device identity differs; a boot on
# an unregistered host fails closed rather than guessing.
LANES = {
    "h103": {
        "lane_id": "lane-a",
        "physical_gpu_index": 7,
        "physical_gpu_uuid": "GPU-6da30cbb-5974-e66f-a406-389946903db7",
        "cpu_affinity": "96-111",
        "cache_root": "/tmp/v-sukmincho-w98/vllm-cache/lane-a",
        "block_ids": [1, 2, 3],
    },
    "h104": {
        "lane_id": "lane-a",
        "physical_gpu_index": 7,
        "physical_gpu_uuid": "GPU-7a8308d6-2a78-2929-99f2-11d2f11a45f3",
        "cpu_affinity": "96-111",
        "cache_root": "/tmp/v-sukmincho-w98/vllm-cache/lane-a",
        "block_ids": [1, 2, 3],
    },
}


class G98EError(RuntimeError):
    """Raised when the D3 grid cannot prove its authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G98EError(message)


def _host() -> str:
    return socket.gethostname().split(".")[0]


def lane_for_block(block_id: int = 1) -> dict[str, Any]:
    host = _host()
    _require(host in LANES, f"host {host!r} has no registered D3 lane")
    return copy.deepcopy(LANES[host])


# The round-1 module binds matrix.lane_for_block to its own box lane at
# import; D3 rebinds it here so the closed G98-C and G98-D authorizations,
# which hash that module, are left untouched.
matrix.lane_for_block = lambda block_id=1: lane_for_block(block_id)


def _load_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _config_key(cfg: Mapping[str, Any]) -> str:
    if cfg.get("action") == OFF_KEY:
        return OFF_KEY
    return r1._config_key(cfg)


def _slug(cfg: Mapping[str, Any], arm: str) -> str:
    return f"{_config_key(cfg).replace('/', '_')}__{arm}"


def grid_configs() -> list[dict[str, Any]]:
    """OFF plus every cell of the frozen lattice."""
    prereg = _load_json(matrix._repository_path(r1.PREREG_MATRIX))
    lattice = prereg["lattice"]
    out: list[dict[str, Any]] = [
        {
            "action": OFF_KEY,
            "quant": "target-matching",
            "window": "off",
            "skip_count": 0,
        }
    ]
    for quant in lattice["quant"]:
        for window in lattice["window"]:
            for skip in lattice["skip_counts"]:
                out.append(
                    {
                        "action": "armed",
                        "quant": quant,
                        "window": window,
                        "skip_count": skip,
                    }
                )
    return out


def boot_environment(cfg: Mapping[str, Any], trace_path: Path, arm: str) -> dict:
    env = r1.boot_environment(
        {k: v for k, v in cfg.items() if k != "action"}, trace_path
    )
    if arm == "legacy":
        env["VLLM_SELF_SPEC_PROFILE_LEGACY_SYNCS"] = "1"
    return env


def _engine_args(cfg: Mapping[str, Any]):
    args = r1._engine_args({k: v for k, v in cfg.items() if k != "action"})
    k = 0 if cfg.get("action") == OFF_KEY else K_ARMED
    args.speculative_config = {
        **args.speculative_config,
        "num_speculative_tokens": K_ARMED,
        "num_speculative_tokens_per_batch_size": [[1, 32, k]],
    }
    return args


def _decode_rate(trace_path: Path, skip: int) -> dict[str, Any]:
    """Committed tokens per second of DECODE time, from the trace itself."""
    committed = 0
    seconds = 0.0
    steps = armed = 0
    with Path(trace_path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < skip:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("record_type") != "koff_engine_step":
                continue
            if record.get("exclusion_reasons"):
                continue
            counters = record.get("counters") or {}
            if not counters.get("H_target_steps"):
                continue
            elapsed = counters.get("decode_time_s")
            if not isinstance(elapsed, (int, float)) or elapsed <= 0:
                continue
            committed += int(counters.get("E_committed", 0))
            seconds += float(elapsed)
            steps += 1
            armed += bool(counters.get("D_armed"))
    _require(steps > 0 and seconds > 0, "no scored decode steps in the interval")
    return {
        "committed_tokens": committed,
        "decode_seconds": round(seconds, 6),
        "decode_tokens_per_s": round(committed / seconds, 6),
        "steps": steps,
        "armed_steps": armed,
    }


def measure_config(cfg: Mapping[str, Any], trace_path: Path) -> dict[str, Any]:
    from vllm import LLMEngine, SamplingParams

    manifest = _load_json(matrix._repository_path(PROMPT_MANIFEST))
    regimes = {r["regime_id"]: r for r in manifest["prompt_plan"]["regimes"]}
    engine = LLMEngine.from_engine_args(_engine_args(cfg))
    observations: dict[str, Any] = {}
    try:
        for regime_id, spec in regimes.items():
            before = _trace_len(trace_path)
            prompts = r1._prompts_for(regime_id, spec["batch"])
            for index, tokens in enumerate(prompts):
                engine.add_request(
                    f"{regime_id}-{index}",
                    {"prompt_token_ids": tokens},
                    SamplingParams(
                        temperature=0.0, max_tokens=MEASURE_TOKENS, ignore_eos=True
                    ),
                )
            while engine.has_unfinished_requests():
                engine.step()
            observations[regime_id] = _decode_rate(trace_path, before)
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()
    return observations


def _trace_len(trace_path: Path) -> int:
    if not Path(trace_path).is_file():
        return 0
    with Path(trace_path).open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def expected_authorization() -> dict[str, Any]:
    sources = {
        "prereg_doc": PREREG_DOC,
        "prompt_manifest": PROMPT_MANIFEST,
        "prompt_bundle": (
            "research/98_selector_demo/data/prereg/w98_prompt_tokens.jsonl.gz"
        ),
        "prereg_matrix": r1.PREREG_MATRIX,
        "round1_runner": "research/98_selector_demo/scripts/run_w98_g98b_round1.py",
        "d3_runner": "research/98_selector_demo/scripts/run_w98_g98e_d3.py",
        "round2_result": "research/98_selector_demo/data/g98_c/round2_result.json",
        "d2_result": "research/98_selector_demo/data/g98_d/d2_result.json",
        "koff_runtime": "vllm/v1/spec_decode/koff_runtime.py",
    }
    source_artifacts = {}
    for name, rel in sources.items():
        path = matrix._repository_path(rel)
        _require(path.is_file(), f"source artifact missing: {rel}")
        source_artifacts[name] = {
            "path": rel,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {
        "schema_version": 1,
        "package_id": PACKAGE_ID,
        "gate": "G98-E",
        "gate_note": (
            "D3 runs as G98-E: the registered letters predate the Round-2 "
            "amendment, which shifted the executed sequence by one"
        ),
        "prerequisite": {
            "cost": "research/98_selector_demo/data/g98_c/round2_result.json",
            "acceptance": "research/98_selector_demo/data/g98_d/d2_result.json",
        },
        "source_artifacts": source_artifacts,
        "runtime": {
            "boot_scope": "w98-lattice",
            "k_values": [0, K_ARMED],
            "ladder": "live K/OFF over the registered schedule",
            "off_is_a_cell": True,
            "draft": "piecewise",
            "kernel": "Marlin",
        },
        "currency": {
            "measure": "committed tokens / summed decode step seconds",
            "decode_only": True,
            "wall_clock_forbidden": (
                "phase 96 I1 retracted a claim to currency mixing; prefill is "
                "46.6% of wall at R5"
            ),
        },
        "equal_work": {
            "ignore_eos": True,
            "max_tokens": MEASURE_TOKENS,
            "rationale": (
                "natural EOS drains the spec and AR arms asymmetrically; phase "
                "96 re-measurement moved one MoE cell from 1.288 to 0.954"
            ),
        },
        "arms": {
            "names": list(ARMS),
            "amendment": "w98_prereg_amendment2_instrument.md, APPROVED (C)",
            "reporting_rule": (
                "both numbers appear wherever the D3 result appears; a dual-arm "
                "run is not licence to quote the kinder arm"
            ),
        },
        "grid": {
            "configurations": len(grid_configs()),
            "mix_independent": True,
            "mix_note": (
                "selector, omniscient and every static single are aggregations "
                "of this grid; the mix is applied at scoring, never at "
                "measurement"
            ),
        },
        "workload_mix": {
            "registered": "equal weight over the six regimes",
            "also_reported": "the mix frontier (which mixes clear the 90% target)",
            "decided": "2026-08-15, before any D3 boot",
        },
        "targets": {
            "omniscient_fraction": 0.90,
            "lcb_rule_over_statics": 0.02,
        },
        "execution_policy": {
            "output_dir": OUTPUT_PATH,
            "create_only": True,
            "resumable": "a completed (configuration, arm) cell is never re-measured",
            "host": _host(),
            "lane": lane_for_block(1),
        },
        "status": "issued",
    }


def validate_authorization(package: Mapping[str, Any]) -> None:
    expected = expected_authorization()
    _require(package.get("package_id") == PACKAGE_ID, f"not {PACKAGE_ID}")
    for name, want in expected["source_artifacts"].items():
        got = (package.get("source_artifacts") or {}).get(name)
        _require(got is not None, f"authorization omits source artifact {name}")
        _require(
            got.get("sha256") == want["sha256"],
            f"source artifact {name} changed since the authorization was issued",
        )
    for field in ("runtime", "currency", "equal_work", "arms", "targets"):
        _require(
            package.get(field) == expected[field],
            f"authorization field {field!r} does not match this runner",
        )


def issue_authorization(path: Path) -> dict[str, Any]:
    path = path.resolve()
    _require(not path.exists(), f"authorization {path} already exists")
    package = expected_authorization()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
    return package


def _run_grid(output_dir: Path, authorization_path: Path) -> None:
    stage_dir = output_dir / GRID_DIR
    stage_dir.mkdir(parents=True, exist_ok=True)
    traces = output_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    lane = lane_for_block(1)
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = set(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        for cfg in grid_configs():
            slug = _slug(cfg, arm)
            target = stage_dir / f"{slug}.json"
            if target.exists():
                continue
            trace = traces / f"{slug}.jsonl"
            trace.unlink(missing_ok=True)
            log = stage_dir / f"{slug}.log"
            env = matrix._boot_child_environment(boot_environment(cfg, trace, arm))
            with log.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--authorization",
                        str(authorization_path),
                        "--output-dir",
                        str(output_dir),
                        "--measure-config",
                        json.dumps(dict(cfg)),
                        "--arm",
                        arm,
                        "--trace",
                        str(trace),
                    ],
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    preexec_fn=lambda: os.sched_setaffinity(0, affinity),
                )
            _require(
                completed.returncode == 0 and target.exists(),
                f"boot {slug} failed; output preserved at {log}",
            )
            print(f"[g98e] measured {slug}", flush=True)


def execute(authorization_path: Path, output_dir: Path) -> int:
    output_dir = output_dir.resolve()
    validate_authorization(_load_json(authorization_path))
    base_env = matrix._boot_child_environment({})
    matrix._preflight_native_sampler(base_env)
    matrix._preflight_inprocess_engine_core(base_env)
    lane = lane_for_block(1)
    matrix._preflight_gpu_identity_and_idle(
        lane["physical_gpu_index"], lane["physical_gpu_uuid"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_grid(output_dir, authorization_path)
    print(json.dumps({"grid_complete": True, "host": _host()}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--issue-authorization", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--measure-config", help=argparse.SUPPRESS)
    parser.add_argument("--arm", help=argparse.SUPPRESS)
    parser.add_argument("--trace", help=argparse.SUPPRESS)
    args = parser.parse_args()
    authorization = (
        args.authorization or matrix._repository_path(AUTHORIZATION_PATH)
    ).resolve()
    output_dir = (args.output_dir or matrix._repository_path(OUTPUT_PATH)).resolve()

    if args.measure_config:
        cfg = json.loads(args.measure_config)
        validate_authorization(_load_json(authorization))
        stage_dir = output_dir / GRID_DIR
        stage_dir.mkdir(parents=True, exist_ok=True)
        observations = measure_config(cfg, Path(args.trace).resolve())
        (stage_dir / f"{_slug(cfg, args.arm)}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "record_type": "w98d3_grid_cell",
                    "package_id": PACKAGE_ID,
                    "config": cfg,
                    "cell": _config_key(cfg),
                    "arm": args.arm,
                    "host": _host(),
                    "observations": observations,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return 0
    if args.issue_authorization:
        issue_authorization(authorization)
        print(json.dumps({"issued": str(authorization)}, indent=2))
        return 0
    if args.dry_run:
        package = expected_authorization()
        print(
            json.dumps(
                {
                    "package_id": package["package_id"],
                    "host": _host(),
                    "gpu": package["execution_policy"]["lane"]["physical_gpu_index"],
                    "configurations": len(grid_configs()),
                    "arms": list(ARMS),
                    "boots": len(grid_configs()) * len(ARMS),
                    "mix": package["workload_mix"],
                },
                indent=2,
            )
        )
        return 0
    return execute(authorization, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
