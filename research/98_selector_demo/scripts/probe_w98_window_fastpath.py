#!/usr/bin/env python3
"""X30 -- does skipping the no-op window rewrite remove the window's overhead?

`results_lever_mechanics.md` measured the KV window making the draft SLOWER at
short context: cost ratios of 1.013-1.020 against the unwindowed draft, where
physics says the saving should be exactly zero. There is nothing to trim, so
the rewrite is pure overhead.

The mechanism is not a device sync -- the FlashAttention metadata builder
never reads `seq_lens_cpu`, so nulling that cache costs nothing. It is
`_apply_draft_kv_window` itself: roughly fourteen small kernel launches plus a
dataclass `replace`, PER DRAFT STEP, so about five times that per armed step.

`VLLM_SELF_SPEC_DRAFT_WINDOW_FASTPATH=1` returns the input metadata unchanged
when sinks + window already cover the longest sequence, which is provably the
same view the slow path would build. This probe asks whether that is (a) real
time and (b) free of any behavioural change.

## Arms

`fastpath_off` is the engine exactly as every scored campaign ran it;
`fastpath_on` differs only in the flag. Arms are INTERLEAVED and repeated so
box drift cannot be mistaken for an effect.

## What decides it

* **Time** -- `draft_chain` mean, and the whole armed step as the
  sync-independent yardstick.
* **Correctness** -- committed tokens per armed step must be IDENTICAL between
  arms. The fast path changes which tensor the attention metadata points at,
  so a difference in acceptance means the two views are not equivalent and the
  flag must not ship, whatever it does to the clock.
* **Did it fire** -- the engine's own hit counters, so "no change" cannot be
  confused with "never taken". R5 (14k) is carried as a NEGATIVE control: the
  window genuinely trims there, so the fast path must NOT fire.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98b_round1 as r1  # noqa: E402

# D3's host-keyed lane table, imported for its side effect: the round-1 module
# binds `matrix.lane_for_block` to h104's device at import, and /h/v-sukmincho
# is one NFS tree shared by both boxes. Importing D3 rebinds it to whichever
# host is actually running, and fails closed on an unregistered one -- which
# is what lets this probe move when a co-tenant takes the lane.
import run_w98_g98e_d3 as d3  # noqa: E402,F401

matrix = r1.matrix

# The window lever alone: no quantization, no skipping, so the measurement is
# attributable to the window path and nothing else.
CELL = {"quant": "target-matching", "window": 1024, "skip_count": 0}
# The no-window control. Comparing the fast path against the SLOW path only
# prices the rewrite; comparing it against window=off prices what is LEFT --
# and the earlier 1.5-2% figure came from the fitted cost model, whose own
# held-out error is 0.53% median, so it needs a direct same-session A/B.
CELL_NOWINDOW = {"quant": "target-matching", "window": "off", "skip_count": 0}
# R1/R6/R8 are short-context, where the fast path should fire on every call.
# R5 is 14k: the window really trims, the fast path must never fire.
REGIMES = ("R1", "R6", "R8", "R5")
# (name, fastpath, cell)
ORDER = [
    ("r0_fastpath_off", False, CELL),
    ("r0_fastpath_on", True, CELL),
    ("r0_window_off", False, CELL_NOWINDOW),
    ("r1_fastpath_off", False, CELL),
    ("r1_fastpath_on", True, CELL),
    ("r1_window_off", False, CELL_NOWINDOW),
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _acceptance(trace: Path, skip: int) -> dict[str, float]:
    """Committed tokens per armed step, from the engine trace."""
    committed = armed = steps = 0
    with Path(trace).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < skip:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("record_type") != "koff_engine_step":
                continue
            counters = record.get("counters") or {}
            if not counters.get("H_target_steps"):
                continue
            committed += int(counters.get("E_committed", 0))
            steps += 1
            armed += bool(counters.get("D_armed"))
    return {
        "committed_tokens": committed,
        "steps": steps,
        "armed_steps": armed,
        "committed_per_step": round(committed / steps, 6) if steps else None,
    }


def measure(
    trace: Path, out_json: Path, fastpath: bool, cell: dict[str, Any] | None = None
) -> None:
    # Default to the windowed cell so an invocation without --window (the form
    # a parent started before this control arm existed uses) still measures
    # what it intended to.
    cell = cell or CELL
    from vllm import LLMEngine, SamplingParams
    from vllm.v1.spec_decode.llm_base_proposer import (
        reset_window_fastpath_stats,
        window_fastpath_stats,
    )
    from vllm.v1.spec_decode.self_spec_profiler import get_profiler

    manifest = r1._load_json(matrix._repository_path(r1.PROMPT_MANIFEST))
    regimes = {r["regime_id"]: r for r in manifest["prompt_plan"]["regimes"]}
    profiler = get_profiler()
    _require(profiler.enabled, "the self-spec profiler is not enabled")
    engine = LLMEngine.from_engine_args(r1._engine_args(cell))
    observations: dict[str, Any] = {}
    try:
        for regime_id in REGIMES:
            spec = regimes[regime_id]
            prompts = r1._prompts_for(regime_id, spec["batch"])
            before = r1._trace_len(trace)
            profiler.reset()
            reset_window_fastpath_stats()
            for index, tokens in enumerate(prompts):
                engine.add_request(
                    f"{regime_id}-{index}",
                    {"prompt_token_ids": tokens},
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=r1.MEASURE_TOKENS,
                        ignore_eos=True,
                    ),
                )
            while engine.has_unfinished_requests():
                engine.step()
            armed = r1._armed_steps(trace, before)
            summary = profiler.summary(warmup=r1.PROFILER_WARMUP)
            chain = summary.get("draft_chain", {})
            stats = window_fastpath_stats()
            observations[regime_id] = {
                "draft_chain_ms": chain.get("mean_ms"),
                "mean_armed_step_s": statistics.mean(armed) if armed else None,
                "armed_step_count": len(armed),
                "batch": spec["batch"],
                "fastpath_calls": stats["calls"],
                "fastpath_hits": stats["hits"],
                "acceptance": _acceptance(trace, before),
            }
    finally:
        with contextlib.suppress(Exception):
            engine.engine_core.shutdown()
    out_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_window_fastpath_ab",
                "config": cell,
                "fastpath": fastpath,
                "observations": observations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_all(output_dir: Path) -> None:
    lane = matrix.lane_for_block(1)
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = sorted(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    traces = output_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    for name, fastpath, cell in ORDER:
        target = output_dir / f"{name}.json"
        if target.exists():
            continue
        trace = traces / f"{name}.jsonl"
        _require(not trace.exists(), f"trace {trace} already exists")
        env = matrix._boot_child_environment(r1.boot_environment(cell, trace))
        env["VLLM_SELF_SPEC_DRAFT_WINDOW_FASTPATH"] = "1" if fastpath else "0"
        print(f"[{name}] fastpath={fastpath} window={cell['window']} ...", flush=True)
        log = output_dir / f"{name}.log"
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--measure",
                    "--trace",
                    str(trace),
                    "--out",
                    str(target),
                    "--window",
                    str(cell["window"]),
                    *(["--fastpath"] if fastpath else []),
                ],
                env=env,
                cwd=str(PHASE.parent.parent),
                stdout=handle,
                stderr=subprocess.STDOUT,
                preexec_fn=lambda: os.sched_setaffinity(0, affinity),
            )
        if completed.returncode != 0 or not target.exists():
            raise RuntimeError(
                f"{name} failed ({completed.returncode}); see {log}"
            )
        print(f"[x30] ok {name} fastpath={fastpath}", flush=True)


def summarise(output_dir: Path) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for name, _fastpath, _cell in ORDER:
        path = output_dir / f"{name}.json"
        if path.exists():
            arms[name] = json.loads(path.read_text())

    groups = {
        "slow": [n for n, _f, _c in ORDER if n.endswith("fastpath_off")],
        "fast": [n for n, _f, _c in ORDER if n.endswith("fastpath_on")],
        "nowindow": [n for n, _f, _c in ORDER if n.endswith("window_off")],
    }

    def collect(group: str, regime: str) -> list[dict[str, Any]]:
        return [
            arms[n]["observations"][regime]
            for n in groups[group]
            if n in arms and regime in arms[n]["observations"]
        ]

    def mean(rs, key):
        vals = [r[key] for r in rs if r.get(key) is not None]
        return statistics.mean(vals) if vals else None

    def pct(new, ref):
        return round((new / ref - 1) * 100, 3) if new and ref else None

    rows: dict[str, Any] = {}
    for regime in REGIMES:
        slow, fast, none = (collect(g, regime) for g in ("slow", "fast", "nowindow"))
        if not slow or not fast:
            continue
        c_slow, c_fast = mean(slow, "draft_chain_ms"), mean(fast, "draft_chain_ms")
        c_none = mean(none, "draft_chain_ms")
        acc_slow = {r["acceptance"]["committed_per_step"] for r in slow}
        acc_fast = {r["acceptance"]["committed_per_step"] for r in fast}
        rows[regime] = {
            "batch": slow[0]["batch"],
            "draft_chain_ms": {
                "window_slow": c_slow,
                "window_fast": c_fast,
                "no_window": c_none,
            },
            # What the fast path recovers, and what is still owed to window=off.
            "fastpath_gain_pct": pct(c_fast, c_slow),
            "total_window_overhead_pct": pct(c_slow, c_none),
            "residual_overhead_pct": pct(c_fast, c_none),
            "fastpath_hit_rate": [
                round(r["fastpath_hits"] / r["fastpath_calls"], 4)
                if r["fastpath_calls"]
                else None
                for r in fast
            ],
            "committed_per_step_slow": sorted(acc_slow),
            "committed_per_step_fast": sorted(acc_fast),
            "acceptance_identical": acc_slow == acc_fast,
            "repeats": {"slow": len(slow), "fast": len(fast), "no_window": len(none)},
        }
    return {
        "record_type": "w98_window_fastpath_summary",
        "schema_version": 1,
        "config": CELL,
        "control": CELL_NOWINDOW,
        "regimes": rows,
        "all_acceptance_identical": all(
            r["acceptance_identical"] for r in rows.values()
        )
        if rows
        else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--fastpath", action="store_true")
    # Absent -> the windowed cell, so an older invocation still means what it
    # meant before the no-window control arm was added.
    parser.add_argument("--window", default=None)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=PHASE / "data/probe_window_fastpath"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.measure:
        cell = None
        if args.window is not None:
            cell = CELL_NOWINDOW if args.window == "off" else {
                **CELL,
                "window": int(args.window),
            }
        measure(args.trace, args.out, args.fastpath, cell)
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_all(args.output_dir)
    summary = summarise(args.output_dir)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print("\ndraft_chain ms. 'overhead' is against the no-window control; "
          "'gain' is what the fast path recovers.")
    print("regime  b     no-win   win-slow   win-fast    overhead     gain  "
          "residual   hits   acc==")
    for regime, row in summary["regimes"].items():
        c = row["draft_chain_ms"]

        def fmt(v, width=8, prec=3):
            return f"{v:{width}.{prec}f}" if v is not None else " " * (width - 1) + "-"

        def fpct(v):
            return f"{v:+8.2f}%" if v is not None else "       -"

        print(
            f"{regime:6s} {row['batch']:3d} {fmt(c['no_window'])} "
            f"{fmt(c['window_slow'])} {fmt(c['window_fast'])} "
            f"{fpct(row['total_window_overhead_pct'])} "
            f"{fpct(row['fastpath_gain_pct'])} "
            f"{fpct(row['residual_overhead_pct'])} "
            f"{row['fastpath_hit_rate']} {row['acceptance_identical']}"
        )
    print(f"\nall acceptance identical: {summary['all_acceptance_identical']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
