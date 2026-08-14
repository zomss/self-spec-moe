# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm the lane's torch.compile cache for every G98-C boot configuration.

Campaign boots set VLLM_SELF_SPEC_KOFF_TRACE to an attempt-numbered trace
path, and vllm hashes registered env vars into its compile-cache key, so
every boot cold-compiled (~14.7 s on h104) and the host-load gate read the
cold cache as a loud host -- three false 'loud' rejections on 2026-08-14
before the cause was found. With that var in envs.py's ignored_factors, one
boot per unique configuration warms the cache and every campaign boot
becomes a cache load, restoring the warm-startup regime the gate thresholds
were calibrated against.

Engine construction only: no measurement, no campaign outputs. Not part of
the authorization; run it before (re-)arming run_g98c_loop.sh on a box with
a cold lane cache.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_g98c_round2 as g98c  # noqa: E402

r1 = g98c.r1
matrix = g98c.matrix

BOOT_SNIPPET = """\
import json
import sys

sys.path.insert(0, sys.argv[1])
import run_w98_g98b_round1 as r1
from vllm import LLMEngine

engine = LLMEngine.from_engine_args(r1._engine_args(json.loads(sys.argv[2])))
print("[warm] engine constructed; exiting", flush=True)
"""


def main() -> int:
    profiles: list[dict] = []
    seen: set[str] = set()
    for cfg in (*g98c.ANCHORS, *g98c.fit_profiles(), *g98c.heldout_profiles()):
        key = g98c._config_key(cfg)
        if key not in seen:
            seen.add(key)
            profiles.append(dict(cfg))

    lane = matrix.lane_for_block(1)
    lo, hi = lane["cpu_affinity"].split("-")
    affinity = set(range(int(lo), int(hi) + 1))
    Path(lane["cache_root"]).mkdir(parents=True, exist_ok=True)
    scratch = Path("/tmp/v-sukmincho-w98/warm_traces")
    scratch.mkdir(parents=True, exist_ok=True)

    failures = 0
    for index, cfg in enumerate(profiles):
        key = g98c._config_key(cfg).replace("/", "_")
        trace = scratch / f"{key}.jsonl"
        trace.unlink(missing_ok=True)
        env = matrix._boot_child_environment(r1.boot_environment(cfg, trace))
        # The compile cache is device-index-agnostic (CUDA_VISIBLE_DEVICES is
        # an ignored hash factor), so warming may borrow any idle same-model
        # GPU while the lane GPU is occupied.
        if os.environ.get("W98_WARM_GPU"):
            env[matrix.DEVICE_PIN_ENV] = os.environ["W98_WARM_GPU"]
        log = scratch / f"{key}.log"
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [sys.executable, "-c", BOOT_SNIPPET, str(SCRIPT_DIR), json.dumps(cfg)],
                cwd=r1.REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                preexec_fn=lambda: os.sched_setaffinity(0, affinity),
            )
        status = "ok" if completed.returncode == 0 else f"rc={completed.returncode}"
        failures += completed.returncode != 0
        print(f"[warm] {index + 1}/{len(profiles)} {key} {status}", flush=True)
    print(f"[warm] done: {len(profiles) - failures}/{len(profiles)} warmed", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
