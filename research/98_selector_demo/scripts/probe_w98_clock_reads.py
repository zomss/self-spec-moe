# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X29b -- count clock reads and munmaps per batch-1 engine step.

Candidate 2 of section 11 (kvm-clock vDSO degrading to syscall) can only
carry the clamp if the engine makes on the order of thousands of clock reads
per step: the clamp is +4-5 ms/step and a degraded read costs ~2 us, so
fewer than ~500 reads/step kills the candidate by arithmetic, no old-box
deployment needed. This measures the count directly: an LD_PRELOAD shim
interposes clock_gettime/gettimeofday (vDSO calls route through the libc
symbol) and munmap, and the driver reads the counters around a counted
window of engine steps.

Run (h104): CUDA_VISIBLE_DEVICES=4 taskset -c 96-111 \
    .venv/bin/python research/98_selector_demo/scripts/probe_w98_clock_reads.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

MEASURE_STEPS = 200
MAX_TOKENS = 1024

_SHIM_C = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <time.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <stddef.h>
static long long n_clock = 0, n_gtod = 0, n_munmap = 0;
static int (*real_cg)(clockid_t, struct timespec *) = 0;
static int (*real_gtod)(struct timeval *, void *) = 0;
static int (*real_munmap)(void *, size_t) = 0;
int clock_gettime(clockid_t c, struct timespec *t) {
    if (!real_cg) real_cg = dlsym(RTLD_NEXT, "clock_gettime");
    __atomic_add_fetch(&n_clock, 1, __ATOMIC_RELAXED);
    return real_cg(c, t);
}
int gettimeofday(struct timeval *tv, void *tz) {
    if (!real_gtod) real_gtod = dlsym(RTLD_NEXT, "gettimeofday");
    __atomic_add_fetch(&n_gtod, 1, __ATOMIC_RELAXED);
    return real_gtod(tv, tz);
}
int munmap(void *addr, size_t len) {
    if (!real_munmap) real_munmap = dlsym(RTLD_NEXT, "munmap");
    __atomic_add_fetch(&n_munmap, 1, __ATOMIC_RELAXED);
    return real_munmap(addr, len);
}
long long shim_clock_reads(void) { return n_clock; }
long long shim_gtod_reads(void) { return n_gtod; }
long long shim_munmaps(void) { return n_munmap; }
"""

_DRIVER = r"""
import ctypes
import json
import sys
import time

sys.path.insert(0, sys.argv[1])
import run_w98_g98b_round1 as r1
from vllm import LLMEngine, SamplingParams

shim = ctypes.CDLL(None)
for f in ("shim_clock_reads", "shim_gtod_reads", "shim_munmaps"):
    getattr(shim, f).restype = ctypes.c_longlong

cfg = {"quant": "target-matching", "window": "off", "skip_count": 0}
engine = LLMEngine.from_engine_args(r1._engine_args(cfg))
tokens = r1._prompts_for("R1", 1)[0]
params = SamplingParams(
    temperature=0.0, max_tokens=int(sys.argv[2]), ignore_eos=True
)
engine.add_request("x29b-0", {"prompt_token_ids": tokens}, params)

warm = 0
while warm < 32 and engine.has_unfinished_requests():
    engine.step()
    warm += 1
c0, g0, m0 = shim.shim_clock_reads(), shim.shim_gtod_reads(), shim.shim_munmaps()
t0 = time.perf_counter()
steps = 0
while steps < int(sys.argv[3]) and engine.has_unfinished_requests():
    engine.step()
    steps += 1
wall = time.perf_counter() - t0
c1, g1, m1 = shim.shim_clock_reads(), shim.shim_gtod_reads(), shim.shim_munmaps()

# Reads made by libcuda's spin-wait do not extend wall time when the clock
# slows (the spin exits on GPU completion); only serial-path reads amplify.
# Measure the spin-loop read rate against a pure GPU wait of known length.
import torch

cycles = 1_000_000
for _ in range(4):
    t = time.perf_counter()
    torch.cuda._sleep(cycles)
    torch.cuda.synchronize()
    cycles = int(cycles * 5.0 / max((time.perf_counter() - t) * 1000, 1e-3))
cs0 = shim.shim_clock_reads()
t = time.perf_counter()
torch.cuda._sleep(cycles)
torch.cuda.synchronize()
spin_ms = (time.perf_counter() - t) * 1000
spin_reads_per_ms = (shim.shim_clock_reads() - cs0) / max(spin_ms, 1e-3)

print("X29B_RESULT " + json.dumps({
    "steps": steps,
    "wall_ms_per_step": round(wall * 1000 / max(steps, 1), 3),
    "clock_reads_per_step": round((c1 - c0) / max(steps, 1), 1),
    "gettimeofday_per_step": round((g1 - g0) / max(steps, 1), 1),
    "munmaps_per_step": round((m1 - m0) / max(steps, 1), 1),
    "spin_reads_per_ms": round(spin_reads_per_ms, 1),
    "spin_window_ms": round(spin_ms, 2),
}))
"""


def main() -> int:
    scratch = Path("/tmp") / f"x29-{os.getuid()}"
    scratch.mkdir(parents=True, exist_ok=True)
    src, lib = scratch / "clockshim.c", scratch / "clockshim.so"
    src.write_text(_SHIM_C)
    rc = os.system(f"cc -O2 -shared -fPIC -o {lib} {src} -ldl")
    if rc != 0:
        print("shim build failed", file=sys.stderr)
        return 1
    import run_w98_g98b_round1 as r1

    cfg = {"quant": "target-matching", "window": "off", "skip_count": 0}
    trace = scratch / "x29b_trace.jsonl"
    trace.unlink(missing_ok=True)
    env = r1.matrix._boot_child_environment(r1.boot_environment(cfg, trace))
    env["LD_PRELOAD"] = str(lib)
    completed = subprocess.run(
        [sys.executable, "-c", _DRIVER, str(SCRIPT_DIR), str(MAX_TOKENS),
         str(MEASURE_STEPS)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in completed.stdout.splitlines():
        if line.startswith("X29B_RESULT "):
            print(line)
            break
    else:
        print(completed.stdout[-2500:])
        return 1
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
