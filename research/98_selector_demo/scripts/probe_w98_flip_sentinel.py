# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X28 -- a continuous sentinel to catch the clamp FLIPPING.

DIAGNOSTIC, NOT SCORED.

Thirteen refuted hypotheses share one method flaw the doc already names: every
instrument observed a STATE (clamped, or clean) and none observed a TRANSITION.
The clamp switches on a timescale of tens of minutes; the cause is whatever
changes on the box at the flip moment, and nothing has ever been watching then.

This watches. Two parts, one process:

  * A CANARY that mimics the draft chain's host profile -- hundreds of tiny
    kernel launches per iteration, then a device synchronize -- because the
    clamp is a fixed per-step HOST delay that scales with dispatch exposure,
    not with GPU work. If the canary's wall time steps up and down with the
    box state, the clamp is not even vLLM-specific and the canary is a
    per-30-seconds clamp meter. If gate probes flip while the canary holds
    flat, the clamp needs something the engine step has and a bare launch loop
    lacks -- either way the canary result discriminates.
  * A BOX SNAPSHOT taken with every burst: per-user process census (so a
    server, monitor, or session appearing or vanishing is on record), GPU
    utilisation, load average, and the pressure files. When the canary steps,
    the snapshot diff at that timestamp is the candidate cause.

Runs on GPU 1 pinned to lane-b CPUs -- the lane X22 showed clamps identically
-- leaving GPU 0 free for gate probes and the campaign.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

LAUNCHES_PER_ITER = 800
ITERS_PER_BURST = 100
WARMUP_ITERS = 20
VMSTAT_KEYS = ("pgfault", "pgmajfault", "numa_hint_faults", "compact_stall")


def canary_burst() -> dict[str, float]:
    """Wall time per iteration of a dispatch-heavy, GPU-light loop."""
    import torch

    x = torch.zeros(256, 256, device="cuda")
    samples: list[float] = []
    for index in range(WARMUP_ITERS + ITERS_PER_BURST):
        start = time.perf_counter()
        for _ in range(LAUNCHES_PER_ITER):
            x.add_(1.0)
        torch.accelerator.synchronize()
        if index >= WARMUP_ITERS:
            samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    n = len(samples)
    return {
        "iter_ms_p10": round(samples[n // 10], 4),
        "iter_ms_median": round(statistics.median(samples), 4),
        "iter_ms_p90": round(samples[(9 * n) // 10], 4),
        "iter_ms_max": round(samples[-1], 4),
    }


MMAP_BYTES = 8 * 1024 * 1024
MMAP_ITERS = 100


def mmap_burst() -> dict[str, float]:
    """Cost of the map-fault-unmap cycle the engine step performs 144x.

    The steal data refuted vCPU preemption (ticks accrue during boots but at
    ~0.2% of wall against the ~11% the clamp requires), which leaves the
    fault/exit class: hypervisor state -- dirty-page logging, KSM, EPT churn
    -- that makes guest memory-mapping operations expensive without
    registering as steal. The engine step does 144 mmap/munmap pairs (one per
    flash-attention call); ~31 us of extra cost per cycle explains the whole
    clamp. This measures that cycle directly: map, touch every page, unmap.
    If this channel steps down at the flip while the launch canary holds
    flat, the mechanism is localised.
    """
    import mmap as mmap_mod

    samples: list[float] = []
    for _ in range(MMAP_ITERS):
        start = time.perf_counter()
        region = mmap_mod.mmap(-1, MMAP_BYTES)
        for offset in range(0, MMAP_BYTES, 4096):
            region[offset] = 1
        region.close()
        samples.append((time.perf_counter() - start) * 1e6)
    samples.sort()
    n = len(samples)
    return {
        "cycle_us_p10": round(samples[n // 10], 2),
        "cycle_us_median": round(statistics.median(samples), 2),
        "cycle_us_p90": round(samples[(9 * n) // 10], 2),
    }


def process_census() -> dict[str, Any]:
    """Who is on the box: enough to diff at a flip, small enough to log."""
    out = subprocess.run(
        ["ps", "-eo", "user:16,pcpu,comm", "--no-headers"],
        capture_output=True,
        text=True,
    ).stdout
    census: dict[str, Any] = {}
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        user, pcpu, comm = parts
        entry = census.setdefault(user, {"procs": 0, "cpu": 0.0, "comms": {}})
        entry["procs"] += 1
        entry["cpu"] = round(entry["cpu"] + float(pcpu), 1)
        if float(pcpu) >= 0.5:
            entry["comms"][comm] = round(entry["comms"].get(comm, 0.0) + float(pcpu), 1)
    return census


def gpu_state() -> list[str]:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    ).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def vmstat_snapshot() -> dict[str, int]:
    values = {}
    for line in Path("/proc/vmstat").read_text().splitlines():
        key, _, value = line.partition(" ")
        if key in VMSTAT_KEYS:
            values[key] = int(value)
    return values


def pressure() -> dict[str, str]:
    out = {}
    for kind in ("cpu", "memory", "io"):
        path = Path(f"/proc/pressure/{kind}")
        if path.is_file():
            out[kind] = path.read_text().splitlines()[0]
    return out


# The box is a KVM guest on kvm-clock, so hypervisor preemption of a vCPU
# shows up as guest wall time with nothing visible consuming CPU -- the
# clamp's exact presentation. Steal ticks per lane are the discriminator:
# steal accruing during a clamped boot means vCPU preemption; a clamped boot
# with flat steal means the fault/exit class (dirty-page logging, KSM, EPT
# pressure), which inflates wall time without registering as steal.
STEAL_LANES = {
    "lane_a": range(0, 16),
    "lane_b": range(96, 112),
    "analysis": range(32, 64),
}


def steal_ticks() -> dict[str, int]:
    per_cpu: dict[int, int] = {}
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("cpu") and line[3].isdigit():
            fields = line.split()
            per_cpu[int(fields[0][3:])] = int(fields[8])
    out = {
        lane: sum(per_cpu.get(c, 0) for c in cpus) for lane, cpus in STEAL_LANES.items()
    }
    out["total"] = sum(per_cpu.values())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval-s", type=float, default=30.0)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    while True:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "canary": canary_burst(),
            "mmap": mmap_burst(),
            "gpus": gpu_state(),
            "loadavg": Path("/proc/loadavg").read_text().split()[:3],
            "pressure": pressure(),
            "vmstat": vmstat_snapshot(),
            "steal": steal_ticks(),
            "census": process_census(),
        }
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            f"[sentinel] {record['ts']} median={record['canary']['iter_ms_median']}ms",
            flush=True,
        )
        time.sleep(args.interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
