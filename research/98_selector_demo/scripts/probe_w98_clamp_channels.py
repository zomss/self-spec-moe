# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X29 -- one channel per surviving clamp mechanism.

DIAGNOSTIC, NOT SCORED. Successor to probe_w98_flip_sentinel (X28), whose
record during a confirmed two-hour clamped stretch showed the launch canary,
the single-threaded mmap probe, and steal ALL flat -- even while the engine
was clamping in the same minutes on the next GPU. Whatever clamps the engine
lives in a channel those probes structurally cannot see. Three candidates
survive that null, and each gets a direct in-guest discriminator here:

  * chase_*   -- physical-host LLC/DRAM latency contention. Serial pointer
                 chase at L2-resident / LLC-sized / DRAM-sized working sets.
                 Contention steps the big chains and spares the small one;
                 the X28 canary was blind because its loop is cache-resident.
  * clock     -- kvm-clock vDSO fast path degrading to syscall. ns per
                 monotonic clock read, with getpid as the syscall baseline.
                 Blind spot: X28 read the clock twice per 5 ms iteration;
                 the engine reads it orders of magnitude more per step.
  * munmap_mt -- synchronous TLB-shootdown cost on a multi-core mm. Toucher
                 threads pinned to distinct CPUs pull the mapping into their
                 TLBs, then the timed munmap must IPI them all. X28's mmap
                 channel was blind because a single-threaded mm sends no
                 shootdowns; the engine's 144 munmaps/step IPI its whole lane.
  * sync_spin / sync_block -- wake-up latency after a real GPU wait: an
                 identical ~2 ms kernel awaited by event polling (pure spin)
                 and by cudaDeviceSynchronize (may sleep). The difference is
                 scheduler/interrupt wake overhead. Shape evidence already
                 disfavours this class; kept as the cheap control.
  * canary / mmap -- X28's original channels, unchanged, as the continuity
                 controls linking this record to the old one.

Every burst also snapshots the box (census, GPUs, load, pressure, vmstat,
steal, clocksource) exactly as X28 did. Run pinned, one GPU:

  CUDA_VISIBLE_DEVICES=5 taskset -c 72-87 python probe_w98_clamp_channels.py \
      --out data/probe_flip_sentinel/channels_<box>.jsonl
"""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap as mmap_mod
import os
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_w98_flip_sentinel as x28  # noqa: E402

CHASE_SETS_MIB = (1, 32, 512)
CHASE_HOPS = 1 << 20
CLOCK_READS = 200_000
GETPID_CALLS = 100_000
MUNMAP_TOUCHERS = 8
MUNMAP_CYCLES = 30
MUNMAP_BYTES = 2 * 1024 * 1024
SYNC_REPS = 20
SYNC_KERNEL_MS = 2.0

_CHASE_C = r"""
#include <stdint.h>
uint32_t chase(const uint32_t *perm, uint64_t hops) {
    uint32_t p = 0;
    for (uint64_t i = 0; i < hops; i++) p = perm[p];
    return p;
}
"""


def _build_chase_helper(scratch: Path):
    """Compile the serial-chase kernel; a python loop's ~100 ns/iter
    interpreter overhead would drown the DRAM latency being measured."""
    src = scratch / "chase.c"
    lib = scratch / "chase.so"
    if not lib.exists():
        src.write_text(_CHASE_C)
        rc = os.system(f"cc -O2 -shared -fPIC -o {lib} {src}")
        if rc != 0:
            return None
    dll = ctypes.CDLL(str(lib))
    dll.chase.restype = ctypes.c_uint32
    dll.chase.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint64]
    return dll


def _sattolo_cycle(n: int, seed: int = 12345):
    """Single-cycle permutation so every hop is a dependent random load."""
    import numpy as np

    rng = np.random.default_rng(seed)
    order = rng.permutation(n).astype(np.uint32)
    perm = np.empty(n, dtype=np.uint32)
    perm[order] = np.roll(order, -1)
    return perm


class ChaseChannel:
    def __init__(self, scratch: Path):
        self.dll = _build_chase_helper(scratch)
        self.chains = {}
        if self.dll is None:
            return
        for mib in CHASE_SETS_MIB:
            n = mib * 1024 * 1024 // 4
            self.chains[mib] = _sattolo_cycle(n)

    def read(self) -> dict[str, float]:
        if self.dll is None:
            return {"unavailable": 1.0}
        out = {}
        for mib, perm in self.chains.items():
            ptr = perm.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
            start = time.perf_counter()
            self.dll.chase(ptr, CHASE_HOPS)
            out[f"ns_per_hop_{mib}MiB"] = round(
                (time.perf_counter() - start) * 1e9 / CHASE_HOPS, 2
            )
        return out


def clock_channel() -> dict[str, float]:
    start = time.perf_counter()
    for _ in range(CLOCK_READS):
        time.monotonic_ns()
    clock_ns = (time.perf_counter() - start) * 1e9 / CLOCK_READS
    start = time.perf_counter()
    for _ in range(GETPID_CALLS):
        os.getpid()
    getpid_ns = (time.perf_counter() - start) * 1e9 / GETPID_CALLS
    return {
        "monotonic_ns_per_read": round(clock_ns, 1),
        "getpid_ns_per_call": round(getpid_ns, 1),
    }


class MultiTLBChannel:
    """Timed munmap of a mapping resident in MUNMAP_TOUCHERS other TLBs."""

    def __init__(self, cpus: list[int]):
        self.cpus = cpus[:MUNMAP_TOUCHERS]
        self.map_ref: mmap_mod.mmap | None = None
        self.go = threading.Barrier(len(self.cpus) + 1)
        self.done = threading.Barrier(len(self.cpus) + 1)
        self.stop = False
        self.threads = [
            threading.Thread(target=self._toucher, args=(c,), daemon=True)
            for c in self.cpus
        ]
        for t in self.threads:
            t.start()

    def _toucher(self, cpu: int) -> None:
        os.sched_setaffinity(threading.get_native_id(), {cpu})
        while True:
            self.go.wait()
            if self.stop:
                return
            m = self.map_ref
            for off in range(0, MUNMAP_BYTES, 4096):
                m[off]
            self.done.wait()

    def read(self) -> dict[str, float]:
        samples = []
        for _ in range(MUNMAP_CYCLES):
            m = mmap_mod.mmap(-1, MUNMAP_BYTES)
            m[0:MUNMAP_BYTES] = b"\x01" * MUNMAP_BYTES
            self.map_ref = m
            self.go.wait()
            self.done.wait()
            start = time.perf_counter()
            m.close()
            samples.append((time.perf_counter() - start) * 1e6)
        samples.sort()
        n = len(samples)
        return {
            "munmap_us_median": round(statistics.median(samples), 2),
            "munmap_us_p90": round(samples[(9 * n) // 10], 2),
            "munmap_us_max": round(samples[-1], 2),
        }


class SyncChannel:
    """Identical GPU wait, spun vs slept; the difference is wake latency."""

    def __init__(self):
        import torch

        self.torch = torch
        torch.cuda.init()
        # Calibrate _sleep cycles for ~SYNC_KERNEL_MS.
        cycles = 1_000_000
        for _ in range(6):
            start = time.perf_counter()
            torch.cuda._sleep(cycles)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - start) * 1000
            cycles = int(cycles * SYNC_KERNEL_MS / max(ms, 1e-3))
        self.cycles = cycles

    def read(self) -> dict[str, float]:
        torch = self.torch
        spin, block = [], []
        event = torch.cuda.Event()
        for _ in range(SYNC_REPS):
            start = time.perf_counter()
            torch.cuda._sleep(self.cycles)
            event.record()
            while not event.query():
                pass
            spin.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            torch.cuda._sleep(self.cycles)
            torch.cuda.synchronize()
            block.append((time.perf_counter() - start) * 1000)
        return {
            "spin_ms_median": round(statistics.median(spin), 4),
            "block_ms_median": round(statistics.median(block), 4),
            "wake_overhead_ms": round(
                statistics.median(block) - statistics.median(spin), 4
            ),
        }


def clocksource() -> str:
    node = Path("/sys/devices/system/clocksource/clocksource0/current_clocksource")
    try:
        return node.read_text().strip()
    except OSError:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval-s", type=float, default=30.0)
    parser.add_argument("--bursts", type=int, default=0, help="0 = run forever")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path("/tmp") / f"x29-{os.getuid()}"
    scratch.mkdir(parents=True, exist_ok=True)

    my_cpus = sorted(os.sched_getaffinity(0))
    chase = ChaseChannel(scratch)
    tlb = MultiTLBChannel(my_cpus)
    sync = SyncChannel()

    burst = 0
    while True:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "clocksource": clocksource(),
            "chase": chase.read(),
            "clock": clock_channel(),
            "munmap_mt": tlb.read(),
            "sync": sync.read(),
            "canary": x28.canary_burst(),
            "mmap": x28.mmap_burst(),
            "gpus": x28.gpu_state(),
            "loadavg": Path("/proc/loadavg").read_text().split()[:3],
            "pressure": x28.pressure(),
            "vmstat": x28.vmstat_snapshot(),
            "steal": x28.steal_ticks(),
            "census": x28.process_census(),
        }
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            "[x29] {ts} chase512={c512} clock={clk} munmap_mt={mm} wake={wk} "
            "canary={cn}".format(
                ts=record["ts"],
                c512=record["chase"].get("ns_per_hop_512MiB"),
                clk=record["clock"]["monotonic_ns_per_read"],
                mm=record["munmap_mt"]["munmap_us_median"],
                wk=record["sync"]["wake_overhead_ms"],
                cn=record["canary"]["iter_ms_median"],
            ),
            flush=True,
        )
        burst += 1
        if args.bursts and burst >= args.bursts:
            return 0
        time.sleep(args.interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
