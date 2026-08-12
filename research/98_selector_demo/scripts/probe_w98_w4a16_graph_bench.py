# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X9b -- W4A16 kernel comparison with host dispatch removed.

DIAGNOSTIC, NOT SCORED.

X9 ran the repo's `benchmark_machete.py` at Qwen3-8B's layer shapes and could
not resolve the question. Its `bench_fns` times a PYTHON loop (`for fn in fns:
fn()`), so every call carries dispatch overhead. Fitting time against weight
bytes across the four shapes exposes the floor:

    machete       12.58 us per-call overhead
    machete_best  14.43 us
    marlin        24.34 us
    torch.matmul   4.41 us

Those intercepts exceed the entire bandwidth-bound ideal for three of the four
shapes (o_proj 2.58 us, qkv 3.87, down 7.75), and Marlin's fitted slope implies
a bandwidth 5x above HBM -- physically impossible, and the signature of a
measurement that is all overhead. Comparing kernels on those numbers would
mostly compare their Python wrappers.

This probe captures N calls into a CUDA graph and replays it, timed with CUDA
events. Replay issues no per-call host work, so the measurement is device time
for the kernel itself -- the quantity that decides whether a decode kernel is
worth writing. It is also the regime the draft actually runs in, since X6
established the chain is captured.

Reading rule, fixed before the data, per (shape, M) against that shape's own
bandwidth bound:
  * a kernel at >=80% of bound   -> use it; no new code needed
  * best kernel 40-80%           -> worth a specialised kernel, gap quantified
  * all <40%                     -> new code clearly justified
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "kernels"))

HIDDEN, INTER, HEADS, KV_HEADS, HEAD_DIM = 4096, 12288, 32, 8, 128
LAYER_SHAPES = [
    ("qkv_proj", HIDDEN, (HEADS + 2 * KV_HEADS) * HEAD_DIM),
    ("o_proj", HEADS * HEAD_DIM, HIDDEN),
    ("gate_up_proj", HIDDEN, 2 * INTER),
    ("down_proj", INTER, HIDDEN),
]
M_VALUES = [1, 2, 4, 8, 16, 32]
GROUP_SIZE = 128
HBM_BW = 3.35e12

GRAPH_CALLS = 32  # calls captured per graph
REPLAYS = 20


def shape_bytes(k: int, n: int, group_size: int) -> float:
    """4-bit weights plus bf16 group scales; activations are negligible at M<=32."""
    return k * n / 2 + (k / group_size) * n * 2


def time_graph(fn, calls: int, replays: int) -> float:
    """Capture `calls` invocations into a graph; return device us per call."""
    import torch

    # Warm up on a side stream before capture, as CUDA graph capture requires.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()

    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    graph.replay()
    torch.accelerator.synchronize()
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) * 1000.0 / (replays * calls)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from benchmark_machete import (  # noqa: E402
        TypeConfig,
        create_bench_tensors,
        machete_create_bench_fn,
        marlin_create_bench_fn,
    )

    from vllm.scalar_type import scalar_types

    types = TypeConfig(
        act_type=torch.bfloat16,
        weight_type=scalar_types.uint4b8,
        output_type=None,
        group_scale_type=torch.bfloat16,
        group_zero_type=None,
        channel_scale_type=None,
        token_scale_type=None,
    )

    rows: list[dict[str, Any]] = []
    for name, k, n in LAYER_SHAPES:
        for m in M_VALUES:
            # One weight set is enough: the graph replays the SAME weights, so
            # they sit in L2 for small shapes. Recorded as a caveat rather than
            # worked around -- L2 residency flatters every kernel equally, and
            # the in-engine figure (38% of bound) is the un-flattered anchor.
            bt = create_bench_tensors((m, n, k), types, GROUP_SIZE)[0]
            entry: dict[str, Any] = {
                "layer": name,
                "m": m,
                "k": k,
                "n": n,
                "bytes": shape_bytes(k, n, GROUP_SIZE),
                "ideal_us": shape_bytes(k, n, GROUP_SIZE) / HBM_BW * 1e6,
                "us_per_call": {},
            }
            # machete_create_bench_fn's out_type default is `torch.dtype` --
            # the CLASS, not a value -- so it must be passed explicitly or the
            # custom op rejects it. (Upstream bug in the benchmark helper; the
            # `bench` path passes it and so never hits this.)
            makers = {
                "machete": lambda b: machete_create_bench_fn(b, out_type=None),
                "marlin": marlin_create_bench_fn,
            }
            for label, maker in makers.items():
                try:
                    fn = maker(bt)
                    entry["us_per_call"][label] = time_graph(fn, GRAPH_CALLS, REPLAYS)
                except Exception as exc:  # noqa: BLE001 - report, do not abort
                    entry["us_per_call"][label] = None
                    entry.setdefault("errors", {})[label] = str(exc)[:200]
            best = [v for v in entry["us_per_call"].values() if v]
            print(
                f"{name:14s} M={m:3d} ideal={entry['ideal_us']:7.2f}us  "
                + "  ".join(
                    f"{lbl}={v:7.2f}({entry['ideal_us'] / v * 100:3.0f}%)"
                    if v
                    else f"{lbl}=FAIL"
                    for lbl, v in entry["us_per_call"].items()
                )
                + (f"  best={min(best):7.2f}us" if best else ""),
                flush=True,
            )
            rows.append(entry)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_w4a16_graph_bench",
                "model": "Qwen3-8B",
                "group_size": GROUP_SIZE,
                "hbm_bw_bytes_per_s": HBM_BW,
                "graph_calls": GRAPH_CALLS,
                "replays": REPLAYS,
                "caveat": (
                    "one weight set replayed, so large shapes stream from HBM "
                    "but small ones may sit in L2; flatters all kernels equally"
                ),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
