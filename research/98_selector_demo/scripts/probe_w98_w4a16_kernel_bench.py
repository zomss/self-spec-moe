# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X9 -- is the W4A16 decode gap a config change, a kernel swap, or new code?

DIAGNOSTIC, NOT SCORED.

X7/X8 left one large opportunity on the draft side: the W4A16 GEMM is 41% of
device time and runs at ~65% of the H100 bandwidth-bound ideal (19.40 us/call
against 12.65), worth ~3.0 ms/step. Before writing a kernel, two cheaper
possibilities have to be excluded:

  1. Machete's `schedule` argument is exposed (`ops.machete_mm(..., schedule=)`)
     and vLLM's wrapper never passes one, so the C++ default heuristic picks --
     tuned for general serving, not for M=1. Nine schedules exist, including a
     narrow `128x16_1x1x1`.
  2. Marlin is the kernel designed for exactly this regime (near-ideal weight
     bandwidth at small batch) and supports our config (group_size 128,
     uint4b8). It sits 4th in the CUDA priority list, behind Machete, on a
     general default ordering rather than a decode-specific one.

This drives the repo's existing `benchmark_machete.py` (which already sweeps
schedules and benchmarks Marlin alongside) with **Qwen3-8B's actual layer
shapes** rather than square matrices, at the M values the draft chain really
runs: batch 1-32 with one token per step.

Reading rule, fixed before the data. Per (shape, M), compare:
  * machete-default   -- what we run today
  * machete-best      -- best of the 9 schedules
  * marlin            -- the small-batch specialist
against the bandwidth-bound ideal for that shape. Then:
  * machete-best ~= ideal          -> config change, pass a schedule
  * marlin ~= ideal, beats machete -> kernel-priority change for the draft
  * all well short of ideal        -> new code is justified, with a baseline
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

# Qwen3-8B, TP1. hidden 4096, intermediate 12288, 32 q heads / 8 kv heads, hd 128.
HIDDEN, INTER, HEADS, KV_HEADS, HEAD_DIM = 4096, 12288, 32, 8, 128
LAYER_SHAPES = [
    ("qkv_proj", HIDDEN, (HEADS + 2 * KV_HEADS) * HEAD_DIM),
    ("o_proj", HEADS * HEAD_DIM, HIDDEN),
    ("gate_up_proj", HIDDEN, 2 * INTER),
    ("down_proj", INTER, HIDDEN),
]
# The draft decodes one token per sequence, so M == batch. X8's range was 1-32.
M_VALUES = [1, 2, 4, 8, 16, 32]
GROUP_SIZE = 128
HBM_BW = 3.35e12  # H100 80GB HBM3


def ideal_us(k: int, n: int, group_size: int) -> float:
    """Weight-streaming lower bound for one W4A16 GEMM at small M.

    At M<=32 activations are negligible (a 4096-wide bf16 row is 8 KB against
    tens of MB of weights), so the bound is set by 4-bit weights plus the bf16
    group scales.
    """
    weight_bytes = k * n / 2  # 4 bits per weight
    scale_bytes = (k / group_size) * n * 2  # bf16 group scales
    return (weight_bytes + scale_bytes) / HBM_BW * 1e6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from benchmark_machete import TypeConfig, bench  # noqa: E402

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
            timers = bench(
                types,
                GROUP_SIZE,
                m,
                k,
                n,
                label="w4a16-decode",
                sub_label=f"{name} MKN=({m}x{k}x{n})",
                sweep_schedules=True,
            )
            entry = {
                "layer": name,
                "m": m,
                "k": k,
                "n": n,
                "ideal_us": ideal_us(k, n, GROUP_SIZE),
                "timings_us": {},
            }
            for t in timers:
                # torch.utils.benchmark Measurement: .median is seconds.
                entry["timings_us"][t.task_spec.description] = t.median * 1e6
            rows.append(entry)
            best = min(entry["timings_us"].values()) if entry["timings_us"] else None
            print(
                f"{name:14s} M={m:3d}  ideal={entry['ideal_us']:7.2f}us  "
                f"best={best:7.2f}us"
                if best
                else f"{name} M={m} no timings",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_w4a16_kernel_bench",
                "model": "Qwen3-8B",
                "group_size": GROUP_SIZE,
                "hbm_bw_bytes_per_s": HBM_BW,
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
