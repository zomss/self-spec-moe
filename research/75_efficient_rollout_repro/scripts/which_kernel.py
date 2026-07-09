#!/usr/bin/env python3
"""Report which mixed-precision linear kernel vLLM will select, under the CURRENT env.

`choose_mp_linear_kernel()` returns silently -- vLLM logs nothing -- so a log-grep
guardrail would be vacuous. This asks the chooser directly, with the same
VLLM_DISABLED_KERNELS the run will use, for the actual weight shapes of the model.

Usage:
  python which_kernel.py --bits 4              # -> MacheteLinearKernel (auto)
  VLLM_DISABLED_KERNELS=MacheteLinearKernel,... python which_kernel.py --bits 4
                                               # -> MarlinLinearKernel (the paper's)
Exit 0 on success; prints one line `KERNEL=<ClassName>`.
"""

import argparse
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, choices=(4, 8), default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--zero-points", action="store_true", help="asymmetric RTN")
    # Qwen2.5-7B FFN down_proj: the largest quantized GEMM (in=18944, out=3584).
    ap.add_argument("--in-features", type=int, default=18944)
    ap.add_argument("--out-features", type=int, default=3584)
    a = ap.parse_args()

    from vllm import envs
    from vllm.model_executor.kernels.linear import choose_mp_linear_kernel
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.scalar_type import scalar_types

    if a.zero_points:
        wtype = scalar_types.uint4 if a.bits == 4 else scalar_types.uint8
    else:
        wtype = scalar_types.uint4b8 if a.bits == 4 else scalar_types.uint8b128

    cfg = MPLinearLayerConfig(
        full_weight_shape=(a.in_features, a.out_features),
        partition_weight_shape=(a.in_features, a.out_features),
        weight_type=wtype,
        act_type=torch.bfloat16,
        group_size=a.group_size,
        zero_points=a.zero_points,
        has_g_idx=False,
    )
    disabled = list(envs.VLLM_DISABLED_KERNELS)
    try:
        kernel = choose_mp_linear_kernel(cfg)
    except ValueError as e:
        print(f"KERNEL=NONE  ({str(e)[:160]})")
        print(f"  disabled={disabled} wtype={wtype} group={a.group_size} zp={a.zero_points}")
        return 3

    print(f"KERNEL={kernel.__name__}")
    print(f"  wtype={wtype} group={a.group_size} zp={a.zero_points} disabled={disabled}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
