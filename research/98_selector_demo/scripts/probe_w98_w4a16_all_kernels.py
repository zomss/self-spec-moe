# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""X10 -- is Marlin actually the best available W4A16 kernel for the draft?

DIAGNOSTIC, NOT SCORED.

X9b compared only Machete and Marlin, which is not enough to justify a kernel
choice. Asking the registry which of the eight CUDA kernels can implement the
draft's config (group_size 128, uint4b8, bf16 activations, SM90) gives four:

    MacheteLinearKernel      YES  (selected today)
    MarlinLinearKernel       YES  (X9b's winner over Machete)
    HummingLinearKernel      YES  (never benchmarked)
    TritonW4A16LinearKernel  YES  (never benchmarked)

and rules out four with reasons: CutlassW4A8 (FP8 activations only), AllSpark
(no SM90 support), Conch (package not installed), Exllama (fp16 activations
only).

This benchmarks all four through their REAL `MPLinearKernel` path -- construct
the layer, run each kernel's own `process_weights_after_loading` repack, then
time `apply_weights`. That is what the engine would run, rather than a
benchmark-only helper, so a kernel that wins here can simply be selected.

Timing is CUDA-graph capture + replay, as in X9b: the draft chain is captured in
production, and per-call host dispatch would otherwise dominate at these shapes
(X9 measured 12-24 us of it, more than the entire bandwidth bound for three of
the four shapes).

Reading rule, fixed before the data: report every kernel at every shape against
that shape's own bandwidth bound. A kernel is only preferable if it wins at
gate_up_proj, which is 52% of the per-layer weight bytes -- winning on the small
projections while losing there is a net loss.
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
M_VALUES = [1, 4, 8, 32]
GROUP_SIZE = 128
HBM_BW = 3.35e12
GRAPH_CALLS = 32
REPLAYS = 20

# Share of per-layer weight bytes, so a per-shape win can be weighted correctly.
LAYER_WEIGHT_SHARE = {
    "qkv_proj": 12.98,
    "o_proj": 8.65,
    "gate_up_proj": 51.90,
    "down_proj": 25.95,
}


def shape_bytes(k: int, n: int, group_size: int) -> float:
    return k * n / 2 + (k / group_size) * n * 2


def time_graph(fn, calls: int, replays: int) -> float:
    import torch

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
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    graph.replay()
    torch.accelerator.synchronize()
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) * 1000.0 / (replays * calls)


def build_scheme_layer(cls, k: int, n: int):
    """Build the layer through the REAL CompressedTensorsWNA16 path.

    A hand-rolled nn.Module does not work: the kernels expect vLLM's parameter
    wrappers (input_dim/output_dim) and attributes like output_partition_sizes
    that only `create_weights` sets up. Forcing the kernel choice by patching
    `choose_mp_linear_kernel` keeps everything else on the production path, so
    each kernel gets exactly the weights and repack the engine would give it.
    """
    import torch

    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_wNa16 as wna16,
    )

    scheme = wna16.CompressedTensorsWNA16(
        strategy="group", num_bits=4, group_size=GROUP_SIZE, symmetric=True
    )
    layer = torch.nn.Module()
    original = wna16.choose_mp_linear_kernel
    wna16.choose_mp_linear_kernel = lambda cfg, *a, **kw: cls
    try:
        scheme.create_weights(
            layer=layer,
            output_size=n,
            input_size=k,
            output_partition_sizes=[n],
            input_size_per_partition=k,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *a, **kw: None,
        )
    finally:
        wna16.choose_mp_linear_kernel = original

    # A real LinearBase carries these; a bare Module does not, and Humming's
    # repack reads them.
    layer.input_size = k
    layer.output_size = n
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n
    layer.output_partition_sizes = [n]
    layer.params_dtype = torch.bfloat16
    layer.has_bias = False  # read by Humming's prepare_humming_layer

    # create_weights allocates empty parameters; fill with valid random data so
    # the repack and the kernel see well-formed input.
    for name, param in list(layer.named_parameters()):
        data = param.data
        if data.dtype in (torch.int32, torch.int64, torch.uint8):
            param.data = torch.randint(
                0, 2**8, data.shape, dtype=data.dtype, device="cuda"
            )
        else:
            param.data = torch.rand(data.shape, dtype=data.dtype, device="cuda") + 0.5
    scheme.process_weights_after_loading(layer)
    return scheme, layer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    # create_weights goes through vLLM's parameter machinery, which needs a
    # tensor-parallel group. TP=1 single process is the draft's actual config.
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:29591",
        local_rank=0,
        backend="nccl",
    )
    from vllm.config import VllmConfig, set_current_vllm_config

    vllm_config = VllmConfig()
    config_ctx = set_current_vllm_config(vllm_config)
    config_ctx.__enter__()
    initialize_model_parallel(1, 1)

    from vllm.model_executor.kernels.linear.mixed_precision.humming import (
        HummingLinearKernel,
    )
    from vllm.model_executor.kernels.linear.mixed_precision.machete import (
        MacheteLinearKernel,
    )
    from vllm.model_executor.kernels.linear.mixed_precision.marlin import (
        MarlinLinearKernel,
    )
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
        TritonW4A16LinearKernel,
    )
    from vllm.scalar_type import scalar_types

    KERNELS = {
        "machete": MacheteLinearKernel,
        "marlin": MarlinLinearKernel,
        "humming": HummingLinearKernel,
        "triton_w4a16": TritonW4A16LinearKernel,
    }

    rows: list[dict[str, Any]] = []
    for name, k, n in LAYER_SHAPES:
        for m in M_VALUES:
            entry: dict[str, Any] = {
                "layer": name,
                "m": m,
                "k": k,
                "n": n,
                "ideal_us": shape_bytes(k, n, GROUP_SIZE) / HBM_BW * 1e6,
                "us_per_call": {},
                "errors": {},
            }
            for label, cls in KERNELS.items():
                try:
                    cfg = MPLinearLayerConfig(
                        full_weight_shape=(k, n),
                        partition_weight_shape=(k, n),
                        weight_type=scalar_types.uint4b8,
                        act_type=torch.bfloat16,
                        group_size=GROUP_SIZE,
                        zero_points=False,
                        has_g_idx=False,
                    )
                    ok, why = cls.can_implement(cfg)
                    if not ok:
                        entry["errors"][label] = f"can_implement: {why}"
                        entry["us_per_call"][label] = None
                        continue
                    # Fresh layer per kernel: the repack mutates in place.
                    scheme, layer = build_scheme_layer(cls, k, n)
                    x = torch.rand((m, k), dtype=torch.bfloat16, device="cuda")

                    # Bind the loop variables; a bare closure would capture
                    # the last iteration's scheme/layer/x for every kernel.
                    def call(_s=scheme, _l=layer, _x=x):
                        return _s.apply_weights(_l, _x, None)

                    entry["us_per_call"][label] = time_graph(call, GRAPH_CALLS, REPLAYS)
                except Exception as exc:  # noqa: BLE001 - report, do not abort
                    entry["us_per_call"][label] = None
                    entry["errors"][label] = f"{type(exc).__name__}: {exc}"[:180]
            got = {k2: v for k2, v in entry["us_per_call"].items() if v}
            best = min(got, key=got.get) if got else None
            print(
                f"{name:14s} M={m:3d} ideal={entry['ideal_us']:6.2f}  "
                + "  ".join(
                    f"{lbl}={v:7.2f}({entry['ideal_us'] / v * 100:3.0f}%)"
                    if v
                    else f"{lbl}=--"
                    for lbl, v in entry["us_per_call"].items()
                )
                + (f"   BEST={best}" if best else ""),
                flush=True,
            )
            rows.append(entry)

    config_ctx.__exit__(None, None, None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_w4a16_all_kernels",
                "model": "Qwen3-8B",
                "group_size": GROUP_SIZE,
                "hbm_bw_bytes_per_s": HBM_BW,
                "layer_weight_share_mb": LAYER_WEIGHT_SHARE,
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
