"""Run the REAL vLLM unquantized TRITON FusedMoE on the same tokens at decode
shape (N small) vs verify shape (N large) and check if per-token greedy argmax of
the MoE output diverges. This isolates whether the MoE kernel itself is
batch-shape sensitive enough to flip tokens (the suspected root cause: draft runs
small N, verify large N, same compiled dynamic graph -> different numerics).

Uses TRITON backend (what the engine logs select) directly, eager (no compile),
to test the KERNEL's shape sensitivity. Then repeats under torch.compile.
"""
import os
import torch

os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")

import vllm  # noqa
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk

torch.manual_seed(0)
DEV, DT = "cuda", torch.bfloat16
H = 2048
INTER = 1408           # Qwen1.5-MoE expert intermediate
E = 60
TOPK = 4

w1 = (torch.randn(E, 2 * INTER, H, device=DEV, dtype=DT) * (H ** -0.5))
w2 = (torch.randn(E, H, INTER, device=DEV, dtype=DT) * (INTER ** -0.5))
gate_w = torch.randn(E, H, device=DEV, dtype=DT) * 0.02

POOL = torch.randn(80, H, device=DEV, dtype=DT)


def moe(x):
    logits = torch.nn.functional.linear(x, gate_w)
    tw, tid, _ = fused_topk(x, logits, TOPK, renormalize=True)
    out = fused_experts(x, w1, w2, tw, tid)
    return out


def greedy_proxy(out):
    # proxy "logits": project to vocab-like space deterministically
    return out  # compare the raw MoE output argmax over hidden dim


for label, fn in [("eager", moe), ("compiled", torch.compile(moe, dynamic=False))]:
    tok16 = POOL[:16].contiguous()
    o16 = fn(tok16.clone())
    o80 = fn(POOL.clone())[:16]
    # per-token argmax over hidden dim as a token-flip proxy
    a16 = o16.float().argmax(-1)
    a80 = o80.float().argmax(-1)
    flip = (a16 != a80).float().mean().item()
    maxdiff = (o16 - o80).abs().max().item()
    reldiff = ((o16 - o80).abs() / (o80.abs() + 1e-3)).mean().item()
    print(f"[{label:>8}] N16-vs-N80(same tokens): argmax-flip={flip:.3f} "
          f"max|d|={maxdiff:.3e} mean-rel={reldiff:.3e}")
