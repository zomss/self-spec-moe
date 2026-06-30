"""Micro-diagnostic: does compiling the SAME MoE-gate graph twice (two separate
torch.compile instances, mimicking draft vs verify) produce numerically
different outputs for the SAME input?

Models the qwen2_moe MoE block's *inductor-visible* surround (RMSNorm + residual
+ gate linear + a stand-in for the opaque expert op): the gate softmax/topk is
the inductor-fused part that selects experts. If two compilations select even one
different top-k expert for some token, the downstream output diverges hard.

Run: PYTHONPATH=/data/smcho/ssm-cc <venv>/python micro_moe_compile.py
"""
import torch

torch.manual_seed(0)
DEV = "cuda"
DT = torch.bfloat16
H = 2048        # hidden
E = 60          # experts
TOPK = 4

# Build a small "decoder layer body" the way inductor sees it around the opaque
# MoE op: rmsnorm -> gate GEMM -> softmax topk. We compare the *router_logits*
# and the *topk expert ids* across two independent compilations.
gate_w = torch.randn(E, H, device=DEV, dtype=DT) * 0.02
norm_w = torch.ones(H, device=DEV, dtype=DT)


def body(x):
    # fused-add-rmsnorm-ish (inductor-fusable)
    var = (x.float() ** 2).mean(-1, keepdim=True)
    xn = (x.float() * torch.rsqrt(var + 1e-6)).to(DT) * norm_w
    logits = torch.nn.functional.linear(xn, gate_w)  # (N, E) gate GEMM
    probs = torch.softmax(logits.float(), dim=-1)
    tw, tid = torch.topk(probs, TOPK, dim=-1)
    return logits, tw, tid


def run(compiled, N):
    x = torch.randn(N, H, device=DEV, dtype=DT)
    # use a FIXED input across both compilations
    return x, compiled(x)


# Two independent compilations of the identical function (draft vs verify).
c1 = torch.compile(body, dynamic=False)
c2 = torch.compile(body, dynamic=False)

# Same fixed input for both.
for N in (1, 16, 80):
    x = torch.randn(N, H, device=DEV, dtype=DT)
    l1, w1, i1 = c1(x)
    l2, w2, i2 = c2(x)
    le, we, ie = body(x)  # eager reference
    logit_max_diff = (l1 - l2).abs().max().item()
    id_mismatch_12 = (i1 != i2).any(dim=-1).float().mean().item()
    id_mismatch_1e = (i1 != ie).any(dim=-1).float().mean().item()
    print(
        f"N={N:>3} | logit |c1-c2|max={logit_max_diff:.3e} | "
        f"topk-id mismatch c1-vs-c2={id_mismatch_12:.3f} | "
        f"c1-vs-eager={id_mismatch_1e:.3f}"
    )
