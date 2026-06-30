"""Pin the divergence: is the compiled gate routing batch-shape dependent, and
does it diverge from eager? Compare topk-id selection (compiled @ shape N) vs
(eager) and vs (compiled @ a different shape) for the SAME per-token inputs.

This emulates draft (small N) vs verify (larger N) seeing the SAME token's hidden
state but routed by gate GEMMs compiled for different shapes.
"""
import torch

torch.manual_seed(0)
DEV, DT = "cuda", torch.bfloat16
H, E, TOPK = 2048, 60, 4
gate_w = torch.randn(E, H, device=DEV, dtype=DT) * 0.02
norm_w = torch.ones(H, device=DEV, dtype=DT)


def gate_route(x):
    var = (x.float() ** 2).mean(-1, keepdim=True)
    xn = (x.float() * torch.rsqrt(var + 1e-6)).to(DT) * norm_w
    logits = torch.nn.functional.linear(xn, gate_w)
    probs = torch.softmax(logits.float(), dim=-1)
    tw, tid = torch.topk(probs, TOPK, dim=-1)
    return logits, tid


cg = torch.compile(gate_route, dynamic=False)

# A pool of token hidden states. We route the SAME first 16 tokens both as a
# small batch (draft) and embedded in a larger batch (verify) under the compiled
# kernel, and compare expert selections.
POOL = torch.randn(80, H, device=DEV, dtype=DT)
tok16 = POOL[:16].contiguous()

# eager reference for tok16
_, id_eager = gate_route(tok16)

# compiled @ N=16 (draft shape)
_, id_c16 = cg(tok16.clone())

# compiled @ N=80 (verify shape), then slice the first 16 (same tokens)
_, id_c80full = cg(POOL.clone())
id_c80 = id_c80full[:16]

# As a set (order-independent) per token, compare selected experts.
def setmismatch(a, b):
    a_s = [set(r.tolist()) for r in a]
    b_s = [set(r.tolist()) for r in b]
    return sum(x != y for x, y in zip(a_s, b_s)) / len(a_s)

print(f"draft(c16) vs eager   set-mismatch: {setmismatch(id_c16, id_eager):.3f}")
print(f"verify(c80) vs eager  set-mismatch: {setmismatch(id_c80, id_eager):.3f}")
print(f"draft(c16) vs verify(c80) set-mismatch: {setmismatch(id_c16, id_c80):.3f}")
print(f"  -> if draft-vs-verify > 0, the SAME token routes to DIFFERENT experts "
      f"in draft vs verify under compile (shape-driven)")
