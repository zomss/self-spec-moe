"""Phase 40: compare all four MoE dumps (A/C x bf16/fp32) at the chosen layer.

Shows the full convergence picture:
  - A_bf16 vs C_bf16: the divergence (Step 1)
  - A_fp32 vs C_fp32: does fp32-accum in BOTH paths make them converge? (Step 2)
  - per-config: how much fp32 moved each path.
Also reports (d) the final-logit argmax flips for bf16 and fp32.
"""
import os

import torch

DATA = "/data/smcho/self-spec-moe/research/40_num_divergence/data"
L = 0


def load_moe(tag):
    return torch.load(os.path.join(DATA, f"dump_{tag}", f"moe_dump_L{L}.pt"),
                      map_location="cpu", weights_only=False)["moe_output"].float()


def load_logits(tag):
    return torch.load(os.path.join(DATA, f"dump_{tag}", "logits_dump.pt"),
                      map_location="cpu", weights_only=False)["logits"]


def rel(x, ref):
    return ((x - ref).norm() / ref.norm()).item()


def flips(a, c):
    m = min(a.shape[0], c.shape[0])
    am = a[:m].argmax(-1)
    cm = c[:m].argmax(-1)
    return int((am != cm).sum()), m


Ab, Af = load_moe("A"), load_moe("A_fp32")
Cb, Cf = load_moe("C"), load_moe("C_fp32")

print(f"=== MoE output (layer {L}) relative diffs ===")
print(f"  A_bf16 vs C_bf16  : {rel(Ab, Cb):.5f}   <- the divergence (Step 1)")
print(f"  A_fp32 vs C_fp32  : {rel(Af, Cf):.5f}   <- after fp32-accum (Step 2)")
print(f"  A_bf16 vs A_fp32  : {rel(Af, Ab):.5f}   (how much fp32 moved A's local sum)")
print(f"  C_bf16 vs C_fp32  : {rel(Cf, Cb):.5f}   (how much fp32 moved C's reduce)")
print(f"  norms: A_bf16={Ab.norm():.3f} A_fp32={Af.norm():.3f} "
      f"C_bf16={Cb.norm():.3f} C_fp32={Cf.norm():.3f}")

print("\n=== Final-logit argmax flips (d) ===")
lAb, lCb = load_logits("A"), load_logits("C")
lAf, lCf = load_logits("A_fp32"), load_logits("C_fp32")
fb, mb = flips(lAb, lCb)
ff, mf = flips(lAf, lCf)
print(f"  A_bf16 vs C_bf16 : {fb}/{mb} flips  (max|d|={ (lAb-lCb).abs().max():.3e})")
print(f"  A_fp32 vs C_fp32 : {ff}/{mf} flips  (max|d|={ (lAf-lCf).abs().max():.3e})")
