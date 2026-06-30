"""Phase 40 Step 1 analysis: diff config A vs config C MoE dumps.

Confirms (a) router logits + (b) expert selection MATCH (same routing), quantifies
(c) the post-combine MoE-output relative error (max/mean), and (d) counts the
final-logit argmax flips (= the rejected draft tokens). With --fp32 it compares
the fp32-accum dumps to show the error drops by orders of magnitude.
"""
import argparse
import os

import torch

DATA = "/data/smcho/self-spec-moe/research/40_num_divergence/data"


def _rel_err(a, b):
    """max/mean relative error of a vs reference b (over nonzero ref)."""
    num = (a - b).abs()
    den = b.abs().clamp_min(1e-6)
    rel = num / den
    return rel.max().item(), rel.mean().item(), num.max().item()


def load(tag, layer):
    d = os.path.join(DATA, f"dump_{tag}")
    moe = torch.load(os.path.join(d, f"moe_dump_L{layer}.pt"),
                     map_location="cpu", weights_only=False)
    lg = torch.load(os.path.join(d, "logits_dump.pt"),
                    map_location="cpu", weights_only=False)
    return moe, lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--fp32", action="store_true",
                    help="compare the *_fp32 dumps instead of bf16")
    args = ap.parse_args()
    sfx = "_fp32" if args.fp32 else ""

    A_moe, A_lg = load("A" + sfx, args.layer)
    C_moe, C_lg = load("C" + sfx, args.layer)

    print(f"=== Config A vs C  (layer {args.layer}, "
          f"{'FP32-accum' if args.fp32 else 'bf16-accum'}) ===")
    print(f"A moe_output dtype (orig): {A_moe['moe_output_dtype']}  "
          f"n_tokens={A_moe['n_tokens']}")
    print(f"C moe_output dtype (orig): {C_moe['moe_output_dtype']}  "
          f"n_tokens={C_moe['n_tokens']}")

    # Restrict to REAL prompt tokens (drop the padded rows the prefill graph
    # adds): a row is real if its MoE output is non-zero in BOTH A and C.
    n = min(A_moe["moe_output"].shape[0], C_moe["moe_output"].shape[0])
    oa_full = A_moe["moe_output"][:n]
    oc_full = C_moe["moe_output"][:n]
    real = ((oa_full.abs().sum(-1) > 0) & (oc_full.abs().sum(-1) > 0))
    n_real = int(real.sum())
    print(f"\nreal (non-padding) tokens compared: {n_real} / {n}")

    # --- (a) router logits ---
    ra = A_moe["router_logits"][:n][real]
    rc = C_moe["router_logits"][:n][real]
    rl_max = (ra - rc).abs().max().item()
    print(f"\n(a) router logits  max|A-C| = {rl_max:.3e}  "
          f"(MATCH if ~0 -> same gate input/output)")

    # --- (b) expert selection ---
    ia, ic = A_moe["topk_ids"][:n][real], C_moe["topk_ids"][:n][real]
    wa, wc = A_moe["topk_weights"][:n][real], C_moe["topk_weights"][:n][real]
    # sort within each token (selection-set equality, order-independent)
    ia_s, _ = torch.sort(ia, dim=-1)
    ic_s, _ = torch.sort(ic, dim=-1)
    id_match = (ia_s == ic_s).all().item()
    id_frac = (ia_s == ic_s).float().mean().item()
    w_max = (wa - wc).abs().max().item()
    print(f"(b) expert ids   set-equal = {id_match}  "
          f"(elementwise-sorted match frac = {id_frac:.4f})")
    print(f"    topk weights max|A-C| = {w_max:.3e}")

    # --- (c) post-combine MoE output ---
    oa, oc = oa_full[real], oc_full[real]
    mx, mn, abs_mx = _rel_err(oa, oc)
    print(f"\n(c) post-combine MoE output  rel-err  max={mx:.3e}  mean={mn:.3e}  "
          f"(abs max={abs_mx:.3e})")
    print(f"    A.norm={oa.norm():.4f}  C.norm={oc.norm():.4f}  "
          f"||A-C||={(oa - oc).norm():.4f}")

    # --- (d) final logit argmax flips ---
    la, lc = A_lg["logits"], C_lg["logits"]
    m = min(la.shape[0], lc.shape[0])
    la, lc = la[:m], lc[:m]
    # Real positions: logit row not all-equal (padded rows are constant/garbage,
    # identical in A and C -> never flip; exclude them from the rate denominator).
    real_l = (la.std(dim=-1) > 1e-3) & (lc.std(dim=-1) > 1e-3)
    m_real = int(real_l.sum())
    am = la[real_l].argmax(dim=-1)
    cm = lc[real_l].argmax(dim=-1)
    flips = (am != cm).sum().item()
    lmx = (la[real_l] - lc[real_l]).abs().max().item()
    print(f"\n(d) final logits (real positions={m_real})  max|A-C| = {lmx:.3e}  "
          f"argmax FLIPS = {flips}/{m_real}  "
          f"(flip rate = {flips / max(m_real, 1):.4f} = per-token rejections)")


if __name__ == "__main__":
    main()
