"""Phase 41 Gate 1: rel-err of the genuine full-replica draft MoE output vs
plain MoE (D) at DP=8 (post-fix) -- should be ~0 (was 0.70x / rel 0.539).

Compares against D-dp1 (single-GPU plain-MoE ground truth). Also prints the
norm table and the A-dp1 vs D-dp1 sanity (must be ~0)."""
import os
import torch

DATA = "/data/smcho/ssm-num/research/41_replica_fix/data"


def load_moe(tag, layer=0):
    d = os.path.join(DATA, f"dump_{tag}")
    return torch.load(os.path.join(d, f"moe_dump_L{layer}.pt"),
                      map_location="cpu", weights_only=False)


def rel_err(a, b):
    num = (a - b).abs()
    den = b.abs().clamp_min(1e-6)
    rel = num / den
    return rel.max().item(), rel.mean().item()


def real_rows(*mats):
    n = min(m.shape[0] for m in mats)
    mask = torch.ones(n, dtype=torch.bool)
    for m in mats:
        mask &= (m[:n].abs().sum(-1) > 0)
    return n, mask


def compare(ref_tag, tag):
    rm = load_moe(ref_tag)["moe_output"]
    am = load_moe(tag)["moe_output"]
    n, real = real_rows(rm, am)
    r, a = rm[:n][real], am[:n][real]
    mx, mn = rel_err(a, r)
    return (int(real.sum()), a.norm().item(), r.norm().item(),
            (a - r).norm().item(), mx, mn)


def main():
    print("=== Gate 1: full-replica draft MoE rel-err vs plain MoE (D) ===\n")
    print(f"{'compare (A=replica, D=plain)':<34} {'norm':>8} {'ref_norm':>9} "
          f"{'||d||':>8} {'rel_max':>10} {'rel_mean':>10}")
    rows = [
        ("A_dp1 vs D_dp1 (sanity, ~0)", "D_dp1", "A_dp1"),
        ("A_dp8 vs D_dp1 (GATE 1)",     "D_dp1", "A_dp8"),
        ("A_dp8 vs D_dp8",              "D_dp8", "A_dp8"),
        ("D_dp8 vs D_dp1 (plain DP inv)","D_dp1", "D_dp8"),
    ]
    for label, ref, tag in rows:
        try:
            nt, an, rn, dn, mx, mn = compare(ref, tag)
            print(f"{label:<34} {an:>8.4f} {rn:>9.4f} {dn:>8.4f} "
                  f"{mx:>10.3e} {mn:>10.3e}   (n={nt})")
        except Exception as e:
            print(f"{label:<34}  ERROR: {e}")

    print("\n--- norm summary (ref plain-MoE D_dp1 norm) ---")
    dn1 = None
    for tag in ["D_dp1", "A_dp1", "D_dp8", "A_dp8"]:
        try:
            m = load_moe(tag)["moe_output"]
            n, real = real_rows(m)
            nv = m[:n][real].norm().item()
            if tag == "D_dp1":
                dn1 = nv
            ratio = f"{nv/dn1:.4f}" if dn1 else "-"
            print(f"  {tag:<8} norm={nv:8.4f}  ratio_to_D_dp1={ratio}")
        except Exception as e:
            print(f"  {tag:<8} ERROR: {e}")


if __name__ == "__main__":
    main()
