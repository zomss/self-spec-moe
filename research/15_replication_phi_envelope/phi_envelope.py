#!/usr/bin/env python3
"""G>=4 replication (EPLB / expert-cache) speedup envelope.

Hierarchical draft at G groups, with each node replicated up to a local expert
fraction phi (native E/G + replicated cache). Acceptance(phi) comes from the
Phase 09 mass-optimal cache sweep (M = phi*E). Comm is latency-bound at decode:
the verify inter-node collective fires regardless of phi (until phi->1), so the
draft removes the full inter-node latency while acceptance rises with phi.

  T_draft  = compute + intra-node A2A         (2*L collectives, NVLink)
  T_verify = compute + inter-node A2A         (2*L collectives, IB)
  speedup  = E_tokens(acc(phi), n) * T_verify / (n*T_draft + T_verify)

Memory: replication factor over native EP at G groups is R = phi * G; per-GPU
local experts = phi*E / gpus_per_node.
"""

from __future__ import annotations

import json
from pathlib import Path

NVLINK_US = 30.0          # measured (phase 12), per collective, decode payloads
GPUS_PER_NODE = 8
NUM_SPEC = 3
IB_US = [50.0, 100.0, 200.0]
G_VALUES = [4, 8]

P9 = Path(__file__).resolve().parents[1] / "09_rebalancing_ceiling" / "data"
MODELS = {
    "Qwen3-30B-A3B": {"json": "qwen3_30b_a3b_ceiling.json", "layers": 48, "compute_ms": 12.0},
    "GPT-OSS-20B": {"json": "gpt_oss_20b_ceiling.json", "layers": 24, "compute_ms": 6.0},
}


def load_acc(path):
    rows = json.load(open(path))["rows"]
    return sorted((r["budget_M"], r["sampled_acceptance"]) for r in rows)


def interp(curve, m):
    if m <= curve[0][0]:
        return curve[0][1]
    if m >= curve[-1][0]:
        return curve[-1][1]
    for (m0, a0), (m1, a1) in zip(curve, curve[1:]):
        if m0 <= m <= m1:
            return a0 + (a1 - a0) * (m - m0) / (m1 - m0)
    return curve[-1][1]


def e_tokens(acc, n):
    return sum(acc**i for i in range(n + 1))


def speedup(acc, compute_ms, layers, ib_us, n):
    coll = 2 * layers
    t_draft = compute_ms + coll * NVLINK_US / 1000.0
    t_verify = compute_ms + coll * ib_us / 1000.0
    return e_tokens(acc, n) * t_verify / (n * t_draft + t_verify)


def main():
    out = {}
    for name, m in MODELS.items():
        curve = load_acc(P9 / m["json"])
        E = curve[-1][0]  # full budget = num experts
        print(f"\n===== {name}  (E={E}, layers={m['layers']}, compute~{m['compute_ms']}ms, n={NUM_SPEC}) =====")
        out[name] = {"E": E, "rows": []}
        # phi grid from the available M points
        phis = [(M, M / E) for M, _ in curve if M / E <= 0.85]
        print(f"{'phi':>5} {'M':>4} {'acc':>6} {'R@G4':>6} {'exp/GPU@G4':>11} | "
              + "  ".join(f"sp(IB={int(ib)})" for ib in IB_US))
        for M, phi in phis:
            acc = interp(curve, M)
            r_g4 = phi * 4
            per_gpu = M / GPUS_PER_NODE
            sps = [speedup(acc, m["compute_ms"], m["layers"], ib, NUM_SPEC) for ib in IB_US]
            print(f"{phi:>5.2f} {M:>4} {acc:>6.3f} {r_g4:>6.1f} {per_gpu:>11.1f} | "
                  + "  ".join(f"{s:>9.2f}" for s in sps))
            out[name]["rows"].append(
                {"phi": round(phi, 3), "M": M, "acc": round(acc, 3),
                 "R_g4": round(r_g4, 2), "exp_per_gpu": round(per_gpu, 1),
                 "speedup": {str(int(ib)): round(s, 3) for ib, s in zip(IB_US, sps)}}
            )
        # break-even phi per IB
        print("  break-even phi (speedup>=1.0):")
        for ib in IB_US:
            be = None
            for M, phi in phis:
                if speedup(interp(curve, M), m["compute_ms"], m["layers"], ib, NUM_SPEC) >= 1.0:
                    be = phi
                    break
            tag = f"phi>={be:.2f} (R@G4={be*4:.1f}, {be*E/GPUS_PER_NODE:.0f} exp/GPU)" if be else "unreachable (<=0.85)"
            print(f"    IB={int(ib):>3}us/coll: {tag}")
            out[name].setdefault("break_even", {})[str(int(ib))] = be

    Path(__file__).resolve().parent.joinpath("data/phi_envelope.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
