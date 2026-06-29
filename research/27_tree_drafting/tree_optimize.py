#!/usr/bin/env python3
"""Find the best tree structure: max accept length per verify cost (comm-bound PCIe).

Inputs (all measured on this node):
  - h(b) from tree_hitrate.py: P(verify-next in draft top-b) along the accepted path.
  - verify cost S_v(T) and comm-free draft cost S_d(T) vs total tokens T=B*nodes,
    fit from the Phase 24 forced-PCIe batch sweeps.
Evaluates tree families and reports the structure maximizing lossless speedup
  speedup = (accept_len+1) * S_v(B*1) / (draft_cost + S_v(B*node_count))
vs the chain (linear k) baseline. Baseline token cost = full-EP verify of 1 token.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# --- measured cost models (Phase 24, forced-PCIe, per global batch B) ---
# verify (full-EP PCIe-SHM): (B=512,1024,2048)->(48,71,124) ms ; T=B*nodes
SV_INT, SV_SLOPE = 22.7, 0.0495      # S_v(T) = 22.7 + 0.0495*T  (ms)
# comm-free draft (skip-A2A): (512,1024,2048)->(23.1,26.9,46.5) ms
SD_INT, SD_SLOPE = 15.3, 0.0152      # S_d(T) = 15.3 + 0.0152*T  (ms)


def s_verify(T): return SV_INT + SV_SLOPE * T
def s_draft(T): return SD_INT + SD_SLOPE * T


def geom(rate, d):
    return rate * (1 - rate ** d) / (1 - rate) if rate < 1 else float(d)


def eval_tree(name, depth, node_count, accept_len, draft_widths, B):
    """draft_widths: frontier width expanded at each draft step (sequential)."""
    draft_cost = sum(s_draft(B * w) for w in draft_widths)
    verify_cost = s_verify(B * node_count)
    cycle = draft_cost + verify_cost
    speedup = (accept_len + 1) * s_verify(B * 1) / cycle
    return {"name": name, "depth": depth, "nodes": node_count,
            "accept_len": round(accept_len, 3), "draft_ms": round(draft_cost, 1),
            "verify_ms": round(verify_cost, 1), "speedup": round(speedup, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hitrate", type=Path,
                    default=Path(__file__).resolve().parent / "data/qwen3_tree_hitrate.json")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--output-json", type=Path,
                    default=Path(__file__).resolve().parent / "data/tree_optimum.json")
    a = ap.parse_args()
    B = a.batch
    H = json.loads(a.hitrate.read_text())["h_pooled"]   # H[b-1]=h(b)
    def h(b): return H[min(b, len(H)) - 1]
    h1 = h(1)
    print(f"[h(b)] " + " ".join(f"b{b}={h(b):.3f}" for b in range(1, len(H) + 1)))
    print(f"[cost@B={B}] baseline verify(1 tok)={s_verify(B):.1f}ms\n")

    cands = []
    # 1) chain depth d (b=1): nodes=d, rate h1
    for d in range(1, 13):
        cands.append(eval_tree(f"chain d={d}", d, d, geom(h1, d), [1] * d, B))
    # 2) full (d,b): every node expands top-b; rate h(b); nodes=sum b^j
    for b in (2, 3, 4):
        for d in range(1, 7):
            nodes = sum(b ** j for j in range(1, d + 1))
            widths = [b ** (j - 1) for j in range(1, d + 1)]  # frontier per draft step
            cands.append(eval_tree(f"full d={d},b={b}", d, nodes, geom(h(b), d), widths, B))
    # 3) spine + catch-w leaves: spine top-1 chain depth d, (w-1) catch leaves/level.
    #    continue prob h1 (spine), token accepted with prob h(w); nodes=d*w
    for w in (2, 3, 4, 6, 8):
        for d in range(1, 13):
            accept = h(w) * (1 - h1 ** d) / (1 - h1)
            cands.append(eval_tree(f"spine d={d},w={w}", d, d * w, accept, [1] * d, B))

    cands.sort(key=lambda c: -c["speedup"])
    best_chain = max((c for c in cands if c["name"].startswith("chain")),
                     key=lambda c: c["speedup"])
    print(f"{'structure':>16} {'depth':>5} {'nodes':>5} {'acc_len':>7} "
          f"{'draft':>6} {'verify':>7} {'speedup':>7}")
    for c in cands[:12]:
        print(f"{c['name']:>16} {c['depth']:>5} {c['nodes']:>5} {c['accept_len']:>7} "
              f"{c['draft_ms']:>6} {c['verify_ms']:>7} {c['speedup']:>7}")
    print(f"\nbest overall: {cands[0]['name']} -> {cands[0]['speedup']}x "
          f"(accept {cands[0]['accept_len']})")
    print(f"best chain  : {best_chain['name']} -> {best_chain['speedup']}x "
          f"(accept {best_chain['accept_len']})")
    print(f"tree gain over chain: {cands[0]['speedup']/best_chain['speedup']:.2f}x")

    a.output_json.write_text(json.dumps(
        {"B": B, "h_pooled": H, "ranked": cands[:20],
         "best": cands[0], "best_chain": best_chain}, indent=2))
    print(f"[saved] {a.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
