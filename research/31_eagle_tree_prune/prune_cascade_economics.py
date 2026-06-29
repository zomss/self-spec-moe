#!/usr/bin/env python3
"""Cascade economics: EAGLE draft -> QUANTIZED comm-free phase-1 prune -> full phase-2 verify.

Two-phase verification (the user's proposal). Does a cheap (quantized, comm-free) phase-1
verifier that prunes a wide tree let the full phase-2 verify a small tree and still beat a
plain chain? Reuses the Phase 31 pruner-fidelity frontier (pruned accept vs node budget)
and the measured Phase-24 cost models. Generous to the cascade: EAGLE draft treated as
FREE (so a loss is robust). Sweeps phase-1 cheapness q (q*S_skip, q=1 bf16; q<1 quantized).
"""

from __future__ import annotations

import json
from pathlib import Path

D = json.loads((Path(__file__).resolve().parent / "data/qwen3_prune_frontier.json").read_text())
NWIDE = D["Nfull"]                                  # 30 nodes
PR = {int(k): v for k, v in D["budgets"].items()}   # N -> {conf,exact}
CHAIN = {2: D["direct"]["chain_k2"]["accept"], 3: D["direct"]["chain_k3"]["accept"],
         4: D["direct"]["chain_k4"]["accept"], 1: 0.88}

def sv(T): return 22.7 + 0.0495 * T                # full-EP verify over T tokens (comm-bound)
def sk(T): return 15.3 + 0.0152 * T                # comm-free forward over T tokens

def chain_speedup(B):
    return max((CHAIN[k] + 1) * sv(B) / sv(B * k) for k in CHAIN)

def cascade_speedup(B, q, pruner="conf"):
    # cost = q*S_skip(B*Nwide) [phase-1, comm-free quantized] + S_verify(B*Nsmall) [phase-2]
    best = 0.0; bn = None
    for N in PR:
        acc = PR[N][pruner]
        cost = q * sk(B * NWIDE) + sv(B * N)
        s = (acc + 1) * sv(B) / cost
        if s > best:
            best, bn = s, N
    return best, bn

QMAP = {"bf16 (q=1.0)": 1.0, "FP8-H100 (q~0.6)": 0.6, "FP4-Blackwell (q~0.4)": 0.4,
        "ideal (q=0.2)": 0.2, "free phase-1 (q=0)": 0.0}

print(f"wide tree = {NWIDE} nodes; cascade = q*S_skip(B*{NWIDE}) phase-1 + full verify(N_small)")
print("(EAGLE draft treated as FREE -> generous to the cascade)\n")
for B in [8, 128, 512]:
    ch = chain_speedup(B)
    print(f"=== B={B} (chain baseline = {ch:.3f}x) ===")
    for label, q in QMAP.items():
        s, n = cascade_speedup(B, q, "conf")
        so, no = cascade_speedup(B, q, "exact")
        verdict = "WIN" if s > ch * 1.10 else ("~chain" if s > ch * 0.97 else "LOSE")
        print(f"  phase-1 {label:>22}: cascade(conf) {s:.3f}x (N={n})  "
              f"| oracle {so:.3f}x  -> {verdict}")
    # threshold q where cascade(conf) == chain
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        s, _ = cascade_speedup(B, mid, "conf")
        if s > ch: lo = mid
        else: hi = mid
    phase1_ms = lo * sk(B * NWIDE)
    print(f"  -> cascade beats chain only if q < {lo:.2f} "
          f"(phase-1 < {phase1_ms:.1f} ms; full verify(1tok)={sv(B):.1f} ms)\n")
