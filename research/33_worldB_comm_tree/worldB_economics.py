#!/usr/bin/env python3
"""World B: comm-aware tree-spec for MoE-EP. EAGLE's default WIDE tree blows up the verify
all-to-all (routes ~tree-size x tokens); a comm-aware chain/small-tree cuts it. Measured
forced-PCIe verify step (data/qwen3_worldB_verify.json) x accept length (Phase 31)."""
import json
from pathlib import Path
V = {r["batch_global"]: r["step_ms"]
     for r in json.load(open(Path(__file__).resolve().parents[1]
       / "24_commbound_throughput/data/qwen3_worldB_verify.json"))["rows"]}
PR = json.load(open(Path(__file__).resolve().parents[1]
       / "31_eagle_tree_prune/data/qwen3_prune_frontier.json"))
BSEQ = 64
base = V[BSEQ]                       # no-spec: verify 1 token at the serving load
def draft(nodes): return 0.1 * nodes # EAGLE head, ~cheap, ~per-node
designs = [
    ("comm-aware chain (N=4)",   4,  3.077, 4),
    ("comm-aware pruned (N=6)",  6,  PR["budgets"]["6"]["conf"], 30),
    ("EAGLE-default wide (N=30)",30, PR["L_full"], 30),
]
print(f"serving load B_seq={BSEQ}; baseline verify(1 tok)={base:.1f} ms\n")
print(f"{'design':<26}{'verify ms':>10}{'accept':>8}{'speedup':>9}")
res = {}
for name, N, acc, nd in designs:
    v = V[BSEQ * N]; cyc = draft(nd) + v
    sp = (acc + 1) * base / cyc
    res[name] = {"N": N, "verify_ms": round(v, 1), "accept": round(acc, 2), "speedup": round(sp, 2)}
    print(f"{name:<26}{v:>10.1f}{acc:>8.2f}{sp:>9.2f}x")
wide = res["EAGLE-default wide (N=30)"]; ch = res["comm-aware chain (N=4)"]; pr = res["comm-aware pruned (N=6)"]
print(f"\nverify all-to-all: wide {wide['verify_ms']}ms vs pruned {pr['verify_ms']}ms = "
      f"{wide['verify_ms']/pr['verify_ms']:.1f}x step; "
      f"bandwidth (minus {base:.0f}ms floor) = {(wide['verify_ms']-base)/(pr['verify_ms']-base):.1f}x")
print(f"end-to-end: comm-aware chain {ch['speedup']}x / pruned {pr['speedup']}x  vs  "
      f"EAGLE-default wide {wide['speedup']}x  -> {ch['speedup']/wide['speedup']:.1f}x better")
json.dump(res, open("data/worldB_economics.json", "w"), indent=2)
