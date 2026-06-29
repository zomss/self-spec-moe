# Phase 33: World B -- comm-aware tree-spec for MoE-EP

**The two worlds (paper structure):**
- **World A** (Phases 18-30): training-free, comm-free local-routing + FP4 self-spec draft
  -- no draft head needed; lossless ~1.2-1.35x single-node PCIe. The training-free option.
- **World B** (this phase): *with* a conventional draft head (EAGLE3/MTP), make tree-spec
  comm-aware on MoE-EP. EAGLE's default WIDE tree (~25-64 nodes) blows up the verify
  all-to-all (it routes ~tree-size x tokens to accept ~4); a comm-aware chain/small tree
  cuts that comm and is ~2x faster end-to-end.

**Objective:** measure, on the forced-PCIe MoE-EP engine, the verify cost vs draft-tree
size, and the end-to-end speedup of comm-aware (chain / pruned) vs EAGLE-default (wide)
trees.

**Method:** forced-PCIe verify step (`24_commbound_throughput/bench_commbound.py`, NVLS-off
recipe) measured across token counts = serving load B_seq=64 x tree size N (global batch
B_seq*N). Accept length from Phase 31 (`tree_verify` / prune frontier) -- the local-routing
tree as the EAGLE stand-in (conservative on the gap EAGLE leaves; the verify comm is
draft-independent so the wide-vs-small comparison is robust). Draft modeled as EAGLE-cheap
(~0.1 ms/node). `worldB_economics.py`.

**Result (B_seq=64):** see `results_worldB.md`. EAGLE-default wide (N=30) 1.26x; comm-aware
pruned (N=6) 2.61x; comm-aware chain (N=4) 2.93x -> 2.3x better; verify all-to-all 2.4x
step / 4.7x bandwidth smaller for the small tree.

**Caveat / next:** accept lengths are the World-A local-routing stand-in for EAGLE; a real
EAGLE head would raise all accepts but the wide-vs-small *ratio* holds (verify comm is
draft-independent). Inter-node IB would widen the gap further (higher comm fraction).
