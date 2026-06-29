# Phase 27: Tree drafting (increase accept length at minimal verify cost)

**Source:** Phase 26 frontier (beat accept-length-per-GB). Tree drafting raises accept
length without more memory -- the extra accept comes from cheap comm-free draft compute,
not GB. But tree size trades against verify cost (the comm-bound all-to-all routes
B*nodes tokens), so we must find the best tree SHAPE, not the biggest tree.

**Objective:** (1) measure the building block h(b)=P(verify-next in draft top-b) along
the accepted path; (2) with the measured verify/draft cost models, find the tree
structure maximizing lossless speedup, as a function of batch.

**Assumptions:** comm-free local-routing draft (C=0.5E). Accept length of a tree from
h(b) via the standard tree model; verify cost = S_v(B*node_count), draft = sum of
S_d(B*frontier) over depth (both from Phase 24 forced-PCIe sweeps).

**Commands:**
- `python tree_hitrate.py --local-files-only --depth 12 --output-json data/qwen3_tree_hitrate.json`
- `python tree_optimize.py --batch {8,128,512}`

**Decision criteria:** best structure + speedup vs chain baseline, per batch. Beat the
Phase 26 chain optimum.

**Next artifact:** results_tree.md; then a real tree-attention end-to-end validation of
the winning small tree (the B1-analog for trees).
