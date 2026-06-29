# Results: tree drafting -- best structure, and a correction to 3f

Date: 2026-06-29. Qwen3-30B-A3B, comm-free local-routing draft (C=0.5E), forced-PCIe
cost models (Phase 24). Accept length from measured h(b) via the standard tree model.

## 1. Building block: branching is very effective (measured)

`h(b) = P(verify's next token in draft's top-b)` along the accepted path (8 prompts,
depth 12, `tree_hitrate.py`):

| b | 1 | 2 | 3 | >=4 |
| --- | ---: | ---: | ---: | ---: |
| h(b) | 0.875 | **0.979** | **1.000** | 1.000 |

The draft's **top-3 essentially always contains verify's greedy token**. So branching
lifts per-level acceptance 0.875 -> ~1.0 -- a tiny amount of width buys near-perfect
per-level acceptance. (Per-depth h_j(b) is flat; no depth decay.)

## 2. The cost that bounds tree size (and corrects 3f)

The verify is ONE forward over all tree nodes -> the MoE all-to-all routes `B*node_count`
tokens -> **verify cost scales with tree size** (and, for a chain, with k). Measured:
`S_v(T) = 22.7 + 0.0495*T` ms (T = B*nodes; from the Phase 24 batch sweep). The
comm-free draft is NOT free either: each draft token is a full forward, `S_d(T) =
15.3 + 0.0152*T` ms, done sequentially over depth.

**Correction to 3f.** The 3f composition used the 1-token verify cost (S_v(B)) for a
k-token verify -- it should be S_v(B*k). With the correct scaling, even *perfect*
acceptance caps the high-batch speedup at ~1.35x (k=1): the comm-free draft (~23 ms at
B=512) nearly equals the comm it saves (~25 ms), so each extra draft token barely beats a
decode. So 3f's 1.39x was optimistic; the real high-batch optimum is **k=1, ~1.27x**.

## 3. Optimal tree structure depends on batch

| batch | best structure | nodes | accept len | speedup | vs best chain |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 (latency) | **full d=2, b=2** | 6 | 1.94 | **1.21x** | +8% |
| 128 | chain d=1 | 1 | 0.88 | 1.18x | +0% |
| 512 (serving) | chain d=1 | 1 | 0.88 | 1.27x | +0% |
| 1 (extrapolated) | full d=3, b=3 | 39 | 3.0 | 1.29x | +15% |

**Trees help only at low batch.** When B is small the verify is latency-bound (fixed
~23 ms; node count is nearly free), so branching (h(b)->1) buys accept length cheaply ->
small balanced trees win. As B grows the verify all-to-all becomes bandwidth-bound
(scales with B*nodes), so every extra node costs -> the optimum collapses to a 1-token
chain and trees give no gain.

**The best tree is small and shaped, never big** (your point, confirmed): the winner is
`full d=2-3, b=2-3` (6-14 nodes), not a large tree -- bigger trees always lose to verify
cost. Branching beats depth (h(2)=0.98 per level vs chaining at 0.875), but only a couple
of levels are worth it.

## 4. Bottom line

- Single-node-PCIe lossless speedup is **~1.2-1.3x** across batch (corrected down from
  3f's 1.39x; the comm-free draft only saves ~half the step, so the ceiling is ~1.35x
  even at perfect acceptance).
- **Tree drafting is a LOW-BATCH (latency) lever**: at B=8 a small `full d=2,b=2` tree
  gives +8% over the best chain; gain grows as batch shrinks. At serving batch it does
  not help -- the comm-bound verify penalizes node count.
- Optimal structure: **small balanced tree (d=2-3, b=2-3) at low batch; 1-token chain at
  high batch.**

## 5. Caveat / next

Accept length uses measured h(b) + the standard geometric tree model; the verify/draft
costs are measured. A real tree-attention end-to-end run (accept length for the winning
small tree, the B1-analog for trees) would validate the structure before any integration.
