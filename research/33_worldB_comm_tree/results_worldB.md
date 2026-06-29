# Results: World B -- comm-aware tree-spec beats EAGLE's default wide tree ~2.3x on MoE-EP

Date: 2026-06-29. Forced-PCIe MoE-EP engine (Qwen3-30B, attention-DP8+EP, NVLS-off recipe).
Verify step MEASURED across token counts = serving load B_seq=64 x tree size N. Accept
length from Phase 31 (local-routing tree as the EAGLE stand-in; verify comm is
draft-independent so the wide-vs-small comparison is robust). Draft modeled EAGLE-cheap.

## The measurement

Forced-PCIe verify step vs verified tokens (B_seq=64):

| verified tokens (B_seq x N) | verify step |
| ---: | ---: |
| 64 (N=1, baseline) | 29.6 ms |
| 256 (N=4) | 40.9 ms |
| 384 (N=6) | 47.0 ms |
| 512 (N=8) | 49.9 ms |
| 1024 (N=16) | 70.2 ms |
| 1920 (N=30) | 110.8 ms |

## End-to-end (verify measured x accept x EAGLE-cheap draft)

| design | tree N | verify | accept | speedup |
| --- | ---: | ---: | ---: | ---: |
| comm-aware chain | 4 | 40.9 | 3.08 | **2.93x** |
| comm-aware pruned | 6 | 47.0 | 3.40 | 2.61x |
| **EAGLE-default wide** | **30** | **110.8** | 3.83 | **1.26x** |

## Headline

- **EAGLE's default wide tree gets only 1.26x on comm-bound MoE-EP** -- its verify
  all-to-all routes ~30x the tokens (to accept ~4), so the comm dominates and the extra
  accept (+0.4 over a chain) doesn't pay.
- **A comm-aware chain/small tree gets 2.6-2.9x -> ~2.3x better.**
- The verify all-to-all of the wide tree is **2.4x the step / 4.7x the bandwidth** of the
  small tree. That is the comm being cut.
- EAGLE trees in practice are 25-64 nodes, so the effect is **>=** what is shown; at
  inter-node IB (higher comm fraction) the gap widens further.

## Why (the mechanism, from Phase 27/31)

The verify is one all-to-all over all tree nodes -> comm scales with tree width. A wide
tree routes many candidate tokens to accept only the best path (~4), so most of its comm
is wasted. A chain (minimal breadth) is the comm-floor; pruning a wide tree by confidence
(Phase 31, near-oracle) brings it down toward the floor (89% of the accept at ~5x fewer
nodes). So on MoE-EP you should draft small/pruned trees, not EAGLE's default wide ones.

## Positioning (World A + World B)

- **World A** -- training-free comm-free local-routing + FP4 self-spec: ~1.2-1.35x
  single-node PCIe, no draft head, lossless. For when you cannot/will not train a head.
- **World B (this)** -- with EAGLE3/MTP, comm-aware (chain/pruned) trees on MoE-EP:
  ~2.3x over the naive wide-tree baseline, by cutting the verify all-to-all.

Both rest on the same analysis -- the verify communication scales with the number of
verified tokens (Phase 27) -- which is the project's draft-agnostic core result. World B is
the concrete systems win it implies for conventional tree-spec on MoE serving.

## Caveats

- Accept lengths are the World-A local-routing stand-in for EAGLE; a real EAGLE head raises
  all accepts, but the wide-vs-small ratio holds (verify comm is draft-independent). A
  measured EAGLE head would sharpen the absolute numbers.
- Verify MEASURED on the forced-PCIe engine; draft modeled (EAGLE-cheap). The win is
  verify-dominated, so the draft model barely moves it.
- B_seq=64 serving load; the wide-tree penalty grows with batch and with inter-node f.
