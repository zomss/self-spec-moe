# Phase 50 — World B with a REAL EAGLE3 head at single node

**Source phases** 33 (World B comm-aware tree-spec: measured with a
local-routing STAND-IN for EAGLE acceptance), 30 (EAGLE-vs-self-spec
comparison: ESTIMATED only), 47-49 (the World A measured baselines on this
testbed).

## Objective

Settle World B's honesty debt before multi-node: run a real trained EAGLE3
head (`Tengyunw/qwen3_30b_moe_eagle3`, base Qwen3-30B-A3B) on the identical
forced-PCIe testbed and measure (1) real EAGLE acceptance + throughput vs the
World A lockstep/overlap numbers at the same operating points, (2) the
comm-aware K/chain sensitivity with a real head.

Candidate matrix (DP4, b64, full-length): EAGLE3 chain K in {2,4} x a2a in
{0, 500, 1000}; compare against measured no-spec, World A lockstep K2/K4,
World A overlap b2 K2/K4.

## Commands

Harness gained env-gated `W7_SPEC_METHOD`/`W7_SPEC_MODEL` (defaults preserve
self-spec). Runner: `STACK=0` skips the self-spec env stack for clean EAGLE
engines.

```
TAG=eg3_k2_a2a0 A2A=0 KS=2 STACK=0 SHADOW=0 \
  EXTRA_ENV="W7_SPEC_METHOD=eagle3 W7_SPEC_MODEL=<head-path>" \
  bash research/49_overlap_impl/scripts/run_ov0b_one.sh
```

## Artifacts

`data/` result JSONs (written via W7_OUT override), `results_worldB_real.md`.
