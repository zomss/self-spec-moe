# Phase 57: large_ep_spec_strategy — the optimal spec-decode strategy at large EP

**Repositioning.** The project's durable, defensible contribution is NOT "a
new self-spec method wins" (on this good fabric it does not — Phases 52-56).
It is a **design rule for practitioners**: *how should you configure
speculative decoding when serving MoE with large expert parallelism
(multi-node EP)?* The answer, grounded in the measured verify-comm-scaling
law, inverts single-GPU tree-spec practice.

**Thesis.** On MoE-EP the verify is one all-to-all over EVERY drafted token,
so the exposed verify communication scales with **draft VOLUME** (tree
size / chain length), not with accepted tokens. Therefore the objective at
large EP is **accept-per-verified-token**, and the optimum is **minimal draft
volume** — a short chain (K\* small, often 1), NOT the wide trees that are
optimal on a single GPU (where extra draft nodes are ~free parallel
attention). As EP scale / comm-fraction f grows, K\* shrinks. This is the
"large-EP spec-decode strategy."

**What is already established (single-node, forced-PCIe EMULATION):**
- Phase 33: verify step vs verified-token volume; comm-aware chain/small tree
  ~2.3x over naive wide-tree EAGLE (accept = local-routing stand-in).
- Phase 50: REAL Qwen3-30B EAGLE3 head (`Tengyunw/qwen3_30b_moe_eagle3`),
  chain K-sweep: accept saturates ~1.67 by K=4 -> **K\*=1 at every emulated f**;
  "accept-per-verified-token IS comm efficiency"; EAGLE wins low-f, World A
  (self-spec) wins high-f. Caveat: literal tree runs never executed; single
  node; f via emulated a2a delay.

**The gap this phase fills = REAL 2-node fabric** (the same upgrade Phases
52-56 gave World A): every World B number is emulated. Measure the law and
K\* on genuine inter-node EP (h107+h106, EP16), and produce the repositioned
paper's central figure.

## Experiments

1. **Smoke**: Qwen3-30B + EAGLE3 head, single-node then 2-node DP/EP16.
   Harness `research/52_two_node_e2e/scripts/w7_2node.py` now has
   `W7_SPEC_METHOD=eagle3` / `W7_SPEC_MODEL`. EAGLE runs WITHOUT the self-spec
   env stack (no VLLM_SELF_SPEC_* flags — use a clean env, cf. Phase 50
   STACK=0).
2. **Verify-comm-scaling on real fabric**: verify step time & effective
   bandwidth vs verified-token volume (K sweep) at real EP16. Replaces Phase
   33's emulation.
3. **Real EAGLE3 K-sweep at 2-node EP16**: tok/s + accept vs K at b8/32/64.
   Locate K\* on the real fabric; compare to the emulated K\*=1.
4. **The money plot — K\* vs EP scale / f**: sweep f by EP width (1-node EP8
   vs 2-node EP16; optionally 236B for higher f) and show K\* shrinking as EP
   grows. This IS the design rule.

## Baseline honesty (the reviewer concern)

The claim is a within-spec-decode CONFIGURATION result, so the baseline must
be the tuned conventional default, not a strawman: report the K-sweep in full
(so the reader sees the whole curve), frame K\* vs the naive large-volume/wide
default, and state the gap honestly (Phase 31: free confidence-pruning is
near-oracle, so vs a tuned dynamic-tree EAGLE-2 the margin is smaller than the
2.3x-vs-wide number). The contribution is the LAW + the K\*(f) rule measured
on real hardware, not a headline multiplier.

## Decision criteria

GO for the repositioned paper if, on real 2-node EP16: (a) verify step scales
~linearly with draft volume (law holds on real fabric), and (b) K\* is small
(chain, not wide) and demonstrably shrinks with f. Either outcome is
publishable as the design rule; a surprise (K\* large on real fabric) would
itself be the finding.

## Expected artifact

`results_large_ep_spec.md` + the K\*(f) figure data. Reuses the 2-node
harness, real fabric, and the cached EAGLE3 head — no new infra.
