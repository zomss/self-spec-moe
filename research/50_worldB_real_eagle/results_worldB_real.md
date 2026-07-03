# Phase 50 results — World B validated with a REAL EAGLE3 head at single node

**Setup.** `Tengyunw/qwen3_30b_moe_eagle3` (real trained EAGLE3 head for
Qwen3-30B-A3B) on the identical forced-PCIe testbed as every World A number
(DP4/EP4 GPUs 0-3, b64 global, full-length runs, greedy). Harness unchanged
except env-gated `W7_SPEC_METHOD`/`W7_SPEC_MODEL`; EAGLE engines run without
the self-spec env stack (`STACK=0`).

## 1. The draft-volume / comm curve with real acceptance (the World B claim)

EAGLE3 chain, K = draft volume knob (verify routes (K+1)·B tokens):

| a2a µs | K=1 | K=2 | K=4 | K=8 |
|---:|---:|---:|---:|---:|
| 0    | **3022** /1.46 | 2687 /1.60 | 2111 /1.67 | 1260 /1.68 |
| 500  | **1209** /1.46 | 1079 /1.60 | 965 /1.67 | 778 /1.68 |
| 1000 | **687** /1.46 | 668 /1.60 | 622 /1.67 | 539 /1.68 |

(tok/s / accept_len.) Accept SATURATES at ~1.67 by K=4 (+0.01 from K4→K8)
while verify volume grows linearly — so every draft token past K≈2 is pure
verify-comm waste. **The comm-aware optimum is K\*=1 at every measured f for
this head**, and over-drafting steepens with volume (K8 = 0.42× K1 at native).
This is Phase 33's claim measured with real acceptance: minimal draft volume
wins on comm-bound MoE-EP; large speculation budgets are counterproductive.
K8 routes 9/1.68 = **5.4 all-to-all tokens per committed token**.

## 2. Cross-candidate table (all single-node candidates, same testbed)

| a2a µs | World A lockstep K2 | World A overlap best | EAGLE3 best (K=1) |
|---:|---:|---:|---:|
| 0    | 2478 (2.91) | 1542 | **3022** (1.46) |
| 500  | **1710** (2.89) | 1208 | 1209 (1.46) |
| 1000 | 1306 (2.91) | 1272 (K4: 3.95) | 687 (1.46) |

- **EAGLE3 wins at native (+22% over World A)** — the near-free trained head
  is the best low-f config measured all campaign.
- **EAGLE3 collapses as f grows** (half of World A at 1000 µs). Mechanism:
  the verify routes (K+1)/accept tokens of all-to-all per committed token —
  **accept-per-verified-token IS comm efficiency**. EAGLE3: 3/1.60 = 1.88;
  self-spec: 3/2.91 = 1.03. The cheap-but-inaccurate head pays ~2× the comm
  per useful token; the expensive-but-near-perfect self-spec wins the
  comm-bound regime — the Phase 32 inversion, now measured with a real head.
- World A's distribution-robustness is demonstrated, not just argued: the
  self-spec draft IS the model (accept 2.91 on this harness's off-distribution
  synthetic text), while the trained head degrades (1.60 vs its ~2.3
  on-distribution reputation).

## 3. Caveats

1. EAGLE3 accept is depressed by off-distribution prompts; on-distribution
   ~2.3-2.5 softens but does not reverse the high-f conclusion (3/2.4 = 1.25
   tokens/token still loses to 1.03).
2. **Literal tree-spec runs (the exact World B configuration) remain
   unexecuted anywhere** — this vLLM branch has no tree-attention drafting.
   SCOPING DECISION: accept the a-fortiori argument — a tree's marginal
   (width) nodes have strictly worse acceptance than a chain's marginal
   (depth) nodes, and this head's accept saturates at 1.67 even on chains,
   so a wide tree is strictly worse per routed token than the measured K=8
   chain (already 0.42-0.78x of K=1). The chain-volume sweep therefore
   bounds the wide-tree conclusion; only the small-pruned-tree UPSIDE at low
   batch (+8% with the Phase-27 stand-in) lacks a real-head number. Literal
   in-engine tree runs = an explicit item for the upstream-vLLM port or the
   multi-node rental stack.
3. K\*=1 is head-specific (this head's accept saturates early); a stronger
   head shifts K\* up but the volume-cost slope is head-independent.

## 4. The complete single-node deployment map (measured)

| regime | best config | tok/s @b64 |
|---|---|---:|
| native (f≈0.4, latency-bound) | **EAGLE3 K=1** | 3022 |
| a2a 500 (f≈0.6) | **World A lockstep K2** | 1710 |
| a2a 1000 (f≈0.7) | World A lockstep K2 (overlap-K4 at parity, 0.97×) | 1306 |
| beyond (multi-node) | World A overlap-K4+ (trend + Phase-32 model) | rental |

Every candidate class is now measured at single node; the multi-node rental
tests the one regime this box cannot produce.
