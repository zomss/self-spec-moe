# Phase 77 results

## Anchor gate: OPEN — offline β reproduces both e2e accept lengths

`scripts/anchor_gate.py` (offline teacher-forced window-draft acceptance,
W7-faithful prompts, draft decode step against target KV sliced to
sinks16+window512 with original RoPE), composed via geometric τ and compared
to the two 76-E3 end-to-end measurements:

| cell | β_greedy (positions) | τ_geom | E3 measured | err | verdict |
|---|---|---|---|---|---|
| dense Qwen2.5-7B, 16k, K=4 | 0.9766 (384) | 4.771 | 4.850 | **−1.6%** | PASS |
| MoE Qwen3-30B, 32k, K=6 | 0.9757 (1152) | 6.510 | 6.493 | **+0.3%** | PASS |

Data: `data/anchor_gate.json`.

**The method is calibrated**: one-step offline β + geometric composition
predicts real spec-decoding accept lengths to ≤2% on both architectures. The
sweep (`score_accept.py`, next) inherits this credibility.

### Measurement requirements learned (bind the sweep design)

1. **≥12 prompts × 96 positions (~1000+) per cell.** Per-prompt β spread is
   large (0.91–1.00 across the same bank); 4 prompts missed by −9% (the first
   gate attempt FAILed on sampling variance alone — running mean climbed
   0.93→0.976 monotonically as prompts accumulated).
2. **Per-prompt β has run-to-run jitter** (same prompt: 1.000 in one process,
   0.927 in another — MoE router argmax flips at numerics boundaries amplify
   close calls; cf. P75-E2's batch-invariance finding). The AGGREGATE over
   ≥1000 positions is the stable statistic; never quote per-prompt β.
3. T=1.0 overlap tracks greedy β closely at these levels (0.9754 vs 0.9757 on
   MoE) — both fall out of the same forwards; report both.
4. The token-exact window slice (vs vLLM's 16-token page granularity) does NOT
   produce a measurable discrepancy at window 512 — the gate closed to 0.3%
   without modeling pages.
