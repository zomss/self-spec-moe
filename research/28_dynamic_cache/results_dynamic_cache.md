# Results: verify-warmed dynamic cache -- the memory axis win

Date: 2026-06-29. Qwen3-30B-A3B (128 experts, top_k 8). Depth-0 acceptance
beta = overlap(verify next-dist, draft next-dist) along a real greedy continuation, with
the **verify-context-KV draft** (full-weight context, local routing for the predicting
token only). Cache policies vs size C (`dynamic_cache.py`).

## The result

| C | mem (FP4) | static (prompt top-C) | verify-warmed (EMA) | last-token (realizable) |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.5 GB | 0.227 | 0.424 | 0.898 |
| 8 | 1.0 GB | 0.449 | 0.641 | **0.992** |
| 16 | 2.0 GB | 0.565 | **0.885** | 0.992 |
| 32 | 4.1 GB | 0.774 | 0.991 | 0.992 |
| 64 | 8.2 GB | 0.903 | 0.990 | 0.992 |

## Headline: 4-8x less memory for the same (or better) acceptance

- **Static** needs C=64 (**8.2 GB**) for beta 0.90.
- **verify-warmed (EMA)** hits **beta 0.885 at C=16 (2.0 GB)** and 0.99 at C=32 (4 GB)
  -> **4x less memory**.
- **last-token cache** (cache = the last committed token's experts) hits **beta 0.992 at
  C=8 (1.0 GB)** -> **8x less memory AND near-perfect acceptance**.

This beats the Phase 26 static frontier decisively (beat-bar was beta~0.84 at <8 GB;
we get 0.99 at 1 GB).

## Why it works (the verify-KV synergy)

With the verify-context-KV draft the context is full-weight, so the draft's local MoE
only affects the PREDICTING token. That token's next-token distribution is shaped by the
**last committed token's** MoE experts -- which verify computed last cycle and therefore
KNOWS. Caching exactly those (top_k=8 experts/layer -> C=8) makes the draft's predicting-
token MoE identical to full -> beta ~ 1. So the "oracle" is **realizable**: it is the
last-committed-token cache, causal, no future info. The EMA policy is worse at small C
because stale history crowds out the fresh experts; the pure last-token policy is best.

(This also explains why the verify-context-KV default mattered: alone it gave only +0.017
beta with a static cache, but it is what makes the predicting-token's experts the *only*
thing the cache must cover -- enabling the tiny last-token cache.)

## Scope / caveat: this is depth-0 (k=1)

Measured for the FIRST draft token (k=1). k=1 is the HIGH-BATCH optimum (Phase 27:
verify-cost scaling), so this directly applies to the serving regime: **k=1 draft, verify-
KV, last-token cache (C=8, 1 GB) -> beta 0.99**. For k>1 / trees (low-batch lever), the
deeper draft tokens condition on UNVERIFIED draft tokens whose experts aren't known, so
the cache must predict them -> beta decays with depth (follow-up). The speedup ceiling is
unchanged (~1.35x at k=1, the comm-free-draft-vs-verify-cost limit); the win here is
MEMORY (8x) and reliability (beta 0.84 -> 0.99), not peak speed.

## Updated memory frontier (depth-0)

| operating point | beta | memory | vs static |
| --- | ---: | ---: | --- |
| static knee (Phase 26) | 0.84 | 8.2 GB | baseline |
| verify-warmed EMA | 0.885 | 2.0 GB | 4x less |
| **last-token cache** | **0.99** | **1.0 GB** | **8x less, higher beta** |

## Next

- Depth>0 decay: measure beta vs draft depth for k>1 trees with the dynamic cache (the
  deeper tokens need predicted experts) -- sets the low-batch tree memory cost.
- Multi-token cache policy: union of last committed token's + draft-routed experts for the
  in-cycle draft tokens.
