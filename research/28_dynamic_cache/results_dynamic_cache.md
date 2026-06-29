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

## Depth>0 decay (k>1 / trees): the win is regime-split

beta_j = overlap at draft depth j along the greedy spine, cycle-fixed caches built from
committed usage (`depth_decay.py`, PRE=8 committed, K=6):

| depth | static C=32 | last-tok C=8 | EMA C=16 | EMA C=32 | oracle |
| ---: | ---: | ---: | ---: | ---: | ---: |
| d0 | 0.872 | **0.994** | 0.923 | 0.989 | 0.994 |
| d1 | 0.837 | 0.364 | 0.513 | 0.720 | 0.942 |
| d2 | 0.653 | 0.441 | 0.382 | 0.768 | 0.844 |
| d3 | 0.732 | 0.368 | 0.398 | 0.535 | 0.631 |
| d4 | 0.656 | 0.280 | 0.694 | 0.770 | 0.830 |
| d5 | 0.614 | 0.214 | 0.478 | 0.638 | 0.815 |
| **accept len** | 2.80 | **1.60** | -- | **2.91** | 3.6 |

Two findings:
1. **The last-token C=8 trick is depth-0 only.** It COLLAPSES after the first token
   (0.99 -> 0.36 -> 0.21): deeper draft tokens predict from UNVERIFIED draft tokens whose
   experts aren't in the 8-expert cache. So it gives accept ~1.6 (good for k=1, useless
   for trees). For k>1 you need a broad cache: **EMA/static C=32 (4 GB) holds beta ~0.6-0.8
   across depth -> accept ~2.9** (covers the cycle's ~20 hot experts, Phase 21).
2. **Even the oracle decays with depth** (0.99 -> 0.94 -> 0.84 -> 0.63 -> ...). Caching
   the prediction-shaping token's exact experts isn't enough, because the in-cycle draft
   tokens are processed LOCALLY (verify hasn't run on them -> no clean KV for them). So the
   accumulating local-draft-token context degrades deep predictions regardless of cache --
   a FUNDAMENTAL cap on comm-free-draft depth (verify-KV cleans the committed context, not
   the in-cycle draft tokens).

## Memory win is regime-split (honest summary)

| regime | k | best cache | memory | beta / accept | vs static |
| --- | ---: | --- | ---: | --- | --- |
| high batch | 1 | last-token C=8 | **1.0 GB** | beta 0.99 | **8x less** |
| low batch / trees | >1 | EMA/static C=32 | 4.0 GB | accept ~2.9 | ~2x less |

So the **8x memory win is specifically the k=1 (high-batch / serving) operating point**
-- which is also the speedup optimum there (Phase 27). For trees (low batch) the win is
~2x (4 GB vs static's 8 GB for similar accept), and deep trees are capped by the
context-degradation limit above.

## Next

- The context-degradation cap suggests the comm-free local draft can't sustain long
  accept lengths regardless of memory -- a real ceiling worth stating in any writeup.
- Cross-model check (DeepSeek shared-expert, GPT-OSS) of the last-token / C=32 results.
