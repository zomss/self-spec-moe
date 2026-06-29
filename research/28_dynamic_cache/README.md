# Phase 28: Verify-warmed dynamic cache (memory axis)

**Source:** Phase 26 frontier (static top-C cache: knee at 0.5E=8 GB, beta 0.84; steep
diminishing past it) + Phase 27 decision to make the **verify-context-KV draft the
default**. Objective 2: minimize the resident draft-expert memory while keeping accept
length.

**Key idea:** with a verify-context-KV draft the context is full-weight, so the draft's
local MoE only affects the PREDICTING token -> the cache only needs that token's experts,
not the whole context's. The predicting token is shaped by the LAST COMMITTED token's
experts, which verify already knows -> a verify-warmed cache (track recently-used experts)
should reach high beta at a much smaller C than the static prompt-top-C frontier.

**Objective:** measure depth-0 acceptance beta = overlap(verify next-dist, draft
next-dist) vs cache size C, for cache policies {static (prompt top-C), verify-warmed (EMA
of verify's used experts), oracle (last committed token's experts)}, with the verify-KV
draft. Beat the Phase 26 static frontier (beat-bar: beta~0.84 at <8 GB / <0.5E).

**Assumptions:** depth-0 (k=1 / first draft token) -- the deeper-draft decay (cache must
predict unverified tokens' experts) is a follow-up. Per-layer cache; EMA decay 0.9;
warmup 4 positions.

**Commands:** `python dynamic_cache.py --local-files-only --depth 20 --output-json data/qwen3_dynamic_cache.json`

**Decision criteria:** does verify-warmed reach static-C=64's beta (0.84) at C<<64?
Memory = C/128 * 16.3 GB (FP4). Next: deeper-draft decay; multi-token cache policy.
