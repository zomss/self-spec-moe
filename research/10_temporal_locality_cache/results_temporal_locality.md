# Results: Temporal Routing Locality and a Verify-Warmed Draft Cache

Date: 2026-06-24

## Status

Real sampled decode sequences (gen 192 tokens, 16 prompts/model). Coverage of the
next position's true top-k by a per-layer LRU expert cache of capacity `C`, warmed
by verified routing, vs a per-request static top-`C`-by-mass cache. Acceptance
validated by masked forward at selected `C`. One-step proxy (multi-token lower).

## Coverage vs cache size

### Qwen3-30B-A3B (E=128, top-8)

| C | C/E | dyn coverage | static coverage | dyn all-top-k-local |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.06 | 0.423 | 0.366 | 0.004 |
| 12 | 0.09 | 0.505 | 0.460 | 0.022 |
| 16 | 0.12 | 0.576 | 0.536 | 0.050 |
| 24 | 0.19 | 0.696 | 0.651 | 0.140 |
| 32 | 0.25 | 0.783 | 0.736 | 0.261 |
| 48 | 0.38 | 0.895 | 0.847 | 0.526 |
| 64 | 0.50 | 0.946 | 0.914 | 0.703 |

### GPT-OSS-20B (E=32, top-4)

| C | C/E | dyn coverage | static coverage | dyn all-top-k-local |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.12 | 0.437 | 0.394 | 0.052 |
| 6 | 0.19 | 0.540 | 0.502 | 0.127 |
| 8 | 0.25 | 0.637 | 0.583 | 0.223 |
| 12 | 0.38 | 0.781 | 0.695 | 0.434 |
| 16 | 0.50 | 0.879 | 0.777 | 0.639 |
| 24 | 0.75 | 0.969 | 0.915 | 0.887 |

## Acceptance validation (masked forward, decode positions)

| Model | C | C/E | sampled acceptance | top-1 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-30B | 16 | 0.12 | 0.382 | 0.467 |
| Qwen3-30B | 32 | 0.25 | 0.741 | 0.767 |
| GPT-OSS-20B | 8 | 0.25 | 0.637 | 0.600 |
| GPT-OSS-20B | 16 | 0.50 | 0.833 | 0.767 |

## Reading

**1. The dynamic cache beats the static one — modestly.** At equal memory the
verify-warmed LRU cache improves coverage by ~0.04-0.10 over per-request static
(Qwen3 C=32: 0.783 vs 0.736; GPT-OSS C=16: 0.879 vs 0.777). Temporal recency is a
real but small additional signal.

**2. The bigger lever is per-request specialization, not recency.** Phase 09's
global static cache needed `M ~= 0.54 E` (Qwen3) / `0.79 E` (GPT-OSS) for
`beta >= 0.8`. A cache specialized to the request (per-request static here) plus
recency reaches `beta >= 0.8` at roughly:

| Model | C for beta>=0.8 | C/E | Phase 09 static M*/E | reduction |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-30B | ~40 | ~0.31 | 0.54 | ~1.7x |
| GPT-OSS-20B | ~15 | ~0.47 | 0.79 | ~1.7x |

So warming the cache from verify roughly **halves** the experts needed. Most of
that comes from making the cache per-request; the LRU/recency part adds the
smaller increment in (1).

**3. It softens the high-EP problem but does not remove it.** The cache is still a
replicated warm set of size `C`, so the replication factor over the plain-EP
budget is `R = C/(E/N) = N * (C/E) ~= 0.3N` (Qwen3) to `0.5N` (GPT-OSS):

| EP size N | R (Qwen3, C/E=0.31) | R (GPT-OSS, C/E=0.47) |
| ---: | ---: | ---: |
| 4 | 1.2x | 1.9x |
| 8 | 2.5x | 3.8x |
| 16 | 5.0x | 7.5x |
| 32 | 10x | 15x |

This is ~half the Phase 09 factor, but still grows linearly with EP. At moderate
EP (2-8) a 2.5-4x warm cache may be affordable; at the cross-node scale
(EP >= 16) it still approaches near-full replication.

## Caveats

- **Novelty lag is real.** Even at large `C`, dynamic coverage stays below 1
  (Qwen3 C=64: 0.946) because a first-use expert cannot be cached yet. Recency
  cannot cover genuinely new routing.
- **One-step proxy.** Multi-token drafting compounds divergence; the all-top-k-
  local rates (Qwen3 C=32: 0.26) show full-coverage at a layer is still uncommon,
  so multi-token acceptance would be lower than the one-step numbers.
- **Warm-replica cost.** The cache holds expert-weight replicas refreshed by
  prefetch during verify; the refresh bandwidth must fit the verify window.

## Conclusion

The verify-warmed / per-request cache is a real improvement: it roughly halves the
draft-cache size needed for useful acceptance versus a static global cache, and it
validates the intuition that the verify step's routing should specialize the
draft cache. But temporal recency itself contributes only a small part, the
needed cache is still ~0.3-0.5 E, and the high-EP replication factor `R ~ 0.3N-0.5N`
still grows with EP. The method becomes plausible at **moderate EP with an
affordable per-request warm cache**, but the cross-node high-EP regime still
requires heavy replication. This moves the viability boundary outward by ~2x
without crossing it for the large-EP target regime.
