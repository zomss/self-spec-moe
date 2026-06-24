# Phase 10: Temporal Routing Locality and a Verify-Warmed Draft Cache

Source phase: [`../09_rebalancing_ceiling`](../09_rebalancing_ceiling)

## Objective

Phase 09 showed a **static** draft cache needs ~0.5-0.8 E experts per device to
reach useful acceptance. But speculative decoding gives a free signal: the verify
step reveals the **true** experts used at each position, and MoE routing is
correlated in time within a sequence. So a draft cache can be **dynamic** -
warmed from recently verified routing - and may cover the next token's experts
with far fewer slots than a static cache.

The phase asks:

```text
Does temporal routing locality let a small verify-warmed cache cover the next
token's experts, so local-draft acceptance no longer needs ~half the pool?
```

## Design

- Decode a real continuation per prompt (sampling), then read per-layer per-
  position true top-k experts from a full forward.
- **Dynamic cache:** a per-layer LRU expert cache of capacity `C`, warmed by the
  true top-k of every position as it is "verified". Before predicting position
  `t`, the cache holds the `C` most-recently-used experts (built from positions
  `< t`). This models the novelty lag honestly: an expert used for the first time
  at `t` is a miss, then enters the cache for later positions.
- **Static baseline:** the per-request top-`C` experts by gate mass (a stronger
  static baseline than Phase 09's global set).
- **Metric:** coverage = fraction of position `t`'s true top-k already in the
  cache, plus the all-top-k-local rate, swept over cache size `C`. Then validate
  the winning `C` with actual masked-forward sampled acceptance.

Reading: if **dynamic coverage** reaches ~0.9 at a **small** `C` (a few x top-k)
while **static** needs much larger `C` for the same coverage, temporal locality
revives the method and the Phase 09 static bound was too pessimistic. If dynamic
~= static, temporal locality is weak and the negative result holds.

## Commands

```bash
.venv/bin/python research/10_temporal_locality_cache/temporal_locality_cache.py \
    --model Qwen/Qwen3-30B-A3B --device cuda:0 --gen-tokens 192 \
    --cache-sizes 8,12,16,24,32,48,64 --validate-cache-sizes 16,32 \
    --output-csv research/10_temporal_locality_cache/data/qwen3_30b_a3b_temporal.csv \
    --output-json research/10_temporal_locality_cache/data/qwen3_30b_a3b_temporal.json

.venv/bin/python research/10_temporal_locality_cache/temporal_locality_cache.py \
    --model openai/gpt-oss-20b --device cuda:0 --gen-tokens 192 \
    --cache-sizes 4,6,8,12,16,24 --validate-cache-sizes 8,16 \
    --output-csv research/10_temporal_locality_cache/data/gpt_oss_20b_temporal.csv \
    --output-json research/10_temporal_locality_cache/data/gpt_oss_20b_temporal.json
```

## Expected artifacts

| Output | Path |
| --- | --- |
| Runner | `temporal_locality_cache.py` |
| Coverage + acceptance CSV/JSON | `data/<model>_temporal.{csv,json}` |
| Logs | `logs/` |
| Summary | `results_*.md` |
