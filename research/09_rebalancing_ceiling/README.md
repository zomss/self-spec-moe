# Phase 09: Rebalancing Ceiling vs Replication Budget

Source phase: [`../07_recent_model_acceptance`](../07_recent_model_acceptance)

## Objective

Phase 07 found local-draft acceptance collapses as EP grows because each device
holds fewer experts (`E/N`). The natural mitigation is locality-aware rebalancing
or replication. Instead of testing heuristics one by one, this phase measures the
**ceiling**: the best acceptance any replicated draft-expert cache can achieve as
a function of its per-device budget `M`.

The phase asks:

```text
How large must a per-device replicated draft-expert cache be to reach
beta >= 0.8, and is that budget affordable relative to E/N (what EP gives free)
and E (full replication, which defeats EP)?
```

## Design

- A **draft-expert cache** of `M` experts is replicated on every device, chosen
  per layer as the top-`M` experts by aggregated gate mass (the mass-optimal
  fixed set -> an upper bound on any fixed placement of that size).
- Because the cache is the same on every device, it is **EP-invariant**: its
  acceptance depends on `M`, not on the EP size `N`. The curve therefore bounds
  what rebalancing can do at any EP.
- Draft routing uses the model's default expert top-k selected from the local
  cache; verification stays exact (lossless), so the draft placement may differ
  from the verify placement.
- Acceptance is the same one-token proxy as Phase 05/07 (optimistic upper bound).

Reference budgets: for EP size `N`, plain EP gives `M = E/N` per device.

| Model | E | E/8 (EP8) | E/4 (EP4) | E/2 (EP2) | E (full) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-30B-A3B | 128 | 16 | 32 | 64 | 128 |
| GPT-OSS-20B | 32 | 4 | 8 | 16 | 32 |

## Reading the result

- `M = E` must give acceptance ~1.0 (sanity: no masking).
- `M = E/N` is the no-replication ceiling at EP `N` (already optimal placement,
  so an upper bound on Phase 07's random/contiguous numbers).
- Let `M*` = smallest `M` reaching `beta >= 0.8`. Then:
  - `M* ≈ E/N` for a useful `N` -> rebalancing alone is enough; engineer it.
  - `E/N << M* << E` -> viable only with replication; the curve gives the memory
    price (`M*` experts per device per layer).
  - `M* ≈ E` -> rebalancing is futile; reaching acceptance needs ~full
    replication, which removes the point of EP. Negative result stands.

## Commands

```bash
.venv/bin/python research/09_rebalancing_ceiling/rebalancing_ceiling.py \
    --model Qwen/Qwen3-30B-A3B --device cuda:0 \
    --budgets 8,16,24,32,48,64,96,128 \
    --output-csv research/09_rebalancing_ceiling/data/qwen3_30b_a3b_ceiling.csv \
    --output-json research/09_rebalancing_ceiling/data/qwen3_30b_a3b_ceiling.json

.venv/bin/python research/09_rebalancing_ceiling/rebalancing_ceiling.py \
    --model openai/gpt-oss-20b --device cuda:0 \
    --budgets 4,8,12,16,24,32 \
    --output-csv research/09_rebalancing_ceiling/data/gpt_oss_20b_ceiling.csv \
    --output-json research/09_rebalancing_ceiling/data/gpt_oss_20b_ceiling.json
```

## Expected artifacts

| Output | Path |
| --- | --- |
| Sweep runner | `rebalancing_ceiling.py` |
| Ceiling CSV/JSON | `data/<model>_ceiling.{csv,json}` |
| Run logs | `logs/` |
| Summary | `results_*.md` |
