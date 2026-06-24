# Phase 07: Local-Draft Acceptance on Recent MoE Models

Source phase: [`../06_local_draft_policy`](../06_local_draft_policy)

## Objective

Test whether the Self-MoE-spec premise (local-only routed draft accepted by exact
full-MoE verification) holds on **recent** MoE checkpoints, instead of concluding
from the older small models in Phases 03-06.

The phase asks:

```text
Do recent MoE architectures give local-only draft acceptance high enough to be
useful, or is the ~0.40 ceiling seen on Qwen1.5-MoE a property of recent models too?
```

## Why this phase

Phases 03-06 measured local-draft acceptance only on `Qwen1.5-MoE-A2.7B` and
`PowerMoE-3b`. Both are small and not in the multi-node target regime, so the
negative result (`beta ~= 0.40` vs the `0.8-0.98` requirement) may be an artifact
of model choice. Before pivoting to a hybrid draft or writing a negative result,
re-test the premise on recent models.

## Models

| Model | Released | Routed experts | top-k | Shared expert | Notes |
| --- | --- | ---: | ---: | --- | --- |
| `Qwen/Qwen3-30B-A3B` | 2025 | 128 | 8 | No | Recent, fine-grained |
| `openai/gpt-oss-20b` | 2025 | 32 | 4 | No | Recent, coarser |

**Caveat:** neither model has a shared expert, which the 00_proposal note calls
the fidelity anchor. So this set isolates *recency* and *expert granularity*, not
the shared-expert hypothesis. If both still fail, the shared-expert architectures
(DeepSeek / Llama-4 style) remain an explicit open caveat, not a tested case.

## Method

Offline, single forward per prompt (one-token acceptance proxy, same as Phase 05;
real multi-token `beta` would be lower, so these numbers are optimistic):

1. Full forward (no patch) -> target next-token distribution `p` and per-layer
   full router probabilities (for `gamma` / top-k overlap / hot experts).
2. For each placement, EP rank, and `draft_top_k`: patch every router to mask
   non-local experts and keep only `draft_top_k` local experts, then forward ->
   draft next-token distribution `q`.
3. Metrics vs `p`:
   - `gamma` = full-router probability mass on local experts,
   - top-k overlap = fraction of true top-k experts that are local,
   - expected acceptance = `sum_i min(p_i, q_i)` (overlap),
   - sampled acceptance = Monte-Carlo `min(1, p/q)` accept rate,
   - `KL(p||q)`, top-1 (greedy) match.

Placements: contiguous, random, and hot-replicated variants. EP sizes 2/4/8.
Prompts are domain-bucketed (general / chat / code / math) because routing
locality is domain-dependent.

## Commands

```bash
# Qwen3-30B-A3B (GPU 0)
.venv/bin/python research/07_recent_model_acceptance/measure_local_draft_acceptance.py \
    --model Qwen/Qwen3-30B-A3B --device cuda:0 \
    --ep-sizes 2,4,8 --draft-top-k-values 1,2,4,8 \
    --output-csv research/07_recent_model_acceptance/data/qwen3_30b_a3b_acceptance.csv \
    --output-json research/07_recent_model_acceptance/data/qwen3_30b_a3b_acceptance.json

# GPT-OSS-20B (GPU 1)
.venv/bin/python research/07_recent_model_acceptance/measure_local_draft_acceptance.py \
    --model openai/gpt-oss-20b --device cuda:1 \
    --ep-sizes 2,4,8 --draft-top-k-values 1,2,4 \
    --output-csv research/07_recent_model_acceptance/data/gpt_oss_20b_acceptance.csv \
    --output-json research/07_recent_model_acceptance/data/gpt_oss_20b_acceptance.json
```

## Go/No-Go Criteria

- If best sampled acceptance on a recent model reaches `beta >= 0.8` (high-`f`
  regime) or `>= 0.9` (moderate / DBO-composed), the premise survives on recent
  models -> proceed to multi-token acceptance, then runtime.
- If both recent models stay near the Qwen1.5-MoE `~0.40` ceiling, the negative
  result is now cross-architecture and recent -> write it up, with the
  shared-expert architectures named as the one untested caveat.

## Expected Artifacts

| Output | Path |
| --- | --- |
| Acceptance runner | `measure_local_draft_acceptance.py` |
| Metrics CSV/JSON | `data/<model>_acceptance.{csv,json}` |
| Run logs | `logs/` |
| Summary | `results_*.md` |
