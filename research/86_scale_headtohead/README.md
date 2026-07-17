# Phase 86 — the KnapSpec head-to-head: all regimes, starting at Qwen3-8B

Source: paper/DIRECTION.md; KnapSpec (2602.20217: up to 1.47x at
Llama3.1-70B, 1.43x Qwen3-32B, **1.28x Qwen3-8B**; accept 93.5% at 32B;
AR baselines; T=0.7 sampling + greedy; batch unstated = b1-latency
class). User directive 2026-07-18: compare ours vs KnapSpec at ALL
regimes, starting with Qwen3-8B, then 32B.

## Pre-registered predictions (from our laws, BEFORE measurement)

P1. **QK-norm rule on a new dense model**: Qwen3-8B HAS QK-norm ->
    draft-only kvq_fp8 beta ~0.97 (vs Qwen2.5-7B's broken 0.60-0.84).
    First out-of-family test of F6 on the dense class.
P2. **Skip beta scale/family-dependence**: Qwen3-8B (36 layers) layer
    profile sits between Qwen2.5-7B (hostile: 0.45@12.5%) and the 32B
    regime KnapSpec's 93.5% accept implies; iterative-greedy sets at
    budget ~4-5 reach beta 0.85-0.92 (better than Qwen2.5's 0.87@3 but
    not yet 32B-class).
P3. **At 8B/b1 the map's winner is quant-led, not skip-led**, pricing
    1.28-1.35x greedy -- MATCHING KnapSpec's 1.28 at equal-or-lower
    search cost; composition adds little at b1/short-ctx (the honest
    regime-dependence claim), more at b1/32k and at batch.
P4. 32B (next): composed (skip-set x W4 x window) prices ~1.5-1.6x
    roofline > their 1.43 -- the composition dividend appears WITH SCALE.

## Plan

- E1 (8B beta column): refs at 16k (ondist; math bank for the
  AIME-comparable rows), singles (q_int4 RTN-matched-to-ckpt, q_fp8,
  win512/128, kvq_fp8 [P1], skip125/25) + leave-one-out layer profile
  (36 layers) + iterative-greedy sets [P2].
- E2 (8B R + e2e): W7 dense TP1 arms at b1 (their setting; greedy AND
  T=0.7) and b8/b32 (our regimes they do not report): AR baseline, W4
  (ckpt exists: ~/ckpts/Qwen3-8B-W4A16-INT4), W4+win512 composed at
  16k/32k. Compare vs their 1.28x at matched scale [P3].
- E3 (32B): same column at Qwen3-32B (W4 ckpt to build, GPTQ) [P4].
- Comparison discipline: matched-setting rows only vs their table;
  batch rows labeled as regimes their method does not address; skip-set
  e2e draft (their lever in OUR harness) needs shared-KV layer-mapping
  plumbing -- deferred unless the 32B search selects it.

## Constraints

- GPUs 6/7 for offline; e2e single-GPU arms as available; caches /data.
- Beta method: 77 harness, disjoint-artifact rule enforced (layer sets
  profiled on refs, selected sets re-measured; nothing scored on its
  own construction data).

## Artifacts

results_h2h.md; data/beta_q3_8b.csv, iter_greedy_q3_8b.csv, e2e jsons.
