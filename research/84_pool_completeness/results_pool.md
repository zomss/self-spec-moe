# Phase 84 results — the six gates: FOUR new live levers, one domination kill

All on 77's paired ondist refs (1152 positions); ngram also replayed on
the math distribution.

| gate | arm | beta | verdict |
|---|---|---|---|
| n-gram drafting | dense / moe | .071 / .130 (math: .150/.124) | dead on general+math text |
| n-gram drafting | **mla** | **.844 (math .827)**, coverage .96/.92, run@k4 3.3 | **ALIVE — distribution-stable**; R=0, floor-free: prices tau(0.83, gamma) ~ 2.5-3x where the map says OFF (1.13 ceiling) |
| MoE top-C routing | topc4 / topc2 | **.8915** / .7248 | ALIVE at C=4 (halves routed FLOPs at -0.08 beta; the fork's own draft_topc, finally mapped) |
| sub-layer skip | mlpskip6 (6 MLPs ~ 4 layers' bytes) | **.8464** | ALIVE — BEATS whole-layer at equal bytes (iter-greedy budget-4 = .817): granularity strictly improves the skip frontier |
| sub-layer skip | attnskip6 (~2 layers' bytes) | .8828 | borderline (whole-layer budget-2 = .908 at byte parity) BUT cuts KV reads/cache too -- different R structure, keep with note |
| vocab restriction | vres16k / vres32k | .9948 / .9957 [CIRCULAR -- see correction] | **RETRACTED then re-gated: coverage-limited** |
| low-rank (SVD, 50% bytes) | svdr50 | .0443 | **DEAD by domination** (int4: 25% bytes at .924) — closes the factorization cell alongside pruning |

## The two map-consequential finds

**1. ngram flips the MLA column.** Model-free prompt-lookup drafting holds
beta_eff 0.83-0.84 on BOTH prompt distributions on V2-Lite — the
architecture where all five model levers are OFF (triply measured). With
R=0 and no draft chain (floor-free by construction), it prices at
tau(0.83, gamma) minus verify-cycle overhead: ~2.5-3x. Honest mechanism
note: this is an OUTPUT-REPETITIVENESS regime (V2-Lite is a base model;
its continuations loop), not an MLA-architecture effect — the selector
gains a new regime FEATURE (output n-gram coverage, measurable in the
same offline pass) rather than a new architecture rule. vLLM ships the
ngram speculator, so an e2e check is harness plumbing away.

**2. vres16k -- RETRACTED (2026-07-18): the gate was CIRCULAR.** The keep
set was built from the same 12-prompt refs it was scored on (the refs
contain only ~12k distinct tokens, so top-16k covered them by
construction). The e2e audit caught it: live continuations cover only
0.69 of the refs-keep; honest C4-frequency keeps reach 0.80 (16k) / 0.90
(32k) live coverage -> honest beta ~0.85-0.88, and the composed trade
LOSES at deep gamma (~1.57x vs 1.91x without). vres survives only where
natural coverage is high (domain/code workloads) or gamma is shallow --
scoped out of the dense map. THE METHODS LESSON: a beta gate whose keep/
profile derives from its own evaluation set is circular; every profiled
lever's artifact must be built on disjoint data (the flr expert sets
pass this check: usage from 4 prompts, scored on 12). Original text
(invalid) follows for the record:
**[retracted] vres16k is nearly free beta AND large composed R.** The W4 checkpoint
quantizes everything EXCEPT lm_head (skipped by the recipe) — so lm_head
(1.09 GB bf16) is ~23% of the composed draft's bytes. Restricting the
draft vocab to the top-16k tokens costs beta 0.9948 (=free within noise)
and cuts those bytes 9x: composed R 0.311 -> ~0.25, repricing the
headline cell b32/16k from 1.91x to ~2.16x and lifting every composed
dense cell. Needs: draft-side logit mask plumbing (small) + e2e.

## Taxonomy after the gates

Every cell of the (cost term x reduction op) matrix is now non-empty:
in-map (7), gated-alive-entering-pool (4: ngram, topc4, mlpskip, vres),
gated-dead with domination argument (3: pruning, svdr, [attnskip absorbed
into mlpskip's frontier note]), measured-excluded with reason (kvq pool),
scoped-out with statement (dynamic token selection as window upper bound;
tree/multi-draft as tau-formula change). The pool-completeness claim is
now BY TAXONOMY.
