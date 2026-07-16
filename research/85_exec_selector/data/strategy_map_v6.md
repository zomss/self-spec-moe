# Strategy map v6 — LCB selection, execution-extended, feasibility-filtered

winner = argmax LCB05; speedup = tau/(gamma(R+phi)+1+psi) at the chain tier's fitted constants; P(win) from 600 MC draws.
ngram priced as geometric beta_eff at gamma<=8 (block-proposal semantics approximated; e2e unvalidated -- wide sigma).


## dense

| cell | winner (LCB05 / median / P(win)) | runner-up | note |
|---|---|---|---|
| b1/2k | **q_int4+win512+vres16k** 1.27/1.46/P0.82 | q_int4 1.22 | 84: lm_head cut (x0.80) |
| b1/16k | **q_int4+win512+vres16k** 1.61/2.02/P0.73 | q_int4+win512 1.46 | 84: lm_head cut (x0.80) |
| b1/32k | **q_int4+win512+vres16k** 2.00/2.61/P0.66 | q_int4+win512 1.76 | 84: lm_head cut (x0.80) |
| b8/2k | **q_int4+win512+vres16k** 1.28/1.48/P0.85 | q_int4 1.23 | 84: lm_head cut (x0.80) |
| b8/16k | **q_int4+win512+vres16k** 1.60/1.81/P0.95 | q_int4+win512 1.46 | 84: lm_head cut (x0.80) |
| b8/32k | **q_int4+win512+vres16k** 1.81/2.42/P0.69 | q_int4+win512 1.62 | 84: lm_head cut (x0.80) |
| b32/2k | **q_int4+win512+vres16k** 1.16/1.32/P0.84 | q_int4 1.11 | 84: lm_head cut (x0.80) |
| b32/16k | **q_int4+win512+vres16k** 1.99/2.26/P0.93 | q_int4+win512 1.83 | 84: lm_head cut (x0.80) |
| b32/32k | INFEASIBLE (residency) | | |

## moe

| cell | winner (LCB05 / median / P(win)) | runner-up | note |
|---|---|---|---|
| b4/2k | **OFF** (best LCB 0.89 q_fp8 | P(OFF-ish)=0.14 | measR/cell |
| b4/16k | **OFF** (best LCB 0.90 win128 | P(OFF-ish)=0.10 | measR/cell |
| b4/32k | **win128** 1.23/1.40/P0.71 | q_fp8 1.15 | measR/cell |
| b8/2k | **OFF** (best LCB 0.93 q_fp8 | P(OFF-ish)=0.29 | measR/cell |
| b8/16k | **win512** 1.07/1.20/P0.80 | win128 0.99 | measR/cell |
| b8/32k | **win512** 1.11/1.23/P0.57 | win128 1.08 | measR/cell |
| b32/2k | **OFF** (best LCB 0.93 q_fp8 | P(OFF-ish)=0.32 | measR/cell |
| b32/16k | **win128** 1.19/1.34/P0.77 | win512 1.11 | measR/cell |
| b32/32k | **win512** 1.61/1.90/P0.70 | win128 1.52 | measR/cell |

## mla

| cell | winner (LCB05 / median / P(win)) | runner-up | note |
|---|---|---|---|
| b4/2k | **OFF** (best LCB 0.90 ngram | P(OFF-ish)=0.28 | e2e MEASURED 0.77-0.78x: psi=2.4 (CPU lookup) kills it on this stack |
| b4/16k | **OFF** (best LCB 0.90 ngram | P(OFF-ish)=0.28 | e2e MEASURED 0.77-0.78x: psi=2.4 (CPU lookup) kills it on this stack |
| b4/32k | **OFF** (best LCB 0.88 ngram | P(OFF-ish)=0.31 | e2e MEASURED 0.77-0.78x: psi=2.4 (CPU lookup) kills it on this stack |
| b8/2k | **OFF** (best LCB 0.91 ngram | P(OFF-ish)=0.27 | e2e MEASURED 0.77-0.78x: psi=2.4 (CPU lookup) kills it on this stack |
| b8/16k | **OFF** (best LCB 0.91 ngram | P(OFF-ish)=0.27 | e2e MEASURED 0.77-0.78x: psi=2.4 (CPU lookup) kills it on this stack |
| b8/32k | **OFF** (best LCB 0.90 ngram | P(OFF-ish)=0.28 | e2e MEASURED 0.77-0.78x: psi=2.4 (CPU lookup) kills it on this stack |
| b32/2k | **OFF** (best LCB 0.91 ngram | P(OFF-ish)=0.26 | e2e MEASURED 0.77-0.78x: psi=2.4 (CPU lookup) kills it on this stack |
| b32/16k | **OFF** (best LCB 0.89 ngram | P(OFF-ish)=0.29 | e2e MEASURED 0.77-0.78x: psi=2.4 (CPU lookup) kills it on this stack |
| b32/32k | **OFF** (best LCB 0.89 ngram | P(OFF-ish)=0.32 | e2e MEASURED 0.77-0.78x: psi=2.4 (CPU lookup) kills it on this stack |
