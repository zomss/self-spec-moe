# Strategy map v6 — LCB selection, execution-extended, feasibility-filtered

winner = argmax LCB05; speedup = tau/(gamma(R+phi)+1+psi) at the chain tier's fitted constants; P(win) from 600 MC draws.
ngram priced as geometric beta_eff at gamma<=8 (block-proposal semantics approximated; e2e unvalidated -- wide sigma).


## dense

| cell | winner (LCB05 / median / P(win)) | runner-up | note |
|---|---|---|---|
| b1/2k | **q_int4** 1.22/1.28/P0.39 | q_int4+win512 1.15 | GPTQ; measR/cell |
| b1/16k | **q_int4+win512** 1.46/1.77/P0.74 | q_int4+win512+vres32kC4 1.28 | measR combo / w4xwinfac |
| b1/32k | **q_int4+win512** 1.76/2.31/P0.77 | q_int4+win512+vres32kC4 1.53 | measR combo / w4xwinfac |
| b8/2k | **q_int4** 1.23/1.30/P0.41 | q_int4+win512 1.15 | GPTQ; measR/cell |
| b8/16k | **q_int4+win512** 1.46/1.58/P0.86 | q_int4+win512+vres32kC4 1.27 | measR combo / w4xwinfac |
| b8/32k | **q_int4+win512** 1.62/2.12/P0.74 | q_int4+win512+vres32kC4 1.41 | measR combo / w4xwinfac |
| b32/2k | **q_int4** 1.11/1.17/P0.30 | q_int4+win512 1.05 | GPTQ; measR/cell |
| b32/16k | **q_int4+win512** 1.83/1.97/P0.92 | q_int4+win512+vres32kC4 1.49 | measR combo / w4xwinfac |
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
