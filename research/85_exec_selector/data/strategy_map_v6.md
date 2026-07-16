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
| b4/16k | **flr50+q_fp8** 1.15/1.35/P0.70 | topc4+flr50 0.97 | 83 realization R per cell (batch-dep) |
| b4/32k | **flr50+q_fp8** 1.43/1.73/P0.75 | win128 1.23 | 83 realization R per cell (batch-dep) |
| b8/2k | **OFF** (best LCB 0.93 q_fp8 | P(OFF-ish)=0.29 | measR/cell |
| b8/16k | **win512** 1.07/1.20/P0.68 | win128 0.99 | measR/cell |
| b8/32k | **win512** 1.11/1.23/P0.26 | win128 1.08 | measR/cell |
| b32/2k | **OFF** (best LCB 0.93 q_fp8 | P(OFF-ish)=0.32 | measR/cell |
| b32/16k | **win128** 1.19/1.34/P0.67 | win512 1.11 | measR/cell |
| b32/32k | **win512** 1.61/1.90/P0.65 | win128 1.52 | measR/cell |

## mla

| cell | winner (LCB05 / median / P(win)) | runner-up | note |
|---|---|---|---|
| b4/2k | **ngram** 2.47/2.90/P1.00 | q_fp8 0.59 | 84: block-4 semantics; e2e UNVALIDATED |
| b4/16k | **ngram** 2.48/2.89/P1.00 | q_fp8 0.55 | 84: block-4 semantics; e2e UNVALIDATED |
| b4/32k | **ngram** 2.43/2.86/P1.00 | q_fp8 0.59 | 84: block-4 semantics; e2e UNVALIDATED |
| b8/2k | **ngram** 2.49/2.88/P1.00 | q_fp8 0.58 | 84: block-4 semantics; e2e UNVALIDATED |
| b8/16k | **ngram** 2.49/2.89/P1.00 | q_fp8 0.58 | 84: block-4 semantics; e2e UNVALIDATED |
| b8/32k | **ngram** 2.46/2.91/P1.00 | win512 0.60 | 84: block-4 semantics; e2e UNVALIDATED |
| b32/2k | **ngram** 2.49/2.90/P1.00 | q_fp8 0.59 | 84: block-4 semantics; e2e UNVALIDATED |
| b32/16k | **ngram** 2.44/2.88/P1.00 | q_fp8 0.60 | 84: block-4 semantics; e2e UNVALIDATED |
| b32/32k | **ngram** 2.45/2.86/P1.00 | q_fp8 0.59 | 84: block-4 semantics; e2e UNVALIDATED |

## Provenance notes

1. Dense 2k cells: the window component is a cost no-op at 2k, so the
   winner is effectively q_int4+vres16k; printed with win512 for config-
   set continuity.
2. MoE b4/16k and b4/32k flr50 wins use the REALIZATION R scaled by
   T_t growth (FLR_CTXFAC — assumption, not measurement): these two
   cells head the e2e measurement queue.
3. MLA ngram: block-4 semantics approximated as geometric beta_eff at
   gamma<=4; e2e UNVALIDATED (vLLM ships the ngram speculator — queue).
4. LCB DEMOTES the v5 moe 2k flips to OFF (best LCB 0.89-0.93) —
   consistent with the measured parity band (E2c 1.03±0.10, 0.63, 0.64):
   the selector now agrees with the measurement it used to contradict.

## v6 measurement queue (from the map's own uncertainty)

- vres16k draft-side plumbing + e2e at dense b32/16k (priced LCB 1.99,
  median 2.26 — the new headline candidate)
- ngram e2e on V2-Lite (priced LCB ~2.45; one harness config if
  W7_SPEC_METHOD passes through)
- flr50+q_fp8 at moe b4/16k (priced LCB 1.15 on assumed ctx scaling)
