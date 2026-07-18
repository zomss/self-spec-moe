# Strategy map v6.3 — Humming-W4A8 reprice (Phase 87 fold)

Delta vs v6: dense gains q_int4+win512+A8humming (kernel-factor transfer, ckpt unbuilt -> queue-gated); new measured Qwen3-8B column. Winner = argmax LCB05.


## dense (Qwen2.5-7B, repriced)

| cell | winner (LCB05/med/P) | runner-up | note |
|---|---|---|---|
| b1/2k | **q_int4** 1.21/1.29/P0.23 | q_int4+win512 1.14 | GPTQ; measR/cell |
| b1/16k | **q_int4+win512** 1.45/1.77/P0.46 | q_int4+win512+vres32kC4 1.31 | measR combo / w4xwinfac |
| b1/32k | **q_int4+win512** 1.78/2.36/P0.50 | q_int4+win512+A8humming 1.50 | measR combo / w4xwinfac |
| b8/2k | **q_int4** 1.22/1.30/P0.21 | q_int4+win512 1.16 | GPTQ; measR/cell |
| b8/16k | **q_int4+win512** 1.47/1.59/P0.40 [CONTESTED] | q_int4+win512+A8humming 1.39 | measR combo / w4xwinfac |
| b8/32k | **q_int4+win512** 1.62/2.14/P0.41 [CONTESTED] | q_int4+win512+A8humming 1.58 | measR combo / w4xwinfac |
| b32/2k | **q_int4** 1.10/1.17/P0.09 | q_int4+win512 1.06 | GPTQ; measR/cell |
| b32/16k | **q_int4+win512** 1.80/1.97/P0.23 [CONTESTED] | q_int4+win512+A8humming 1.76 | measR combo / w4xwinfac |
| b32/32k | INFEASIBLE (residency) | | |

## dense_q38b (Qwen3-8B, measured column, 16k)

| cell | winner (LCB05/med/P) | runner-up | rest |
|---|---|---|---|
| b1/16k | **w4win** 1.37/1.42/P0.53 | w4ffn25win 1.25/P0.03 | w4a8win_humming* 1.13 |
| b8/16k | **w4a8win_humming** 1.84/1.90/P0.94 | w4win 1.75/P0.06 | w4a8win_cutlass 1.55; w4ffn25win 1.46 |
| b16/16k | **w4a8win_humming** 2.12/2.19/P1.00 | w4win 1.73/P0.00 | w4a8win_cutlass 1.72; w4ffn25win 1.48 |

## measurement queue (emitted)

- Q2.5-7B W4A8 arm at b8/16k CONTESTED (gap 0.08 < model error; nominate-confirm rule)
- Q2.5-7B W4A8 arm at b8/32k CONTESTED (gap 0.04 < model error; nominate-confirm rule)
- Q2.5-7B W4A8 arm at b32/16k CONTESTED (gap 0.04 < model error; nominate-confirm rule)

Standing queue items: w4a8 b32 8B cell unmeasured. RESOLVED 2026-07-18: 32B W4A8 arm MEASURED -- b8 K5 1.22x vs w4win 1.28x (no flip; kernel factor ~1.05 at 32B/TP2 vs 0.945 at 8B -> scale/TP-dependent); b16/16k capacity-infeasible at TP2 (KV 185k < 262k, 7th residency incident).
