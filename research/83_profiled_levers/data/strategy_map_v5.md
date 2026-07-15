# Strategy map v5 — profiled levers (Phase 83)

q_int4 beta source: RTN 0.924 (E3 pending).
Profiled columns: dense ls3 0.869; moe flr50 0.9531 / flr25
0.8229 / skip6p 0.742. mla unchanged. delivery=1 pricing;
e2e verdicts noted where measured (E2b/E2c).


## dense

| cell | v5 winner | speedup* | v4 | note |
|---|---|---|---|---|
| b1/2k | **q_int4** | 1.28× | q_int4 1.28 |  |
| b1/16k | **q_int4** | 1.28× | q_int4 1.28 |  |
| b1/32k | **q_int4+win512** | 1.31× | q_int4+win512 1.31 |  |
| b8/2k | **q_int4** | 1.30× | q_int4 1.30 |  |
| b8/16k | **q_int4+win512** | 1.55× | q_int4+win512 1.55 |  |
| b8/32k | **q_int4+win512** | 1.61× | q_int4+win512 1.61 |  |
| b32/2k | **q_int4+win128** | 1.27× | q_int4+win128 1.27 |  |
| b32/16k | **q_int4+win512** | 1.91× | q_int4+win512 1.91 |  |
| b32/32k | (over-capacity) | | | |

## moe

| cell | v5 winner | speedup* | v4 | note |
|---|---|---|---|---|
| b4/2k | **flr50+q_fp8** (marginal/OFF region) | 1.12× | lr50+q_fp8 1.04 | FLIP. e2e DELIVERED 1.03x (E2c) |
| b4/16k | **flr50+q_fp8+win512** (marginal/OFF region) | 1.11× | win128 1.05 | FLIP.  |
| b4/32k | **win128** | 1.58× | win128 1.58 |  |
| b8/2k | **flr50+q_fp8** (marginal/OFF region) | 1.13× | q_fp8 1.06 | FLIP. e2e 0.63x -- chain-blocked (E2c) |
| b8/16k | **win512** | 1.27× | win512 1.27 |  |
| b8/32k | **win512** | 1.37× | win512 1.37 |  |
| b32/2k | **flr50+q_fp8** | 1.15× | q_fp8+win512 1.07 | FLIP. e2e 0.64x -- chain-blocked (E2c) |
| b32/16k | **win128** | 1.44× | win128 1.44 |  |
| b32/32k | **win512** | 2.22× | win512 2.22 |  |

## mla

| cell | v5 winner | speedup* | v4 | note |
|---|---|---|---|---|
| b4/2k | **q_fp8** (marginal/OFF region) | 1.08× | q_fp8 1.08 |  |
| b4/16k | **skip125** (marginal/OFF region) | 1.02× | skip125 1.02 |  |
| b4/32k | **q_fp8** (marginal/OFF region) | 1.07× | q_fp8 1.07 |  |
| b8/2k | **skip125** (marginal/OFF region) | 1.02× | skip125 1.02 |  |
| b8/16k | **skip125** (marginal/OFF region) | 1.03× | skip125 1.03 |  |
| b8/32k | **win512** (marginal/OFF region) | 1.05× | win512 1.05 |  |
| b32/2k | **q_fp8** (marginal/OFF region) | 1.07× | q_fp8 1.07 |  |
| b32/16k | **q_fp8** (marginal/OFF region) | 1.13× | q_fp8 1.13 |  |
| b32/32k | **q_fp8** (marginal/OFF region) | 1.04× | q_fp8 1.04 |  |

## Flips vs v4: 4

- moe b4/2k: lr50+q_fp8 1.04 -> **flr50+q_fp8 1.12** e2e DELIVERED 1.03x (E2c)
- moe b4/16k: win128 1.05 -> **flr50+q_fp8+win512 1.11** 
- moe b8/2k: q_fp8 1.06 -> **flr50+q_fp8 1.13** e2e 0.63x -- chain-blocked (E2c)
- moe b32/2k: q_fp8+win512 1.07 -> **flr50+q_fp8 1.15** e2e 0.64x -- chain-blocked (E2c)
