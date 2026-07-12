# Strategy map v4 — composed configurations (search over measured physics)


## dense

| cell | best config | speedup* | γ | source | v3 winner |
|---|---|---|---|---|---|
| b1/2k | **q_int4** | 1.28× | γ3 | prodβ/measR | w4marlin 1.28 |
| b1/16k | **q_int4** | 1.28× | γ3 | prodβ/measR |  |
| b1/32k | **q_int4+win512** | 1.31× | γ3 | measβ/modelR |  |
| b8/2k | **q_int4** | 1.30× | γ3 | prodβ/measR |  |
| b8/16k | **q_int4+win512** | 1.55× | γ4 | measβ/measR | win128 1.20 |
| b8/32k | **q_int4+win512** | 1.61× | γ5 | measβ/modelR |  |
| b32/2k | **q_int4+win128** | 1.27× | γ3 | prodβ/modelR |  |
| b32/16k | **q_int4+win512** | 1.91× | γ6 | measβ/measR | win128 1.56 |
| b32/32k | (bf16 over-capacity) | | | | |

## moe

| cell | best config | speedup* | γ | source | v3 winner |
|---|---|---|---|---|---|
| b4/2k | **lr50+q_fp8** | 1.04× | γ1 | measβ/modelR |  |
| b4/16k | **win128** | 1.05× | γ2 | prodβ/measR |  |
| b4/32k | **win128** | 1.58× | γ8 | prodβ/measR |  |
| b8/2k | **q_fp8** | 1.06× | γ4 | prodβ/measR |  |
| b8/16k | **win512** | 1.27× | γ5 | prodβ/measR | win 1.27 |
| b8/32k | **win512** | 1.37× | γ7 | prodβ/measR |  |
| b32/2k | **q_fp8+win512** | 1.07× | γ2 | measβ/modelR |  |
| b32/16k | **win128** | 1.44× | γ6 | prodβ/measR |  |
| b32/32k | **win512** | 2.22× | γ8 | prodβ/measR | win 2.22 |

## mla

| cell | best config | speedup* | γ | source | v3 winner |
|---|---|---|---|---|---|
| b4/2k | **q_fp8** | 1.08× | γ6 | prodβ/measR |  |
| b4/16k | **skip125** | 1.02× | γ1 | prodβ/modelR |  |
| b4/32k | **q_fp8** | 1.07× | γ5 | prodβ/measR |  |
| b8/2k | **skip125** | 1.02× | γ1 | prodβ/modelR |  |
| b8/16k | **skip125** | 1.03× | γ1 | prodβ/modelR |  |
| b8/32k | **win512** | 1.05× | γ1 | prodβ/measR |  |
| b32/2k | **q_fp8** | 1.07× | γ5 | prodβ/measR |  |
| b32/16k | **q_fp8** | 1.13× | γ7 | prodβ/measR | fp8marlin 1.13 |
| b32/32k | **q_fp8** | 1.04× | γ4 | prodβ/measR |  |

## Registered E3 candidates (composed cells worth an e2e check)

- dense b32/16k: **q_int4+win512** predicted 1.91× (γ6, measβ/measR)
- dense b8/32k: **q_int4+win512** predicted 1.61× (γ5, measβ/modelR)
- dense b8/16k: **q_int4+win512** predicted 1.55× (γ4, measβ/measR)

**Notes**: measβ/prodβ = measured-combo vs product-law β; measR/modelR = measured combo arm vs term-edit prediction (windowed-MoE modelR is conservative pending the h-term refinement). Deep-γ cells are rooflines (delivery ≤86% at γ6).
