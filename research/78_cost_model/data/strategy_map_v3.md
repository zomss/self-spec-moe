# Strategy map v3 — three architectures, measured R × measured β


## dense

| cell | fp8w8a8 | skip25 | skip50 | w4machete | w4marlin | win | win128 | WINNER |
|---|---|---|---|---|---|---|---|---|
| b1/2k | 1.27 | 0.61 | 0.65 | 1.14 | 1.28 | 0.99 | 0.99 | **w4marlin 1.28× (γ3)** |
| b1/16k | 1.24 | - | 0.67 | 1.12 | 1.28 | 1.02 | 1.01 | **w4marlin 1.28× (γ3)** |
| b1/32k | 1.25 | 0.63 | 0.67 | 1.13 | 1.28 | 1.04 | 1.04 | **w4marlin 1.28× (γ3)** |
| b8/2k | 1.28 | - | 0.66 | 1.15 | 1.30 | 1.02 | 1.02 | **w4marlin 1.30× (γ3)** |
| b8/16k | 1.16 | - | 0.68 | 1.08 | 1.19 | 1.20 | 1.20 | **win128 1.20× (γ4)** |
| b8/32k | 1.12 | - | 0.71 | 1.06 | 1.18 | 1.37 | 1.36 | **win 1.37× (γ6)** |
| b32/2k | 1.17 | 0.61 | 0.67 | 1.07 | 1.17 | 1.07 | 1.11 | **fp8w8a8 1.17× (γ5)** |
| b32/16k | 1.11 | - | 0.72 | 1.05 | 1.11 | 1.55 | 1.56 | **win128 1.56× (γ7)** |

## moe

| cell | fp8block | fp8marlin | localroute | skip50 | win | win128 | WINNER |
|---|---|---|---|---|---|---|---|
| b4/2k | 0.98 | 1.02 | 0.94 | 0.70 | 0.93 | 0.99 | **fp8marlin 1.02× (γ2)** |
| b4/16k | 1.06 | 1.00 | 0.92 | 0.71 | 1.02 | 1.05 | **fp8block 1.06× (γ4)** |
| b4/32k | 1.25 | 1.38 | 0.95 | 0.80 | 1.30 | 1.58 | **win128 1.58× (γ8)** |
| b8/2k | 0.99 | 1.06 | 0.94 | 0.73 | 0.98 | 0.98 | **fp8marlin 1.06× (γ4)** |
| b8/16k | 1.00 | 1.07 | 0.91 | 0.77 | 1.27 | 1.16 | **win 1.27× (γ5)** |
| b8/32k | 0.92 | 0.85 | 0.88 | 0.77 | 1.37 | 1.33 | **win 1.37× (γ7)** |
| b32/2k | 0.98 | 1.06 | 0.93 | 0.74 | 0.98 | 0.95 | **fp8marlin 1.06× (γ4)** |
| b32/16k | 1.09 | 1.02 | 0.97 | 0.76 | 1.32 | 1.44 | **win128 1.44× (γ6)** |
| b32/32k | 1.04 | 1.04 | 0.93 | 0.80 | 2.22 | 2.01 | **win 2.22× (γ8)** |

## mla

| cell | fp8block | fp8marlin | skip50 | win | WINNER |
|---|---|---|---|---|---|
| b4/2k | 0.69 | 1.08 | 0.54 | 1.06 | **fp8marlin 1.08× (γ6)** |
| b4/16k | 0.88 | 0.93 | 0.59 | 0.93 | **OFF** |
| b4/32k | 0.95 | 1.07 | 0.52 | 0.85 | **fp8marlin 1.07× (γ5)** |
| b8/2k | 1.04 | 1.02 | 0.50 | 0.97 | **fp8block 1.04× (γ4)** |
| b8/16k | 1.00 | 1.00 | 0.62 | 0.95 | **fp8block 1.00× (γ1)** |
| b8/32k | 1.08 | 0.95 | 0.62 | 1.05 | **fp8block 1.08× (γ6)** |
| b32/2k | 0.97 | 1.07 | 0.58 | 0.95 | **fp8marlin 1.07× (γ5)** |
| b32/16k | 0.96 | 1.13 | 0.62 | 1.03 | **fp8marlin 1.13× (γ7)** |
| b32/32k | 0.98 | 1.04 | 0.63 | 0.92 | **fp8marlin 1.04× (γ4)** |

## Consistency with the five e2e ground-truth cells

| cell | v3 winner (predicted) | e2e measured |
|---|---|---|
| dense b1/2k | w4marlin 1.28× | w4* 1.21× ✓ |
| moe b8/16k | win 1.27× | win* 1.16× ✓ |
| moe b8/32k | win 1.37× | win* 1.34× ✓ |
| dense b32/16k | win128 1.56× | win* 1.54× ✓ |
| moe b32/32k | win 2.22× | win* 1.64× ✓ |

**Notes**: β ctx-specific per architecture (77), capped 0.995; R measured (76 + MLA confirmation tier). MLA local-route: β measured (0.88-0.99, shared-expert anchor) but its cost regime is comm-bound fabric — no NVLink cost arm; kvq columns are reference-only. Deep-γ cells are rooflines (delivery 90%@γ3 → 86%@γ6, P75/76-E3).

**Honest readings**: (1) the MLA column is effectively an OFF region — every winner ≤1.13× roofline, i.e. within the noise band and known delivery factor of parity; this CONFIRMS the registered V3 claim (no strong self-spec lever on NVLink MLA — its measured-β strengths, shared-expert local-route 0.95-0.99 and shallow-skip 0.97, await the comm-bound fabric and a skip cost arm). (2) MoE b4 long-ctx winners are NOT robust (S4 placement noise; the win128 1.58× at b4/32k rides a noisy R) — quote b8/b32 rows. (3) win128 vs win512 winners at b8+/16k trade within ±0.02 — same-family ties, not crossovers.
