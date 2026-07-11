
## dense: strategy map  (winner | speedup* | gamma*)

| cell | d_fp8w8a8 | d_w4machete | d_w4marlin | d_win | WINNER | d_kvq |
|---|---|---|---|---|---|---|
| b1/2k | 1.19 (g3) | 1.11 (g2) | 1.24 (g3) | 0.97 (g1) | **w4marlin 1.24x** | 0.95 |
| b1/16k | 1.17 (g3) | 1.11 (g2) | 1.25 (g3) | 1.00 (g1) | **w4marlin 1.25x** | 0.95 |
| b1/32k | 1.18 (g3) | 1.10 (g2) | 1.24 (g3) | 1.02 (g1) | **w4marlin 1.24x** | 0.96 |
| b8/2k | 1.19 (g3) | 1.13 (g2) | 1.25 (g3) | 0.99 (g1) | **w4marlin 1.25x** | 0.96 |
| b8/16k | 1.12 (g2) | 1.07 (g1) | 1.17 (g2) | 1.13 (g2) | **w4marlin 1.17x** | 1.00 |
| b8/32k | 1.08 (g2) | 1.05 (g1) | 1.15 (g2) | 1.27 (g3) | **win 1.27x** | 1.03 |
| b32/2k | 1.11 (g2) | 1.06 (g1) | 1.14 (g2) | 1.03 (g1) | **w4marlin 1.14x** † | 0.98 |
| b32/16k | 1.07 (g2) | 1.05 (g1) | 1.09 (g2) | 1.40 (g4) | **win 1.40x** | 1.02 |

### dense: layer-skip break-even beta (P17: measured accept COLLAPSES)

| cell | d_skip25 beta>=1.0x | d_skip50 beta>=1.0x | d_skip25 beta to beat winner | d_skip50 beta to beat winner |
|---|---|---|---|---|
| b1/2k | 0.78 | 0.57 | >0.999 | 0.88 |
| b1/16k | - | 0.55 | - | 0.87 |
| b1/32k | 0.76 | 0.53 | 0.99 | 0.84 |
| b8/2k | - | 0.57 | - | 0.89 |
| b8/16k | - | 0.51 | - | 0.77 |
| b8/32k | - | 0.46 | - | 0.80 |
| b32/2k | 0.77 | 0.54 | 0.96 | 0.76 |
| b32/16k | - | 0.44 | - | 0.86 |

## moe: strategy map  (winner | speedup* | gamma*)

| cell | m_fp8block | m_fp8marlin | m_localroute | m_win | WINNER | m_kvq |
|---|---|---|---|---|---|---|
| b4/2k | 0.96 (g1) | 1.00 (g1) | 1.01 (g1) | 0.93 (g1) | **localroute 1.01x** | 0.91 |
| b4/16k | 1.02 (g1) | 0.98 (g1) | 1.01 (g1) | 1.00 (g1) | **fp8block 1.02x** | 0.99 |
| b4/32k | 1.16 (g3) | 1.25 (g3) | 1.03 (g1) | 1.20 (g3) | **fp8marlin 1.25x** | 1.15 |
| b8/2k | 0.97 (g1) | 1.02 (g1) | 1.01 (g1) | 0.98 (g1) | **fp8marlin 1.02x** | 0.92 |
| b8/16k | 0.98 (g1) | 1.03 (g1) | 0.99 (g1) | 1.21 (g3) | **win 1.21x** | 1.05 |
| b8/32k | 0.90 (g1) | 0.83 (g1) | 0.95 (g1) | 1.24 (g3) | **win 1.24x** | 1.01 |
| b32/2k | 0.96 (g1) | 1.03 (g1) | 1.00 (g1) | 0.97 (g1) | **fp8marlin 1.03x** † | 0.93 |
| b32/16k | 1.04 (g1) | 0.99 (g1) | 1.07 (g1) | 1.24 (g3) | **win 1.24x** | 0.98 |
| b32/32k | 1.01 (g1) | 1.01 (g1) | 1.01 (g1) | 1.90 (g6) | **win 1.90x** | 1.14 |

### moe: layer-skip break-even beta (P17: measured accept COLLAPSES)

| cell | m_skip50 beta>=1.0x | m_skip50 beta to beat winner |
|---|---|---|
| b4/2k | 0.64 | 0.66 |
| b4/16k | 0.63 | 0.66 |
| b4/32k | 0.44 | 0.76 |
| b8/2k | 0.57 | 0.61 |
| b8/16k | 0.51 | 0.80 |
| b8/32k | 0.49 | 0.81 |
| b32/2k | 0.56 | 0.60 |
| b32/16k | 0.52 | 0.84 |
| b32/32k | 0.44 | 0.99 |

**Legend**: speedup* = max over gamma of tau_beta(gamma)/(gamma*R+1) (P75-validated roofline; ~90% delivered at b1 dense, P75 E3). † = compute-bound cell -- formula optimistic there (P74: real high-batch self-spec lost; verify tokens are not free). kvq/bf16 columns are REFERENCE only (kvq is global, not draft-only; bf16 R=1 shows plain self-drafting never pays).
