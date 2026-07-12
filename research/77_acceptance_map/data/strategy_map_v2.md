# Strategy map v2 — measured R × measured β (77)


## dense

| cell | d_fp8w8a8 | d_skip25 | d_skip50 | d_w4machete | d_w4marlin | d_win | WINNER v2 | v1 winner | Δ |
|---|---|---|---|---|---|---|---|---|---|
| b1/2k | 1.27 | 0.61 | 0.65 | 1.14 | 1.28 | 0.99 | **w4marlin 1.28× (γ3)** | w4marlin 1.24× (γ3) |  |
| b1/16k | 1.24 | - | 0.67 | 1.12 | 1.28 | 1.02 | **w4marlin 1.28× (γ3)** | w4marlin 1.25× (γ3) |  |
| b1/32k | 1.25 | 0.63 | 0.67 | 1.13 | 1.28 | 1.04 | **w4marlin 1.28× (γ3)** | w4marlin 1.24× (γ3) |  |
| b8/2k | 1.28 | - | 0.66 | 1.15 | 1.30 | 1.02 | **w4marlin 1.30× (γ3)** | w4marlin 1.25× (γ3) |  |
| b8/16k | 1.16 | - | 0.68 | 1.08 | 1.19 | 1.20 | **win 1.20× (γ4)** | w4marlin 1.17× (γ2) | **WINNER CHANGED** |
| b8/32k | 1.12 | - | 0.71 | 1.06 | 1.18 | 1.37 | **win 1.37× (γ6)** | win 1.27× (γ3) | Δ+0.10× |
| b32/2k | 1.17 | 0.61 | 0.67 | 1.07 | 1.17 | 1.07 | **fp8w8a8 1.17× (γ5)** | w4marlin 1.14× (γ2) | **WINNER CHANGED** |
| b32/16k | 1.11 | - | 0.72 | 1.05 | 1.11 | 1.55 | **win 1.55× (γ8)** | win 1.40× (γ4) | Δ+0.15× |

_reference (not implementable draft-only today): kvq_fp8 composed = 0.87, 0.92, 0.80, 0.88, 0.97, 0.86, 0.90, 0.98 across cells_

## moe

| cell | m_fp8block | m_fp8marlin | m_localroute | m_skip50 | m_win | WINNER v2 | v1 winner | Δ |
|---|---|---|---|---|---|---|---|---|
| b4/2k | 0.98 | 1.02 | 0.94 | 0.70 | 0.93 | **fp8marlin 1.02× (γ2)** | localroute 1.01× (γ1) | **WINNER CHANGED** |
| b4/16k | 1.06 | 1.00 | 0.92 | 0.71 | 1.02 | **fp8block 1.06× (γ4)** | fp8block 1.02× (γ1) |  |
| b4/32k | 1.25 | 1.38 | 0.95 | 0.80 | 1.30 | **fp8marlin 1.38× (γ8)** | fp8marlin 1.25× (γ3) | Δ+0.13× |
| b8/2k | 0.99 | 1.06 | 0.94 | 0.73 | 0.98 | **fp8marlin 1.06× (γ4)** | fp8marlin 1.02× (γ1) |  |
| b8/16k | 1.00 | 1.07 | 0.91 | 0.77 | 1.27 | **win 1.27× (γ5)** | win 1.21× (γ3) |  |
| b8/32k | 0.92 | 0.85 | 0.88 | 0.77 | 1.37 | **win 1.37× (γ7)** | win 1.24× (γ3) | Δ+0.12× |
| b32/2k | 0.98 | 1.06 | 0.93 | 0.74 | 0.98 | **fp8marlin 1.06× (γ4)** | fp8marlin 1.03× (γ1) |  |
| b32/16k | 1.09 | 1.02 | 0.97 | 0.76 | 1.32 | **win 1.32× (γ5)** | win 1.24× (γ3) |  |
| b32/32k | 1.04 | 1.04 | 0.93 | 0.80 | 2.22 | **win 2.22× (γ8)** | win 1.90× (γ6) | Δ+0.32× |

_reference (not implementable draft-only today): kvq_fp8 composed = 0.95, 1.02, 1.27, 0.95, 1.10, 1.06, 0.96, 1.01, 1.25 across cells_


**Legend**: composed speedup* = max_γ τ_β(γ)/(γ·R+1); β measured (77, ctx-specific, greedy, β capped 0.995 for the geometric model); R measured (76 E1). v1 = borrowed-β map (76 E2). MLA awaits its cost tier.
