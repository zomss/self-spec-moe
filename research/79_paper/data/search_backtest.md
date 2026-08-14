# Profiling-budget backtest — MLA as the new architecture

Ground truth = full-information pricing (measured beta/R).
Regret = true speedup of the chosen config vs the true best
(chosen config re-priced with full information).

| level | GPU-min | winners match | near-opt (reg<=0.02) | OFF/ON match | median regret | max regret |
|---|---|---|---|---|---|---|
| L0 | 0 | 1/9 | 1/9 | 4/9 | +0.077 | +0.223 |
| L1 | 5 | 1/9 | 1/9 | 4/9 | +0.077 | +0.223 |
| L2 | 26 | 3/9 | 3/9 | 4/9 | +0.057 | +0.213 |
| L3 | 46 | 3/9 | 3/9 | 5/9 | +0.064 | +0.213 |
| L4 | 91 | 7/9 | 8/9 | 8/9 | +0.000 | +0.026 |
| FULL | 305 | 9/9 | 9/9 | 9/9 | +0.000 | +0.000 |

## Per-cell detail

### L0

| cell | predicted | true | regret |
|---|---|---|---|
| b4/2k | OFF 1.00x | q_fp8 1.08x | +0.083 |
| b4/16k | OFF 1.00x | skip125 1.02x | +0.024 |
| b4/32k | win512 1.01x | q_fp8 1.07x | +0.223 |
| b8/2k | OFF 1.00x | skip125 1.02x | +0.022 |
| b8/16k | win512 1.01x | skip125 1.03x | +0.077 |
| b8/32k | win512 1.02x | win512 1.05x | +0.000 |
| b32/2k | OFF 1.00x | q_fp8 1.07x | +0.068 |
| b32/16k | win512 1.02x | q_fp8 1.13x | +0.106 |
| b32/32k | win512 1.04x | q_fp8 1.04x | +0.121 |

### L1

| cell | predicted | true | regret |
|---|---|---|---|
| b4/2k | OFF 1.00x | q_fp8 1.08x | +0.083 |
| b4/16k | OFF 1.00x | skip125 1.02x | +0.024 |
| b4/32k | win512 1.01x | q_fp8 1.07x | +0.223 |
| b8/2k | OFF 1.00x | skip125 1.02x | +0.022 |
| b8/16k | win512 1.01x | skip125 1.03x | +0.077 |
| b8/32k | win512 1.02x | win512 1.05x | +0.000 |
| b32/2k | OFF 1.00x | q_fp8 1.07x | +0.068 |
| b32/16k | win512 1.02x | q_fp8 1.13x | +0.106 |
| b32/32k | win512 1.04x | q_fp8 1.04x | +0.121 |

### L2

| cell | predicted | true | regret |
|---|---|---|---|
| b4/2k | skip125 1.02x | q_fp8 1.08x | +0.064 |
| b4/16k | skip125 1.02x | skip125 1.02x | +0.000 |
| b4/32k | skip125 1.03x | q_fp8 1.07x | +0.074 |
| b8/2k | skip125 1.02x | skip125 1.02x | +0.000 |
| b8/16k | skip125 1.03x | skip125 1.03x | +0.000 |
| b8/32k | skip125 1.03x | win512 1.05x | +0.052 |
| b32/2k | q_fp8+skip125 1.04x | q_fp8 1.07x | +0.057 |
| b32/16k | q_fp8+skip125 1.07x | q_fp8 1.13x | +0.103 |
| b32/32k | q_fp8+skip125 1.07x | q_fp8 1.04x | +0.213 |

### L3

| cell | predicted | true | regret |
|---|---|---|---|
| b4/2k | skip125 1.02x | q_fp8 1.08x | +0.064 |
| b4/16k | skip125 1.02x | skip125 1.02x | +0.000 |
| b4/32k | OFF 1.00x | q_fp8 1.07x | +0.068 |
| b8/2k | skip125 1.02x | skip125 1.02x | +0.000 |
| b8/16k | skip125 1.03x | skip125 1.03x | +0.000 |
| b8/32k | q_fp8 1.00x | win512 1.05x | +0.104 |
| b32/2k | q_fp8+skip125 1.04x | q_fp8 1.07x | +0.057 |
| b32/16k | q_fp8+skip125 1.07x | q_fp8 1.13x | +0.103 |
| b32/32k | q_fp8+skip125 1.03x | q_fp8 1.04x | +0.213 |

### L4

| cell | predicted | true | regret |
|---|---|---|---|
| b4/2k | q_fp8 1.06x | q_fp8 1.08x | +0.000 |
| b4/16k | skip125 1.02x | skip125 1.02x | +0.000 |
| b4/32k | q_fp8 1.05x | q_fp8 1.07x | +0.000 |
| b8/2k | q_fp8 1.01x | skip125 1.02x | +0.004 |
| b8/16k | q_fp8 1.00x | skip125 1.03x | +0.026 |
| b8/32k | win512 1.05x | win512 1.05x | +0.000 |
| b32/2k | q_fp8 1.05x | q_fp8 1.07x | +0.000 |
| b32/16k | q_fp8 1.10x | q_fp8 1.13x | +0.000 |
| b32/32k | q_fp8 1.02x | q_fp8 1.04x | +0.000 |

### FULL

| cell | predicted | true | regret |
|---|---|---|---|
| b4/2k | q_fp8 1.08x | q_fp8 1.08x | +0.000 |
| b4/16k | skip125 1.02x | skip125 1.02x | +0.000 |
| b4/32k | q_fp8 1.07x | q_fp8 1.07x | +0.000 |
| b8/2k | skip125 1.02x | skip125 1.02x | +0.000 |
| b8/16k | skip125 1.03x | skip125 1.03x | +0.000 |
| b8/32k | win512 1.05x | win512 1.05x | +0.000 |
| b32/2k | q_fp8 1.07x | q_fp8 1.07x | +0.000 |
| b32/16k | q_fp8 1.13x | q_fp8 1.13x | +0.000 |
| b32/32k | q_fp8 1.04x | q_fp8 1.04x | +0.000 |
