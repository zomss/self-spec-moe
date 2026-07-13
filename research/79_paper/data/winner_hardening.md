# Winner hardening — exhaustive combo search over ALL cells

Pricing: measured combo beta/R first; else product-law beta
(UPPER bound; measured exceptions are destructive) x term-edit R.
delivery = 1 (Phase-81 floor-free chain). kvq excluded (not
draft-only implementable). `!` = combo in the measured-destructive
family (mla, ctx>=32k, skip x context-lever) — its price is
optimistic by ~0.10-0.12 beta.


## dense (26 configs x gamma<=8 per cell)

### b1/2k — max 1.28x (q_int4) -> **argmax (see verdict table)**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| q_int4 | 1.28x | 3 | 0.932 | 0.604 | prodB/measR |
| q_fp8 | 1.27x | 6 | 0.984 | 0.709 | prodB/measR |
| q_int4+win512 | 1.26x | 3 | 0.925 | 0.615 | measB/modelR |
| q_int4+win128 | 1.24x | 3 | 0.917 | 0.615 | prodB/modelR |
| q_fp8+win512 | 1.22x | 4 | 0.977 | 0.732 | measB/modelR |
| q_fp8+win128 | 1.20x | 4 | 0.968 | 0.731 | prodB/modelR |
| win512 | 0.99x | 1 | 0.983 | 1.000 | prodB/measR |
| win128 | 0.99x | 1 | 0.984 | 1.003 | prodB/measR |

### b1/16k — max 1.28x (q_int4) -> **argmax (see verdict table)**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| q_int4 | 1.28x | 3 | 0.924 | 0.595 | prodB/measR |
| q_int4+win512 | 1.26x | 3 | 0.917 | 0.598 | measB/modelR |
| q_int4+win128 | 1.24x | 2 | 0.901 | 0.597 | prodB/modelR |
| q_fp8 | 1.24x | 5 | 0.978 | 0.720 | prodB/measR |
| q_fp8+win512 | 1.20x | 3 | 0.957 | 0.712 | prodB/modelR |
| q_fp8+win128 | 1.19x | 3 | 0.955 | 0.711 | prodB/modelR |
| win512 | 1.02x | 1 | 0.978 | 0.949 | prodB/measR |
| win128 | 1.01x | 1 | 0.976 | 0.952 | prodB/measR |

### b1/32k — max 1.31x (q_int4+win512) -> **argmax (see verdict table)**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| q_int4+win512 | 1.31x | 3 | 0.926 | 0.580 | measB/modelR |
| q_int4 | 1.28x | 3 | 0.933 | 0.607 | prodB/measR |
| q_int4+win128 | 1.27x | 3 | 0.907 | 0.579 | prodB/modelR |
| q_fp8 | 1.25x | 5 | 0.980 | 0.713 | prodB/measR |
| q_fp8+win512 | 1.25x | 4 | 0.968 | 0.690 | measB/modelR |
| q_fp8+win128 | 1.21x | 3 | 0.953 | 0.689 | prodB/modelR |
| win512 | 1.04x | 2 | 0.975 | 0.901 | prodB/measR |
| win128 | 1.04x | 2 | 0.972 | 0.905 | prodB/measR |

### b8/2k — max 1.30x (q_int4) -> **argmax (see verdict table)**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| q_int4 | 1.30x | 3 | 0.932 | 0.596 | prodB/measR |
| q_fp8 | 1.28x | 6 | 0.984 | 0.705 | prodB/measR |
| q_int4+win512 | 1.26x | 3 | 0.925 | 0.615 | measB/modelR |
| q_int4+win128 | 1.25x | 3 | 0.917 | 0.609 | prodB/modelR |
| q_fp8+win512 | 1.22x | 5 | 0.977 | 0.725 | measB/modelR |
| q_fp8+win128 | 1.21x | 4 | 0.968 | 0.720 | prodB/modelR |
| win512 | 1.02x | 1 | 0.983 | 0.952 | prodB/measR |
| win128 | 1.02x | 1 | 0.984 | 0.954 | prodB/measR |

### b8/16k — max 1.55x (q_int4+win512) -> **argmax (see verdict table)**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| q_int4+win512 | 1.55x | 4 | 0.917 | 0.435 | measB/measR |
| q_int4+win128 | 1.38x | 3 | 0.901 | 0.500 | prodB/modelR |
| q_fp8+win128 | 1.36x | 4 | 0.955 | 0.591 | prodB/modelR |
| q_fp8+win512 | 1.36x | 4 | 0.957 | 0.596 | prodB/modelR |
| win128 | 1.20x | 4 | 0.976 | 0.744 | prodB/measR |
| win512 | 1.20x | 4 | 0.978 | 0.750 | prodB/measR |
| q_int4 | 1.19x | 2 | 0.924 | 0.669 | prodB/measR |
| q_fp8 | 1.16x | 4 | 0.978 | 0.779 | prodB/measR |

### b8/32k — max 1.61x (q_int4+win512) -> **argmax (see verdict table)**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| q_int4+win512 | 1.61x | 5 | 0.926 | 0.421 | measB/modelR |
| q_fp8+win512 | 1.60x | 7 | 0.968 | 0.496 | measB/modelR |
| q_int4+win128 | 1.56x | 4 | 0.907 | 0.417 | prodB/modelR |
| q_fp8+win128 | 1.54x | 5 | 0.953 | 0.492 | prodB/modelR |
| win512 | 1.37x | 6 | 0.975 | 0.626 | prodB/measR |
| win128 | 1.36x | 5 | 0.972 | 0.623 | prodB/measR |
| q_int4 | 1.18x | 2 | 0.933 | 0.692 | prodB/measR |
| q_fp8 | 1.12x | 4 | 0.980 | 0.820 | prodB/measR |

### b32/2k — max 1.27x (q_int4+win128) -> **argmax (see verdict table)**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| q_int4+win128 | 1.27x | 3 | 0.917 | 0.593 | prodB/modelR |
| q_int4+win512 | 1.26x | 3 | 0.925 | 0.613 | measB/modelR |
| q_fp8+win128 | 1.25x | 4 | 0.968 | 0.688 | prodB/modelR |
| q_fp8+win512 | 1.25x | 5 | 0.977 | 0.707 | measB/modelR |
| q_fp8 | 1.17x | 5 | 0.984 | 0.787 | prodB/measR |
| q_int4 | 1.17x | 2 | 0.932 | 0.701 | prodB/measR |
| win128 | 1.11x | 4 | 0.984 | 0.845 | prodB/measR |
| win512 | 1.07x | 3 | 0.983 | 0.879 | prodB/measR |

### b32/16k — max 1.91x (q_int4+win512) -> **argmax (see verdict table)**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| q_int4+win512 | 1.91x | 6 | 0.917 | 0.311 | measB/measR |
| q_fp8+win128 | 1.81x | 7 | 0.955 | 0.395 | prodB/modelR |
| q_fp8+win512 | 1.79x | 7 | 0.957 | 0.407 | prodB/modelR |
| q_int4+win128 | 1.74x | 5 | 0.901 | 0.341 | prodB/modelR |
| win128 | 1.56x | 7 | 0.976 | 0.530 | prodB/measR |
| win512 | 1.55x | 8 | 0.978 | 0.541 | prodB/measR |
| q_fp8 | 1.11x | 3 | 0.978 | 0.830 | prodB/measR |
| q_int4 | 1.11x | 2 | 0.924 | 0.756 | prodB/measR |


## moe (53 configs x gamma<=8 per cell)

### b4/2k — max 1.04x (lr50+q_fp8) -> **OFF HOLDS**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| lr50+q_fp8 | 1.04x | 1 | 0.836 | 0.768 | measB/modelR |
| lr50+q_fp8+win512 | 1.03x | 1 | 0.816 | 0.761 | measB/modelR |
| lr50+win512 | 1.03x | 1 | 0.821 | 0.775 | measB/modelR |
| q_fp8 | 1.02x | 2 | 0.990 | 0.952 | prodB/measR |
| lr50+q_fp8+win128 | 1.02x | 1 | 0.786 | 0.759 | prodB/modelR |
| lr50+win128 | 1.01x | 1 | 0.794 | 0.773 | prodB/modelR |
| lr50 | 1.01x | 1 | 0.837 | 0.826 | prodB/measR |
| win128 | 0.99x | 1 | 0.949 | 0.965 | prodB/measR |

### b4/16k — max 1.05x (win128) -> **MARGINAL — measure**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| win128 | 1.05x | 2 | 0.969 | 0.883 | prodB/measR |
| lr50+q_fp8+win512 | 1.04x | 1 | 0.822 | 0.749 | measB/modelR |
| lr50+win512 | 1.04x | 1 | 0.824 | 0.761 | measB/modelR |
| lr50+q_fp8+win128 | 1.03x | 1 | 0.795 | 0.747 | prodB/modelR |
| lr50+win128 | 1.02x | 1 | 0.801 | 0.759 | prodB/modelR |
| win512 | 1.02x | 1 | 0.971 | 0.931 | prodB/measR |
| q_fp8+win512 | 1.02x | 1 | 0.965 | 0.932 | prodB/modelR |
| q_fp8+win128 | 1.02x | 1 | 0.962 | 0.931 | prodB/modelR |

### b4/32k — max 1.58x (win128) -> **argmax (see verdict table)**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| win128 | 1.58x | 8 | 0.978 | 0.528 | prodB/measR |
| q_fp8 | 1.38x | 8 | 0.990 | 0.659 | prodB/measR |
| win512 | 1.30x | 6 | 0.983 | 0.684 | prodB/measR |
| q_fp8+win512 | 1.06x | 3 | 0.982 | 0.891 | measB/modelR |
| lr50+q_fp8+win512 | 1.05x | 1 | 0.829 | 0.740 | measB/modelR |
| lr50+win512 | 1.05x | 1 | 0.830 | 0.749 | measB/modelR |
| q_fp8+win128 | 1.05x | 2 | 0.968 | 0.889 | prodB/modelR |
| lr50+q_fp8+win128 | 1.04x | 1 | 0.800 | 0.739 | prodB/modelR |

### b8/2k — max 1.06x (q_fp8) -> **MARGINAL — measure**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| q_fp8 | 1.06x | 4 | 0.990 | 0.904 | prodB/measR |
| lr50+q_fp8 | 1.05x | 1 | 0.836 | 0.750 | measB/modelR |
| lr50+q_fp8+win512 | 1.05x | 1 | 0.816 | 0.737 | measB/modelR |
| lr50+q_fp8+win128 | 1.03x | 1 | 0.786 | 0.734 | prodB/modelR |
| q_fp8+win512 | 1.01x | 1 | 0.949 | 0.927 | measB/modelR |
| lr50+win512 | 1.01x | 1 | 0.821 | 0.807 | measB/modelR |
| q_fp8+win128 | 1.01x | 1 | 0.939 | 0.924 | prodB/modelR |
| lr50 | 1.01x | 1 | 0.837 | 0.826 | prodB/measR |

### b8/16k — max 1.27x (win512) -> **argmax (see verdict table)**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| win512 | 1.27x | 5 | 0.971 | 0.677 | prodB/measR |
| win128 | 1.16x | 3 | 0.969 | 0.763 | prodB/measR |
| lr50+q_fp8+win512 | 1.13x | 1 | 0.822 | 0.617 | measB/measR |
| lr50+win512 | 1.09x | 1 | 0.824 | 0.674 | measB/measR |
| q_fp8 | 1.07x | 5 | 0.993 | 0.902 | prodB/measR |
| q_fp8+win512 | 1.06x | 2 | 0.965 | 0.862 | prodB/modelR |
| q_fp8+win128 | 1.06x | 2 | 0.962 | 0.860 | prodB/modelR |
| lr50+q_fp8+win128 | 1.04x | 1 | 0.795 | 0.723 | prodB/modelR |

### b8/32k — max 1.37x (win512) -> **argmax (see verdict table)**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| win512 | 1.37x | 7 | 0.983 | 0.645 | prodB/measR |
| win128 | 1.33x | 6 | 0.978 | 0.658 | prodB/measR |
| q_fp8+win512 | 1.12x | 4 | 0.982 | 0.821 | measB/modelR |
| q_fp8+win128 | 1.10x | 3 | 0.968 | 0.820 | prodB/modelR |
| lr50+q_fp8+win512 | 1.06x | 1 | 0.829 | 0.719 | measB/modelR |
| lr50+q_fp8+win128 | 1.05x | 1 | 0.800 | 0.717 | prodB/modelR |
| lr50+win512 | 1.04x | 1 | 0.830 | 0.757 | measB/modelR |
| lr50+win128 | 1.03x | 1 | 0.808 | 0.755 | prodB/modelR |

### b32/2k — max 1.07x (q_fp8+win512) -> **MARGINAL — measure**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| q_fp8+win512 | 1.07x | 2 | 0.949 | 0.825 | measB/modelR |
| lr50+q_fp8+win512 | 1.07x | 1 | 0.816 | 0.694 | measB/modelR |
| q_fp8+win128 | 1.07x | 2 | 0.939 | 0.816 | prodB/modelR |
| q_fp8 | 1.06x | 4 | 0.990 | 0.900 | prodB/measR |
| lr50+q_fp8 | 1.06x | 1 | 0.836 | 0.728 | measB/modelR |
| lr50+q_fp8+win128 | 1.06x | 1 | 0.786 | 0.685 | prodB/modelR |
| lr25+q_fp8+win128 | 1.00x | 1 | 0.679 | 0.685 | prodB/modelR |
| lr50 | 0.99x | 1 | 0.837 | 0.847 | prodB/measR |

### b32/16k — max 1.44x (win128) -> **argmax (see verdict table)**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| win128 | 1.44x | 6 | 0.969 | 0.571 | prodB/measR |
| win512 | 1.32x | 5 | 0.971 | 0.647 | prodB/measR |
| q_fp8+win512 | 1.16x | 3 | 0.965 | 0.759 | prodB/modelR |
| q_fp8+win128 | 1.16x | 3 | 0.962 | 0.754 | prodB/modelR |
| lr50+q_fp8+win512 | 1.07x | 1 | 0.822 | 0.696 | measB/modelR |
| lr50+q_fp8+win128 | 1.06x | 1 | 0.795 | 0.691 | prodB/modelR |
| lr50 | 1.05x | 1 | 0.826 | 0.737 | prodB/measR |
| lr50+win512 | 1.03x | 1 | 0.824 | 0.765 | measB/modelR |

### b32/32k — max 2.22x (win512) -> **argmax (see verdict table)**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| win512 | 2.22x | 8 | 0.983 | 0.348 | prodB/measR |
| win128 | 2.01x | 8 | 0.978 | 0.389 | prodB/measR |
| lr50+q_fp8+win512 | 1.50x | 3 | 0.829 | 0.353 | measB/measR |
| lr50+win512 | 1.49x | 3 | 0.830 | 0.357 | measB/measR |
| q_fp8+win512 | 1.23x | 5 | 0.982 | 0.736 | measB/modelR |
| q_fp8+win128 | 1.19x | 4 | 0.968 | 0.733 | prodB/modelR |
| lr50+q_fp8+win128 | 1.06x | 1 | 0.800 | 0.693 | prodB/modelR |
| lr50+win128 | 1.04x | 1 | 0.808 | 0.737 | prodB/modelR |


## mla (53 configs x gamma<=8 per cell)

### b4/2k — max 1.08x (q_fp8) -> **MARGINAL — measure**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| q_fp8 | 1.08x | 6 | 0.999 | 0.895 | prodB/measR |
| win512 | 1.06x | 2 | 0.967 | 0.871 | prodB/measR |
| skip125 | 1.02x | 1 | 0.966 | 0.930 | prodB/modelR |
| lr50+skip125 | 1.00x | 1 | 0.939 | 0.947 | measB/modelR |
| lr50+skip125+win512 | 0.99x | 1 | 0.930 | 0.945 | prodB/modelR |
| lr50 | 0.99x | 1 | 0.995 | 1.020 | prodB/modelR |
| lr25+skip125 | 0.99x | 1 | 0.922 | 0.947 | prodB/modelR |
| skip125+win512 | 0.98x | 1 | 0.899 | 0.928 | measB/modelR |

### b4/16k — max 1.02x (skip125) -> **OFF HOLDS**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| skip125 | 1.02x | 1 | 0.966 | 0.921 | prodB/modelR |
| lr50+skip125 | 1.00x | 1 | 0.942 | 0.935 | prodB/modelR |
| skip125+win512 | 0.99x | 1 | 0.891 | 0.901 | prodB/modelR |
| lr25+skip125 | 0.98x | 1 | 0.904 | 0.935 | prodB/modelR |
| lr50 | 0.98x | 1 | 0.975 | 1.016 | prodB/modelR |
| lr50+skip125+win512 | 0.98x | 1 | 0.868 | 0.915 | prodB/modelR |
| q_fp8+skip125 | 0.96x | 1 | 0.958 | 1.037 | measB/modelR |
| lr25 | 0.96x | 1 | 0.936 | 1.016 | prodB/modelR |

### b4/32k — max 1.07x (q_fp8) -> **MARGINAL — measure**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| q_fp8 | 1.07x | 5 | 0.995 | 0.909 | prodB/measR |
| skip125 | 0.99x | 1 | 0.902 | 0.913 | prodB/modelR |
| lr50 | 0.97x | 1 | 0.954 | 1.014 | prodB/modelR |
| q_fp8+skip125 | 0.95x | 1 | 0.902 | 1.011 | measB/modelR |
| lr50+q_fp8 | 0.93x | 1 | 0.953 | 1.092 | measB/modelR |
| lr25 | 0.93x | 1 | 0.878 | 1.014 | prodB/modelR |
| skip25 | 0.93x | 1 | 0.701 | 0.827 | prodB/modelR |
| lr25+skip125 | 0.93x | 1 | 0.792 | 0.925 | prodB/modelR |

### b8/2k — max 1.02x (skip125) -> **OFF HOLDS**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| skip125 | 1.02x | 1 | 0.966 | 0.924 | prodB/modelR |
| q_fp8 | 1.02x | 3 | 0.999 | 0.967 | prodB/measR |
| lr50+skip125 | 1.00x | 1 | 0.939 | 0.944 | measB/modelR |
| lr50+skip125+win512 | 0.99x | 1 | 0.930 | 0.940 | prodB/modelR |
| skip125+win512 | 0.99x | 1 | 0.899 | 0.920 | measB/modelR |
| lr25+skip125 | 0.99x | 1 | 0.922 | 0.944 | prodB/modelR |
| lr50 | 0.99x | 1 | 0.995 | 1.023 | prodB/modelR |
| lr25+skip125+win512 | 0.97x | 1 | 0.891 | 0.940 | prodB/modelR |

### b8/16k — max 1.03x (skip125) -> **OFF HOLDS**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| skip125 | 1.03x | 1 | 0.966 | 0.911 | prodB/modelR |
| lr50+skip125 | 1.01x | 1 | 0.942 | 0.926 | prodB/modelR |
| skip125+win512 | 1.01x | 1 | 0.891 | 0.880 | prodB/modelR |
| q_fp8 | 1.00x | 1 | 1.000 | 0.990 | prodB/measR |
| lr25+skip125 | 0.99x | 1 | 0.904 | 0.926 | prodB/modelR |
| lr50+skip125+win512 | 0.99x | 1 | 0.868 | 0.895 | prodB/modelR |
| lr50 | 0.98x | 1 | 0.975 | 1.017 | prodB/modelR |
| lr50+q_fp8+skip125 | 0.98x | 1 | 0.942 | 0.985 | prodB/modelR |

### b8/32k — max 1.05x (win512) -> **MARGINAL — measure**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| win512 | 1.05x | 1 | 0.813 | 0.725 | prodB/measR |
| skip125 | 1.00x | 1 | 0.902 | 0.903 | prodB/modelR |
| q_fp8+skip125 | 0.97x | 1 | 0.902 | 0.954 | measB/modelR |
| lr50 | 0.97x | 1 | 0.954 | 1.013 | prodB/modelR |
| lr50+q_fp8 | 0.95x | 1 | 0.953 | 1.046 | measB/modelR |
| q_fp8 | 0.95x | 1 | 0.995 | 1.106 | prodB/measR |
| skip25 | 0.94x | 1 | 0.701 | 0.805 | prodB/modelR |
| lr25+skip125 | 0.94x | 1 | 0.792 | 0.914 | prodB/modelR |

### b32/2k — max 1.07x (q_fp8) -> **MARGINAL — measure**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| q_fp8 | 1.07x | 5 | 0.999 | 0.909 | prodB/measR |
| skip125 | 1.03x | 1 | 0.966 | 0.913 | prodB/modelR |
| lr50+skip125+win512 | 1.01x | 1 | 0.930 | 0.905 | prodB/modelR |
| q_fp8+skip125 | 1.01x | 1 | 0.956 | 0.934 | measB/modelR |
| lr50+skip125 | 1.01x | 1 | 0.939 | 0.917 | measB/modelR |
| q_fp8+skip125+win512 | 1.01x | 1 | 0.934 | 0.921 | prodB/modelR |
| lr50+q_fp8+skip125+win512 | 1.00x | 1 | 0.929 | 0.923 | prodB/modelR |
| lr25+skip125 | 1.00x | 1 | 0.922 | 0.917 | prodB/modelR |

### b32/16k — max 1.13x (q_fp8) -> **MARGINAL — measure**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| q_fp8 | 1.13x | 7 | 1.000 | 0.850 | prodB/measR |
| skip125 | 1.04x | 2 | 0.966 | 0.893 | prodB/modelR |
| skip125+win512 | 1.03x | 1 | 0.891 | 0.831 | prodB/modelR |
| q_fp8+skip125 | 1.03x | 1 | 0.958 | 0.903 | measB/modelR |
| q_fp8+skip125+win512 | 1.03x | 1 | 0.891 | 0.841 | prodB/modelR |
| win512 | 1.03x | 1 | 0.922 | 0.873 | prodB/measR |
| lr50+skip125 | 1.02x | 1 | 0.942 | 0.895 | prodB/modelR |
| lr50+q_fp8+skip125 | 1.02x | 1 | 0.942 | 0.904 | prodB/modelR |

### b32/32k — max 1.04x (q_fp8) -> **OFF HOLDS**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| q_fp8 | 1.04x | 4 | 0.995 | 0.941 | prodB/measR |
| skip125 | 1.01x | 1 | 0.902 | 0.886 | prodB/modelR |
| lr50+q_fp8 | 0.98x | 1 | 0.953 | 1.000 | measB/modelR |
| lr50 | 0.98x | 1 | 0.954 | 1.002 | prodB/modelR |
| skip25 | 0.96x | 1 | 0.701 | 0.772 | prodB/modelR |
| q_fp8+skip125+win512! | 0.95x | 1 | 0.730 | 0.814 | prodB/modelR |
| q_fp8+win512 | 0.95x | 1 | 0.818 | 0.910 | measB/modelR |
| q_fp8+skip25 | 0.95x | 1 | 0.698 | 0.785 | prodB/modelR |


## Measurement queue (challengers above 1.05x optimistic)

- moe b32/32k: win512 priced 2.22x (gamma8, prodB/measR)
- dense b32/16k: q_int4+win512 priced 1.91x (gamma6, measB/measR)
- dense b8/32k: q_int4+win512 priced 1.61x (gamma5, measB/modelR)
- moe b4/32k: win128 priced 1.58x (gamma8, prodB/measR)
- dense b8/16k: q_int4+win512 priced 1.55x (gamma4, measB/measR)
- moe b32/16k: win128 priced 1.44x (gamma6, prodB/measR)
- moe b8/32k: win512 priced 1.37x (gamma7, prodB/measR)
- dense b1/32k: q_int4+win512 priced 1.31x (gamma3, measB/modelR)
- dense b8/2k: q_int4 priced 1.30x (gamma3, prodB/measR)
- dense b1/2k: q_int4 priced 1.28x (gamma3, prodB/measR)
- dense b1/16k: q_int4 priced 1.28x (gamma3, prodB/measR)
- moe b8/16k: win512 priced 1.27x (gamma5, prodB/measR)
- dense b32/2k: q_int4+win128 priced 1.27x (gamma3, prodB/modelR)
- mla b32/16k: q_fp8 priced 1.13x (gamma7, prodB/measR)
- mla b4/2k: q_fp8 priced 1.08x (gamma6, prodB/measR)
- moe b32/2k: q_fp8+win512 priced 1.07x (gamma2, measB/modelR)
- mla b4/32k: q_fp8 priced 1.07x (gamma5, prodB/measR)
- mla b32/2k: q_fp8 priced 1.07x (gamma5, prodB/measR)
- moe b8/2k: q_fp8 priced 1.06x (gamma4, prodB/measR)
- moe b4/16k: win128 priced 1.05x (gamma2, prodB/measR)
- mla b8/32k: win512 priced 1.05x (gamma1, prodB/measR)

## Verdict (2026-07-14): 26/26 winners CONFIRMED; menu extensions beta-gated

- Exhaustive argmax (extended sets, full class grid, gamma<=8) matches the
  registered map v4 winner at every cell, all three architectures. The
  added levers (skip25, lr25, and all their combos) never win a cell.
- Menu extension gate (beta_menu_ext.csv, dense ondist refs):
  * weight pruning DEAD by domination: 2:4 magnitude beta 0.416, 50%
    unstructured 0.600 -- vs int4 RTN (also uncalibrated) at MORE byte cut
    and beta 0.924. No calibrated prune can enter the menu without beating
    quant on both axes at once.
  * non-contiguous layer sets (KnapSpec's lever): leave-one-out profile
    beta 0.836-0.954 (early-mid droppable, late critical); greedy set
    {2,5,6} beta 0.849 vs contiguous skip125 0.448 -- SET SELECTION
    DOUBLES skip beta at equal budget (and 5.6x at budget 7: 0.507 vs
    0.09). Product law prices SHALLOW sets (budget 3: 0.849 vs 0.857
    predicted) and degrades super-multiplicatively with DEPTH regardless
    of adjacency -- the adjacency hypothesis was tested and REFUTED
    (non-adjacent 7-set 0.435 < adjacency-blind greedy 0.507; both far
    below their ~0.62-0.65 products). Depth, not contiguity, breaks
    composition; deep sets must be measured, not priced.
  * Even doubled, skip does not flip any dense winner (best composed use
    ~1.14x vs q_int4 1.28x) -- winners robust to the menu extension.
- Standing discretization caveats: window {128,512}, skip budgets {3,7},
  kvq excluded (implementability), gamma<=8, <=1 lever/class, no q_int4
  arm on moe/mla (no checkpoint; class muted by F2 parity).
