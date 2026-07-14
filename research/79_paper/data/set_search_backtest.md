# Set-search backtest — prior-guided allocation over the measured pool

Pool: 17 measured sets; prior = product of the
24 leave-one-out singles. Regret = pool-best beta minus chosen
beta. profile-solve = prior argmax with zero measurements.


## budget 3 — pool 6, best beta 0.855, prior-vs-measured Spearman rho=0.94

| m measured | prior-guided regret | random regret (exp.) |
|---|---|---|
| 0 | +0.006 | +0.086 |
| 1 | +0.006 | +0.086 |
| 2 | +0.000 | +0.026 |
| … | prior-guided CONVERGED at m=2 (2×45 s ≈ 1 min) | |

| set | prior | measured |
|---|---|---|
| {2,5,6} | 0.857 | 0.849 |
| {2,4,5} | 0.855 | 0.855 |
| {4,12,19} | 0.776 | 0.819 |
| {3,11,20} | 0.747 | 0.779 |
| {8,17,23} | 0.714 | 0.753 |
| {20,22,24} | 0.604 | 0.557 |

## budget 5 — pool 5, best beta 0.758, prior-vs-measured Spearman rho=0.80

| m measured | prior-guided regret | random regret (exp.) |
|---|---|---|
| 0 | +0.001 | +0.158 |
| 1 | +0.001 | +0.158 |
| 2 | +0.000 | +0.047 |
| … | prior-guided CONVERGED at m=2 (2×45 s ≈ 1 min) | |

| set | prior | measured |
|---|---|---|
| {2,3,4,5,6} | 0.758 | 0.757 |
| {2,4,5,6,12} | 0.758 | 0.758 |
| {3,7,13,18,22} | 0.602 | 0.572 |
| {6,10,15,19,24} | 0.589 | 0.616 |
| {19,21,22,23,25} | 0.455 | 0.296 |

## budget 7 — pool 6, best beta 0.507, prior-vs-measured Spearman rho=0.60

| m measured | prior-guided regret | random regret (exp.) |
|---|---|---|
| 0 | +0.000 | +0.111 |
| 1 | +0.000 | +0.111 |
| … | prior-guided CONVERGED at m=1 (1×45 s ≈ 0 min) | |

| set | prior | measured |
|---|---|---|
| {2,3,4,5,6,7,12} | 0.663 | 0.507 |
| {2,5,7,9,12,14,16} | 0.622 | 0.435 |
| {2,4,8,11,15,19,23} | 0.521 | 0.352 |
| {2,6,9,13,16,20,24} | 0.507 | 0.395 |
| {3,6,10,14,17,21,24} | 0.506 | 0.464 |
| {17,19,20,21,22,23,24} | 0.355 | 0.222 |
