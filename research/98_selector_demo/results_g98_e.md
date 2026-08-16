# G98-E (D3): the selector reaches 95% of omniscient — and barely beats one static

First record of the completed D3 campaign, 2026-08-16, on h103 (bare metal)
GPU 7 / NUMA 1. 62 boots: 31 configurations x 2 instrument arms, each boot
measuring all six regimes. Numbers from `data/g98_e/d3_result.json`.
Interpretation against `w98_prereg.md` §D3 belongs to the researcher; this
records what was measured.

## Verdict: one clause passes, the other depends on a reading

| clause | corrected arm | legacy arm |
| --- | --- | --- |
| >= 90% of the omniscient composite | **95.1%** PASS | **94.8%** PASS |
| beats every static by +2% (all 31 configurations) | **FAIL** (one) | **FAIL** (one) |
| beats every static by +2% (single-lever + OFF) | **PASS** (1.141x) | **PASS** (1.154x) |
| selector over static-OFF | **1.371x** | **1.311x** |

Both amendment-2 arms agree on every clause, so the result is not an
instrument artifact. Both are reported per the registered rule.

**The ambiguity is real and load-bearing.** §D3 requires the LCB to beat
"every static single configuration including OFF". Read as *any one fixed
configuration*, the selector fails: `w4a16-quantized/w1024/skip4` held
constant across all six regimes scores 589.6 tok/s against the selector's
597.7 — a lead of only **+1.4%**, inside the +2% margin. Read as *every
single-LEVER configuration plus OFF* — the phase's term of art in "single-lever
profiles" — the selector passes comfortably at 1.141x over the best single
lever. The cell that defeats it is a three-lever composition, which is
exactly what separates the readings.

The stricter reading is recommended as the headline: it is conservative, and
it asks the question a deployment actually faces — *is per-regime selection
worth more than picking one good configuration and leaving it alone?*

## What the selector got wrong

It differs from the omniscient pick in **5 of 6 regimes**, so this is a real
test rather than a relabelled ceiling. Its one large miss is decisive:

| regime | selector's pick | measured | omniscient pick | measured |
| --- | --- | --- | --- | --- |
| R1 | w4a16/woff/skip4 | 178.2 | w4a16/w1024/skip8 | 178.6 |
| **R4** | **w4a16/woff/skip0** | **522.3** | **off** | **684.9** |
| R5 | w4a16/w512/skip4 | 1007.5 | w4a16/w1024/skip0 | 1016.2 |
| R5cot | w4a16/w512/skip4 | 1144.2 | w4a16/w512/skip4 | 1144.2 |
| R6 | w4a16/woff/skip8 | 4203.0 | w4a16/w256/skip4 | 4575.4 |
| R8 | w4a16/woff/skip4 | 2450.6 | w4a16/w512/skip4 | 2452.8 |

**At R4 speculation loses outright and the selector armed anyway** — 522
against OFF's 685, a 24% loss on that regime. Four of its other five misses
cost under 1%. So the selector's error is not diffuse mis-ranking; it is a
single failure to fail closed, in the regime where the cost model was
already weakest (R4 carried the largest fitted `F`, and its predicted margin
over OFF was the thinnest in the map at 1.03x). That is a C3-shaped finding
arriving in D3: the selector needed to know *when not to arm*, and its
predicted margin was already telling it so.

## The frontier: where per-regime selection is worth anything

Sweeping each regime's share from 0 to 1 over 126 mixes:

* **the 90% target holds in 112 of 126 mixes (89%)** — robust to the mix;
* **the selector beats every static by +2% in only 9 of 126 (7%)**.

Concentration matters in the direction the misses predict. As R4's share
rises the fraction falls from 0.995 to **0.763** (the armed pick loses to
OFF there); as R1's share rises it climbs from 0.891 to 0.998; R6 is nearly
flat, 0.951 to 0.919.

**This is the number that sizes Phase 97.** Per-regime selection is close to
optimal, but on almost every mix a single well-chosen static configuration
lands within 2% of it. That is the Phase-82 dwell-time law restated in a
different phase's currency: switching pays only where the mix makes regimes
differ enough to matter, and on this workload family it mostly does not.
Before more engineering goes into a runtime switching engine, this says the
gain being bought is small.

## The instrument moves which static wins

Amendment 2's dual arm earned its cost here. The two instruments do not just
shift magnitudes — they change **which configuration is the best static**:
`w4a16/w1024/skip4` under the corrected instrument, `w4a16/w512/skip4` under
the legacy one. The verdict is unchanged either way, but a single-arm run
would have reported a different best-static baseline with no way to know it
was instrument-dependent.

## Protocol

* **Decode currency only** — committed tokens over summed decode step time.
  Never mixed with wall clock (Phase 96 I1 retracted a claim to exactly that).
* **Equal work** — `ignore_eos`, fixed budget; every configuration emits
  identical committed-token counts per regime, verified in the grid.
* **Aggregation is time-weighted**, `1 / sum(v_R / rate_R)`, not an
  arithmetic mean of rates. Measured OFF rates span 125 tok/s (R1, batch 1)
  to 3448 (R6, batch 32), so the two differ by ~7x and the verdict would
  follow whichever was used.
* **Prediction barrier** — the selector's picks came from
  `d3_predictions.json` (digest `45eddcdf933e0dfb`), built from Round-2's
  fitted cost model and G98-D's acceptance read at the deployed depth K=4,
  and committed before the grid was scored. Acceptance was measured on
  content seeds 4-5 while D3 evaluates on seeds 2-3, so the selector
  predicted unseen content.
* Two scorer defects were found and fixed before the verdict: the arithmetic
  aggregation above, and a silent fallback that substituted the measured
  optimum when the prediction map failed to load, reporting a vacuous 100%.
  Both are recorded in the commit history; the second is why the scorer now
  refuses rather than repairs.
