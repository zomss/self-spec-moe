# Round-2 model and lattice specification

Date: 2026-08-13
Status: proposal. Requires a **new preregistration** — it changes both the
registered model form and the fit set, so it is not an amendment.

## 1. Why Round 1 cannot absorb this

The registered form places every term inside the `keep_frac` product:

```text
D(quant, w, k) ~= (1 - k/L) * (W_bytes(quant)*kappa_w + KV_bytes(w)*kappa_kv + c0)
```

Two terms are missing, and both were measured in v6:

* **A non-layer cost `F`.** `W_bytes` includes the bf16 `lm_head` and
  embeddings — 2.508 GB of 16.400 (15.3%) for the bf16 draft, and 2.519 GB of
  6.100 (**41.3%**) for the quantized one, because quantizing the body leaves
  them a much larger share. None of it scales with skipped layers. Fitted from
  the three skip singles alone, `F` = **3.66 ms at R1**, 3.51 at R8, 2.59 at R6
  — 8–12% of the draft chain — and that two-parameter fit reproduces the
  held-out middle point to 0.1 ms. The kernel profile corroborates it:
  `nvjet_tst_384x8` at 412.8 us/call x 4 draft calls is the lm_head GEMM at 90%
  of bandwidth-bound.
* **A window overhead `f_win`.** At 431 tokens the bf16 window singles measure
  `woff` 30.36 ms against 31.25–31.63 windowed, so windowing costs ~1 ms while
  reading *fewer* KV bytes. The registered model can only express that as a
  **negative `kappa_kv`**, which is what it does at R1 (-6.87e-12).

**The correction cannot be retrofitted.** Fitting the corrected form to the same
8 singles was tested directly:

| regime | F fitted | worst held-out | registered model |
| --- | --- | --- | --- |
| R1 | 0.08 ms | 0.1063 | 0.1077 |
| R4 | -1.03 ms | 0.2200 | 0.1991 (worse) |
| R6 | -0.66 ms | 0.0946 | 0.0848 (worse) |
| R8 | 0.08 ms | 0.1443 | 0.1456 |

`F` collapses to ~0 not because it is absent but because the fit set cannot see
it: **6 of 8 singles sit at `keep = 1.0`**, where `c_layer` and `F` are the same
column, and along the skip axis (all `woff`, fixed context) `keep*KV` is exactly
proportional to `keep`, so those two points cannot separate KV from `c_layer`
either.

## 2. The Round-2 model

```text
D(quant, w, k) = keep * ( W_layer(quant)*kappa_w
                        + KV_bytes(w)*kappa_kv
                        + c_layer
                        + f_win * [w > 0] )
               + F

keep       = 1 - k/L
W_layer    = quantised BODY bytes only   (13.892 GB bf16 / 3.581 GB w4a16)
F          = lm_head + embeddings + sampling; NOT scaled by keep,
             and NOT quant-dependent, because lm_head stays bf16 in both
```

Five parameters: `kappa_w`, `kappa_kv`, `c_layer`, `f_win`, `F`.

`F` is quant-independent by construction. That follows from the decision to keep
`ignore=["lm_head"]` — aligning with EfficientRollout, and additionally required
by the serving stack: vLLM builds `lm_head` as `ParallelLMHead`, whose scheme
lookup does not match `targets=["Linear"]`, so a checkpoint with a quantized
`lm_head` fails to load with `no module or parameter named
'lm_head.weight_packed'`. Verified by building one.

## 3. The fit set, chosen by conditioning rather than by taste

Rank alone is not the criterion — the current 8 singles are already formally
rank-5. What matters is the weakest independent direction, which is what
determines whether `F` survives estimation:

| fit set | points | weakest direction | keep levels |
| --- | --- | --- | --- |
| current | 8 | 0.0622 | 0.778, 0.889, 1.000 |
| A: + windowed skip, + quant crosses | 12 | 0.0955 | 0.778, 0.889, 1.000 |
| B: A + skip16 | 14 | 0.1686 | **0.556**, 0.778, 0.889, 1.000 |
| **C: B + quant at deep skip** | **15** | **0.1939** | 0.556, 0.778, 0.889, 1.000 |

**The decisive addition is a deeper skip level, not more crosses.** A -> B
(adding skip16) is worth more than all four crosses in A, because the current
keep range spans only 22% (0.778–1.0) and an intercept cannot be separated from
a slope over that span at the measured noise level.

### Proposed fit set (design C, 15 points)

```text
window axis, bf16, keep=1 : woff, w128, w256, w512, w1024
skip axis, bf16, woff     : skip4, skip8, skip16
skip axis, bf16, w256     : skip4, skip8, skip16      <- breaks keep*KV ~ keep
quant                     : woff/skip0, woff/skip8, w256/skip0, w256/skip16
```

What each group buys:

* the **windowed skip** points break the `keep*KV` proportional-to-`keep`
  degeneracy that makes the current skip axis uninformative about `c_layer`;
* **skip16** doubles the keep range, which is what rescues `F`;
* **quant at keep<1** tests directly whether the quant benefit scales with
  layers kept — the assumption the registered model makes and never tests;
* **quant at a windowed setting** tests quant x KV, the interaction the v6
  residuals are concentrated in.

`skip16` requires extending `W98_SKIP_COUNTS` from `{0,4,8}`, which is part of
the new preregistration.

## 4. Where acceptance enters — D1, D2, D3 are different

`F` behaves differently in each claim, and conflating them is easy:

| claim | what F does |
| --- | --- |
| **D1** cost per step | must be a term OUTSIDE `keep_frac`; this is the fix above |
| **D2** acceptance (tau) | **no effect** — `lm_head` is bf16 in every arm, so it cannot confound the quant comparison the knapsack identity rests on |
| **D3** end-to-end, decode currency | **amortised by acceptance**: cost per committed token is `D / (1 + A)`, so higher tau spreads `F` over more tokens |

So D2's registered claims (product bound, knapsack identity, u-axis resolution)
are unaffected by this correction, and D3 is the only place cost and acceptance
interact through `F`.

## 5. The consequence worth stating in the paper

Because `lm_head` stays bf16, **`F` is identical at every lever setting**. No
combination of quantization, window, or skip reduces it. It is an irreducible
floor:

* 3.66 ms at R1 — **12%** of the unlevered draft chain (30.36 ms)
* and **18%** of the best measured composed configuration (~20 ms)

That bounds what the selector can ever achieve, and it is a cleaner statement of
the limit than the registered model can make — that model instead hands the
floor both a `keep_frac` discount and a quantization discount, neither of which
it earns. The 4–23% under-prediction of quantized composed cells in v6 is
precisely the size of that unearned discount.

## 6. What this does not change

* Round 1's result stands as measured and is not rescored.
* Amendment 1 (the envelope) carries over unchanged; it is orthogonal.
* The runtime decision (piecewise + Marlin, Option A) carries over; it was
  chosen because the frozen lattice cannot support a whole-chain fit, and this
  proposal does not revisit that.
* `epsilon_arm = 0.015` and the elimination rule are untouched.
