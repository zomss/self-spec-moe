# W98-R2 preregistration — corrected cost model, re-conditioned lattice

Date: 2026-08-13

Status: **frozen.** No GPU command is authorized here; each scored run still
needs its own source-bound authorization. Amending anything below after scored
data exists invalidates the claim it supports.

Supersedes `w98_prereg.md` §D1 and its fit set. **Carried over unchanged**:
D2, D3, `epsilon_arm = 0.015`, the gate order, the prompt manifest, and
amendment 1 (the envelope). Round 1's measured result stands and is not
rescored.

| artifact | sha256 (first 16) |
| --- | --- |
| `data/prereg2/w98r2_matrix.json` | `4660518c6cc65eb8` |
| `data/prereg2/w98r2_fit.json` | `f8810b79bd0af938` |
| `data/prereg2/w98r2_heldout.json` | `ebebcfe950c369d0` |

## 1. Why a new preregistration and not an amendment

Round 1 measured a systematic under-prediction of quantized composed cells
(4–23%, one-directional). It was root-caused, and the cause is **not** a
runtime effect and **not** a bad envelope:

* `W_bytes` in the registered model includes the bf16 `lm_head` and embeddings
  — 15.3% of the bf16 draft but **41.3%** of the quantized one, since
  quantizing the body leaves them a larger share — and none of it scales with
  skipped layers, yet the model multiplies all of it by `keep_frac`.
* The registered form has no window-overhead term, so it expresses one as a
  **negative `kappa_kv`** (−6.87e−12 at R1), which is unphysical.

Both fixes change the registered functional form, and the corrected form is
**not estimable on the Round-1 fit set** — tested directly, `F` collapses to
~0 (R1 0.08 ms, R4 −1.03, R6 −0.66) with held-out residuals unchanged or
worse. So the fit set changes too. That is a new preregistration by definition.

## 2. The model

```text
D(quant, w, k) = keep * ( W_layer(quant)*kappa_w
                        + KV_bytes(w)*kappa_kv
                        + c_layer
                        + f_win * [w > 0] )
               + F

keep      = 1 - k/L,  L = 36
W_layer   = quantised BODY bytes only: 13.892 GB (bf16), 3.581 GB (w4a16)
KV_bytes  = min(context, w + sinks) * 147456,  sinks = 16
F         = lm_head + embeddings + sampling
```

Five parameters per regime: `kappa_w`, `kappa_kv`, `c_layer`, `f_win`, `F`.

**`F` is quant-independent by construction**, because `lm_head` and embeddings
stay bf16 in both drafts. That follows from the standing decision to keep
`ignore=["lm_head"]` — aligned with EfficientRollout, and independently forced
by the serving stack: vLLM builds `lm_head` as `ParallelLMHead`, whose scheme
lookup does not match `targets=["Linear"]`, so a quantized-`lm_head` checkpoint
fails to load (`no module or parameter named 'lm_head.weight_packed'`).
Verified by building one.

**Registered consequence:** since no lever touches `F`, it is an irreducible
floor on draft cost — measured at 3.66 ms at R1, 12% of the unlevered draft
chain and 18% of the best composed configuration. The model must express it as
a floor, and D1' is a test of whether it does.

## 3. Frozen lattice

* **quant**: `target-matching`, `w4a16-quantized`
* **window**: `128, 256, 512, 1024, off`
* **skip counts**: `0, 4, 8, 16` — **skip16 is new** and extends
  `W98_SKIP_COUNTS` from `{0,4,8}`
* **regimes**: `R1, R4, R5, R5cot, R6, R8`
* **prompts**: the existing manifest, seeds `2, 3`, unchanged — so the cost
  re-measurement is on byte-identical prompts to Round 1 and the two are
  directly comparable

### Fit set — 15 points, chosen by conditioning

```text
window axis (bf16, keep=1) : woff, w128, w256, w512, w1024
skip axis (bf16, woff)     : skip4, skip8, skip16
skip axis (bf16, w256)     : skip4, skip8, skip16
quant                      : woff/skip0, woff/skip8, w256/skip0, w256/skip16
```

The Round-1 set was **formally rank-5 already**; rank was never the problem.
The criterion is the weakest independent direction, which is what decides
whether `F` survives estimation:

| fit set | points | weakest direction | keep levels |
| --- | --- | --- | --- |
| Round 1 | 8 | 0.0622 | 0.778, 0.889, 1.000 |
| + windowed skip, + quant crosses | 12 | 0.0955 | unchanged |
| + skip16 | 14 | 0.1686 | **adds 0.556** |
| **this fit set** | **15** | **0.1939** | 0.556, 0.778, 0.889, 1.000 |

**The decisive addition is skip16, not the crosses.** Adding it is worth more
than all four crossed points combined, because the Round-1 keep range spanned
only 22% and an intercept cannot be separated from a slope over that span at
the measured noise (`sigma_repro` 0.002–0.014).

Each group is registered with what it identifies:

* **windowed skip** breaks the degeneracy that `keep*KV` is exactly
  proportional to `keep` when skip is sampled only at `woff`;
* **skip16** doubles the keep range and is what rescues `F`;
* **quant at keep<1** tests whether the quant benefit scales with layers kept —
  the assumption the Round-1 model made and never tested;
* **quant at a windowed setting** tests quant × KV, where the Round-1 residual
  concentrated.

### Held-out set — 8 composed cells, frozen now

In `w98r2_heldout.json`. Machine-checked at freeze time: **no cell appears in
the fit set**, **every cell varies at least two axes**, and the split is
balanced — 4 quantized / 4 bf16, skip counts {4:3, 8:2, 16:3}, windows
{128:2, 512:2, 1024:3, off:1}.

That second check is registered explicitly because Round 1's split failed it:
`target-matching/woff/skip8` was held out while varying a single axis *and*
sitting in the fit set, and had to be excluded after the fact.

## 4. D1' — Round-1 soundness, restated

Model fitted from the 15 fit points only. Claim: intervals computed under
**amendment 1** cover the held-out composed configurations, with **zero false
eliminations** under `(K+1)/q_lo < 1 + epsilon_arm`, `epsilon_arm = 0.015`.

Envelope, unchanged from amendment 1:

```text
envelope(x) = sqrt( fit_term(x)^2 + (2 * sigma_repro(x))^2 ) * 2.0
```

`sigma_repro` from ≥3 repeats of two fit-set anchors **bracketing** the
campaign. Section-5 validity condition carries over: where
`2*sigma_repro >= fit_term`, the regime is reported **NOT RESOLVABLE**, not
covered.

**Registered in advance:** Round 1 found R5 and R5cot not resolvable on
piecewise+Marlin. If they are again, D1' is reported over the resolvable
regimes and the others are named, not quietly dropped.

## 5. What D2 and D3 inherit

Unchanged, but the relationship to `F` is registered so it cannot be
reinterpreted later:

| claim | how F enters |
| --- | --- |
| **D1'** cost per step | `F` is a term outside `keep_frac`; this is the fix |
| **D2** acceptance (tau) | **no effect** — `lm_head` is bf16 in every arm, so it cannot confound the quant comparison the knapsack identity rests on |
| **D3** end-to-end | **amortised by acceptance**: cost per committed token is `D / (1 + A)` |

D2's acceptance and confirmation measurements use **fresh content seeds `4, 5`**,
disjoint from the `2, 3` used for the cost lattice, so acceptance is not
measured on prompts already consumed by the cost fit. Cost uses seeds `2, 3`
precisely so Round-1 and Round-2 cost are comparable.

## 6. Measurement contract

* **runtime**: piecewise, `WHOLECHAIN=0`, `FULLCG=0`
* **kernel**: Marlin, via `VLLM_DISABLED_KERNELS=MacheteLinearKernel`
* **d_measured**: the profiler's `draft_chain` region, warmup 20
* **why this runtime**: whole-chain requires `window > 0`, so it cannot sample
  the skip and quant axes at `woff`; a mixed fit is invalid because
  "uncaptured" and "unwindowed" are the same indicator. Marlin is what makes
  the campaign resolvable — it lowered `sigma_repro` 3–10x versus Machete.
  Registered cost of this choice: the scored runtime is **1.21x slower than
  whole-chain at batch 1** and ~1% slower at batch ≥ 32.

## 7. Assumptions that can still invalidate this

1. **skip16 may collapse acceptance.** It is a cost-model fit point, and D1'
   does not depend on its acceptance — but if the configuration cannot boot or
   produces too few armed steps to measure, the fit set loses the point that
   carries most of `F`'s identifiability, and the conditioning reverts toward
   the Round-1 value.
2. The corrected model may still under-predict quantized composed cells. If it
   does, the cause is **not** the two terms added here and the residual is a
   genuine composition effect rather than a specification error.
3. Two capture-path weight-equality assertions (`koff_runtime.py:1927`,
   `:2503`) remain unresolved and will be hit by Round 2's KMAX streams.
