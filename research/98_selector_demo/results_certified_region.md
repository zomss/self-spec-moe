# The surrogate's foundation: a recall guarantee, not a validity region

Record of 2026-08-16. Analysis over the measured G98-D factorial; no new
measurement. Script: `scripts/analyze_w98_certified_region.py`; record:
`data/g98_d/certified_region.json`.

## The question

The two-round search orders compositions by the product of single-lever
retentions and confirms a shortlist by measurement. That ordering rule is a
heuristic — `paper/c2.md` measures top-1 mis-rank at 75-100% for *every*
rule including ours, and a KnapSpec-style additive knapsack beats our
aggregation outright on llama (0.79% vs 1.61% regret at confirm-5). So the
fair criticism is that KnapSpec has an exact algorithm and we have an
ordering hunch.

What would answer it is a **certificate**: something computable from
single-lever measurements alone, known *before* any composed measurement,
that bounds what the surrogate can cost you.

## The hypothesis that failed

The product-of-singles is a multiplicative-independence approximation, so it
should hold while each lever's damage is small and break once the levers
destroy enough of the draft to interact. That suggests a **damage budget**
computable from singles:

```
D(set) = sum_i -log( retention_i ),   retention_i = tau(lever_i) / tau(base)
```

**Registered hypothesis:** there exists `D*` such that the surrogate ranks
correctly for every composition with `D <= D*`.

**REFUTED.** Restricting to a damage band and re-ranking inside it, the
correlation does not fall — it is flat or rising — and shortlist recall is
perfect in every band including the worst:

| damage band | n | mean rho | recall@4 |
| --- | --- | --- | --- |
| [0.0, 0.3) | 112 | 0.8386 | 8/8 |
| [0.3, 0.6) | 74 | 0.8590 | 7/7 |
| [0.6, 1.0) | 32 | 0.8475 | 5/5 |
| [1.0, 1.6) | 24 | 0.8356 | 3/3 |
| [1.6, inf) | 22 | **0.9488** | 2/2 |

There is no validity region because the surrogate never leaves validity. The
hypothesis was the wrong shape.

## What the damage budget actually certifies

It certifies the **bound's tightness**, which is a different and still useful
thing:

| damage band | median abs log error | mean SIGNED log error |
| --- | --- | --- |
| [0.0, 0.3) | 0.0173 | **+0.0058** |
| [0.3, 0.6) | 0.0866 | **+0.0799** |
| [0.6, 1.0) | 0.1802 | **+0.1845** |
| [1.0, 1.6) | 0.2514 | **+0.2763** |
| [1.6, inf) | 0.5557 | **+0.6065** |

`rho(D, |log error|) = +0.875` (`tau_eff`), `+0.853` (`tau_k4`), n=264.

Two things to read off. The signed error is **positive in every band** — the
product is a lower bound and stays one, which is D2(a)'s direction result
reproduced across the whole factorial rather than at a single screen. And
the looseness is a **monotone, predictable function of a quantity known from
singles**: at `D < 0.3` the bound is tight to 1.7%, at `D > 1.6` it
under-predicts by 55%.

That also explains why the ranking survives: the error is **systematic, not
noise**. A bias that grows smoothly with `D` shifts values but largely
preserves order among the compositions being compared.

## The foundation that does hold: bounded shortlist recall

The search never consumes a top-1 pick; it confirms a shortlist. The
quantity that matters is whether the shortlist *contains* the optimum.

Over 12 independent (regime, seed) groups, 22 compositions each:

| | recall@1 | recall@2 | recall@3 | **recall@4** | top-1 regret mean / max |
| --- | --- | --- | --- | --- | --- |
| `tau_eff` | 8/12 | 10/12 | 11/12 | **12/12** | 0.32% / 1.99% |
| `tau_k4` | 8/12 | 10/12 | 11/12 | **12/12** | 0.22% / 1.86% |

Both acceptance statistics agree, and recall@4 is perfect in every damage
band as well as overall.

**The claim the search can make.** Measure the singles (7 boots on this
lattice), order the 22 compositions by the product, confirm the top 4:

* sample complexity `O(#levers + 4)` rather than `O(#compositions)` — 11
  measurements against 29, on a lattice where the space is small enough that
  the saving is real but modest, and grows with the lattice;
* the optimum was in the confirmed set in **12 of 12 groups**;
* and if the budget collapses to confirm-1, the loss is bounded by the
  measured **2.0% worst case**, not by the surrogate's 33% top-1 miss rate.

## How this compares to KnapSpec, stated fairly

KnapSpec solves an exact knapsack over premises this record falsifies
(additive value, additive cost, cosine-proxy inputs). We run an approximate
ordering over premises that are measured or bounded. Neither is "more
rigorous" in the abstract — an optimality guarantee is worth exactly what
its premises are worth, and `paper/c2.md`'s own head-to-head shows the
knapsack *aggregation* is fine (2.9-4.4% regret) while the *proxy inputs*
fail (14-32%).

What this record adds is that our side is no longer only a hunch. The
ordering carries a measured recall guarantee at a stated shortlist size, and
the value bound carries a certified error bar that is a monotone function of
a pre-computable quantity. That is a weaker guarantee than an exact solver
and a stronger one than a heuristic, and it is the honest description.

## Limitations

* **12 groups.** recall@4 = 12/12 carries a one-sided 95% lower bound of
  **0.78** — the point estimate is 1.00, the guarantee is not.
* **One model, one box, one lattice** (Qwen3-8B, H100, 3 levers / 22
  compositions). The shortlist size 4 is measured here, not derived; a wider
  lattice may need a wider shortlist and the scaling is unmeasured.
* **Skip-16 is absent** from this factorial (G98-D confirms only skip4 and
  skip8), so D2(b)'s k=16 ranking inversion is not reproduced or explained
  here. The coherent reading — that D2(b) compares sets at *equal* damage,
  where the systematic component cancels and only the interaction remains —
  is an interpretation this data does not test.
* Acceptance only. The cost half is D1's, and is separately sound.
