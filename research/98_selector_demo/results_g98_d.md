# G98-D (W98-D2): acceptance measured — the product bound fails on one lever

First record of the completed D2 acceptance campaign, 2026-08-15. Numbers
are from `data/g98_d/d2_result.json` (authorization v1). Interpretation
against the registered claim belongs to the researcher's read of
`w98_prereg.md` §D2; nothing is asserted here beyond what was measured.

## D2(a) screen honesty: FAILS, and the failure has one address

**6 violations in 132 scored (cell, regime) rows — 95.5% honest.** Every
violation is listed:

| cell | regime | bound tau | confirmed tau | shortfall |
| --- | --- | --- | --- | --- |
| target-matching/w128/skip4 | R6 | 8.018 | 6.407 | +1.611 |
| w4a16-quantized/w128/skip4 | R6 | 7.794 | 6.825 | +0.969 |
| target-matching/w128/skip4 | R1 | 7.083 | 6.426 | +0.656 |
| w4a16-quantized/w128/skip4 | R1 | 6.711 | 6.487 | +0.223 |
| target-matching/w128/skip8 | R6 | 7.134 | 6.949 | +0.185 |
| w4a16-quantized/w128/skip8 | R6 | 6.935 | 6.751 | +0.184 |

**All six carry window = 128**, the narrowest window, always combined with
layer skipping, and always at a SHORT-context regime (R6 four times, R1
twice). No violation occurs at any wider window, at any long-context
regime, or for any lever alone. The registered inequality
`f_set >= prod f_i` therefore has a precise boundary rather than a diffuse
error: it fails where the narrowest KV window and layer skipping are
stacked on short prompts.

**Composition is otherwise CONSTRUCTIVE, not destructive.** Across all
confirmations the interaction ratio (actual `f_set` / product of `f_i`) has
a median ABOVE 1 for every lever set — quant+window 1.061, window+skip
1.078, quant+window+skip 1.165 — and by regime:

| regime | median ratio | min |
| --- | --- | --- |
| R1 | 0.985 | 0.864 |
| R6 | 0.997 | 0.766 |
| R5cot | 1.082 | 1.010 |
| R8 | 1.087 | 0.975 |
| R5 | 1.175 | 1.042 |
| **R4** | **1.670** | 1.253 |

At R4 (8.8k context) the bound is not merely valid but very conservative:
each window factor alone is punishing (~0.33), so their product is tiny,
yet the composition retains 1.67x that. The losses OVERLAP rather than
compound, because a window that already discards most of the context
removes the very information that layer skipping would otherwise have
degraded. At short context the two levers damage different things, so they
stack closer to — and at w128, worse than — multiplicatively.

**The practical direction is the benign one.** A violated lower bound means
the screen is OVER-optimistic: it admits compositions that do not deserve
admission, and confirmation catches them. It does not produce the dangerous
failure — eliminating a good composition — because any cell the bound rates
poorly is in reality poorer still. The preregistration's own framing
("admit-only: it may let a bad composition through to confirmation, never
eliminate a good one") describes exactly this tolerance, so the screen
remains sound as a selection device even though its stated inequality is
false in six cells.

## D2(c) u-axis: resolves in 107 of 264 adjacent pairs

Adjacent u-buckets separate by INTERVAL (not point estimate) in **107 of
264** adjacent bucket pairs across all confirmed (cell, regime) streams —
40.5%. Two buckets were populated per stream at 640 generated tokens
(crossing the provisional 256 edge). The final bucket boundaries remain a
scored OUTPUT of this run rather than an input, per the preregistration;
placing them where `tau(w, g, u)` actually crosses is the next analysis
step, not a measurement.

## How it ran

h104 GPU 7, NUMA node 1, `w98-d2` scope at KMAX = 8, seeds 4 and 5 on the
disjoint D2 bundle. 16 singles, then the screen, then 44 confirmations —
60 boots.

**Barrier integrity.** All 16 singles were measured, the product-bound
screen was committed with digest `25bbcc62ac40d654`, and only then were the
22 admitted compositions booted. `commit_screen` refuses once any
confirmation exists and `score_reveal` refuses without a digest-matching
screen, so the bound could not have been fitted to the confirmations it is
scored against.

One boot (`w4a16-quantized/woff/skip8`, seed 4) failed on CUDA OOM when a
co-tenant took 41.9 GiB of the lane GPU mid-campaign. The campaign is
resumable and never re-measures a completed cell, so relaunching after the
co-tenant left completed exactly the two missing cells and scored. That is
an environmental interruption, not a measurement condition.

**No timing gate, by design.** The G98-C measurement-shape gate judges
draft-chain timing; acceptance is boot-deterministic (calibration: two
boots agreed on 17/17 requests and 0/210 steps), so the clamp cannot bias
it and the gate would only have rejected good data. Both that and the
absence of repeat-boot anchors are recorded in the authorization with their
rationale.

## Not scored here

**D2(b)**, the knapsack identity against count-matched controls, is not
scored: `knapsack_select`, `top_k_by_retention` and `worst_k_by_retention`
all consume a per-layer retention vector `r` that the preregistration never
specifies a protocol for, and the natural leave-one-out estimator is
forbidden by the frozen lattice's skip counts `{0, 4, 8, 16}`. The three
routes are recorded in `design_g98d_d2_campaign.md`; choosing one after
seeing this campaign's acceptance data is precisely what preregistration
exists to prevent, so it awaits a recorded decision.
