# B — acceptance past 1K of generation: the window lever inverts

First measurement of acceptance beyond the calibration range. G98-D generated
640 tokens, so with u-edges `[256, 1024, 3072]` it populated **only buckets 0
and 1**; the refined evaluation's LO cell generates 5–25K. Everything the
selector believes about acceptance at LO lengths has, until now, been
extrapolation from a scalar fitted below 1K.

Measured here on LO content (AIME, the Phase-100 registered bundle), 8192
generated tokens, batch 8, KMAX=8 stream read at k=4, u-edges extended to
`[256, 1024, 3072, 8192]`.

## The curves

tau at k=4, by generated-suffix length:

| cell | u<256 | 256–1K | 1K–3K | 3K–8K | trend |
| --- | --- | --- | --- | --- | --- |
| `woff/skip0` | 4.962 | 4.933 | 4.955 | 4.974 | flat |
| `woff/skip4` | 4.553 | 4.556 | 4.598 | 4.709 | **+3.4%** |
| `woff/skip8` | 3.642 | 3.701 | 3.986 | **4.106** | **+12.7%** |
| `w512/skip4` | 4.553 | 4.397 | 4.228 | **4.266** | **−6.3%** |

Armed steps per bucket run 235–410 (u<256) to 4599–6539 (3K–8K), so the long
buckets — the ones that matter for LO — are the best sampled, not the worst.

## Two findings

**1. Unwindowed acceptance RISES with generation length, and it rises most
where the draft is most damaged.** skip8 gains 12.7% from the first 256 tokens
to the 3K–8K range, skip4 gains 3.4%, skip0 is flat. Long chain-of-thought
becomes more predictable as it proceeds, and a weakened drafter has the most
to gain from that. The consequence for the selector is direct: its scalar,
calibrated below 1K, **under-rates deep skip at LO lengths by up to 12.7%**.

**2. The windowed cell is the only one that FALLS, and it falls
monotonically.** Holding skip fixed at 4, the window's acceptance penalty
against the unwindowed draft is:

| u | u<256 | 256–1K | 1K–3K | 3K–8K |
| --- | --- | --- | --- | --- |
| `w512/skip4` vs `woff/skip4` | 0.0% | −3.5% | −8.0% | **−9.4%** |

At short generation a 512-token window costs nothing, because the generated
suffix still fits inside it. As generation runs past the window the draft
loses sight of its own recent output, and acceptance decays. The penalty is
zero by construction at u < 512 and grows without bound in principle.

## Why this matters for the refined grid

Campaign 1 found the window family winning the LO cell at high batch
(`w1024` 1.271 at b16, `magicdec512` 1.504 at b32). That is a **cost** win:
a bounded KV read is cheaper per draft step. This measurement supplies the
other half of the trade — the window's **acceptance** cost, which is ~0 at
short generation and −9.4% by 8K, and still growing.

So the LO cell contains a genuine crossover in both directions:

* the window pays more as batch rises (cost side, measured by Campaign 1);
* the window pays less as generation lengthens (acceptance side, measured
  here).

A single fixed configuration cannot sit on the right side of both. That is
the most concrete switching case the phase has produced, and it is
qualitatively different from the R-regime result where per-regime selection
was worth only +1.4%.

## Status of the input this replaces

The selector currently consumes one scalar tau per (cell, regime), pooled from
below 1K. On the refined grid that input is wrong in two directions at once:
it under-rates deep skip and over-rates windows, and both errors grow with
generation length. **Acceptance must enter the prediction as tau(u), not as a
scalar**, before the selector is scored on LI/LO/LIO/SS.

## Scope

* Acceptance is deterministic given checkpoints and prompts — G98-F measured
  bit-identical accept patterns across boxes — so **these curves transfer to
  h103/h104 unchanged**. Unlike this box's throughput numbers, this result is
  not VM-local.
* One content set (AIME/LO), one model, k=4 read from a KMAX=8 stream.
  `ignore_eos` is used deliberately as an instrument, per the Phase-100
  protocol, so that every request reaches the long-u buckets rather than
  stopping early and leaving them empty.
* 8192 tokens covers the low end of LO's 5–25K range. The 3K–8K bucket is
  the longest measured; whether the trends continue past 8K is not shown.
