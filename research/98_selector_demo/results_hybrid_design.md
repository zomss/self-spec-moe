# X14 — one boot config for both small and large batch

Date: 2026-08-13
Status: complete. **`FULLCG=1 + WHOLECHAIN=1 + Marlin` is best or within 1% at
every batch from 1 to 64.** No per-step gating needed. But the win is the kernel
swap, not the graph, which corrects how this phase has been describing it.

## The question

X13 left an apparent dilemma: whole-chain wins ~1.4x at batch 1 and ~1% at batch
>= 32, so a system serving both regimes seemed to need two deployments or a
per-step switch.

The two flags are not the same kind of thing, which is the opening:

* **`FULLCG`** — boot-time. Swaps paged FA3 for scratchpad+splitkv.
* **`WHOLECHAIN`** — per-step, already keyed on batch. Captures the K-step chain
  as one graph.

Every prior measurement was `(0,0)` or `(1,1)`. `(1,0)` — the cheaper kernel with
a host-driven chain — had never been run.

## Result

R1, window 256, Marlin forced, K=4, 40 timed steps, no profiler:

| batch | piecewise (0,0) | fullcg_only (1,0) | **wholechain (1,1)** | WC vs best other |
| --- | --- | --- | --- | --- |
| 1 | 26.99 ±0.09 | 22.41 ±0.05 | **22.31 ±0.02** | wins 0.5% |
| 8 | 28.30 ±0.78 | **24.46 ±0.08** | 24.70 ±0.11 | loses 1.0% |
| 32 | 32.63 ±0.27 | 34.49 ±0.56 | **32.33 ±0.25** | wins 0.9% |
| 64 | 32.54 ±0.27 | 33.95 ±0.17 | **32.31 ±0.25** | wins 0.7% |

Whole-chain is best at three of four batches and within 1% at the fourth, so one
configuration covers the range. The hybrid per-step gate proposed before this
measurement is unnecessary.

## The win is the kernel swap, not the graph

```text
batch  1: total +4.68 ms = kernel swap +4.58 + graph +0.10
batch  8: total +3.61    = kernel swap +3.84 + graph -0.24
batch 32: total +0.30    = kernel swap -1.86 + graph +2.15
batch 64: total +0.23    = kernel swap -1.41 + graph +1.64
```

**At small batch `FULLCG` alone delivers essentially all of it.** The graph adds
0.10 ms at batch 1 and is slightly negative at batch 8.

**At large batch the roles invert.** splitkv is *worse* than paged FA3 (−1.86 ms
at b32) and the graph compensates (+2.15). They nearly cancel, which is why
b32/b64 look flat across all three modes.

Keeping both flags is what makes a single config work: each covers the regime
where the other fails. That is a different reason from the one this phase has
been giving.

### Correction to how this phase described the fix

X6 through X13 called this "the whole-chain fix" and attributed it to removing
~116 host round-trips per step via graph capture. The decomposition shows the
splitkv kernel does the work at small batch. The host-starvation diagnosis (X3,
X4) stands — the GPU was idle and the host was the cause — but the remedy that
matters is the cheaper attention path, not the capture.

## Correction to the headline number

The "1.41x at batch 1" figure was measured with **Machete**. With Marlin,
piecewise at b1 is 26.99 ms (not 35.34), so whole-chain's margin is **1.21x**,
not 1.41x.

Marlin absorbed **8.35 ms** on the piecewise path against only **2.70 ms** on
the whole-chain path (X12) — consistent with Marlin issuing fewer host launches,
which matters most precisely when the runtime is host-bound.

**The two fixes overlap and do not compose additively.** Machete+piecewise 35.34
→ Marlin+whole-chain 22.31 is **1.58x total**, not 1.4x × 1.12x.

## Recommendation

Deploy `FULLCG=1 + WHOLECHAIN=1 + Marlin` for all batch sizes, carrying three
constraints already established:

1. Windowing must be on, and must genuinely bind at long context — a
   non-binding window costs 5.8–20.8 ms/step of extra attention (X13).
2. Sampled decoding falls back automatically: whole-chain requires `all_greedy`.
3. Graph memory scales with captured `(batch, table_width, K)` shapes; coverage
   across the deployed batch range is unmeasured.

## Consequence for the lattice decision

Since the win is now attributed to the kernel rather than the graph, and Marlin
helps the piecewise path *more* than the whole-chain path, **Option A
(re-measure the lattice on piecewise) is materially more attractive than when it
was first framed**: piecewise+Marlin trails by only 1.21x at batch 1 and by ~1%
at batch >= 32, it carries no `window > 0` constraint, and the lattice blocker
(skip and quant sampled only at `woff`) disappears entirely.
