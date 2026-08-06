# W9 results — MoE b32 confound isolated, and P-W7d RETRACTED

Pre-registration: commit d45da0fb2 (before results). Artifacts
`data/w9/w9_moe_*.json` (4 TP2 boots), plus the two banked corners.

## 1. The 2×2 (MoE b32, R2, K=4, TP2, notune)

S = uncond/off, per corner:

| | max_num_seqs=32 | max_num_seqs=64 |
|---|---|---|
| **natural EOS** | 1.265 (W9) | **1.087 (W7 certified)** |
| **fixed-256** | 0.928 (W8b+W8d) | 0.943 (W9) |

- **max_num_seqs is NOT the cause**: at fixed width it moves S by
  +1.6% (0.928 → 0.943). Pre-registered reading "~0.93 → drain
  explains it" is the one that fired.
- **The drain is the whole effect**: natural-vs-fixed is +36% at
  ms=32 and +15% at ms=64.

## 2. Why the drain inflates S — repeatable but NOT comparable

The natural-EOS measurement is extremely stable: W7's 8 off rounds
across 2 boots span 0.9% (1179.2–1189.5). Stability was never the
problem. The problem is that **the two arms do different amounts of
work**:

| W7 MoE R2 b32 | out p50 | out p95 |
|---|---|---|
| off (AR) | 209 | **1332** |
| uncond (spec) | 203 | **925** |

At T=0 speculative decoding is distribution-preserving, so the text
*should* be identical. It is not: verify processes K+1 tokens per
step, so the target forward runs at different batch shapes than AR,
floating-point reduction order changes, and near-tie argmaxes diverge.
Once the text diverges, EOS lands elsewhere and the length
distributions separate — here by 44% at p95.

That matters because R2's lengths are heavy-tailed (p50 203, p95 1332)
and a fixed 32-prompt batch DRAINS: most wall time is spent at low
effective batch finishing a few long requests. The arm with the
shorter tail exits the inefficient low-batch phase sooner and posts a
higher e2e rate — independently of any per-step speedup. The off arm's
own p95 also changes with max_num_seqs (1332 at ms=64, 958 at ms=32),
confirming the workload itself is configuration-dependent.

**Methodological finding: at b32 under natural EOS, e2e throughput is
repeatable to 1% yet not comparable across arms or configurations.**
The fixed-length protocol removes this by construction — every request
emits exactly 256 tokens, so both arms perform identical work.

## 3. RETRACTION — P-W7d's refutation does not stand

`results_w7.md` reported P-W7d REFUTED because MoE b32/R2 armed at
S = 1.087 "certified on both boots, on natural content", and
`results_w8.md` §7 promoted it as the cell "Round 1 kept alive that
only Round 2's measured f could bank". **Both claims are withdrawn.**

Under a controlled workload the cell loses at both engine
configurations: S = 0.928 (ms=32) and 0.943 (ms=64), with tight rounds
and τ unchanged (4.45 vs 4.43 — this was never an acceptance effect).
The oracle's OFF prediction for MoE b32 was **correct**; the apparent
win was a drain artifact of unequal generated lengths.

What survives from W7: the {OFF} verdicts below b32 (acceptance-
independent, unaffected — those cells lose under every protocol), and
MLA b32's ARM, which reproduces under both protocols (1.064 fixed-256
unperturbed, 1.129 natural) and is therefore protocol-robust.

## 4. Consequences

- **MoE has no armed cell** at any measured batch. Its Round-2 story
  is now: gate OFF everywhere, matching Round 1's prediction exactly.
- **The paper loses its "two-round design recovers a win the oracle
  missed" exhibit.** The honest replacement is stronger as a
  methods contribution: a protocol that is repeatable to 1% still
  produced a wrong ARM verdict, and only a workload-controlled
  re-measurement caught it. Cost-side cells must be decided on
  equal-work protocols.
- **Every prior e2e S at natural EOS with heavy-tailed regimes and a
  draining batch is suspect** where arms diverge in length. Affected:
  cost-side (R) verdicts near S ≈ 1. Not affected: acceptance (τ,
  f) measurements, and cells whose verdict is far from 1 or
  acceptance-independent. A sweep of prior phases for near-1 e2e
  verdicts under drain conditions is the follow-up.
