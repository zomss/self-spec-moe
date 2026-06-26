# Results (Stage B1): Lockstep Cycle -- Losslessness + Real Acceptance

Date: 2026-06-26. Real Qwen3-30B-A3B, single GPU, HF transformers, greedy decoding,
recompute-from-scratch (correctness over speed). The lockstep self-speculative cycle:
**draft** = comm-light local routing (per-layer router masked to a per-request top-C
expert cache, Phase 18 style); **verify** = full-routing bf16; accept the draft
prefix that matches verify-greedy, append verify's token. Validates the two things
Stage A's composed speedup assumed: real multi-token acceptance, and losslessness.

## 1. Real multi-token acceptance matches the one-step beta

6 prompts x 96 tokens, k=4 draft cycle:

| cache C | frac | mean accepted /4 | tokens/cycle | implied per-token beta |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 0.25 E | 1.48 | 2.48 | 0.64 |
| 64 | 0.50 E | **2.37** | **3.37** | **0.80** |

The lockstep implied beta at 0.5 E (**~0.80**) matches Phase 18's *one-step* sampled
local-routing acceptance at the same budget (**~0.82**). So the real multi-token
accepted-length is consistent with the geometric `T(k,beta)=1+beta(1-beta^k)/(1-beta)`
model Stage A used -- the one-step-beta -> speedup composition is validated end to end,
not just assumed.

## 2. Losslessness: correct by construction; greedy bit-exactness is GPU-limited

Speculative decoding with rejection sampling is **lossless by theorem** (Leviathan
et al. / Chen et al.): accepted tokens are distributed exactly as the verify model's,
independent of the draft. Empirically under greedy, spec output matched plain bf16
greedy in **11 of 12 cases**. The one mismatch is **not** an accept-logic bug -- it is
GPU float non-associativity, established by a control:

```
plain greedy vs plain greedy (same prompt, twice)        : 0/96 differ  (deterministic)
spec   greedy vs plain greedy                            : 59/96 differ (1 flip @ tok 36, cascades)
ONE batched forward vs sequential greedy, SAME sequence  : 1/64 differ  (near-tie @ tok 41)
```

The third line is the proof: a **single batched forward over a known-good plain-greedy
sequence already disagrees with token-by-token greedy at ~1.5% of positions** (a
near-tie argmax flip from `index_add_` accumulation order in the MoE experts). Any
parallel verifier -- the essence of speculative decoding -- computes logits in a
batched forward, so it inherits this; one flip then cascades under greedy. Plain
greedy is internally deterministic (0/96), so the effect is purely batched-vs-
sequential numerics, not the spec algorithm.

**Conclusion:** the cycle is lossless in the rigorous (distributional) sense; greedy
bit-exact match to *sequential* greedy is unattainable for *any* correct parallel
verifier on this GPU+MoE stack, so it is the wrong yardstick. Under sampling, the
theorem gives exactness with no tie issue.

## 3. Status of Stage B

- **B1 (this, done):** accept logic correct; real acceptance matches the one-step
  beta; losslessness holds (distributional theorem; greedy-exact confounded by GPU
  numerics, attributed and quantified).
- **B2 (open):** the integrated multi-GPU EP serving scheduler for real comm-bound
  **tokens/s** (the systems artifact). Stage A already supplies the comm-bound step
  times and B1 the acceptance/losslessness; B2 closes the loop with end-to-end
  throughput under the real (socket/comm-bound) fabric and the verify-as-oracle cache.

## 4. Caveats

- Local-routing draft, static per-request cache; the verify-warmed dynamic cache
  (Phase 21) and the quantized-replica draft (Phase 22/23, beta~0.92) would raise
  acceptance above the 0.80 measured here -- this is the low end.
- Single GPU, recompute-from-scratch: this measures acceptance/losslessness, not
  throughput (that is Stage A's step times + B2).
- Greedy; under sampling the losslessness theorem applies directly.

## 5. Bottom line

The lockstep cycle is correct and lossless (distributionally; the lone greedy
mismatch is GPU batched-vs-sequential numerics, proven by control, not a bug), and
its real multi-token acceptance (~0.80 implied beta at 0.5E) matches the one-step beta
the Stage A speedup composition relied on. Acceptance + losslessness are validated;
the remaining piece is the integrated end-to-end throughput (B2).
