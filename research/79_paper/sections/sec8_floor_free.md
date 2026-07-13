# §8 Cashing the frontier: the floor-free chain (draft)

> Source: Phase 81 (`results_floor.md`, E0–E3 + MoE check) and the Phase 79
> MLA post-mortem. Numbers final. ~1.5 pages.

## 8.1 Where §7 left the composed draft

Composition cut the dense draft's bytes to R = 0.31, and delivery
collapsed to 77% at b32 — because each draft step pays a fixed launch
cost that now dominates a byte-cheap step. Anatomy (Kineto traces of the
exact composed config) makes the floor concrete: the chain step is 6.47 ms
at only 61% GPU-active — 2.26 ms/step is launch idle spread across ~370
eager kernels — and cycle arithmetic bounds the recoverable payoff at
1.98×. CUDA-graph capture is the textbook remedy, and it is exactly what
the record says fails: naive FA3 chain capture collapses acceptance (5.69
→ 1.93 in our control), the known freeze trap. This section is about why
it fails, the two laws that fix it, and what the fixed chain delivers.

## 8.2 F11 — the constant-geometry capture law

FA3's host-side kernel schedule (work distribution, splits) is decided at
capture time. A captured graph replays that schedule verbatim, so capture
is replay-safe **iff the attention geometry is constant** — and a draft
chain's sequence GROWS each step, which is why every naive chain capture
in the record collapses acceptance (FA3 5.69→1.93; TRITON_ATTN 5.69→1.95
— refuting our own earlier hypothesis that a seq-len-independent grid
would be safe). The law's contrapositive is the fix: the window scratchpad
CLAMPS geometry (sinks+window = 528 keys, constant by construction), so
the entire chain step becomes capture-stable. One implementation subtlety
carries the win: the scratchpad's attention must be a fused paged-FA3 call
— torch SDPA at q_len=1 with a mask silently dispatches a mem-efficient
decomposition (~0.25 ms/layer of gemv soup where the fused kernel needs
~30 µs), which is also the post-mortem of our own earlier scratchpad
prototype (and, we suspect, of others' "graphs don't help" readings).
Result: chain step 6.47 → 3.70 ms at 94% GPU-active, acceptance
bit-preserved (5.655 vs 5.685).

## 8.3 F12 — the compacted step-0 law

With the chain fixed, profiling re-attributes the wall to step-0: the
draft re-ingests the K+1 verify-span tokens whose KV verify already wrote
(under shared KV) — a ~20 ms ragged forward whose outputs are discarded
except at one position. Semantically, step-0 is a q=1 decode of the
appended token. Compacting it that way, however, costs −0.55 acceptance
naively, and the cause is a law worth stating generally: **padded-path
seq_lens span rejected slots, which a causal ragged forward masks
implicitly and a q=1 decode does not.** The compacted token attends the
stale KV of the just-rejected draft tokens; under a KV window those stale
keys sit among the ~window most recent — maximal attention mass — so the
window regime AMPLIFIES the poisoning (−0.55) that full context dilutes
to noise (−0.04, matching an earlier shelved reading of the same
mechanism). The fix is one line of metadata: trim per-request seq_lens by
the cycle's rejection count. Acceptance restores bit-clean (5.672).

## 8.4 F13 — the payoff, end to end

Re-running §7's cells on the fixed chain (same method, same baselines):

| cell | roofline | broken chain | fixed chain | delivery |
|---|---|---|---|---|
| dense b32/16k K=6 | 1.91× | 1.48× (77%) | **4152 tok/s = 1.91×** | **≈100%** |
| dense b32/16k K=4 | — | 1.48× | 1.85× | |
| dense b8/16k K=4 | 1.55× | 1.41× (91%) | 1.64× | >100%* |
| dense b8/16k K=6 | — | — | 1.52× | |

The registered roofline is measured EXACTLY at the headline cell — the
delivery discount was the launch floor and nothing else. The map's γ*
structure survives (b8: K=4 > K=6, as selected), and the baseline is not
handicapped: async-scheduling helps nospec by only 4% (headline 1.84×
against the async baseline). (*b8 slightly exceeds its registration
because the fixed chain's R is better than the standalone R the map used.)

## 8.5 F14 — the laws travel

Architecture generality: the identical stack engages under DP4/EP4 MoE —
compaction on all four ranks, acceptance unchanged (3.766 vs 3.764) — and
lifts the MoE window cell 615 → 688 tok/s (+12%). The chain fix is not a
dense special case.

Prediction beyond the derivation set: the MLA challenger of §6's OFF
hardening initially collapsed (acceptance 2.00) on a path none of §8's
work touched. F11 called it: FLASH_ATTN_MLA is FA3-family, its captured
chain schedule freezes exactly as the law says, and the one-line
reclassification restores acceptance to 5.92 by default. A law that
diagnoses failures in code it was not derived from is doing the work we
claim for it.

Delivery(γ,R) closes as a design rule: delivery is a function of (γ, R),
the launch floor binds precisely when composition makes the draft
byte-cheap, and a constant-geometry chain removes the floor — after
which the selector may rank on raw τ_β(γ)/(γR+1). Remaining headroom is
known and bounded: verify still idles 31% of its forward, and step-0's
residual over a pure chain step is the last ~6% to the 1.98× ceiling.
