# §2 Background and related work (draft — table + positioning prose)

> Source: `duplicate_check.md` (fail-closed gate 2026-07-13) + phase record.
> The three required sharpenings from the check are implemented here:
> MagicDec named as the seed framing, SparseSpec cited as the window lever's
> full treatment, and the selection-axis taxonomy stated explicitly.

## 2.1 Setting

Speculative decoding accepts draft tokens by rejection sampling against the
target, preserving the output distribution; per-cycle payoff is
τ_β(γ)/(γR + 1). SELF-speculation instantiates the draft from the target
itself — no trained head, no second model to deploy — by cheapening some
part of the forward pass. Each cheapening is a *lever*: weight quantization
(W4/fp8), sparse/window attention over shared KV, draft-only KV-cache
quantization, layer skip, and (on MoE) restricted expert routing. Every
lever has a literature that studies it alone; §4–5 measure them as one
portfolio.

## 2.2 The selection-axis taxonomy (Table 1)

Three selection axes exist in the 2026 literature. They compose rather than
compete, and ours is the third:

| axis | representative work | chooses | from what signal |
|---|---|---|---|
| speculation depth (γ) | SmartSpec, Learning-to-Draft (RL), PACER, DEL (exit layer) | how far to draft, per step | goodput / online reward |
| drafter identity | Not-a-Bandit (no-regret online) | which trained drafter, per query | online accept feedback |
| **lever × architecture (ours)** | — | which self-spec lever (incl. OFF), per deployment regime | **offline measured R and β surfaces** |

Depth adaptation lives INSIDE one lever (our γ* falls out of the same
composition formula); drafter selection assumes a zoo of trained drafts.
Neither answers the practitioner deciding what to deploy for model M at
(batch, context) — the lever question — and none carries an OFF region.

## 2.3 Related work by lever and the deltas

**The seed framing.** MagicDec posed the bottleneck-aware question — where
speculation pays as batch and sequence length move — for dense models with a
KV-sparse self-draft, including "select drafting strategy" language. Our
paper is MagicDec's regime question answered with a measured multi-lever,
multi-architecture selector; the bottleneck framing is theirs, the map,
the acceptance-portability results, the composition law, and the delivery
analysis are ours.

**Window/sparse attention.** SparseSpec develops our window lever into a
full system (PillarAttn, verification-informed token selection, serving
co-design) on dense reasoning models. We cite it as the lever's strongest
form: our fixed sinks+window arm is a LOWER BOUND on what their selection
buys, and our contribution is the lever's PLACE in the map — including
where it loses (short context: strict no-op; MLA: the cost×accept double
weakness of §5-F8), which single-lever treatments cannot see. Our
window-size β-insensitivity on GQA (win128 ≈ win2048 ⇒ smallest window
dominates) is likewise absent there.

**Weight quantization.** EfficientRollout (reproduced in our record as the
dense weight-quant win) and QuantSpec-adjacent work; our delta is the
parity verdict on sparse MoE (18 cells, two kernels — §4-F2) and fp8's
status as the ONLY architecture-portable β (§5-F5).

**KV-cache quantization.** The KIVI line studies K/V asymmetry for the
TARGET; our draft-only fp8 read flips sign across architectures, and the
QK-norm rule (§5-F6) gives the mechanism plus a design rule (V-only on
un-normed models) that KIVI-style rescaling cannot rescue (it makes β
worse).

**Layer skip.** LayerSkip/SWIFT/DEL search or adapt skip sets, and
KnapSpec (2602.20217) is the lever's strongest form — non-contiguous layer
SELECTION solved as a knapsack over offline-profiled per-layer
latency/acceptance. Our fixed middle-block skip is a lower bound on that
lever (the same relation our window arm has to PillarAttn). KnapSpec is
dense-only, single-lever, with no OFF notion and no regime axis — and its
knapsack-over-profiles methodology independently validates the
offline-profiling-then-search approach our §7 applies ACROSS levers. Our
measured β curve (§5-F7) is the caution its cost side needs: skip
collapses super-linearly on dense, runs +0.24 higher on MoE, and cliffs at
50% on MLA.

**MoE-specific mechanisms.** SS-MoE / MoE-Spec / SP-MoE and utility-driven
variants each build one MoE mechanism (expert-subset draft, verify-side
expert budgeting, prefetch). None selects among levers or maps regimes;
several compose with any draft lever our map picks (e.g., MoE-Spec's
verification budgeting).

**Serving-side overlap.** Speculative-speculative decoding overlaps
drafting with verification; our delivery(γ,R) analysis and the floor-free
chain (§8) address the complementary serving question — whether the
SELECTED config's roofline survives execution — and our two execution laws
(constant-geometry capture; compacted-step rejection trimming) are, to our
knowledge, unreported.

**Camera-ready check**: one login-walled OpenReview entry ("rethinking
high-throughput speculative decoding") remains unread; re-verify before
submission (flag carried from the duplicate check).
