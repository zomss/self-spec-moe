# Paper outline — phases 76-81 (updated 2026-07-13 with the Phase-81 headline)

**Working title**: *No Universal Draft: A Measured Selector for Training-Free
Self-Speculative Decoding Across Batch, Context, and Architecture*

**Thesis (one sentence)**: which self-speculative draft lever pays — and
whether any does — depends on (batch, context, architecture) in ways that are
measurable once, predictable off-grid, and composable; we build and
e2e-validate that selector.

**Positioning** (per `duplicate_check.md`, fail-closed gate passed
2026-07-13): MagicDec posed the bottleneck-aware drafting question for dense
models; single-lever papers each answer for one lever (SparseSpec: sparse
attention; QuantSpec: KV-quant; SWIFT/DEL/LayerSkip: skip; SS-MoE/MoE-Spec:
MoE mechanisms); existing "selectors" choose depth (SmartSpec/LTD) or drafter
identity (Not-a-Bandit). Ours is the third axis: lever × architecture from
measured cost×accept physics, including OFF.

**Venue shape**: MLSys-style measurement+systems paper (a law + a validated
selector + honest negative regions), not a single-method paper.

---

## 1. Introduction

- The practitioner's question: "I serve model M at batch b, context c —
  should I self-speculate, and with what?" No trained head, no new weights.
- Why the answer is non-obvious: 5 levers × regimes × architectures; each
  lever cuts a different term; acceptance is lever- AND arch-dependent.
- Contributions (each with its number):
  C1 The measured cost map R (3 architectures × 9 cells × 7-9 levers) with
     two clean crossovers and an OFF region (§4).
  C2 The measured acceptance map β + the portability verdict: only
     fp8-weight-quant is regime- and architecture-portable; the QK-norm rule
     for KV-cache quantization (§5).
  C3 The selector: speedup = τ_β(γ)/(γR+1), validated e2e on 5 cells
     (2.5-16% err); cost model with held-out-arm (2.2%) and forward (2.9%)
     validation; one-anchor transfer to an unswept architecture (R 9.3%) (§6).
  C4 The composition law: β_combo ≈ Πβ_i (median dev 0.01) with two
     structured exceptions; composed configs win the dense long-ctx map;
     the delivery(γ, R) systems law (§7).
  C5 The floor-free chain: two execution laws that CASH the composed
     frontier — (i) FA3's captured schedule is replay-safe iff attention
     geometry is constant (the window scratchpad clamps it: chain step
     6.47→3.70 ms, accept bit-preserved); (ii) a compacted q=1 step-0 must
     trim seq_lens by the cycle's rejections (non-causal decode attends the
     rejected tail; window-amplified, −0.55 accept → fixed). Result:
     **1.91× measured at dense b32/16k = the registered roofline, delivery
     ≈ 100%**; same stack gives the MoE window cell +12% (§8).
- Anti-contribution framing (honesty as a feature): three map regions say
  "turn it OFF"; the delivery discount that gated the composed frontier is
  fully attributed (launch floor) and fully recovered (C5).

## 2. Background and related work

- Spec decoding + rejection sampling; self-speculation (no training).
- The levers, each with its literature: sparse/window attention (SparseSpec
  — cite prominently; our fixed window is a lower bound on their PillarAttn),
  weight-quant (P75-repro'd EfficientRollout; QuantSpec adjacent), KV-quant
  (KIVI-line for the K/V asymmetry), layer skip (LayerSkip/SWIFT/DEL),
  expert-subset/local routing (SS-MoE; our P24/25 lineage).
- Selection axes taxonomy (depth / drafter / lever×arch) — table.
- MagicDec as the seed framing; what the MoE/parallelism axis changes.

## 3. Measuring the two surfaces (methodology)

- **R** (cost): serve-mode TPOT, one server per arm; warm pass (prefill-
  contamination trap, ×4.7 inflation shown); over-capacity detection;
  infra-parity preflight (E0: backend/kernel/CG assertions — the silent-flip
  trap catalog: fp8_e5m2→FlashInfer, MLA+fp8KV→FLASHMLA, TP-EP has no a2a);
  anchored to an independent offline slope method at 2%.
- **β** (accept): offline teacher-forced rejection-sampling acceptance
  against target refs; anchor gate = geometric composition must reproduce
  two e2e accept lengths (dense −1.6%, MoE +0.3%); sample-size requirements
  (≥12 prompts × 96 pos; per-prompt jitter).
- Both surfaces are PAIRED (same refs / same denominators) and every claim
  in the paper traces to one of the two, or to an e2e run.
- Fig: pipeline diagram (new). Table: the S-gates.

## 4. The cost map (Phase 76)

- Fig 1 (heatmaps), Fig 2 (crossovers). Findings:
  F1 crossovers exist (W4→window along ctx×batch on dense; OFF→window on MoE)
     — the selection problem is real.
  F2 weight-quant is a parity band on sparse MoE at every cell, BOTH kernels
     (extends P74/E4 to full coverage).
  F3 the over-capacity envelope: levers enable operating points bf16 cannot
     serve (dense b32/32k).
  F4 law refinement: local-route is NOT free on NVLink under DP-EP
     (0.74-0.94) — the a2a exists; prior "no-op" readings measured TP-EP,
     which has no dispatch collective at all.
- Measurement notes sidebar (warm pass, DP placement bimodality → medians).

## 5. The acceptance map (Phase 77)

- Three-architecture β table. Findings:
  F5 portability verdict: of 12 levers × 3 archs, ONLY fp8 weight-quant is
     portable (±0.03); kvq INVERTS across archs; skip swings +0.5.
  F6 the QK-norm rule: K-outlier magnitude (amax 426/150/28, max/rms
     206/70/16) anti-correlates with draft-only fp8-K viability
     (β unstable/.97/.999); V-only quantization is free (0.99) — design rule
     for a draft-only KV pool.
  F7 the skip curve: super-linear collapse (0.47/0.09/0.03 dense), measured
     — cost-only skip claims are not decision-grade (Fig 5, the (R,β) plane).
  F8 window: size-insensitive on GQA (smallest window dominates) but
     size-decisive and ctx-decaying on MLA — window is doubly weak on MLA
     (with 76-E4's cost story).
- Early-exit vs middle-skip aside (β=0.000 for early exit — same R, dead β).

## 6. The selector (Phase 78)

- The composition formula (P75-validated, ~90% delivery at γ3) + the term-
  decomposed cost model (levers = term edits; archs = config constants;
  the h/KV-management term found by structured residuals).
- Validation ladder (table): V0 in-grid (dense 3.5%, 8/8 crossovers) → V1
  held-out arm (window from bytes alone: 2.2%/6.9%) → V2 forward (win128 R
  2.9%; b16×8k 8.0%) → V3 registered transfer FAIL → V3b one-anchor
  refinement (R 9.3% on 53 cells; TPOT needs per-stack constants — honest
  split; the skip arms PROVE the floor split).
- Strategy map v3 (Fig 3) + 5/5 e2e consistency (Fig 4 updated with the
  2 forward cells).
- The refined transfer claim: new architecture = one 5-minute anchor cell.

## 7. Lever composition (Phase 80)

- The pruning/MPQ framing (offline profiling + composition law + search +
  argmax validation); SWIFT delineation (skip-set search, single lever).
- F9 the composition law: β_combo ≈ Πβ_i, median dev 0.009-0.013 (48 paired
  cells); Exception 1: window RESCUES kv-quant on dense (+0.13-0.26 —
  levers cutting a term cut co-levers' error exposure); Exception 2:
  MLA-only destructive skip×context at 32k (composition portability tracks
  the arch verdict).
- Combo pricing by term edits: the falsification cell (kvq+win priced
  cost-moot, measured ±5%); dense combo measured R 0.311.
- The search → map v4: dense long-ctx goes composed (q_int4+win512,
  measured-β × measured-R provenance).
- F10 e2e: composed config confirmed at b8/16k (1.41×, 91% delivery); the
  b32 miss ⇒ **delivery(γ, R)**: the fixed per-draft-step launch floor
  binds precisely when composition makes the draft byte-cheap (K4≈K6
  signature) — composition shifts the bottleneck from bytes to launch.
  This sets up §8: the floor is not fate.

## 8. Cashing the frontier: the floor-free chain (Phase 81)

- Anatomy first (Kineto): the composed dense chain step is 6.47 ms at 61%
  GPU-active — 2.26 ms/step is launch idle across ~370 eager kernels; cycle
  arithmetic puts the recoverable payoff at up to 1.98×.
- F11 **the constant-geometry capture law**: FA3's captured schedule is
  replay-safe iff attention geometry is CONSTANT. The known failure (P35-
  style accept collapse, reproduced as a control: 5.69 → 1.93) needs a
  growing sequence; the window scratchpad clamps KV length (sinks+window),
  so the whole chain step replays as one graph. Implementation: paged FA3
  varlen over the compacted window block table replacing an SDPA that
  silently dispatched mem-efficient decomposition (~0.25 ms/layer → ~30 µs).
  Chain step 6.47 → 3.70 ms (94% active), accept bit-preserved (5.655 vs
  5.685). Negative results kept: TRITON_ATTN and naive FA3 chain-CG both
  collapse accept — the A/B accept gate is methodology, not paranoia.
- F12 **the compacted step-0 correctness law**: under shared KV, step-0 is
  semantically a q=1 decode of the appended token — but padded-path
  seq_lens span rejected slots, which a causal ragged forward masks and a
  q=1 decode does NOT. Untrimmed, the appended token attends the stale KV
  of the cycle's rejected draft tokens; a KV window AMPLIFIES this (up to K
  wrong keys among the ~window most recent → −0.55 accept; at full context
  it dilutes to noise — reconciling the earlier −0.04 reading). Fix: trim
  per-request seq_lens by num_rejected. Accept restored bit-clean (5.672).
- F13 the payoff, e2e (same cells/method as §7): dense b32/16k K6
  **4152 tok/s = 1.91× — exactly the registered roofline, delivery ≈ 100%**
  (was 1.48×/77%); b8/16k K4 1.64× (registered 1.55×); γ* structure of the
  map survives (b8: K4 > K6). Baseline-fairness control: nospec+async +4%,
  headline still 1.84× against it.
- F14 cross-architecture: the identical stack engages under DP4/EP4 MoE
  (compaction on all ranks) — window cell 615 → 688 tok/s (+12%) at
  identical accept. The chain fix is architecture-generic, not a dense
  special case.
- Delivery(γ,R) closes: with the floor removed, delivery ≈ 1 at both
  measured γ* — the selector drops the delivery discount for dense composed
  configs and ranks on raw τ_β(γ)/(γR+1).

## 9. Discussion, limitations, and open surface

- Single box (H100 ×4), single serving stack; TPOT absolutes don't transfer
  (2-3 stack constants) — the R-selector does.
- DP4 placement noise floor (medians, flagged cells); deep-γ rooflines.
- Deferred: the comm-bound fabric (PCIe/multi-node) where the MoE triple
  (β 0.82-0.83 measured) and MLA's shared-expert local-route (β 0.95-0.99)
  are parked — the original thesis, now with its accept side fully measured.
- The draft-only KV pool: motivated (1.25× on QK-normed MoE) and scoped
  (V-only on un-normed models) but unbuilt.
- Remaining headroom on the fixed chain: verify forward idles 31%
  (4.8 ms/cycle) — a second, independent dial; step-0 residual vs a pure
  chain step is the last ~250 tok/s to the 1.98× ceiling.
- Runtime lever SWITCHING (the map as a scheduler policy) is scoped as
  follow-on work (Phase 82) — this paper establishes the maps, the
  selector, and that the selected configs deliver.

## 10. Conclusion

- The map exists, the selector predicts it, composition extends it,
  "OFF" is a first-class answer — and the selected frontier configs
  deliver their rooflines end to end on a floor-free chain.

---

## Claims → evidence traceability (internal QA table)

| claim | evidence | artifact |
|---|---|---|
| crossovers | 76 E1 + repair | fig2, summary.csv |
| 5/5 e2e | P74/75 + 76-E3 | fig4, results_strategy_map.md |
| portability / QK-norm | 77 sweep + kvq probe | beta.csv, results_accept.md |
| held-out/forward/transfer | 78 V1/V2/V3b | fit.json, v2/v3b_verify.json |
| composition law | 80 E0 (48 paired cells) | beta_combo.csv |
| composed e2e | 80 E3 | logs/e3c_*, results_composition.md |
| delivery(γ,R) | 80 E3 K4≈K6 + 76-E3 γ-trend | same |
| capture law (F11) | 81 E1/E1b A/B + traces | results_floor.md, data/trace_* |
| step-0 law (F12) | 81 E2b isolation ladder (spsd/spsd2/spw0/spw0sd/spsdfix) | results_floor.md, logs/e1_* |
| 1.91× headline (F13) | 81 E3 (7 cells, parallel) | data/e3/, results_floor.md |
| MoE generality (F14) | 81 MoE check (noreg + fixed arms) | data/moe_check/ |

## Figures plan

- Fig 1 cost heatmaps (have; regenerate with MLA row) · Fig 2 crossovers
  (have) · Fig 3 strategy map (regenerate as v3/v4 two-panel evolution) ·
  Fig 4 validation scatter (add the 2 forward cells + combo cells) · Fig 5
  (R, β) plane (add measured skip points + combo points) · NEW Fig 6:
  three-arch β portability (slopegraph) · NEW Fig 7: composition law
  (measured vs product scatter with the two exception clusters) · NEW
  Fig 8: the floor-free chain — (a) chain-step anatomy before/after
  (6.47 ms/61% active → 3.70 ms/94%), (b) delivery bars per cell
  (77% → 100% at b32/16k; the b8 cells), (c) the step-0 poisoning
  schematic (causal span vs q=1 key set under the window).

## Writing order

related-work table → §4/§5 (data sections, assets ready) → §6/§7 → §8
(fresh, numbers final) → §3 methodology → §1/§9/§10 last. Re-check the
login-walled OpenReview paper before submission (duplicate_check.md flag).
