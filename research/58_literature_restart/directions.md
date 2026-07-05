# Phase 58: sharpened research directions (post-literature)

## What the literature confirms (our findings are correct, and now cited)

- **Comm-bound wide-EP MoE decode** is established (DeepSeek-V3 2505.09343;
  MegaScale-Infer 2504.02263). Our machine-balance argument matches.
- **SD is a memory-BW-for-compute arbitrage; net-negative past a short-context
  batch crossover (~32)** and serving systems should disable it under load
  (Synergy 2310.18813; SmartSpec 2406.14066). **This validates our Phase-57
  serving-batch loss.**
- **The verify forward over committed tokens is irreducible under exact SD** —
  the field's own consensus, matching our (k+1)-irreducibility theorem.
- **Layer-skip degrades super-linearly, worse on MoE** (2603.23701) —
  validates Phase 17.
- **The comm axis of an MoE spec DRAFT, and the multi-node all-to-all of an MoE
  VERIFY, are white space.** All three sub-reviews converge on this: every
  MoE-SD paper (MoESD, EVICT, MoE-Spec, SP-MoE, MoE-SpeQ, Cascade) models the
  draft-tree expert blow-up as a **single-GPU memory-I/O** cost; none studies
  or optimizes the **multi-node EP all-to-all**, and none exploits a
  **communication-cheap (local/on-rank-expert) draft**. World A's node-local
  comm-free draft has **no precedent** (nearest: EasySpec 2502.02493 "fuzzy"
  de-sync, aimed at GPU utilization, not eliminating collectives).

## What the literature reveals that we MISSED (this reopens the question)

Our "SD loses at serving batch" was measured at **short context (1024-2048
tokens), multi-node EP, no DeepEP** — the adversarial corner. Two serving-batch
WIN regimes we never tested:

1. **LONG CONTEXT (MagicDec, ICLR2025 2408.11049).** KV-cache load scales with
   `batch x seqlen` and does NOT amortize like weights, so decode stays
   **memory-bound at ANY batch** for long context -> SD helps at batch 32-256,
   32k-100k ctx, **speedup INCREASES with batch**. The exact opposite of our
   short-context result. **We never tested long context.**
2. **MoE is FAVORABLE on the memory axis** (MoESD 2505.19645): sparse MoE
   benefits MORE than dense at moderate batch; sparser -> broader winning batch
   range (expert FFNs memory-bound, idle compute -> verify tokens ~free).
3. **DeepEP-hidden comm** (DeepSeek-V3 MTP 2412.19437): K=1 MTP gets ~1.8x in
   production because DeepEP/DBO hide the a2a. EAGLE-3 (2503.01840) claims
   1.38x @ batch 64 — but dense/NVLink, no inter-node EP a2a.

**The reconciliation:** our negative is real but regime-specific. The comm
penalty that sinks SD at serving batch is a **short-context, exposed-a2a**
phenomenon. At long context the KV read dominates and the comm penalty is a
smaller fraction; whether SD then WINS on multi-node MoE-EP is **untested and
unknown** — the collision of MagicDec (memory-bound, single-node) with our
comm-bound-EP finding has never been measured.

## Direction A -- cheap draft, SHARPENED

Open problem: a draft cheap on **all three axes** (compute + memory-I/O + comm).
The literature has each axis alone but never all three:
- memory-I/O: 4-bit self-draft, >90% accept, near-free (QuantSpec 2502.10424).
- comm: node-local / on-rank-expert draft -- **unbuilt** (World A is the only
  instance, and it pays full compute).
- compute: the backbone floor (~0.6x, Phase 46) is irreducible for a self-draft
  (layer-skip fails on MoE); only a separate small head (EAGLE/MTP) beats it.

**The sharpened, potentially-winning idea: a comm-cheap draft in the
LONG-CONTEXT regime.** At long context + serving batch the target is KV-bound
(MagicDec), so the draft's compute-backbone floor matters LESS (compute is not
the bottleneck) while comm-avoidance matters MORE. World A's comm-free draft
lost at short context because compute dominated; at long context the balance may
finally favor it. **This is the untested sweet spot** where our node-local /
FP4-replica comm-free draft could actually win -- and it is white space.

## Direction B -- cheap verify, SHARPENED

Lossless verify-**comm** reduction is impossible (our theorem + field
consensus). But the literature's active lossless frontier reduces the verify's
**memory-I/O** (expert union loaded), not comm:
- **EVICT (2605.00342)**: losslessly truncate the draft tree before verify ->
  fewer unique experts LOADED (memory), up to 2.35x, MoE-aware, single-GPU.
- SP-MoE / MoE-SpeQ: losslessly HIDE the expert transfer (prefetch/overlap).

None touches the multi-node all-to-all. The field explicitly flags the open
problem: *"a lossless method that lowers the intrinsic compute-or-communication
of a committed MoE verify token (e.g., verifying with a strict subset of
experts)."* Two sharpened sub-directions:
- **B1 (lossless): EP-comm-aware speculation-length / goodput gating.** The
  goodput-gated adaptive-length idea (SmartSpec, SMART 2604.09731) has never
  been made **all-to-all-cost-aware** on multi-node EP. Co-design K (and
  tree shape) with the measured exposed a2a -> the deployable version of our
  Phase-57 K*(EP) rule. Lossless, real, incremental.
- **B2 (bounded-lossy): strict-subset-EP verify.** Verify committed tokens over
  a REDUCED expert set / narrower EP (comm-cheap), with a periodic full-EP
  exact correction bounding the error. The lossy frontier the theorem permits;
  novel if the error bound beats the linear comm<->quality rate.

## The unifying new hypothesis (what this session + the literature jointly point to)

**Communication-aware speculative decoding for multi-node MoE-EP in the
LONG-CONTEXT serving regime** is simultaneously (i) white space in the
literature, (ii) the regime where our comm-free-draft work (World A) could
finally win because decode is KV-bound not compute-bound, and (iii) testable
now. Concretely: at 32k-128k context, serving batch, multi-node EP, does a
comm-free / node-local draft (World A) net a win, where it lost at short
context? MagicDec says the target is memory-bound (favorable); our work says
the a2a is a penalty (unfavorable); the balance at long context is unmeasured.

## Concrete next experiments (cheap, decisive)

1. **Re-run the World A / EAGLE serving-batch test at LONG context** (32k+,
   Qwen3-30B, 2-node EP16). Does the short-context serving-batch loss become a
   win when KV-bound? This single measurement decides whether Direction A is
   alive. (Reuses everything; only max_model_len + long prompts change.)
2. If long-context is favorable: **comm-free draft (node-local/FP4) at long
   context** -- the World A mechanism in its true regime.
3. **B1: a2a-aware goodput-gated K** on the real fabric -- the lossless,
   deployable, incremental win.

## Positioning (honest)

The paper's spine is now: *the first characterization of speculative decoding
on communication-bound multi-node MoE-EP* -- the verify-comm irreducibility
theorem, the K*(EP) rule, the measured short-context regime map (latency-win /
serving-loss), AND the long-context regime (test pending) that determines
whether a comm-cheap draft has a serving-batch home. Novelty is the
multi-node-EP-comm angle that the entire MoE-SD literature has treated as
single-GPU memory-I/O.
