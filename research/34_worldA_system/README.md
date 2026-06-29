# Phase 34: World A integrated system (the actual decoder)

Goal: a real, lossless, end-to-end self-speculative decoder on the forced-PCIe MoE-EP
engine, producing a **measured tokens/s** speedup vs the no-spec baseline. This is the
"B2b-full" build deferred throughout Phases 24-33.

## What we already have (components, NOT integrated)
- Acceptance beta + real multi-token accept length (single-GPU recompute harnesses). [24 B1, 25, 22]
- Comm-free draft STEP TIME via `VLLM_SELF_SPEC_SKIP_A2A` -- but this is a TIMING stand-in:
  it tiles the local chunk to valid expert ids, output is GARBAGE by design. [24 B2a]
- Verify step time (real vLLM EP decode). [24 3e]
- Dynamic-cache / skip-cold beta, union growth (offline analysis). [28]
- Lockstep correctness + losslessness on ONE GPU, recompute-from-scratch. [24 B1]
- Composed (not measured) end-to-end speedup.

**The gap:** no integrated distributed lockstep decoder -> no real tokens/s, no real
system-level losslessness.

## Remaining implementation (dependency order)

1. **Correct comm-free local-routing MoE (the draft).** Replace the timing-only
   `SKIP_A2A` with a CORRECT forward: per-device router MASKED to the resident expert set,
   compute those experts locally, **no all-to-all**, skip-cold for misses (route to top
   resident + renorm; the harness `local_forward` logic). Must produce correct draft
   outputs, not garbage. *Touches vLLM fused-MoE + the AgRs all2all path. Highest risk.*
   [task: B2b-1]

2. **Resident draft-expert cache.** The draft routes to a per-device resident set that is
   LARGER than the EP shard (the shard alone -> poor coverage -> low beta). Options:
   - MVP: **full replica** per device (simplest). Fits in bf16 only for a SMALL MoE
     (DeepSeek-V2-Lite 16B / Qwen1.5-MoE) alongside the verify shard.
   - Scale: **FP4** replica (16 GB) or **globally-hot top-C + skip-cold** (8 GB,
     batch-independent) for large models. [28]

3. **Per-step draft<->verify mode switch across DP workers.** Toggle local-routing(draft)
   vs full-EP(verify) each phase; `SKIP_A2A` is read live, but switching across DP
   processes needs a control signal (file-flag pattern, or the driver broadcasts the mode).

4. **Lockstep driver.** draft k tokens (local mode) -> verify (full EP, one forward) ->
   rejection-sample -> commit -> repeat. Two integration paths:
   - (A) custom step-level loop on the engine (full control, reuses B1 logic);
   - (B) plug into vLLM's spec-decode framework as a "self-draft" method (reuses its
     scheduler / rejection / KV -- IF the API admits draft = target in local mode).
   *Investigate (B) first to avoid reinventing the scheduler.* [task: B2b-2]

5. **KV correctness.** Commit the VERIFY's KV for accepted tokens (source of truth);
   draft KV is scratch; roll back rejected. (Draft/verify KV diverge after the 1st MoE
   layer -- the established correctness point.) [task: B2b-3]

6. **Rejection sampling.** The accept logic on the verify logits over the k draft tokens
   -- reuse B1.

7. **Measurement.** Real tokens/s vs no-spec baseline on the forced-PCIe engine
   (`NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1`); system losslessness
   (per-cycle fresh-reference greedy match ~0.99, B1 caveat).

## Staged plan
- **Stage 1 (MVP, the make-or-break):** small MoE (DeepSeek-V2-Lite) + **bf16 full-replica**
  local-routing draft + full-EP bf16 verify + lockstep (k=1-4) + forced-PCIe -> first real
  tokens/s vs baseline. Validates the entire World A loop end-to-end. No FP4, no cache
  policy yet.
- **Stage 2 (memory):** FP4 draft experts + globally-hot top-C + skip-cold; scale to
  Qwen3-30B; confirm the 8 GB / batch-independent memory story.
- **Stage 3 (low batch / breadth):** verify-warmed dynamic cache; cross-model; tree/k sweep.

## Reuse
B1 lockstep + rejection logic; harness `local_forward` (masking); the `SKIP_A2A`
scaffolding (extend to correct); forced-PCIe recipe; `bench_commbound` timing path.

## Risks / effort
- Correct local-mode MoE in vLLM internals (main effort).
- Lockstep vs vLLM continuous batching (path A vs B).
- KV correctness across modes; memory budget (verify shard + draft replica + KV).
- ~1.5-2 weeks, high risk -- the long-deferred integration.

## Decision criteria
Stage 1 GO = measured lockstep tokens/s on forced-PCIe exceeds the no-spec baseline AND
per-cycle losslessness ~0.99. Then memory (Stage 2), then breadth (Stage 3).

---

## vLLM spec-decode scoping -- DECISION: reuse the framework (path B), no custom driver

Scoped the v1 spec-decode API (Explore agent). Findings + decision:

- **Two engine paths.** MoE targets (DeepSeek-V2, Qwen2Moe, Qwen3) default to the **V2
  runner** (`vllm/v1/worker/gpu/model_runner.py`), whose spec methods are limited to
  eagle/eagle3/mtp/dflash; non-eagle methods fall back to the V1 runner. So we implement
  against the **V2 `AutoRegressiveSpeculator`** (`vllm/v1/worker/gpu/spec_decode/
  autoregressive/speculator.py`).
- **The framework decouples rejection/KV/scheduler from how drafts are produced.** The
  speculator runs IN-PROCESS in the same `GPUModelRunner.execute_model`; `propose()` just
  returns draft tokens, and `RejectionSampler` + block-table/KV management + scheduler
  accept/reject accounting + cudagraphs are all free. **No custom lockstep driver needed.**
- **EAGLE/MTP are the self-spec precedent:** they share the target's embed/lm_head/KV pool,
  reuse the target's hidden states, and are distinct `nn.Module`s -- no separate model
  server. Our "same MoE in comm-free mode" is a variant where the draft module IS the
  target (+ a resident expert cache).
- **Clean integration:** subclass `AutoRegressiveSpeculator` (mirror `mtp/speculator.py`);
  `load_draft_model` returns the target (sharing backbone) + our resident expert cache;
  register a method literal (`SpeculativeMethod`, `vllm/config/speculative.py:60`) + the
  **V2 allow-list** (`vllm/config/vllm.py:2031` -- else silent fallback to V1) + a branch in
  `init_speculator` (`.../spec_decode/__init__.py:8`).
- **The local-routing flag has a clean channel:** `ForwardContext.additional_kwargs`
  (`vllm/forward_context.py:180`), set in the speculator's `_run_model` `set_forward_context`,
  read by the FusedMoE forward to (i) skip dispatch/combine (the all-to-all) and (ii) mask
  the router to resident experts. **This replaces the SKIP_A2A env hack with a per-step,
  per-forward signal.**
- **Bonus:** `SpeculativeConfig.moe_backend` already overrides the *drafter's* MoE kernel
  ("useful when drafter and generator require different MoE kernels") -- the framework
  anticipates a different draft MoE path.

### Revised work split (much of W3/W4/W6 is now free)
- **W1 (main work):** the comm-free local-routing MoE bypass in `fused_moe/modular_kernel.py`
  (+ EP gating `config.py:1018`), signalled via `ForwardContext.additional_kwargs`. Largest,
  most invasive piece; orthogonal to spec-decode.
- **W2:** the draft's resident FP4 expert cache (draft module shares target attention/
  embed/lm_head, holds its own experts -- mirrors EAGLE's draft-specific weights).
- **W4/W6:** FREE -- reuse `AutoRegressiveSpeculator` + `RejectionSampler`. Work reduces to
  the speculator subclass + method registration.
- **W3:** subsumed -- the mode signal is the `ForwardContext` flag, not a DP control plane.
- **W5 (subtlest risk):** our self-draft reuses the target's OWN attention layers (unlike
  EAGLE's draft-only layers), so `draft_attn_layer_names` is empty; we must wire the
  draft-step attention to the **speculative slot mappings** so draft KV lands in scratch
  blocks and never corrupts committed target KV (the verify-context-KV pattern). Reuses the
  framework's `compute_slot_mappings` but deviates from the draft-only-layer assumption.

### Precision & dual-residency (FP4 draft + bf16 verify) -- explicit

The two passes use **different expert weights at different precisions**, both resident:

- **Verify = the target's bf16 experts, EP-sharded** (E/num_devices per device). This is the
  exact target -> losslessness is w.r.t. the bf16 model. Unchanged from normal serving.
- **Draft = a SEPARATE resident FP4 expert cache** (globally-hot top-C / replica, sized for
  local-routing coverage -- *larger* than the EP shard), sharing the target's bf16
  attention / embed / lm_head. The draft is lossy (FP4 + local routing); the bf16 verify
  corrects it -> still lossless.
- **Dual-residency:** the device holds BOTH the bf16 verify shard AND the FP4 draft cache;
  the FP4 cache is the *extra* memory (1-16 GB by cache size, Phase 26/28). The draft does
  NOT reuse the verify's bf16 experts (it has its own FP4 weights), so the speculator's
  draft module is "shared bf16 backbone + draft-specific FP4 experts" -- exactly EAGLE's
  shared-backbone + draft-specific-weights pattern, where the draft-specific weights are the
  FP4 expert cache.
- **The ForwardContext flag gates BOTH** the routing mode (local, comm-free) AND the weight
  set (FP4 draft cache vs bf16 verify shard). W1 wires the flag; W2 loads the FP4 weights.

Staging note: FP4 enters at **W2**. W0/W1 use bf16 experts (reused from the target) to
de-risk the framework wiring and the comm-free path first; W2 swaps in the separate FP4
resident cache (the quantized-draft + memory story).

### Revised first steps (de-risk the integration before the invasive MoE work)
0. **Scaffold + sanity:** speculator subclass + registration with draft = target in FULL
   mode (full routing). Confirms the framework wiring runs and is lossless (beta=1, no
   speedup yet). Cheap, isolates integration bugs from MoE bugs.
1. Add the **local-routing flag** (W1) -> comm-free draft.
2. Add the **resident FP4 cache** (W2).
3. **KV-slot correctness** (W5).
4. **Measure** (W7).

---

## W0 CODE READ -- CORRECTION: the framework is HEAD-oriented; a full-model self-draft does NOT fit cleanly

Read the actual V2 speculator code (`autoregressive/speculator.py`, `speculator.py`,
`mtp/speculator.py`, `eagle/utils.py`). The Explore summary's "subclass and return the
target model" is NOT viable. Two hard mismatches for a FULL-MODEL self-draft:

1. **Forward signature is a draft-HEAD signature.** `_run_model` calls
   `self.model(input_ids, positions, hidden_states=..., inputs_embeds=...)`
   (autoregressive/speculator.py:315-330) -- the draft is CONDITIONED on the target's
   hidden states (EAGLE/MTP head). A full target model's `forward(input_ids, positions)`
   does not accept `hidden_states` -> `TypeError`. A full-model self-draft would need a
   wrapper that ignores the fed hidden_states and runs the full forward from input_ids.

2. **The framework assumes the draft has its OWN attention layers/KV.** `load_model`
   computes `draft_attn_layer_names = all_attn_layers - target_attn_layer_names`
   (speculator.py:159); `set_attn` inits the draft attn backend over those (line 169-174).
   For a self-draft that REUSES the target's attention layers, this set is **EMPTY** -> the
   draft attn machinery collapses. The only framework-native alternative is a SEPARATE draft
   instance with its own attention layers -> a SECOND full KV cache (48 layers) -> ~2x KV
   memory. EAGLE works because its draft is ONE layer (tiny extra KV); a full-model draft is
   not that.

**Conclusion:** the v1 spec framework is architected for small draft HEADS (cheap,
hidden-state-conditioned, own small KV). World A's full-model self-draft is an outlier and
does not fit cleanly. Two real paths remain, both substantial:
- **B2 (intricate framework reuse):** self-draft wrapper sharing the target's attention
  layers + OVERRIDE `draft_attn_layer_names` to the target's layers + drive draft KV into
  speculative slots on those shared layers + cudagraph interplay. Reuses rejection/scheduler/
  cudagraphs but is subtle (shared layers used in two modes).
- **A (custom lockstep driver):** single model, two modes (draft=local/comm-free,
  verify=full), single KV cache with draft KV in scratch slots, reuse B1 rejection logic.
  Full control; reimplements the scheduler/KV glue; loses cudagraphs.

**This also mirrors the perf finding:** the spec-decode ecosystem assumes cheap draft heads,
and World A's full-model draft fights that in BOTH performance (dominated by EAGLE) and
implementation (doesn't fit the framework).

**Sequencing implication:** **World B is both the stronger result (~2.3x) AND the far easier
build** -- it reuses the EXISTING EAGLE spec-decode path unchanged and only changes the
draft-TREE SIZE (a comm-aware small/pruned tree), no custom draft, no framework fight, no
2x KV. World A's full-model self-draft is the harder build for the weaker (training-free-only)
result. Recommend building **World B first**; pursue World A only if its training-free
novelty justifies the custom-driver cost.
