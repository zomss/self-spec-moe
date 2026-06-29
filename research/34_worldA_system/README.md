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
