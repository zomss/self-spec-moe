# W7-dp: self-spec acceptance drop — isolation + root cause + fix

**Model for isolation:** `Qwen/Qwen1.5-MoE-A2.7B` (Qwen2-MoE arch: 60 experts,
top-4, 24 layers, FLASH_ATTN / non-MLA, sigmoid-gated shared expert). draft_model
self-spec (draft = target), greedy, K=4, batch 16, OUTLEN 128, forced-PCIe.
Reference points: Qwen3-8B (dense GQA, FLASH_ATTN) and DeepSeek-V2-Lite (MLA-MoE).

All numbers are mean accept_len (= 1 + accepted/ndrafts); per-token = accepted/drafted.
Source of truth is the engine's `SpecDecoding metrics` log line per run (some
`data/*.json` aggregates show null where the rank-0 result didn't reach the
parent queue; the log metrics are authoritative).

## 0. TL;DR

The Qwen3-30B self-spec acceptance collapse is **NOT** FP8, **NOT** DP-routing
token-handling, and **NOT** model-intrinsic. It is **two stacked numerical
divergences between the draft's MoE and the verify's MoE**, both specific to
**non-MLA (FLASH_ATTN) MoE** models:

1. **torch.compile divergence (the dominant effect, DP-independent).** The draft
   model is compiled separately from the verify (`set_model_tag("draft_model")`).
   For this MoE arch the two compilations are numerically inconsistent: with
   compile ON, the draft's greedy tokens diverge from the verify's
   (**per-token 0.94 eager → 0.14 compiled**, accept 4.78 → 1.57 at DP=1).
   Dense (Qwen3-8B) and MLA-MoE (V2-Lite) are unaffected. `enforce_eager`
   fully recovers it (4.78). New opt-in env `VLLM_SELF_SPEC_DRAFT_EAGER=1`
   loads only the draft uncompiled.

2. **Full-replica (non-EP) draft vs EP verify divergence (the DP≥2 effect).**
   With the World-A comm-free full-replica draft (`use_ep=False`, all experts
   per rank, no cross-rank reduction) the draft's MoE output diverges from the
   EP verify (sharded experts + all-gather/reduce-scatter) by floating-point
   reduction order; over 24 layers acceptance drops **eager DP=1 4.78 →
   eager DP=2 2.77**. The EP-shard draft (`use_ep=True`, same path as verify)
   does NOT diverge: **eager DP=2 = 4.88**.

## 1. Isolation table (the 3 cells the task asked for) + controls

| config (Qwen1.5-MoE, K=4, b16) | accept_len | per-token | reads as |
|---|---:|---:|---|
| **DP=1, bf16 draft (full-replica, PIECEWISE)** | **1.57** | 0.14 | broken |
| **DP=2, bf16 draft (full-replica, PIECEWISE)** | **1.76** | 0.19 | broken (no DP drop here — both bugs saturate) |
| **DP=2, FP8 draft (full-replica, PIECEWISE)**  | **1.83** | 0.21 | broken — FP8 ≈ bf16, NOT the cause |
| DP=1 bf16 full-replica, **EAGER** (no compile) | **4.78** | 0.94 | ✅ the true reference |
| DP=2 bf16 full-replica, **EAGER**              | **2.77** | 0.44 | DP drop (non-EP vs EP) |
| DP=2 bf16 **EP-shard draft**, EAGER            | **4.88** | 0.97 | ✅ EP draft matches verify |
| DP=2 bf16 **EP-shard draft**, COMPILED         | **1.53** | 0.13 | ⚠️ EP draft ALSO breaks under compile |
| **Qwen3-8B dense GQA, DP=1, compiled (control)** | **4.97** | 0.99 | ✅ harness + GQA-attn correct |

**Verdict from the 3 cells:** DP2-bf16 ≈ DP1-bf16 ≈ DP2-FP8 (all ~1.6-1.8, all
broken). Per the task's decision tree this is the "both ≈ baseline → not a clean
DP/FP8 split, dig further" branch. Digging (eager controls) splits it cleanly
into the two bugs above.

### Controls that pinned bug #1 to torch.compile (all DP=1, full-replica)

| variant | accept_len | conclusion |
|---|---:|---|
| EAGER (compile off + cudagraph off)      | **4.78** | works |
| compile ON, **cudagraph_mode=NONE**      | 1.54 | broken → it's COMPILE, not cudagraph |
| compile ON, PIECEWISE-only               | 1.51 | broken (FULL vs PIECEWISE irrelevant) |
| compile ON, FULL_DECODE_ONLY             | 1.70 | broken |
| K=1 (step-0 only, no chain)              | 1.37 | broken already at the first draft token |
| shared-expert aux-stream OFF             | 1.57 | not the multi-stream overlap |
| VLLM_USE_LAYERNAME=0                      | 1.59 | not the MoE layer-name registry path |
| combo_kernels OFF                        | 1.57 | not inductor combo-kernel fusion |
| plain (no full-replica / no local-route) | 1.57 | not the World-A flags |

→ Every compiled variant is broken; only eager works. The bug is the draft's
inductor-compiled forward, MoE-specific (dense is fine), independent of cudagraph
mode, DP, FP8, and all World-A self-spec flags.

### Controls that pinned bug #2 to non-EP-replica vs EP-verify (DP=2, eager)

| draft kind @ DP=2 eager | use_ep | accept_len | per-token |
|---|---|---:|---:|
| full-replica (World-A, comm-free) | False | **2.77** | 0.44 |
| EP-shard (same path as verify)    | True  | **4.88** | 0.97 |

→ The drop is the full-replica draft's MoE not matching the EP verify's MoE
numerically. The EP-shard draft matches → 4.88. (At DP=1 there is no EP, so this
bug is absent there; the 4.78 is the no-divergence ceiling.)

## 2. Why GQA-MoE and not MLA-MoE / dense

- **Dense (Qwen3-8B):** no MoE → compiled draft == compiled verify → 4.97. ✅
- **MLA-MoE (V2-Lite):** intrinsic self-spec accept ~2.7 (per-token ~0.85); the
  compile/EP FP divergences are below its rejection margin, so it stays healthy.
- **GQA-MoE (Qwen2/Qwen3-MoE):** intrinsic accept very high (per-token ~0.94),
  so it is acutely sensitive — small MoE numerical drift between draft and verify
  rejects most drafts. This is why the symptom is "GQA-MoE-specific."

## 3. The fix

`VLLM_SELF_SPEC_DRAFT_EAGER=1` (new, default off) loads the draft model
UNCOMPILED while the verify keeps its compile config — fixes bug #1.
Implemented in `vllm/v1/spec_decode/draft_model.py::_get_model` by temporarily
flipping the SHARED `compilation_config.mode`→NONE / `cudagraph_mode`→NONE during
draft construction only (the verify is already built, so it is unaffected; the
`static_forward_context` dict stays shared so the draft's attn/MoE layers
register where the proposer's runtime forward context looks them up).

**Before/after (Qwen1.5-MoE, DP=1, full-replica):**

| | accept_len | per-token |
|---|---:|---:|
| baseline (draft compiled)             | 1.57 | 0.14 |
| `VLLM_SELF_SPEC_DRAFT_EAGER=1`        | 1.89 | partial |
| `enforce_eager` (draft+verify eager)  | **4.78** | 0.94 |

Note: eager-draft alone (1.89) only partially recovers because the **verify is
still compiled** and the compiled verify's MoE diverges from the eager draft.
Full recovery of bug #1 needs the verify eager too (`enforce_eager`), since the
root issue is that the compiled MoE is numerically inconsistent with both eager
and with a second compilation. `VLLM_SELF_SPEC_DRAFT_EAGER` is the scoped knob;
`enforce_eager` is the full mitigation.

Bug #2 (DP≥2 full-replica vs EP) is structural to the comm-free World-A design:
the EP-shard draft (`use_ep=True`) is the clean 4.88 alternative but is not
comm-free. The comm-free full-replica draft trades a fixed amount of acceptance
at DP≥2 for zero MoE all-to-all.

## 4. Qwen3-30B-A3B DP=2 (the real model) — before / after

FP8 full-replica comm-free draft, target bf16, forced-PCIe DP+EP, greedy, K=4,
b16, OUTLEN 96.

| config | accept_len | per-token |
|---|---:|---:|
| baseline (compiled) on THIS branch | **2.59** | 0.40 |
| **enforce_eager** (bug #1 removed) | **2.83** | 0.46 |

Two things stand out:

1. **On Qwen3-30B the compile bug (#1) is MILD** (2.59 compiled vs 2.83 eager),
   unlike Qwen1.5-MoE where it is severe (1.57 vs 4.78). The compile bug's
   severity is model/shape-dependent. So on Qwen3-30B the dominant loss is the
   structural full-replica-vs-EP divergence (#2), not compile.

2. **This branch does NOT reproduce the cited ~1.65** for Qwen3-30B DP=2; it
   measures **2.59 compiled**. The branch already carries the phase-35 GQA
   FULL-CG fix and the other merged self-spec fixes, which evidently lifted the
   earlier 1.65. (Conditions here: FP8 full-replica, K=4, b16, OUTLEN 96,
   PIECEWISE; the original 1.65 predates these merges / used different settings.)

**Headline qualification:** the task expected the fix to take Qwen3-30B
1.65 -> 4-5. On this branch Qwen3-30B DP=2 is already at ~2.6 (compiled) / ~2.83
(eager). Reaching 4-5 is NOT attainable with the comm-free full-replica draft —
that ceiling needs the EP-shard (non-comm-free) draft (Qwen1.5-MoE: EP-shard
eager DP=2 = 4.88). The comm-free full-replica draft is structurally capped at
~2.8 at DP≥2 for these high-sensitivity GQA-MoE models.
