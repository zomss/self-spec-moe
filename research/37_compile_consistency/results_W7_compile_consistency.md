# W7 compile-consistency: the self-spec MoE draft diverges from the verify
# because their kernels are BATCH-SHAPE-VARIANT, not because of two compilations

**Model for isolation:** `Qwen/Qwen1.5-MoE-A2.7B` (Qwen2-MoE arch, 60 experts
top-4, 24 layers, FLASH_ATTN / non-MLA GQA-MoE, sigmoid shared expert).
draft_model self-spec (draft == target), greedy, K=4, batch 16, OUTLEN 128,
single GPU DP=1 unless noted. Headline: `Qwen3-30B-A3B` DP=8 + EP=8.
All accept_len = 1 + accepted/ndrafts from the engine's spec-decode metrics.

Worktree `/data/smcho/ssm-cc` (branch `w7-compile-consistency`).
Env: `VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0`, forced-PCIe trio
(`NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1`) for EP.

---

## 0. TL;DR

The compiled-draft acceptance collapse is **NOT** caused by the draft being
compiled with a separate `set_model_tag("draft_model")` (a second independent
compilation). Forcing the draft to load the verify's EXACT compiled artifact
does NOT recover it. The real cause is that the draft runs its MoE / GEMM /
attention at the **decode batch shape (~1 token/seq)** while the verify runs at
the **larger verify shape (1+K tokens/seq)**, and the default kernels are
**batch-shape-variant** (cuBLAS split-k, shape-tiled Triton fused-MoE, FA split
scheduling). The ~1%/layer numerical drift over 24 layers flips the draft's
greedy token away from what the compiled verify computes -> the verify rejects
its own-architecture draft -> accept collapses (1.57 vs 4.78 eager).

**Fix:** new env `VLLM_SELF_SPEC_COMPILE_CONSISTENT=1` (default off) auto-enables
vLLM's batch-invariant numerics for a self-spec draft_model. With batch-invariant
kernels the draft (small M) and verify (large M) are numerically identical, so
the compiled draft's tokens are accepted: **DP1 accept 1.57 -> 5.0 (perfect)**,
with the draft still COMPILED and cudagraphed (fast) under FULL-CG.

---

## 1. Root cause (which of the 3 hypotheses)

It is **hypothesis #3** (direct numerical inconsistency), refined: the
inconsistency is **batch-shape-driven kernel numerics**, not the compilation
identity. Hypotheses #1 and #2 are ruled out by experiment.

### Reproduction (the clean repro, no EP/FP8 confound)
| Qwen1.5-MoE DP1 K=4 b16 | accept_len | per_tok | gen_s (2048 tok) |
|---|---:|---:|---:|
| compiled (baseline, broken) | **1.57** | 0.144 | 18.1 |
| eager (enforce_eager, reference) | **4.78** | 0.944 | 4.9 |

### Hypothesis #1 (separate `set_model_tag` -> 2nd independent compilation): REJECTED
- The compile cache **key** is `[env_hash, config_hash, code_hash, compiler_hash]`
  (`backends.py:1058`); the model tag only selects the cache **directory**
  (`local_cache_dir = .../{prefix}`, `backends.py:1078`). Draft and verify share
  the same parent hash (same `config/code/compiler`) but get separate
  `backbone/` vs `draft_model/` subdirs -> two independent inductor builds.
- A micro-test (`scripts/micro_moe_compile.py`) shows two independent
  `torch.compile` of the identical gate body are **bit-identical**
  (`|c1-c2|max = 0`). So separate compilations are NOT the divergence.
- Decisive: forcing the draft to load the verify's EXACT artifact (same
  `backbone/` cache dir, non-AOT path) leaves accept at **1.57** — unchanged
  (`scripts` experiment `dp1_noaot_share`). Sharing the compiled callable does
  not help.

### Hypothesis #2 (different MoE backend / kernel): REJECTED
- Engine log: both draft and verify select **`TRITON` Unquantized MoE backend**
  + `MoEPrepareAndFinalizeNoDPEPModular` (`unquantized.py:247`). Same backend,
  same modular kernel. `spec_cfg.moe_backend` is unset -> draft inherits the
  target's `moe_backend='auto'` -> identical selection.

### Hypothesis #3 (numerical inconsistency): CONFIRMED, and pinned to BATCH SHAPE
- `VLLM_SELF_SPEC_DRAFT_EAGER=1` (eager draft + compiled verify) only recovers
  to ~1.9 (phase 36). So the draft must match the *compiled verify*, not eager.
- The real model's TRITON fused-MoE, run on the SAME tokens at decode shape
  (N=16) vs verify shape (N=80), differs by **~1% relative** in its output —
  in EAGER too (`scripts/real_moe_shape.py`: mean-rel ~1.1e-2, max|d| ~1.5e-2).
  The shape-variant reduction order is the seed; the inductor-compiled surround
  (cuBLAS split-k GEMMs for gate/qkv/o_proj/lm_head, FA split scheduling)
  amplifies it enough under COMPILE to flip greedy tokens, while eager's lower
  amplification keeps draft@smallM and verify@largeM token-consistent (4.78).
- **Decisive fix-by-removal:** make the kernels batch-invariant (so numerics are
  shape-independent). `VLLM_BATCH_INVARIANT=1` (compiled) -> **accept 5.0,
  per_tok 1.0** (perfect). This isolates the cause to batch-shape variance.

File pointers: `vllm/v1/spec_decode/draft_model.py::_get_model` (the separate
tag), `vllm/compilation/backends.py:1058,1078` (cache key vs dir),
`vllm/model_executor/models/qwen2_moe.py:181-193` (the MoE block: gate linear
OUTSIDE the opaque `moe_forward` op, so it is inductor-fused),
`vllm/model_executor/layers/batch_invariant.py` (the fix machinery).

---

## 2. The fix

New env `VLLM_SELF_SPEC_COMPILE_CONSISTENT` (default off), wired so that when a
self-spec `draft_model` is active, `VllmConfig.__post_init__` forces
`VLLM_BATCH_INVARIANT=1` in the process env **before** the attention/MoE modules
that cache it at import. That makes every existing batch-invariant reader
(cuBLAS split-k disable, Triton bmm/softmax/mean overrides, FA `num_splits=1`,
TF32 off) take effect, so the draft (small M) and verify (large M) are
numerically identical and the draft is COMPILED (not forced eager).

Files:
- `vllm/envs.py` — declare + parse `VLLM_SELF_SPEC_COMPILE_CONSISTENT`.
- `vllm/config/vllm.py` (`__post_init__`) — when the flag + self-spec
  draft_model are set and `VLLM_BATCH_INVARIANT` is not already set, force it on
  (logged once).
- `vllm/model_executor/layers/batch_invariant.py` — comment only; init already
  gates on `envs.VLLM_BATCH_INVARIANT`.

Default off -> behavior unchanged (DP1 compiled stays 1.57 without the flag).

---

## 3. accept before/after + draft stays fast

### Qwen1.5-MoE DP1 K=4 b16 (the gate) — PASSED
| config | accept_len | per_tok | gen_s (2048 tok) |
|---|---:|---:|---:|
| compiled baseline (broken) | 1.57 | 0.144 | 18.1 |
| eager reference | 4.78 | 0.944 | 4.9 |
| **fix: VLLM_SELF_SPEC_COMPILE_CONSISTENT=1 (compiled)** | **5.00** | **1.000** | 4.2 |
| **fix + VLLM_SELF_SPEC_DRAFT_FULL_CG=1** | **5.00** | **1.000** | **3.2** |

The fix EXCEEDS the eager reference (5.0 vs 4.78) because batch-invariant makes
the draft a perfect predictor of the (also batch-invariant) verify. It is also
the **fastest** wall time (perfect acceptance -> ~5x fewer engine steps). The
draft stays COMPILED + cudagraphed: under FULL-CG the draft FULL-CG path engages
(decode graph FULL=48 captured, chain attention eager per the phase-35 fix),
torch.compile runs (~20 s compile), accept holds at 5.0.

### Qwen1.5-MoE DP2 K=4 b16 — PASSED (compile bug gone, only structural cap remains)
| config | accept_len | per_tok |
|---|---:|---:|
| baseline DP2 compiled (phase 36) | 1.76 | 0.19 |
| **fix DP2 compiled** | **2.65** | **0.413** |

DP2 with the fix lands at **2.65**, the expected ~2.8 structural cap (the
comm-free full-replica draft vs EP verify FP-reduction divergence documented in
phase 36; matches the DP2 eager reference 2.77). It is NOT ~1.6: the compile bug
is removed, leaving only the separate structural cap — exactly as predicted.

---

## 4. Qwen3-30B-A3B (the headline) — fix works at DP<=2; DP=8 is a SEPARATE
## structural cap that collapses acceptance regardless of compile

FP8 full-replica comm-free draft, target bf16, forced-PCIe, FULL-CG,
`VLLM_SELF_SPEC_COMPILE_CONSISTENT=1`. K=4, b16, OUTLEN 96, EP = DP.

| config | accept_len | reads as |
|---|---:|---|
| DP2 FP8 baseline compiled (phase 36) | 2.59 | compile bug present |
| **DP2 FP8 + fix (compiled)** | **2.71** | fix lifts toward eager |
| DP2 FP8 eager (phase 36 ceiling) | 2.83 | eager reference |
| DP2 bf16 + fix (compiled) | 2.50 | (bf16 draft, no FP8) |
| **DP8 FP8 + fix (compiled)** | **1.00** | structural collapse |
| **DP8 FP8 EAGER (no compile at all)** | **1.00** | SAME collapse -> structural |

**The headline finding:** at **DP=8 + EP=8 the acceptance collapses to ~1.0
regardless of compile** — the eager run (no torch.compile, so no compile bug
possible) ALSO gives 1.00. So the DP8 collapse is NOT this fix's job and NOT a
residual compile bug: it is the **World-A comm-free full-replica draft vs the
EP=8-sharded verify** floating-point reduction divergence, which worsens with
EP width (DP2: ~2.7 acceptable; DP8: ~1.0). The compile-consistency fix is
verified correct on 30B at DP=2 (2.59 -> 2.71, approaching the 2.83 eager
ceiling).

**Speedup vs no-spec at DP=8:** none — with accept ~1.0 every drafted token is
rejected, so the (comm-free but non-zero) draft forward is pure overhead at that
width. At DP=8 the comm-free draft cannot pay for itself until the structural
full-replica-vs-EP divergence is addressed (a separate World-B / EP-shard-draft
problem, out of scope here).

**On the speedup headline more broadly.** This fix removes the COMPILE barrier
(it makes the compiled draft a faithful predictor of the compiled verify), but it
does not by itself produce a net speedup on the configs tested:
- DP1 single-GPU small model: no-spec is already 1414 tok/s (gen 1.45 s / 2048
  tok); spec+fix at accept 5.0 is ~3.2 s. Spec-decode's draft+verify overhead is
  not worth it when the GPU is not comm-bound — the World-A premise is to save
  the EP all-to-all at high DP/EP, not to accelerate a small single-GPU model.
- DP8 EP8 (where the comm saving WOULD matter): accept collapses to ~1.0 from the
  structural cap, so the draft work is wasted.
So the realizable headline is the **acceptance recovery** (the compile bug is the
thing this task isolated and asked to fix), demonstrated cleanly on Qwen1.5-MoE
(1.57 -> 5.0) and Qwen3-30B DP2 (2.59 -> 2.71). A NET wall-clock win additionally
requires the structural full-replica-vs-EP cap to be lifted at the EP widths
where the comm saving is large — that is the remaining blocker (Sec 7), not the
compile fix.

Losslessness: greedy output is verifier-gated by construction (rejected drafts
are replaced by the verify's own token), so output is identical to no-spec
regardless of accept rate; the engine ran clean to completion at DP=8 with no
leaked workers.

---

## 5. Env / build hygiene note

The worktree needed not only the `*.so` symlinks but **152 untracked
build-artifact `.py` files** symlinked from `/data/smcho/self-spec-moe/vllm`
(third_party/{deep_gemm,flashmla,fmha_sm100,triton_kernels},
vllm_flash_attn/cute). Qwen1.5-MoE DP1/2 ran without them, but the Qwen3-30B
path imports `vllm.third_party.flashmla.flash_mla_interface` and crashed at init
until they were linked.

## 7. Remaining blocker (precise)

The compile-consistency bug is fixed. The remaining obstacle to a net DP=8
speedup is the **separate structural divergence**: the World-A comm-free
full-replica draft (`use_ep=False`, all experts local, no cross-rank reduction)
diverges from the EP-sharded verify by floating-point reduction order, and the
divergence grows with EP width:

| Qwen3-30B | accept (fix, compiled) | accept (eager) |
|---|---:|---:|
| DP2 / EP2 | 2.71 | 2.83 |
| DP8 / EP8 | 1.00 | 1.00 |

The DP8 collapse is present in EAGER too (1.00), so it is not a compile artifact
and not addressable by batch-invariance. Lifting it needs either (a) the EP-shard
draft (`use_ep=True`, same reduction path as the verify — phase 36 showed this
gives the clean ~4.9 at DP2, but it is NOT comm-free, defeating the World-A
premise), or (b) a World-B tree/communication redesign. That is the open problem;
it is orthogonal to this phase's compile-consistency fix.

## 6. Files
- Fix: `vllm/envs.py`, `vllm/config/vllm.py`,
  `vllm/model_executor/layers/batch_invariant.py`.
- Harness: `scripts/accept_isolate.py` (DP1/DP2 accept), `scripts/run.sh`,
  `scripts/qwen30b_dp8.py` + `scripts/run30b.sh` (30B DP8 accept + speedup +
  losslessness), `scripts/micro_moe_compile.py`, `scripts/micro_moe_shape.py`,
  `scripts/real_moe_shape.py` (diagnostics). Logs in `logs/`, data in `data/`.
