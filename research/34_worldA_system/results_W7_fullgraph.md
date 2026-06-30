# W7 full-graph: the draft was EAGER, not "fundamental compute" — FULL cudagraphs cut the draft forward ~4x

**Goal.** results_W7_microbench.md attributed the World-A single-node loss (0.42x,
V2-Lite, DP=2+EP, forced-PCIe + 100 us A2A, batch 64, K=2) to *fundamental* draft
compute: `t_draft_forward ~= 46 ms ~= 2.5x` the verify (~20 ms) of the same model,
and concluded it was **unfixable** ("no CUDA-graph capture ... can shrink the actual
GPU FLOPs of K dense-64-expert forwards"). This phase tests that conclusion by
forcing the draft to use FULL cudagraphs and re-measuring `t_draft_forward`.

**Headline: the prior conclusion was wrong about the cause.** The `draft_model`
proposer never had its cudagraph dispatcher keys initialized, so the draft ran
**fully eager** (no cudagraphs at all — not even PIECEWISE). ~34 ms of the ~46 ms
"draft forward" was **kernel-launch overhead**, not GPU FLOPs. With FULL cudagraphs
wired in, `t_draft_forward` drops **47.96 ms -> 11.79 ms (4.07x)** — below the
verify (20 ms). The "draft is 2.5x the verify" gap was an eager-vs-graphed
artifact, not a fundamental property of the draft.

## What was actually broken (root cause)
`gpu_model_runner.py::_check_and_update_cudagraph_mode` only calls
`self.drafter.initialize_cudagraph_keys(...)` when the proposer is Eagle /
ExtractHiddenStates — the condition **excludes `uses_draft_model()`**. For the
`DraftModelProposer` the dispatcher therefore stayed at `keys_initialized=False`,
so every draft `dispatch()` returned `CUDAGraphMode.NONE` and the draft forward ran
eager. Confirmed at runtime by instrumenting the decode-step dispatch:

```
[W7-DBG] draft decode-step: mode=NONE  ... wrapper=CUDAGraphWrapper entries=0   (before fix)
[W7-DBG] draft decode-step: mode=FULL  ... wrapper=CUDAGraphWrapper entries=23  (after fix)
```

So the prior microbench's 46 ms draft forward was an **eager** forward. (This is
also why PIECEWISE-vs-"FULL" looked identical in a first pass: with keys never
initialized, flipping the mode in `initialize_cudagraph_keys` was a no-op because
the method was never called.)

## The change (behind `VLLM_SELF_SPEC_DRAFT_FULL_CG`, default off)
1. `gpu_model_runner._check_and_update_cudagraph_mode` (~6960): also init the
   drafter's cudagraph keys for `uses_draft_model()` when the proposer opts in.
2. `gpu_model_runner._maybe_wrap_in_cudagraph` (~5304): wrap the draft model in a
   FULL `CUDAGraphWrapper` (mirroring the verify model) when the flag is set.
3. `gpu_model_runner._dummy_run` (~5990): capture the draft's FULL graph on the
   FULL (uniform-decode) capture pass, sized at `num_reqs` tokens (each draft
   decode step is 1 token/seq, vs the verify's 1+K).
4. `llm_base_proposer.initialize_cudagraph_keys`: emit FULL_AND_PIECEWISE keys
   (FULL for decode steps, PIECEWISE for the variable-shape step-0) with
   `uniform_decode_query_len=1`.
5. `llm_base_proposer._determine_batch_execution_and_padding` /
   `propose()` / `dummy_run`: dispatch the decode-step forwards with
   `uniform_decode=True` and thread the FULL `batch_descriptor` into the draft's
   `set_forward_context` (the FULL `CUDAGraphWrapper` keys on it).

`envs.py`: new `VLLM_SELF_SPEC_DRAFT_FULL_CG` (default 0). PIECEWISE/eager remains
the default, unchanged.

It needed **more than flipping the mode**: the dispatcher keys had to be
initialized for draft_model at all, the draft model had to be FULL-wrapped, the
FULL graph captured on the right pass at the right (num_reqs) shape, and the
batch_descriptor threaded through the forward context.

## Numbers — V2-Lite, batch 64, K=2, forced-PCIe + 100 us A2A (the 0.42x point)

| run | draft_fwd_first | draft_forward | verify | draft_chain | accept_len |
|---|---:|---:|---:|---:|---:|
| PIECEWISE/eager baseline (ref, `_b`) | 46.33 | **46.10** | 19.86 | 97.59 | 2.766 |
| eager control (`_pw`, re-run) | 48.32 | **47.96** | 19.35 | 100.30 | 2.724 |
| FULL cudagraph (`_fullcg`) | 15.22 | **11.79** | 20.66 | 31.35 | **1.991** |

`t_nospec_step = 17.95 ms`.

- **draft_forward: 47.96 -> 11.79 ms (4.07x faster).** draft_chain 100.3 -> 31.3 ms.
  verify unchanged (19.3 -> 20.7 ms). This **confirms** the speed hypothesis: the
  draft forward was launch-overhead-bound, not FLOP-bound.

## The catch: accept_len collapses 2.72 -> 1.99 (FULL-draft correctness bug)
With the FULL-captured draft, `accept_len` drops from 2.72 to 1.99 — the 2nd draft
token (produced by the FULL decode-step forward; the 1st comes from the PIECEWISE
step-0) is systematically rejected. Spec decoding stays **lossless** (the verify
rejects bad drafts; final output tokens are identical, confirmed on a greedy
smoke), but the *efficiency* degrades. Root cause: the draft's per-decode-step
attention metadata is built fresh each step via `build_for_drafting(fast_build=
True)` and is **not in the persistent buffers the captured FULL graph reads** —
the replayed graph attends against capture-time metadata. Backend = FLASH_ATTN_MLA.
Making the draft's per-step attention metadata cudagraph-capturable (in-place
persistent buffers, as the main runner does for the verify) is the remaining work;
it is a real vLLM feature, not a flag.

## Implied single-node speedup
`speedup = accept_len * t_nospec_step / (K * t_draft_forward + t_verify)`,
`t_nospec_step = 17.95`, `K = 2`.

| run | accept_len | t_draft_fwd | K*t_draft + verify | **implied speedup** | chain-based |
|---|---:|---:|---:|---:|---:|
| eager control | 2.724 | 47.96 | 115.87 | **0.42x** | 0.41x |
| FULL, observed (collapsed accept) | 1.991 | 11.79 | 44.24 | **0.81x** | 0.69x |
| FULL, **IF accept_len preserved (2.724)** | 2.724 | 11.79 | 44.24 | **1.11x** | 0.94x |

**The cudagraph fix alone nearly doubles the implied speedup (0.42 -> 0.81), and a
*correct* FULL draft (accept_len preserved) crosses 1.0 (1.11x by K*t+verify; 0.94x
by the real chain).** So single-node World A is recoverable-to-borderline with the
cudagraph fix **once the draft FULL-capture correctness (accept_len) is fixed** —
the prior "fundamental, unfixable" verdict does not hold.

## Outcome
**(a), with a correctness caveat.** draft_forward dropped toward ~12 ms (4x) —
CONFIRMED the launch-overhead hypothesis; the ~46 ms was eager, not fundamental.
Implied speedup with preserved acceptance is **1.11x (>1)**. The measured speedup
today is 0.81x only because the FULL-captured draft loses ~0.7 accept tokens to an
attention-metadata staleness bug (the "real feature" part of (b)).

## Files
- Change: `vllm/envs.py` (`VLLM_SELF_SPEC_DRAFT_FULL_CG`),
  `vllm/v1/spec_decode/llm_base_proposer.py`,
  `vllm/v1/worker/gpu_model_runner.py`.
- Data: `data/mb_spec_v2lite_{pw,full,fullcg}_b64_K2_a2a100us.json`
  (`_full` = the keys-not-initialized no-op pass, kept for the record; `_fullcg` =
  the real FULL draft), `data/mb_profiles_spec_v2lite_{pw,full,fullcg}_*/`.
- Harness: `scripts/w7_microbench.py` (unchanged).
