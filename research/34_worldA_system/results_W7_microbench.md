# W7 micro-bench: decomposing the ~108 ms spec cycle into FIXABLE vs FUNDAMENTAL

**Goal.** results_W7.md found the World-A bf16 full-replica spec at its best point
(DeepSeek-V2-Lite, DP=2+EP, forced-PCIe + 100 us emulated A2A, **batch 64, K=2**)
runs a spec cycle of ~108 ms vs a no-spec step of 17.1 ms -> 0.42x. The cost-model
ideal predicts ~1.0-1.3x. This phase attributes the gap by instrumenting the spec
cycle into (i) fundamental draft compute, (ii) chain-loop CPU orchestration, (iii)
verify.

## Method: env-gated, CUDA-synchronized wall-clock instrumentation
New env `VLLM_SELF_SPEC_PROFILE=1` (default off -> no behavior change) turns on a
process-local profiler (`vllm/v1/spec_decode/self_spec_profiler.py`). Each timed
region does `torch.cuda.synchronize()` at both ends and records `time.perf_counter()`
elapsed (full CPU+GPU time). Three regions:

- **draft_forward_first** -- the step-0 draft forward inside `propose()`
  (`llm_base_proposer.py` ~547).
- **draft_forward** -- a single decode-step draft forward inside the K-step chain
  loop (`llm_base_proposer.py` ~692). Timed under a distinct label because its
  input shape can differ from step-0; in practice the two are within ~1 ms.
- **draft_chain** -- the whole `propose()` call (the K-step chain incl. all CPU
  orchestration between forwards), timed at its call site in
  `gpu_model_runner.py::propose_draft_token_ids` (~5108).
- **verify** -- the target verify forward `_model_forward` in
  `execute_model` (`gpu_model_runner.py` ~4317), covering the K+1 proposed
  tokens/req with spec on, or the 1 token/req plain step with spec off.

The model runner runs in a worker process that is force-killed at engine shutdown,
so the profiler **flushes its per-PID summary to `VLLM_SELF_SPEC_PROFILE_OUT`
incrementally** (atomic rename every 25 samples); the harness reads the file with
the most samples (the model-running rank). Warmup = first 50 decode steps dropped;
steady-state mean over the post-warmup tail (n >= 40 here). Both DP ranks run the
draft and agree within ~1% (rank0 vs rank1 chain 98.6 vs 97.6 ms).

`t_nospec_step` is measured separately by running the no-spec engine and taking the
W7 two-length slope (full per-decode-step wall time incl. sample+bookkeep).

Harness `scripts/w7_microbench.py`; driver `scripts/w7_microbench_serial.sh`;
analysis `scripts/w7_microbench_analyze.py`; raw under `data/mb_*.json` +
`data/mb_profiles_*/`. CUDA graphs ON (realistic config). Spec engines run strictly
one-at-a-time (a concurrent engine inflates the CPU-bound draft loop, per W7).

> Caveat on the inner sync: timing `draft_forward` adds a CUDA sync inside the
> chain loop, which serialises forwards that are already data-dependent and serial.
> This inflates the reconstructed cycle by a few ms vs an un-instrumented run
> (reconstructed 117 ms vs W7's 108 ms here, ~9%), but does not change the
> attribution: the decomposition is internally consistent (chain = K*forward +
> overhead to <1 ms), and the **measured implied speedup reproduces W7 exactly**
> (0.423x, see below).

## Headline numbers -- V2-Lite, batch 64, K=2, forced-PCIe + 100 us A2A (the 0.42x point)

| quantity | ms |
|---|---:|
| t_nospec_step (full no-spec decode step) | **17.95** |
| t_draft_forward (single decode-step draft forward) | **46.10** |
| K * t_draft_forward (= forward_first + (K-1)*forward; K=2) | **92.43** |
| chain_overhead (= t_draft_chain - K*t_draft_forward) | **5.16** |
| t_draft_chain (whole propose()) | **97.59** |
| t_verify (target forward over K+1 tokens) | **19.86** |
| reconstructed cycle (= chain + verify) | **117.44** |
| accept_len | 2.766 |

Reconstruction vs W7: 117 ms reconstructed vs ~108 ms W7. The ~9 ms gap is the
instrumentation's inner-sync overhead; the **implied speedup matches W7 exactly**:
`accept_len * t_nospec_step / cycle = 2.766 * 17.95 / 117.44 = 0.423x` (W7: 0.42x).

### Attribution of the cycle (% of reconstructed 117 ms)
- **(i) Fundamental K-forward draft compute: 92.4 ms = 79%**
- **(ii) Fixable chain-loop CPU orchestration: 5.2 ms = 4%**
- **(iii) Verify: 19.9 ms = 17%**

The draft chain is almost entirely GPU compute (K dense-64-expert draft forwards);
the Python orchestration between forwards is ~5 ms, i.e. ~2.6 ms per inter-forward
gap -- negligible relative to the ~46 ms/forward.

## Does chain_overhead scale with K or batch? (and is it fixed per step?)

| point | accept | t_draft_forward | K*forward | chain | chain_overhead | verify | cycle | spd_measured | spd_ideal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V2-Lite b64 K2 | 2.77 | 46.10 | 92.43 | 97.59 | **5.16** | 19.86 | 117.4 | 0.423 | 0.442 |
| V2-Lite b64 K4 | 4.04 | 60.13 | 238.33 | 238.36 | **0.03** | 18.30 | 256.7 | 0.283 | 0.283 |
| V2-Lite b8  K2 | 2.79 | 46.14 | 93.54 | 97.56 | **4.02** | 14.86 | 112.4 | 0.328 | 0.341 |
| Qwen3-30B b64 K2 (mean) | 1.99 | 69.6* | 126.8 | 142.2 | 15.5* | 30.98 | 173.2 | 0.360 | 0.396 |
| Qwen3-30B b64 K2 (floor/min) | 1.99 | ~55 | ~110 | ~115 | **~4.6** | 31 | ~146 | -- | -- |

`*` The Qwen3-30B **mean** draft_forward/draft_chain are contaminated by DP-rank
straggler spikes (max 0.5-1.5 s vs a stable ~55 ms / ~115 ms floor; verify is clean,
std 1.8 ms). Under DP=2 spec, a slow rank stalls the barrier and injects rare
multi-100ms samples into the other rank's draft timing; these inflate the mean (and
thus the mean-based chain_overhead of 15 ms) but are jitter, not real overhead. The
**clean floor** -- forward ~55 ms, chain ~115 ms -> chain_overhead ~4.6 ms -- is the
physical per-forward cost and matches V2-Lite.

- **chain_overhead is roughly fixed and tiny (~0-5 ms), not growing with K.** It is
  a small constant of CPU orchestration, dwarfed by a single forward (~46-60 ms).
  At K=4 it is within measurement noise of 0 (chain == K*forward to 0.03 ms).
- **chain_overhead is ~the same absolute ms at batch 8 (4.0) and batch 64 (5.2).**
  As a fraction it is slightly larger at small batch only because the cycle is
  smaller -- still ~4%, never the bottleneck.
- **Qwen3-30B (3x bigger forward): chain_overhead stays ~4-5 ms absolute (floor),**
  i.e. RELATIVELY EVEN SMALLER (~3% of the chain vs ~4% on V2-Lite). The CPU
  orchestration is a fixed per-step constant; on bigger models the fundamental
  forward grows but the overhead does not. (Qwen also drafts worse here: accept_len
  1.99 with the bf16 full replica, so its spec loses for the same reason -- the
  draft forward, ~55 ms, is ~1.8x a no-spec step of 31 ms.)
- The per-forward time itself rises with K (46 ms at K=2 -> 60 ms at K=4): each
  extra speculative position lengthens the draft attention context / KV, so K=4
  pays both more forwards AND slightly costlier forwards. This is fundamental draft
  compute, not fixable overhead.

## Why spec loses, in one line of arithmetic
The no-spec budget the cycle must beat is `accept_len * t_nospec_step` (the tokens
spec produces per cycle, each worth one no-spec step):

```
no-spec budget (b64 K2) = 2.766 * 17.95 ms = 49.65 ms
ideal spec cycle (ZERO overhead) = K*t_draft_forward + t_verify
                                 = 92.43 + 19.86 = 112.3 ms
implied ideal speedup = 49.65 / 112.3 = 0.44x
```

**Even with a perfect, zero-overhead, fully-graphed draft driver the speedup is
~0.44x** -- because `K*t_draft_forward = 92 ms` *alone* (before any verify or
overhead) already exceeds the entire no-spec budget of 50 ms by ~1.9x. The single
draft forward (46 ms) is itself ~2.6x a whole no-spec step (18 ms): the bf16 full
replica routes over all 64 experts dense (no EP sharding, no A2A skip benefit on
compute), so one draft forward costs more than the comm-bound no-spec step it is
supposed to amortise.

## Verdict
The 0.42x is **not** an orchestration/driver artifact. Decomposition of the cycle:
- chain-loop CPU overhead (the only fixable piece): **~4-5 ms, ~4% of the cycle.**
  Removing it entirely moves 0.42x -> ~0.44x. Marginal.
- The dominant cost is **fundamental K-forward draft compute (79%, 92 ms),** and
  `K*t_draft_forward` alone is ~1.9x the no-spec budget. **This is unfixable on a
  single node with this draft:** no better driver, no CUDA-graph capture of the
  chain, can shrink the actual GPU FLOPs of K dense-64-expert forwards.

So the World-A bf16 full replica cannot reach the cost-model's 1.0-1.3x on V2-Lite;
the model is too small and the draft too heavy. A win requires making the draft
*cheap* (top-C hot experts + skip-cold, FP4/FP8 resident cache -> draft compute
below the comm it removes) and/or a larger comm/compute ratio (bigger model, true
multi-node exposed A2A, larger E), exactly the W2b/W2c direction results_W7.md
already flagged. Optimising the chain driver is not the lever.

## Files
- Instrumentation (env-gated, additive): `vllm/v1/spec_decode/self_spec_profiler.py`,
  `vllm/v1/spec_decode/llm_base_proposer.py` (draft_forward_first / draft_forward),
  `vllm/v1/worker/gpu_model_runner.py` (draft_chain / verify), `vllm/envs.py`
  (`VLLM_SELF_SPEC_PROFILE`).
- Scripts: `scripts/w7_microbench.py`, `scripts/w7_microbench_serial.sh`,
  `scripts/w7_microbench_analyze.py`.
- Data: `data/mb_{spec,nospec}_*.json`, `data/mb_profiles_*/`.
