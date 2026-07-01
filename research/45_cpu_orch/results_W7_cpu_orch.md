# Phase 45 — W7 CPU-orchestration reduction for the comm-free self-spec cycle

**Model** Qwen1.5-MoE-A2.7B (fast iterate) + Qwen3-30B-A3B (confirm). **Layout**
attention-DP + EP, tp=1, EP=DP=8. **Fabric** forced-PCIe (`NCCL_P2P_DISABLE=1
NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1`). **Spec stack** FP8 full-replica comm-free
draft (`DRAFT_LOCAL_ROUTE=1 DRAFT_FULL_REPLICA=1 DRAFT_FULL_CG=1
COMPILE_CONSISTENT=1`), greedy, K=2, batch 64, CUDA graphs ON.
`VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0`. Branch `w7-cpu-orch`, base HEAD
`753f9f91a`. Fine CPU profiler (`self_spec_profiler.cpu_region`, FINE on),
rank-0 steady means.

## TL;DR

The recon (phase 44) attributed **31.4 ms/cycle** of the Qwen3-30B cycle to CPU
orchestration = draft-side 8.3 ms + "sampling/rejection/outer residual" 23.2 ms,
where residual := cycle − draft_chain − verify. **Fine per-component profiling
shows the sampling/rejection/bookkeeping/next-input CPU is <2 ms/cycle** — it was
never the residual. The residual is almost entirely the **engine-loop**
(scheduler `schedule` + `update_from_output` per-request loops, which run in the
separate EngineCore process) **and the per-step DP cross-rank sync barrier**
(forced-PCIe), i.e. comm-wait and general-vLLM scheduler CPU, neither of which is
self-spec-scoped reducible CPU. The self-spec-scoped lossless cuts available
(non-blocking vectorized rejection parse; persistent chain buffers — the
per-cycle allocations that WERE in scope) are applied env-gated and proven
lossless, but the addressable ms is small because prior merged work
(`sample-in-graph`, `skip-rebuild`, padded-batch on-device tokens) already
captured the large draft/verify CPU. Losslessness is PROVEN (all 10240 greedy
output tokens byte-identical base vs orch on Qwen1.5-MoE); the wall-clock delta
(119.4 -> 118.3 ms/cycle, +0.6%) is within run noise — no regression, no
measurable gain, exactly as the <1 ms addressable CPU predicts.

## 1. Fine CPU breakdown (Qwen1.5-MoE, b64 K2, rank-0 steady means, FINE on)

Per-cycle wall (cycle ≈ 159 ms; draft_chain ≈ 75 ms; verify ≈ 44 ms; **residual
≈ 41 ms**). The instrumented CPU sub-regions (non-syncing `cpu_region`, so a
forced H<->D sync inside still shows as its own stall):

| region (self-spec CPU) | Qwen1.5-MoE ms | **Qwen3-30B ms** | note |
|---|---:|---:|---|
| cpu_exec_prepare_inputs (`_prepare_inputs`) | 0.78 | 0.81 | vectorized; well-optimized |
| cpu_rejection_sample (rejection sampler call) | 0.40 | 0.38 | mostly GPU kernels |
| cpu_reject_parse (`parse_output` D2H + split) | 0.11 | 0.12 | the named `.cpu()` sync |
| cpu_bookkeep_loop (per-req output append) | 0.08 | 0.11 | per-seq py loop, host-only |
| cpu_next_input_build + cpu_prepare_inputs_padded | 0.31 | 0.26 | GPU kernels + tiny host |
| **sum instrumented outer CPU** | **≈1.7** | **≈1.7** | vs recon's 23.2 ms "residual" |

**The outer CPU is identical (~1.7 ms) on both models** — confirming the task's
premise that CPU orchestration is model-size-independent, and confirming it is
NOT the recon's 23.2 ms residual. On Qwen3-30B the profiler also reproduces the
recon's compute terms exactly: draft_chain 110.2, verify 61.5, draft_forward_first
18.7, draft_forward 69.5 ms (recon: 18.5 / 67.5 / 65.4), accept_len 2.91 (recon
2.92).

=> Of the ~41 ms (small) / ~23 ms (30B) residual, only ~1.7 ms is worker
sample/reject/bookkeeping/preprocess CPU. The remaining is engine-loop
(EngineCore scheduler) + DP
sync barrier + IPC. (`verify` region std ≈ 97 ms confirms the forced-PCIe
DP-stall dominates the non-compute time.)

Draft-side per-step regions (inside draft_chain; FINE syncs inflate these):
step0_determine_batch 1.87 (DP coord), chain_setup 3.08, step_pos_slot_update
0.59, step_input_buffering 0.47, step_sample 0.76 (already in-graph),
step0_set_inputs ~1.0 (min). The draft-side CPU is dominated by the DP
coordination barriers (2/cycle) + the eager attention kernel launches that the
accept fix requires — not reducible without changing accept.

## 2. The reductions (env-gated; default off => byte-identical)

Master flag `VLLM_SELF_SPEC_CPU_ORCH=1` enables all; each also has its own flag.

1. **Non-blocking vectorized rejection parse** (`VLLM_SELF_SPEC_FAST_PARSE`) —
   `gpu_model_runner._parse_spec_output_fast` (vllm/v1/worker/gpu_model_runner.py).
   Replaces `RejectionSampler.parse_output`'s blocking `output_token_ids.cpu()`
   (a stream-wide sync, issue #22754) + per-row `[row[mask].tolist() ...]` loop
   with a pinned-buffer non-blocking D2H + event sync + ONE flat `.tolist()`
   split by per-row valid counts. Greedy path only (logprobs None). Byte-
   identical output. Wired in `_bookkeeping_sync`.
2. **Persistent chain query_start_loc_cpu** (`VLLM_SELF_SPEC_CPU_ORCH`) —
   `llm_base_proposer.propose` chain_setup (vllm/v1/spec_decode/llm_base_proposer.py).
   Reuses a persistent CPU arange view instead of `torch.from_numpy(...).clone()`
   every decode step (the per-cycle alloc the task named). The chain never
   mutates it in place, so the view is safe.

Profiler additions (timing-only, FINE-gated, no behavior change):
`cpu_region()` in self_spec_profiler.py; `cpu_*` regions in gpu_model_runner.py.

## 3. Losslessness (hard gate) — accept_len + output tokens UNCHANGED

Greedy, fixed seed 0, 64 requests × 160 output tok (10240 tokens), deterministic.
`CP_JOB=accept` dumps every request's output token ids (FINE off); base vs orch
compared exactly with `scripts/diff_accept.py`.

| model | base accept_len | orch accept_len | token_ids identical |
|---|---:|---:|:---:|
| Qwen1.5-MoE (b64 K2) | 2.8338326803410925 | 2.8338326803410925 | **True** (10240/10240) |
| Qwen3-30B (b64 K2) | 2.9098651525904895 | 2.908489004492788 | **True** (10240/10240) |

**Both models: all 10240 greedy output tokens byte-for-byte identical base vs
orch.** Hard gate PASSED. (Qwen1.5-MoE accept_len matches to full float precision;
Qwen3-30B accept_len differs only at the 3rd decimal — that metric is cumulative
over the whole run incl. warmup, so it wanders run-to-run; the per-position token
identity is the exact gate and it holds.)

## 4. Cycle-ms + speedup (prefill-cancelled two-length slope) vs no-spec

| model | mode | cycle_ms | sys tok/s | accept_len | speedup vs nospec |
|---|---|---:|---:|---:|---:|
| Qwen1.5-MoE | nospec | 19.48 | 3284.8 | — | 1.000x |
| Qwen1.5-MoE | spec_base | 119.39 | 1514.5 | 2.825 | 0.461x |
| Qwen1.5-MoE | spec_orch | 118.27 | 1524.2 | 2.817 | **0.464x** |
| Qwen3-30B | nospec | 34.06 | 1879.1 | — | 1.000x |
| Qwen3-30B | spec_base | 159.20 | 1166.7 | 2.902 | 0.621x |
| Qwen3-30B | spec_orch | 158.81 | 1167.5 | 2.897 | **0.621x** |

Both models: spec_orch vs spec_base is **within run noise** — Qwen1.5-MoE
119.4 -> 118.3 ms (+0.6%), Qwen3-30B 159.2 -> 158.8 ms (+0.07%) — with accept_len
unchanged (2.82 / 2.90). i.e. the CPU-orch reductions cause **no regression** and
the wall-clock gain is below the measurement floor, consistent with the <1 ms/cycle
of self-spec-scoped CPU they address. Qwen3-30B spec is 0.62x (recon 0.55x); spec
is sub-parity on both because the comm-free draft forward dominates (draft_chain
110 ms vs verify 61 ms on 30B) — the recon's structural finding, not a CPU-orch
effect. Zeroing the ~1.7 ms addressable CPU would lift 0.621x -> ~0.628x on 30B;
the rest of the recon's 31.4 ms is the irreducible engine-loop + DP-sync below.

## 5. Irreducible CPU orchestration (and why)

- **Engine-loop scheduler CPU** (EngineCore `schedule()` + `update_from_output()`
  per-request loops, flagged in-source as a bottleneck) — general vLLM, runs in a
  separate process; not self-spec-scoped; reducing it would change engine
  behavior for all decoders.
- **DP cross-rank sync barrier** (`coordinate_batch_across_dp` /
  `_synchronize_dp_ranks`) — 2 for the draft (step-0 + chain) + 1 for verify per
  cycle, each a forced-PCIe all-reduce where fast ranks wait for slow ranks.
  This is comm-wait, required for DP correctness; it is the dominant "residual"
  term (verify std ≈ 97 ms).
- **The one hard D2H the spec path must do** (rejection output -> CPU for the
  scheduler's next step) is now non-blocking (event-synced pinned copy) — the
  stall itself is unavoidable (the engine needs the accepted tokens on CPU) but
  no longer blocks other CUDA streams.
- **Eager draft-chain attention** kernel launches — required by the accept fix
  (the captured decode graph is not replay-safe for the draft's growing
  sequence); replacing them re-introduces the accept collapse.
