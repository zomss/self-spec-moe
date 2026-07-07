# Phase 71 findings — "full-CG forces dense MoE" is FALSE (and the 25.2 ms is misattributed)

Read-only investigation. Verdict on the Phase-70 leading hypothesis and the
reviewer's objection. No `vllm/` code changed.

## TL;DR verdict

1. **vLLM's normal CUDA-graphed decode MoE is SPARSE** (routing-driven; only the
   activated `top_k` experts' weight rows are read). The reviewer is correct.
2. **"Full cudagraph forces dense MoE" is FALSE.** There is no code path where
   FULL-vs-PIECEWISE swaps in a dense (all-experts) MoE. The same sparse Triton
   `fused_moe_kernel` is used in both modes, and the FP8 full-replica draft uses
   that same sparse path (top-8, not 128).
3. **The 25.2 ms is MISATTRIBUTED.** The `draft_forward` region times the
   *entire* 48-layer model forward (attention + MoE + norms + fp8 quant + KV
   write), not the MoE. Nothing in P69/P70 measures MoE in isolation.
4. **The real mechanism is a per-forward LATENCY floor of a deep single-token
   (b1) forward, not weight traffic.** 25 ms exceeds even the read-the-whole-30B
   -model ceiling (~10-12 ms fp8), so it cannot be any weight-bandwidth story,
   dense or sparse. The apparent "5x" (5.3 -> 25.2 ms) is a metric artifact:
   5.3 ms was kernel-*busy*-time, 25.2 ms is a CUDA-*synced wall*. On a
   same-metric basis the draft forward is ~the same (~20-25 ms) under PIECEWISE
   and FULL — FULL-CG neither caused nor cured it.
5. **Not fixable by cudagraph coverage.** The bottleneck is the serial
   dependency chain of a b1 forward across 48 layers, which the FULL graph does
   not remove (it removes only host launch overhead). Real levers: a shallower/
   smaller draft, fewer experts/token (TOPC), or more tokens per rank.

---

## Q1 — Is vLLM's normal CUDA-graphed decode MoE sparse or dense? SPARSE.

The unquantized/FP8 fused-MoE Triton path is routing-driven. Tokens are sorted
by their routed expert and only the routed experts' weight rows are read:

- `vllm/model_executor/layers/fused_moe/moe_align_block_size.py:19-27,74` —
  `moe_align_block_size` sorts the `[num_tokens, top_k]` routing assignments into
  per-expert blocks and marks experts not on this rank as `-1`. The padded token
  budget is `topk_ids.numel() + num_experts*(block_size-1)` — proportional to
  **activated (token,expert) pairs**, not to `num_experts`.
- `vllm/model_executor/layers/fused_moe/fused_moe.py:152-154` — each program
  early-exits (`return`) when `pid_m*BLOCK_SIZE_M >= num_tokens_post_padded`, a
  value loaded from a **device tensor at runtime**.
- `fused_moe.py:160-177` — `off_experts == -1` => write zeros and return (experts
  not resident on this rank do **no** GEMM).
- `fused_moe.py:185-191` — the weight pointer is `b_ptr + off_experts*stride_be`:
  only the routed expert's rows are loaded.

So the weight read is `~ num_tokens*top_k` (activated experts), never
`num_experts`. This is the kernel vLLM uses for ordinary decode.

## Q2 — Does CUDA-graph capture change the MoE kernel or its weight traffic? NO.

**Same kernel, fixed launch shape, runtime-gated work.** The kernel grid is
over-launched to the *capture-time* maximum `EM` and each block self-gates on the
device-side `num_tokens_post_padded`:

- `fused_moe.py:749-764` — `EM = sorted_token_ids.size(0)` (a capture-time
  constant) and `grid = cdiv(EM, BLOCK_SIZE_M) * cdiv(N, BLOCK_SIZE_N)`. The
  block count is fixed (cudagraph-safe), but over-launched blocks hit the
  `num_tokens_post_padded` early-exit above and do nothing.

**FULL vs PIECEWISE only changes attention handling, never MoE.** The splitting
ops are the attention ops only (`vllm/config/compilation.py:1097-1157`,
`set_splitting_ops_for_v1`; the `_attention_ops` / `unified_kv_cache_update`).
MoE is **not** a splitting op, so it is captured inside the graph in *both*
PIECEWISE and FULL modes. PIECEWISE runs attention eagerly between compiled
pieces; the MoE kernel is identical.

**The FP8 full-replica draft uses the same sparse path.** Traced end to end:

- `vllm/model_executor/layers/fused_moe/config.py:1211-1230` — the
  `VLLM_SELF_SPEC_DRAFT_FULL_REPLICA` branch returns a parallel config with
  `ep_size=1, ep_rank=0, use_ep=False`.
- `config.py:1039-1040` — `use_all2all_kernels = dp_size>1 and use_ep` => **False**
  (use_ep is False). `config.py:1067-1069` — `use_batched_activation_format`
  requires all2all kernels => **False**.
- `vllm/model_executor/layers/fused_moe/oracle/fp8.py:259-263,366-377` — Standard
  (not Batched) activation format => backend `TRITON` => `TritonExperts`
  (`experts/triton_moe.py:55`), the sparse contiguous impl. `make_fp8_moe_kernel`
  builds it **without** `max_num_tokens` padding (`oracle/fp8.py:597-610`).
- `experts/triton_moe.py:294` -> `fused_moe.py` `_prepare_expert_assignment` ->
  `moe_align_block_size` (the Q1 sparse sort). Weight read `~ top_k`.
- `expert_map_manager.py:90-91,192` — full-replica => `expert_map = None`, so
  `local_route.py:109-110` `mask_router_logits_to_resident` is a no-op and the
  draft routes the normal top-8. The optional TOPC prune
  (`fused_moe/local_route.py:45-81`) only *reduces* experts/token to `C <= top_k`.
- FP8 dequant is per-tile **inside** the GEMM (`use_fp8_w8a8`,
  `triton_moe.py:351`); there is no full-weight dequant of all 128 experts.

Cudagraph mode does not re-select the experts impl: `VLLM_SELF_SPEC_DRAFT_FULL_CG`
is read only in the spec orchestration / capture layer
(`llm_base_proposer.py:239,348-349`; `gpu_model_runner.py:6159,7161`), never in
MoE kernel selection (fixed at load time in `Fp8MoEMethod`).

**One genuine per-rank difference (still sparse):** an EP verify rank computes
only the subset of a token's 8 routed experts that are locally resident (the rest
map to `-1` and are offloaded via all-to-all); the full-replica draft rank
computes **all 8** routed experts locally (`expert_map=None`, nothing skipped).
That is ~8x the expert GEMM *work per rank* versus one EP rank — but it is still
`top_k` sparse experts per token (not 128, not batched-padded), and it is the
intended comm-free trade (local compute instead of all-to-all). This is
**independent of cudagraphs** and does not make the draft "dense."

## Q3 — Re-attribute the 25.2 ms: what the region really measures

**The `draft_forward` region encloses the whole model forward, not MoE.**

- `vllm/v1/spec_decode/llm_base_proposer.py:1584-1595` — the region wraps exactly
  one call, `self.model(**model_kwargs)` (the `else` branch at :1590;
  `sample_in_graph` is forced False because `_disable_sample_in_graph=True` under
  `_draft_fullcg`, `:388`). `set_forward_context(...)` is entered at :1570,
  *outside* the region.
- Sampling (`step_sample`, :1611), window compaction (`step_build_attn_md`,
  :1434), pos/slot update (:1406) and input buffering (:1519) are **separate**
  regions, each **< 1 ms** (see data below). The scratchpad gather + masked SDPA
  run *inside* the model forward
  (`attention.py:776-781` -> `scratchpad_attn.py:87-114`), so they are inside the
  25 ms, not measured separately.
- `vllm/v1/spec_decode/self_spec_profiler.py:87-113` — `region()` brackets both
  ends with `torch.cuda.synchronize()`, so 25.2 ms is a **CUDA-synced wall**
  (CPU + drained GPU work), not kernel-busy-time.

So `draft_forward` = the entire 48-layer draft forward replayed as one FULL
cudagraph (embed + 48x[norm -> windowed SDPA attention + KV write -> norm -> MoE]
+ final norm + fp8 quant). **No region or trace in P69/P70 isolates the MoE
contribution.** P70's MoE attribution (results.md:110-116) is explicitly "the
leading hypothesis," argued only from token-scaling (verify 40 tok ~ 40 ms vs
draft 1 tok ~ 25 ms => weight/fixed-bound). That argument supports "not
compute-bound," but does **not** distinguish MoE weights from attention/proj
weights, and does not establish density.

### Fine-region data (P70 fixed FULL-CG, K4 b8, 16k/W512/cap544, per rank)

`research/70_fullcg_coverage/data/prof_fine_fullcg/` (rank 7 / rank 0):

| region | mean ms | what it is |
|---|---|---|
| `draft_forward` (chain step) | 25.4 / 25.3 | whole model forward, FULL-graph replay |
| `draft_forward_first` (step-0) | 14.3 / 14.5 | whole model forward, step-0 |
| `verify` (target, 5 tok/seq) | 40.8 / 40.9 | target verify forward |
| `step_sample` | 0.64 | eager sample (logits+argmax) |
| `step_build_attn_md` | 0.48 | window compaction only |
| `step_pos_slot_update` | 0.29 | |
| `step_input_buffering` | 0.23 | |

`draft_forward` variance is large (min 10.6, max 30.6, std 7.5 ms) — unlike a
fixed-shape graph doing fixed work. The fine-profile pids (826477+) are the
FULL-CG **replay** path (they log "FULL-CG replay reads capture-time pointers",
`llm_base_proposer.py:1507`), not an eager fallback — so this is a genuine
replay, yet still 25 ms.

### The weight-traffic ceiling rules out any dense/sparse weight story

Qwen3-30B-A3B: 48 layers, 128 experts, top-8, ~1.8B active expert params/token.
- **Sparse b1 MoE** (8 experts x 48 layers, fp8): ~1.8 GB => ~0.6-0.9 ms.
- **Dense b1 MoE** (all 128 experts, fp8): ~29 GB => ~10-12 ms.
- **Read the ENTIRE 30B model once** (fp8, ~30 GB @ ~2.5-3 TB/s): **~10-12 ms**.

The measured 25.4 ms **exceeds the read-everything ceiling by ~2x**. No
weight-bandwidth explanation — dense MoE included — can produce it. The cost is
therefore **latency-bound**, not bandwidth-bound: a single-token forward down a
48-layer stack is a long serial chain of many small, latency-bound kernels
(per-layer attention, KV write, MoE align/gate/act + two grouped GEMMs, RMSNorms,
fp8 quant). Corroborating: `verify` (5 tok, full 16k paged attention) is only
1.6x the draft (1 tok, cap544 attention) — both are dominated by a **fixed
per-forward cost** independent of token count, not by expert count or KV size.

### The "5x" is a metric mismatch, not a real regression

- P65 "~5.3 ms/draft forward" = 21.1 ms CUDA **kernel-busy-time** per K=4 propose
  / 4 (`research/65_draft_overhead_opt/results_overhead_opt.md:99-101`;
  `data/trace_f123_w512k4_b6/profiler_out_0.txt`). Kernel-busy-time excludes the
  GPU-idle gaps between kernels.
- P70 "25.2 ms" = a CUDA-**synced wall** of one graph replay (includes those gaps).
- Same-metric comparison: the PIECEWISE draft *step* wall is D ~ 20-23 ms/step
  (P69 Stage 0; P70 baseline table 19.9 ms). The FULL-CG whole step is
  draft_forward(25.4) + sub-regions(~1.7) ~ 27 ms. So under the same "wall"
  metric the draft step is **~flat to slightly worse** under FULL-CG, **not 5x
  worse and not better**.

Reconciliation: in PIECEWISE the forward wall ~ 5 ms GPU-busy + ~16 ms host
launch bubble ~ ~21 ms. FULL-CG removes the ~16 ms host bubble but the wall is
still ~25 ms — i.e. the eliminated host time is replaced by GPU-side inter-kernel
latency the cudagraph does **not** remove. The FULL graph removes host launch
overhead; it cannot remove the GPU-side per-kernel dispatch/dependency latency of
a deep b1 forward. That is why FULL-CG did not drop the forward from ~21 ms to
the ~5 ms kernel-busy floor.

## Fixability

- **Not a cudagraph-coverage problem.** The P70 coverage fix (b1 replays a b1
  graph) was correct and necessary, but the residual is a GPU-side latency floor,
  not the host launch bubble the FULL graph targets. P70's stated next lever
  ("make the FULL-graph MoE as cheap as the eager sparse path") rests on the
  false premise — the MoE is already sparse and identical in both modes.
- **Real levers** (reduce the serial-kernel latency floor of a b1 forward):
  a shallower / smaller draft (fewer layers => fewer serialized kernels);
  `VLLM_SELF_SPEC_DRAFT_TOPC` to route fewer experts/token; or more tokens per
  rank to amortize the fixed per-forward latency (pinned to b1 by DP8 at 16k).
  These are structural (draft-model / batch shape), not cudagraph tuning.

## Limitations / what would make this airtight

No kernel-level trace of the **FULL-CG draft forward** exists in P69/P70 (only
`SelfSpecProfiler` region timing; the sole kernel-level trace, P65
`profiler_out_0.txt`, is whole-cycle + eager draft). The MoE-vs-attention split
of the 25 ms is therefore not directly measured. The verdict does not depend on
that split: the arithmetic ceiling (25 ms > read-everything ~12 ms) already
falsifies any dense/sparse **weight** attribution, and the kernel code + dispatch
trace already prove the MoE is sparse in both modes. A one-off nsys/torch trace
of a single FULL-CG draft replay would confirm the latency-floor decomposition
(kernel count x per-kernel GPU dispatch latency) if a precise per-op attribution
is wanted.

## Answers to the deliverable questions

- **Normal CUDA-graphed decode MoE: SPARSE** (`fused_moe.py:152-191`,
  `moe_align_block_size.py:19-27`).
- **"Full-CG forces dense MoE": FALSE and the 25.2 ms is MISATTRIBUTED.** No
  dense path exists; the FP8 full-replica draft uses the same sparse
  `TritonExperts` (`config.py:1211-1230` -> `oracle/fp8.py:259-263,366-377` ->
  `triton_moe.py:294`). The 25.2 ms region times the whole forward
  (`llm_base_proposer.py:1584-1595`), not MoE.
- **What actually costs 25 ms:** a per-forward **latency floor** of a deep b1
  single-token forward (serial chain of many small latency-bound kernels across
  48 layers), which the FULL graph does not remove (it removes only host launch
  overhead). It is *not* dense MoE, *not* a padding artifact (b1 replays b1),
  *not* the scratchpad (< 1 ms). The "5x" is an artifact of comparing
  kernel-busy-time (5.3 ms) to a synced wall (25 ms); same-metric, the forward is
  ~flat PIECEWISE<->FULL.
- **Fixable?** Structural, not a cudagraph fix: shallower/smaller draft, fewer
  experts/token, or larger per-rank batch. FULL-CG coverage is not the lever.
