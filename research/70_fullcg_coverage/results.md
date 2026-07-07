# Phase 70 results — draft-chain FULL cudagraph: capture/replay COVERAGE

Single-node h107 DP8/EP8, 16k, arm-B (fp8 comm-free replica + shared-KV + W512
sinks16 + P65 stack). Everything behind `VLLM_SELF_SPEC_DRAFT_FULLCG` (default
off). Debug via `W7_FULLCG_DBG=1`.

## Step 1 — the descriptor diff: the P69 premise was WRONG

P69 concluded the cap544/16k draft-chain FULL graph is "CAPTURED but NOT
REPLAYED (falls back to eager)". Instrumenting the cudagraph dispatch/lookup
path (`vllm/compilation/cuda_graph.py` `CUDAGraphWrapper.__call__` +
`llm_base_proposer.py` propose) shows the opposite:

```
[fullcg-dbg] rank=0 chain dispatch: batch_size=1 mode=FULL
  last_desc=BatchDescriptor(num_tokens=5, num_reqs=5, uniform=True ...)
  full_captured=[5,10,20,25,35,40,50,60,65,75,80,90,100]
  model_wrapper_captured=[(5,5,True),(10,10,True),...]
[fullcg-dbg] wrapper=<draft.model> REPLAY ctx_desc=(num_tokens=5,num_reqs=5,uniform=True)
```

The draft-chain forward **DOES replay a captured FULL graph**. The real bug:
the per-rank draft batch is **1**, but the dispatched descriptor is
`(num_tokens=5, num_reqs=5)` — batch_size=1 pads UP to num_reqs=5 (=1+K).

### Root cause (file:line)

`vllm/config/compilation.py:1502` `adjust_cudagraph_sizes_for_spec_decode`
rounds every shared `cudagraph_capture_sizes` UP to a multiple of the verify
query len `vq = 1 + num_speculative_tokens` (5 for K4). Each draft decode-step
forward is **one token per seq**, so the drafter's dispatcher
(`uniform_decode_query_len = 1`, `llm_base_proposer.py:813`) keys its
uniform-decode FULL graphs by `num_reqs == num_tokens`. With the (1+K)-rounded
sizes the **smallest** draft FULL graph is `num_reqs = 1+K` (5). So a per-rank
draft batch of 1 replays a **5-sequence** graph — (1+K)x the window-scratchpad
gather (`index_select` of cap544 pages, 48 layers) and MoE-expert work every
chain step. At 2k/W64/cap96 (the DP4 canary) that waste is tiny (the +93% win
survives); at 16k/W512/cap544 it dominates (draft step 23 -> 52 ms).

## Step 2 — the fix

`vllm/v1/spec_decode/llm_base_proposer.py`
`_augment_draft_capture_sizes_for_num_reqs` (called from
`initialize_cudagraph_keys`): when the window-scratchpad chain is on
(`_draft_fullcg`, q=1 FULL_AND_PIECEWISE), augment the **draft** dispatcher's
capture sizes with the small raw num_reqs values (`< 1+K`, i.e. 1,2,[4]) that
the runner already drives during capture. The b1/b2 chains then replay an exact
bN graph instead of padding up to b(1+K). Scoped to `_draft_fullcg` so the plain
DRAFT_FULL_CG / PIECEWISE-chain path stays byte-identical (adding uncaptured
small FULL keys there faults a later FULL dispatch — regression avoided).

Capturing the new small draft shapes triggered a second bug:
`vllm/v1/worker/gpu_model_runner.py:capture_model` — inductor's lazy attention-
pattern init (`_sfdp_init`, `init_once_fakemode`) ran for the FIRST time INSIDE
the cudagraph capture context (the scratchpad uses
`F.scaled_dot_product_attention`), doing an unpinned CPU->GPU copy ->
"Cannot copy between CPU and CUDA tensors during CUDA graph capture". Fix: prime
`torch._inductor.fx_passes.joint_graph.lazy_init(device)` once BEFORE the
capture loop (no-op if already cached).

## Step 3 — single-node validation (DP8, 16k, arm-B, b8)

REPLAY confirmed at the fixed descriptor:
```
[fullcg-dbg] rank=0 chain dispatch: batch_size=1 mode=FULL
  last_desc=BatchDescriptor(num_tokens=1, num_reqs=1, uniform=True ...)
[fullcg-dbg] wrapper=<draft.model> REPLAY ... num_tokens=1, num_reqs=1  (x8)
```

| config | K2 tok/s (accept) | K4 tok/s (accept) | F ms | D ms/step |
|---|---|---|---|---|
| **same-session baseline (FULLCG off)** | **265.9 (2.902)** | **290.7 (4.617)** | 47.6 | **19.9** |
| P69 FULLCG unfixed (b1 -> b(1+K) pad) | 160.3 (2.878) | 150.8 (4.666) | 39.6 | 52.0 |
| **P70 FULLCG fixed (b1 replays b1)** | **192.6 (2.859)** | **170.5 (4.638)** | 19.8 | **49.5** |

(cycle = accept x batch x 1000 / tok_s = F + K*D.) Accept preserved (K4 4.638
vs 4.617 baseline — parity). REPLAY confirmed at `(num_tokens=1, num_reqs=1)`.

### Verdict (single-node b8)

The coverage fix is CORRECT and helps: the b1 chain now replays a b1 graph, and
FULLCG tok/s recovers ~20% of what padding cost it (160->193 K2, 151->171 K4).
The fix collapses the FIXED cost F 39.6 -> 19.8 ms (the step-0 draft forward no
longer runs a b(1+K)-padded graph).

**But FULLCG still LOSES to the PIECEWISE baseline at 16k / W512 / cap544 / b8**
(193/171 vs 266/291 tok/s). The per-draft-step D is HIGHER under FULLCG
(49.5 vs 19.9 ms) even though the forward replays a single graph. The make-or-
break gate (draft step -> ~8 ms) is NOT met. The mechanism only wins at small
cap (P69 DP4/2k/W64/cap96 canary: +93%); at cap544 the residual per-step cost
(measured below) exceeds the PIECEWISE launch-bubble the FULL graph removes.

### FINE region breakdown (fixed FULLCG, K4 b8, rank-0, 508 chain-step samples)

`data/prof_fine_fullcg/` (`VLLM_SELF_SPEC_PROFILE_FINE=1`), mean ms:

| region | mean ms | note |
|---|---|---|
| `draft_forward` (chain step) | **25.2** | the FULL-graph replay itself — dominant |
| `draft_forward_first` (step-0) | 14.3 | step-0 draft forward |
| `verify` | 40.2 | target verify (b8, K+1 tok/seq) |
| `step_sample` | 0.64 | eager draft sample (compute_logits+argmax) |
| `step_build_attn_md` | 0.51 | window compaction (`_apply_draft_kv_window`) |
| `step_pos_slot_update` | 0.31 | |
| `step_input_buffering` | 0.24 | |

**The per-step cost FULLCG was designed to remove is already <1 ms each**
(sample 0.64, window 0.51, metadata 0.31). The residual is the draft **forward**
itself: a single b1 replay of the 30B-A3B draft costs **25 ms**. It is
weight-bound, not launch-bound — `verify` (40 tokens) is 40 ms vs `draft_forward`
(1 token) 25 ms, i.e. token count barely moves it. The leading hypothesis: the
CUDA-graph (FULL) MoE must be fixed-shape, so it cannot use the eager path's
sparse (active-experts-only) routing and pays much more weight traffic per token
than the PIECEWISE baseline's eager sparse MoE. This cost is what the FULL graph
cannot amortize at 16k; the removed PIECEWISE launch bubble (~16 ms) is smaller
than the extra forward cost.

## Verdict + honest conclusion

- **P69's premise was wrong**: the cap544/16k draft-chain FULL graph IS replayed
  (not "captured-but-eager"). Corrected with the descriptor-diff instrumentation.
- **Real bug found + fixed (file:line)**: `adjust_cudagraph_sizes_for_spec_decode`
  (`compilation.py:1502`) rounds shared capture sizes to multiples of (1+K); the
  q=1 draft keys by num_reqs so its smallest FULL graph is b(1+K), and a per-rank
  b1 chain padded up to a b(1+K) replay. Fixed by
  `_augment_draft_capture_sizes_for_num_reqs` (`llm_base_proposer.py`, gated on
  `_draft_fullcg`). Also fixed the capture-time inductor `_sfdp_init` CPU-copy
  crash (`gpu_model_runner.py:capture_model` lazy_init prime) the new small
  shapes exposed. b1 now replays a `(1,1,True)` graph — confirmed.
- **Result**: the fix recovers ~20% of the FULLCG tok/s lost to padding
  (160->193 K2, 151->171 K4; F 39.6->19.8 ms) with accept preserved
  (K4 4.638 vs 4.617). But **FULLCG still loses to the PIECEWISE baseline at
  16k/W512/cap544/b8** (193/171 vs 266/291 tok/s) because the draft FORWARD is
  25 ms (weight-bound), not the per-step launch overhead. The make-or-break gate
  (draft step -> ~8 ms) is NOT met.
- **Step 4 (2-node E2E): NOT RUN / NOT WARRANTED.** Single-node fixed FULLCG is
  slower than the same-session PIECEWISE baseline, so the 2-node arm cannot beat
  the no-spec denominator; measuring it would only burn the peer node (same
  reasoning as P69). Warranted only if the draft-forward cost itself is cut
  (e.g. a CUDA-graph-capturable SPARSE MoE for the draft, or a smaller draft).
- **What a real win needs**: the residual is the FULL-graph draft forward's MoE
  weight traffic, not the scratchpad attention or the per-step host bubble. The
  window-scratchpad + b1-coverage machinery is correct and lands the forward as
  one replayed graph; the next lever is making that graph's MoE as cheap as the
  eager sparse path (out of this phase's scope).

Data: `data/w72n_p69_{baseline,fix2_fullcg}_spec_cg_K{2,4}.json`,
`data/prof_fine_fullcg/`. Logs in `logs/`. Code behind
`VLLM_SELF_SPEC_DRAFT_FULLCG` (default off); debug via `W7_FULLCG_DBG=1`.
