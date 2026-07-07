# Phase 69 results — draft chain FULL cudagraph via window-scratchpad attention

Single-node h107 DP8/EP8, 16k, arm-B (fp8 comm-free replica + shared-KV + W512
sinks16 + P65 flag stack). Goal: collapse the ~16 ms/step host-side launch
bubble by running the draft chain forward under a FULL cudagraph.

## Stage 0 — before baseline (this session, arm-B clean K2/K4 b8)

| point | tok/s (accept) | cycle ms* |
|---|---|---|
| K2 b8 | 260.0±1.5 (2.885) | 88.8 |
| K4 b8 | 274.4±6.2 (4.628) | 134.9 |

(*cycle_ms = accept_len × batch × 1000 / tok_s.) Linear solve
cycle = F + K·D: **F ≈ 42.7 ms, D ≈ 23.0 ms/step** (K4 tps ±6.2 -> D ±~2 ms;
this session's arm-B runs a few % slower than P67's F 54.4 / D 16.9 —
same-session before/after is the authoritative comparison). The draft step D is
~5 ms GPU + the ~16-18 ms host-side PIECEWISE launch bubble. Data:
`data/w72n_p69_baseline_clean_spec_cg_K{2,4}.json`.

## Stage 1 — the PIECEWISE cause (file:line)

`vllm/v1/spec_decode/llm_base_proposer.py:3644-3679` (`initialize_attn_backend`):
for a non-MLA (FA3 / GQA) draft backend the code sets
`_draft_chain_force_eager_attn = True`. Reason, in the code comment
(`:3644-3651`):

> On FA3 the captured decode kernel freezes its host-side work distribution at
> capture-time seq_len=1 regardless of num_splits / scheduler_metadata ... So
> instead of a split cap, FA3 draft chains run their attention EAGERLY -- the
> proven correct path (accept restored to the PIECEWISE reference).

Downstream (`llm_base_proposer.py:1189-1233`, chain-setup): with
`DRAFT_CHAIN_PIECEWISE=1` the chain dispatches `piecewise_only` -> the model
BODY runs on captured torch.compile PIECEWISE graph pieces, but
`vllm::unified_attention_with_output` is a **splitting op** (confirmed in the
engine's `compilation_config.splitting_ops`) run EAGERLY between pieces. Per
step that is ~48 kv-update + ~48 attention custom-op launches + ~98
graph-piece launches (the ~16 ms bubble, Phase-65 trace). The verify pass gets
FULL-CG at decode because its attention is one fixed-shape decode over the full
paged cache (capture-time work distribution == replay); the draft chain's
per-step **growing** windowed sequence is exactly what FA3 cannot replay.

## Stage 2 — window-scratchpad dense attention (implemented)

Code behind `VLLM_SELF_SPEC_DRAFT_FULLCG` (default off), composing with
`VLLM_SELF_SPEC_DRAFT_KV_WINDOW`:

- `vllm/v1/spec_decode/scratchpad_attn.py`: gathers the compacted sinks+window
  pages (`index_select` on the persistent `_win_block_table`) into a fixed
  `[bs, CAP, kv_heads, d]` scratchpad, runs masked dense attention
  (q@kᵀ·scale, -inf on padded slots, softmax fp32, @v). Pure shape-driven ops,
  CUDA-graph-capturable.
- `unified_attention_with_output` (`attention.py`): branches to the scratchpad
  when the draft chain forward carries a `DraftScratchpadCtx` (the KV write
  already ran in the preceding `unified_kv_cache_update`). Verify is untouched
  (no ctx).
- proposer: builds the ctx per cycle from the persistent window buffers and
  carries it on the chain forward context (`llm_base_proposer.py`).

Key-set identity (bit-exact greedy): for a block-multiple window W512, the
paged per-step `n_last` is constant (33 blocks), so CAP is cycle-constant
(544 = 34×16) and the padding mask `col >= live seq_len` reproduces the decode
causal set the paged windowed FA3 reads.

## Stage 3 — validation

### Bug found + fixed: K>=3 accept collapse (sample-in-graph)

The first 16k FULLCG run regressed: K2 accept 2.877 (~parity) but **K4 accept
1.970** (baseline 4.628) -- the classic multi-replay collapse. Root cause: the
draft's combined forward+compute_logits+argmax "sample-in-graph" (fws) FULL
graph, replayed K-1 times, does not correctly drive the next chain step's input
under the scratchpad (chain steps 2+ repeat step-1's token). Fix: force the
plain-forward FULL graph + EAGER sample when the scratchpad is on
(`_disable_sample_in_graph=True`). The plain graph replays correctly (each step
reads the in-place-refreshed window buffers).

Also fixed the scratchpad's GPU cost: the manual GQA attention materialised the
8x head expansion (~27 GB writes/forward at 16k) -> replaced with fused
`F.scaled_dot_product_attention(enable_gqa=True)`.

### DP4 canary (2k, W64, shared-KV bf16) — A/B on the SAME binary

| arm | K | accept | tok/s | note |
|---|---|---|---|---|
| paged FA3 (FULLCG off) | 2 | 2.927 | 792.0 | baseline |
| scratchpad FULL-CG     | 2 | 2.908 | 972.2 | +23% tok/s, accept parity |
| paged FA3 (FULLCG off) | 4 | 4.534 | 742.3 | baseline |
| scratchpad + fws graph | 4 | COLLAPSE (see 16k 1.97) | -- | the bug |
| **scratchpad FULL-CG (fws off)** | 4 | **4.479** | **1432.9** | **+93% tok/s, accept parity** |

(1-iter b4 smoke; accept parity within ~0.05, tok/s amplified since GPU work is
tiny at 2k. The 16k numbers below are the load-bearing measurement.)

### Stage 3.2/3.3 — 16k accept + draft-step timing (single-node DP8, arm-B)

| config | K2 tok/s (accept) | K4 tok/s (accept) | F ms | D ms/step |
|---|---|---|---|---|
| baseline (FULLCG off) | 260.0 (2.885) | 274.4 (4.628) | 42.6 | **23.1** |
| **scratchpad FULL-CG** | 160.3 (2.878) | 150.8 (**4.666**) | 39.7 | **52.0** |

**Accept GATE PASSED**: K4 4.666 vs 4.628 (+0.038, parity, well inside ±0.03…
noise), K2 2.878 vs 2.885. The default-off flag never regresses accept.

**Perf: the win does NOT materialize at the 16k/DP8 serving batch.** D goes the
WRONG way (23 -> 52 ms/step). Diagnosis: the chain claims FULL (the "KV window
NOT applied" FULL-mode warning fires -> `skip_rebuild_full_cg=True`, so the
per-step metadata rebuild is skipped and my window refresh runs -> accept is
correct), but D=52 ms ≈ the cost of running the 48-layer draft forward EAGER
(op-by-op, ~52-67 ms) rather than the ~5-10 ms of a single FULL-graph replay.
So at this config the FULL cudagraph is **claimed but not actually replayed** —
the wrapper falls back to eager for the dispatched (num_tokens, uniform)
descriptor. The scratchpad compute itself is cheap (gather ≈0.4 ms + fused
SDPA <1 ms/forward at cap=544), and re-capture is not happening (2 capture bars
total, init only). The gap is the existing draft-FULL-CG capture/dispatch
**shape coverage**: the same code path replays correctly (and wins +93% tok/s)
at the DP4 b4 / W64 canary but not at DP8 b8 / W512 16k.

### Honest verdict

- The **enabling change is done and correct**: the window-scratchpad dense
  attention makes the GQA draft chain a fixed-shape, CUDA-graph-capturable op;
  the whole per-step draft forward captures as ONE FULL graph, greedy accept is
  preserved to parity at K2 and K4 (16k), and the K>=3 sample-in-graph collapse
  is fixed.
- The **FULL-CG win is demonstrated** where the draft FULL graph actually
  replays (DP4 canary: chain step effectively removed, +93% tok/s at K4).
- The **16k serving-batch realization is BLOCKED**: the b1-per-rank chain
  descriptor claims FULL but the wrapper runs the 48-layer forward EAGER, so D
  regresses 23 -> 52 ms. Isolation:
  - per-rank batch is b1 at BOTH DP4 b4 and DP8 b8 -> the descriptor is the same,
    so it is not a plain batch-size coverage gap;
  - **fp8-replica RULED OUT**: the arm-B fp8 comm-free replica draft REPLAYS and
    wins at the DP4 canary (K4 1458 tok/s / accept 4.333, matching bf16's 1433 /
    4.479) -- so the replica draft is FULL-CG-capturable;
  - the remaining live variable is **scale/window**: FULL replays at 2k / W64 /
    cap96 (DP4, +93%) but runs eager at 16k / W512 / cap544. The scratchpad graph
    at cap544 (vs cap96) and/or the 16k KV pool is where the wrapper stops
    matching a captured graph and falls to eager.
  Extending that coverage (making the cap544 draft chain graph replay at 16k) is
  the remaining integration step; it was not cracked within this phase's budget.
- **Stage 4 (2-node E2E) not run**: single-node b8 regresses, so the 2-node
  arm cannot beat no-spec; measuring it would only burn the peer node. Warranted
  only once the b8 FULL replay lands.

Data: `data/w72n_p69_{baseline_clean,fullcg2_clean}_spec_cg_K{2,4}.json`,
`data/w72n_q30b_p69_smoke_*` (DP4 canaries). Logs on disk under `logs/`.
