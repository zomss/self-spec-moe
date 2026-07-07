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

### Stage 2 validation — TBD
### Stage 3 draft-step timing — TBD
### Stage 4 2-node E2E — TBD
