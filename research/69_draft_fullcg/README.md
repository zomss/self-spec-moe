# Phase 69 — draft chain under FULL cudagraph (window-scratchpad attention)

Source: Phase 65/66/67/68. Established config (arm-B): fp8 comm-free replica +
shared-KV + W512 sinks16 + P65 flag stack (DP_COORD_CPU + CHAIN_LIGHT_MD +
SKIP_DP_COORD). At 16k the draft step = ~5.3 ms GPU compute + ~16 ms CPU
dispatch. The device-side wins (window KV read 30x cut, zero draft comm, fp8
experts) all engage; the residual is host-side kernel-launch overhead because
the draft chain runs `cudagraph_mode=PIECEWISE` (attention eager).

## Objective

Collapse the ~16 ms/step host-side launch bubble by running the draft chain
forward under a FULL cudagraph (~1 launch/step), dropping the draft step from
~21 ms toward its ~5-7 ms GPU floor.

## The PIECEWISE blocker (Stage 1 diagnosis)

`llm_base_proposer.py:3644-3679` (`initialize_attn_backend`): for a non-MLA
(FA3 / GQA) draft backend the code sets `_draft_chain_force_eager_attn = True`
because the FA3 decode kernel **freezes its host-side work distribution at
capture-time** and does not re-derive it from the live `seqused_k` at replay
(Phase 35: accept 4.9 -> 1.9, insensitive to num_splits / scheduler_metadata /
capture seq_len). Consequences:

- `propose()` chain loop (`llm_base_proposer.py:1189-1194`): with
  `DRAFT_CHAIN_PIECEWISE=1` the chain dispatches `piecewise_only` -> the model
  BODY runs on captured PIECEWISE graph pieces, but `unified_attention_with_
  output` is a **splitting op** run eagerly between pieces (~48 attention + 48
  kv-update ops + ~98 graph-piece launches/step -> the ~16 ms bubble).
- The target VERIFY pass gets FULL cudagraph at decode because its attention is
  a single fixed-shape decode over the full paged cache (capture-time work
  distribution == replay); the draft chain's per-step **growing** windowed
  sequence is what FA3 cannot replay.

## The fix (Stage 2): window-scratchpad dense attention

Because `DRAFT_KV_WINDOW` bounds the draft's KV read to sinks S + window W +
drafted <= K, and W is a block multiple, each chain step attends over a
**cycle-constant** number of pages (W512 s16 K4 -> n_kept=34 blocks, CAP=544).
Replace the paged FA3 read with:

1. a fixed-shape GATHER of the compacted sinks+window pages from the (shared)
   paged cache into a dense scratchpad `[bs, CAP, kv_heads, d]`
   (`index_select` on the persistent `_win_block_table`);
2. a masked dense attention (q@kᵀ·scale + (-inf on padded slots) -> softmax
   -> @v) — pure shape-driven tensor ops, no host-side scheduling.

Fully CUDA-graph-capturable -> flip `_draft_chain_force_eager_attn = False` so
the existing FULL-CG machinery captures the whole per-step forward as ONE graph.
Bit-exact greedy: same window key set, same causal-at-end mask, same scale.

Gated behind `VLLM_SELF_SPEC_DRAFT_FULLCG` (default off), composes with
`VLLM_SELF_SPEC_DRAFT_KV_WINDOW`. Code:
- `vllm/v1/spec_decode/scratchpad_attn.py` (gather + masked attention)
- `unified_attention_with_output` branch (`attention.py`)
- proposer wiring (`llm_base_proposer.py`).

## Validation gates

1. Canary (`scripts/run_canary.sh`): shared-KV + FULLCG, W64 @2k, K2 — accept
   matches the PIECEWISE run to ~3 decimals (P66 ref 2.689).
2. 16k accept (`scripts/run_1node.sh` W512 K4): reproduce ~4.58 (±0.03).
3. Draft-step wall-clock before -> after; #launches/step after (~1).
4. 2-node E2E (h107+h108) vs a fresh no-spec denominator.

Results: `results_fullcg.md`. Data: `data/`, logs on disk in `logs/`.
