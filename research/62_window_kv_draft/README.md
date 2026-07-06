# Phase 62 — Window-KV drafting: accept ablation at 16k (Qwen3-30B-A3B)

Source phases: 52 (harness + self-spec stack), 57 (on-dist prompts),
59 (long-context methodology).

## Objective

At long context the self-spec draft step is KV-bound. A draft that attends
only to attention sinks + the last W tokens (StreamingLLM-style) reads
5-10x less KV; hypothesis: per-token accept stays ~0.9. Nobody has measured
window-KV draft accept on an MoE. This phase (1) adds a draft-side KV-window
knob to the self-spec stack, (2) proves it correct with a canary, (3)
measures the accept ablation at 16k on Qwen3-30B-A3B, single-node DP8/EP8.

## Mechanism (vllm/ change)

`VLLM_SELF_SPEC_DRAFT_KV_WINDOW` (int tokens, 0=off) and
`VLLM_SELF_SPEC_DRAFT_KV_SINKS` (int tokens, default 16) in `vllm/envs.py`.

In `SpecDecodeBaseProposer` (`vllm/v1/spec_decode/llm_base_proposer.py`),
for DRAFT forward steps only, `_apply_draft_kv_window` compacts each
request's block-table row to the first ceil(SINKS/block) blocks + the last
blocks covering (W + tokens drafted) tokens, and shrinks the draft-side kv
seq_len to the tokens actually present in the kept pages (partial last block
counted). Paged KV carries true RoPE'd positions and FA aligns the causal
mask at the END of the KV sequence, so this is exactly sinks+window
attention — no position surgery. The compacted view lives in proposer-owned
persistent buffers referenced by a shallow-copied metadata object; the TRUE
block table / seq_lens stay untouched, so slot mappings (KV writes) remain
exact and the verify pass stays full-KV lossless. Recomputed per chain step
(the newest pages are always kept). Applied at the step-0 draft forward only
for decode-shaped proposes (prompt-ingest passes stay full-KV, so prompt KV
is written exactly); requires the eager/PIECEWISE chain (our
`VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1` stack) — under a FULL-CG
skip-rebuild chain it warns and stays off.

## Commands

```bash
bash scripts/run_canary.sh        # Part 2: ~2k ctx, K=2, OFF vs W=64
bash scripts/run_ablation.sh 4    # Part 3: 16k ctx arms A,B1,B2,B3,C,D
```

Harness: `research/52_two_node_e2e/scripts/w7_2node.py` (unchanged), env
`scripts/env_1node.sh`, per-arm retry runner `scripts/run_arm.sh`
(run_ksweep.sh pattern; refuses to launch over foreign GPU PIDs).

## Decision criteria

- Canary OFF: accept_len ~3.0 (K=2; greedy same-model draft is always
  right). Canary W=64 at ~2k: accept_len < 3.0 measurably, > 2.0.
- B2 (W=512 at 16k): is per-token r >= ~0.85 on MoE?
- C vs B2 x (D/A): do the window and FP8 factors compose multiplicatively?

## Expected next artifact

`results_window_accept.md` + per-arm JSONs in `data/`.
