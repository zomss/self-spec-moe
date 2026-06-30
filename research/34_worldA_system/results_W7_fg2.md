# W7-fg2: the draft FULL-cudagraph attention was a captured zero-fill stub — capturing the REAL kernel (with 1 split) restores accept_len 2.08→2.81

**Source phase.** results_W7_fullgraph.md FULL-CG'd the `draft_model` draft and dropped
the draft forward 47.96→11.79 ms, but **accept_len collapsed 2.72→1.99** and the
implied speedup stuck at 0.81×. That phase attributed the collapse to "the draft's
per-decode-step attn metadata is built fresh each step and is NOT in the persistent
buffers the captured FULL graph reads." This phase fixes it and finds the cause was
deeper (and the metadata-staleness story was only half-right).

All numbers: DeepSeek-V2-Lite, DP=2 + EP, forced-PCIe + 100 µs emulated A2A,
batch 64, K=2, `t_nospec_step = 17.95 ms`. Harness: `scripts/w7_microbench.py`
(unchanged), `MB_STEADY=250`, env `VLLM_SELF_SPEC_DRAFT_FULL_CG=1`.

## Root cause (what was actually broken)

1. **The captured FULL graph never contained attention at all.** The draft's
   `dummy_run` capture pass called `set_forward_context(None, ...)` (attn_metadata
   = None). With `attn_metadata is None`, `MLAAttention.forward_impl`
   (`mla_attention.py:643`) short-circuits to `return output.fill_(0)` — so the
   draft's FULL graph captured a **zero-fill stub**, not the FlashAttnMLA decode
   kernel. At replay the 2nd+ draft token thus attended against *zeros* → garbage →
   rejected by verify (lossless, but accept_len ≈ 1 + 1 = ~2.0). The verify model's
   own `_dummy_run` does build real metadata for capture
   (`gpu_model_runner.py:5884`); the draft did not.

2. **Naively capturing the real kernel made it WORSE (accept 1.9 < stub 2.08).**
   With real metadata at capture, the FlashAttnMLA decode runs with the engine
   default `flash_attn_max_num_splits_for_cuda_graph = 32`. The FA3 split-combine,
   captured into the draft's FULL graph, reads per-split partial buffers whose
   layout is tied to the capture-time (padded, uniform, seq_len≈3) schedule and does
   **not** replay correctly for the per-step growing draft sequences → corrupted
   attention. (max_seq_len upper-bound alone did not help: 1.90 → 1.91.)

3. **One split = exact attention.** Capping the draft attention builder to
   `max_num_splits = 1` removes the split-combine entirely; the captured kernel is
   exact and reads its bounds from the live per-step `seq_lens` / recomputed
   `scheduler_metadata`. accept_len jumps to 2.78–2.81 (≈ the correct 2.85) and the
   draft forward stays fast (12.4 ms).

## The change (behind `VLLM_SELF_SPEC_DRAFT_FULL_CG`, default off → unchanged)

`vllm/v1/spec_decode/llm_base_proposer.py` only:
- Store `self.runner` so the proposer can reach the runner's persistent
  block-table / seq-len buffers at capture time.
- `_build_draft_decode_capture_metadata(batch_size)`: build a real
  `CommonAttentionMetadata` for the uniform 1-token/seq draft decode shape from the
  runner's persistent `block_table` + `seq_lens`, the proposer's `arange` /
  `_slot_mapping_buffer`, and a large step-invariant `max_seq_len = max_model_len`;
  return `build_for_cudagraph_capture(...)` per layer.
- `dummy_run`: when capturing the FULL (uniform-decode) draft graph, pass that real
  metadata into `set_forward_context` instead of `None`, so the real FlashAttnMLA
  kernel is recorded against the persistent-buffer pointers replay also reads.
- `initialize_attn_backend`: under the FULL-CG flag, cap the draft attention
  builder's `max_num_splits` to 1 (env `W7_FG2_DRAFT_SPLITS` overrides for A/B).

Scoped to FLASH_ATTN_MLA (the backend V2-Lite self-spec uses). No verify/main-model
or default-path changes; flag off ⇒ byte-identical to before (capture still passes
None, no split cap).

## Numbers — V2-Lite b64 K2 forced-PCIe + 100 µs A2A (clean A/B, same machine/code)

| run | accept_len | df_first | draft_fwd | draft_chain | verify | cycle=chain+verify | implied speedup* |
|---|---:|---:|---:|---:|---:|---:|---:|
| PIECEWISE (correct attn, eager) | 2.853 | 46.32 | 45.73 | 97.29 | 19.83 | 117.12 | 0.437 |
| FULL **stub** (before, 2nd-tok zero attn) | 2.079 | 15.52 | 11.78 | 31.49 | 20.88 | 52.37 | 0.712 |
| FULL real + 32 splits (naive) | 1.912 | 14.43 | 11.68 | 30.22 | 18.99 | 49.21 | 0.697 |
| FULL real + **1 split** (after, final) | **2.814** | 15.05 | 12.35 | 31.43 | 19.65 | 51.08 | **0.989** |

*implied speedup = accept_len · 17.95 / cycle (cycle = draft_chain + verify).
By the K·t+verify denominator: stub = 2.079·17.95/(2·11.78+20.88) = **0.84×**;
after = 2.814·17.95/(2·12.35+19.65) = **1.14×**.

**Decisive numbers (before → after):**
- **accept_len 2.08 → 2.81** (≈ the 2.85 correct-attention reference; the linchpin).
- **implied speedup 0.71 → 0.99 (chain) / 0.84 → 1.14 (K·t+verify)** — crosses ~1.0.
- **chain_overhead ≈ unchanged (~4.2 ms).** P1's per-step `build_for_drafting` is
  retained (it refreshes the persistent `scheduler_metadata` each step), so the
  ~1.8 ms rebuild was NOT removed — see below.

## Losslessness — PASS

Greedy (temp=0), 4 prompts, 96 tokens each, comparing FULL-CG-spec vs no-spec vs
PIECEWISE-spec (correct-attention spec reference):

| prompt | FULL-CG == no-spec | PIECEWISE-spec == no-spec | FULL-CG == PIECEWISE |
|---|---|---|---|
| 0 | ✗ | ✗ | ✗ |
| 1,2,3 | ✓ | ✓ | ✓ |

Prompt 0 flips for **both** FULL-CG-spec and PIECEWISE-spec (correct eager attention)
at the same point — i.e. it is an inherent spec-decode-vs-no-spec near-tie greedy
flip (the verify forward processes K+1 tokens with different numerics than no-spec's
1 token), **not** introduced by the FULL-CG fix. FULL-CG-spec exhibits *identical*
divergence behavior to the correct-attention PIECEWISE-spec path → the captured
attention is correct. Verdict: lossless to the same degree as the reference spec
path.

## What did NOT land (and why)

- **P2 (capture compute_logits + sampling in the graph, ~0.9 ms):** skipped to not
  risk P1 (invasive; V2's `_generate_draft` does it via a persistent GPU step scalar
  + persistent output buffers — a much larger change).
- **The ~1.8 ms chain_overhead removal:** replay still calls `build_for_drafting`
  each step because FlashAttnMLA's `scheduler_metadata` must be recomputed per step
  for the growing sequence; skipping it risks stale-schedule correctness. The
  accept_len recovery (0.71→0.99) is the dominant win; the rebuild-removal is a
  separate follow-up.

## Files
- Change: `vllm/v1/spec_decode/llm_base_proposer.py` (only).
- Data: `data/mb_spec_v2lite_fg2final_b64_K2_a2a100us.json` (after, real+1split),
  `data/mb_spec_v2lite_fg2stub_b64_K2_a2a100us.json` (before, stub),
  `data/mb_spec_v2lite_fg2pw_b64_K2_a2a100us.json` (PIECEWISE correct-attn ref),
  and `data/mb_profiles_spec_v2lite_fg2*_*/`.
- Harness: `scripts/w7_microbench.py` (unchanged).
