# W7-gqa: the draft FULL-CG attention is a frozen captured kernel — FA3/GQA can't replay the draft's intra-loop growing sequence; run its chain attention eagerly

**Source.** `34_worldA_system/results_W7_fg2.md` (MLA fix: real capture metadata +
`max_num_splits=1` restored V2-Lite accept 2.08→2.81) claimed the fix was
backend-agnostic. `results_W7_qwen30b.md` §3a then found the **GQA (Qwen3,
FLASH_ATTN/FA3) draft FULL-CG accept COLLAPSES** (DP=2: PIECEWISE 2.0 → FULL-CG
1.6) and the fg2 split-cap does not carry to FA3. This phase reproduces it cheaply,
pins the GQA-specific root cause, fixes it, and validates.

All single-GPU runs: `Qwen/Qwen3-8B` (dense GQA, FA3), draft_model self-spec
(draft = Qwen3-8B), greedy, K=4, CUDA graphs ON, batch 16, OUTLEN 128,
`VLLM_SELF_SPEC_DRAFT_FULL_CG` toggled. Harness `scripts/repro_qwen3_8b.py`.

## 1. Reproduced on Qwen3-8B (the cheap path — no MoE/EP needed)

| config | accept_len |
|---|---:|
| PIECEWISE (reference, draft attn eager) | **4.92** |
| FULL-CG (broken, before fix) | **1.99** |
| **FULL-CG (after fix)** | **4.91** |

Clean A/B on one GPU, ~1–2 min/run. The collapse reproduces without MoE/EP, so the
bug is purely in the **draft's FA3 attention capture**, as hypothesized.

## 2. Root cause (GQA-specific)

The W7 draft FULL-CG path captures ONE decode cudagraph and **replays it K-1 times
inside a single `propose()` call, with the draft sequence GROWING by 1 each replay**
(`llm_base_proposer.py::propose` chain loop). This intra-call growth is unique to the
draft: the verify model replays its decode graph once per engine step.

The FA3 (`FlashAttentionImpl.forward` → `flash_attn_varlen_func`) decode kernel
**derives its host-side work distribution at kernel-launch from the capture-time
`seqused_k` (=1) and FREEZES it into the captured graph**. At replay the live,
growing `seqused_k` is ignored → the 2nd+ draft token attends over a truncated
context → wrong draft tokens → verify rejects them → accept_len ≈ 1 + (bonus) ≈ 2.0.

### Evidence (surgical)

- **Per-step divergence (`scripts/diag_steps.py`, W7_GQA_DEBUG_STEPS).** For the
  IDENTICAL first committed prefix (in_id=13, seq_len=20, same positions/slot),
  the captured FULL-CG forward produces a DIFFERENT draft token than eager:
  PIECEWISE step0→576, FULL-CG step0→21718. Divergence starts at the very first
  captured chain step. (Both modes still emit the same final tokens — verify
  corrects the bad drafts — so it is lossless but accept collapses.)

- **Insensitive to every FA3 schedule knob** (all give accept ≈ 1.99, b16 K4):
  - `max_num_splits` ∈ {0 (heuristic), 1 (fg2 cap), 2, 32}: 1.990 / 1.990 / 1.990 / 1.990
  - strip the AOT `scheduler_metadata` at capture (None → runtime schedule): 1.992
  - seed capture-time `seq_lens`=128 instead of 1: 1.990
  - refresh `scheduler_metadata` per step via `build(fast_build=False)` (the verify
    mechanism): 1.991, and with a consistent `max_seq_len` it HANGS.
  → the fg2 remedies (real capture metadata, 1-split) do not touch the FA3 freeze;
    the bug is the captured kernel's frozen launch, not the schedule contents.

- **Decisive isolation: run the SAME chain eagerly.** Forcing the chain forward to
  `cudagraph_runtime_mode=NONE` (eager attention; all other FULL-CG machinery
  unchanged) restores **accept_len 1.99 → 4.91** = the PIECEWISE reference. So the
  bug is the captured graph replay, NOT the per-step input prep (seq_lens /
  positions / slot_mapping are all correctly live-aliased to `runner.seq_lens` /
  the persistent buffers).

- **PIECEWISE (correct) draft FA forward** logs `scheduler_metadata=None`,
  `num_splits=32`, live `seqused_k=46`, real block table — i.e. the correct path
  does host-side scheduling fresh every eager launch, which a frozen graph cannot.

## 3. The fix (behind `VLLM_SELF_SPEC_DRAFT_FULL_CG`, default off)

`vllm/v1/spec_decode/llm_base_proposer.py`:

1. **`initialize_attn_backend`:** detect the draft attention backend.
   `isinstance(builder, MLACommonMetadataBuilder)` ⇒ MLA (keep the fg2 path: cap
   `max_num_splits=1`, FULL graph). Otherwise (FA3 / GQA, or any non-MLA) ⇒ set
   `self._draft_chain_force_eager_attn = True`. (Env `W7_GQA_FORCE_EAGER_ATTN`
   overrides for A/B.)

2. **`propose` chain setup:** when `_draft_chain_force_eager_attn`, request the
   chain via `_determine_batch_execution_and_padding(..., use_cudagraphs=False)`
   so the chain forwards run EAGER (NONE). Passing `use_cudagraphs=False` (rather
   than overriding the mode afterward) keeps the DP batch coordination consistent
   — overriding after coordination desyncs `num_tokens_across_dp` and trips the
   `set_forward_context` DP assert (`AssertionError: 5 1`) at DP>1 — this WAS the
   reported DP=8 engine error.

3. **`dummy_run`:** skip building the real FA capture metadata and the
   sample-in-graph capture when `_draft_chain_force_eager_attn` (the FA chain
   never replays a captured decode graph, so don't capture one).

The per-step `build_for_drafting` (eager metadata, live seq_lens) now runs for the
FA chain because `cudagraph_runtime_mode==NONE` makes `skip_rebuild_full_cg` False
— the correct path, identical to PIECEWISE attention.

The model body still benefits from torch.compile; only the per-step CUDA-graph
*replay* of the draft decode is dropped on FA3. MLA is unchanged.

## 4. accept_len before/after (the gate)

### Qwen3-8B (single GPU, b16 K4) — GATE PASSED
| | PIECEWISE | FULL-CG before | FULL-CG after |
|---|---:|---:|---:|
| accept_len | 4.92 | **1.99** | **4.91** |

### Qwen3-30B-A3B (DP=2, EP=2, FP8 full-replica draft, forced-PCIe, b16 K4)

| | accept_len | gen wall (s) |
|---|---:|---:|
| PIECEWISE (correct-attn reference) | 1.654 | 28.7 |
| FULL-CG (after fix) | **1.651** | **18.9** |

**FULL-CG accept == PIECEWISE accept (1.65 == 1.65)** -> the FA capture bug is
fixed: the draft attention is now correct under FULL-CG. FULL-CG is also 1.5x
faster wall (18.9 vs 28.7 s) at the same accept (the compiled model body still
wins; only attention is eager).

The *absolute* ceiling (~1.65, not 4–5) is the SEPARATE DP>=2 full-replica-draft
accept collapse documented in `results_W7_qwen30b.md` §3b (present for BOTH
PIECEWISE and FULL-CG, independent of attention capture). On single-GPU Qwen3-8B
(no full-replica DP confound) the fix yields the full 4.91.

**DP=8 (the reported engine error).** Before: FULL-CG died with
`AssertionError: 5 1` in `set_forward_context` (the draft chain overrode the
cudagraph mode AFTER DP coordination, desyncing `num_tokens_across_dp`). The fix
coordinates via `use_cudagraphs=False` -> **DP=8 no longer crashes** (30B DP=8
b16 K4 ran to completion: accept_len 1.0, gen 24.8 s, clean shutdown, no leaked
workers). Accept at DP=8 is ~1.0 (the same full-replica artifact; accepted=0 /
1008 drafts), so there is no speedup to monetize at DP=8 -- but the engine crash
is resolved.

## 5. MLA regression check + a NEW finding

My change does not touch MLA (it keeps the fg2 FULL path; `force_eager` log NOT
emitted for FLASH_ATTN_MLA). DeepSeek-V2-Lite single-GPU:

| K | PIECEWISE | FULL-CG (MLA, kept FULL) |
|--:|---:|---:|
| 2 | 2.99 | 2.79 (≈ fg2's 2.81 — OK) |
| 4 | 4.95 | **2.00 (collapse!)** |

**Finding:** the captured-graph freeze affects MLA too — it just SCALES WITH K. fg2
validated only K=2 (collapse masked). At K=4 MLA FULL-CG also collapses. So the
"MLA is fixed" claim holds only at small K; the same eager-attention remedy would
fix MLA at all K (at the cost of the FULL win fg2 reported). Left as-is here to
respect the fg2/task scope; flagged as a remaining issue (§7).

## 6. Files
- Fix: `vllm/v1/spec_decode/llm_base_proposer.py`.
- Harness: `scripts/repro_qwen3_8b.py`, `scripts/diag_steps.py`,
  `scripts/qwen30b_dp2_accept.py`.
- Logs: `logs/`.

## 7. Remaining issues
- **MLA FULL-CG collapses at K>2** (same root cause, K-scaling). Not addressed
  (out of scope; respects fg2). The eager-attention remedy generalizes if desired.
- **FA3 FULL-CG draft decode loses the per-step graph-replay win** (attention runs
  eager). A true FULL fix would need V2's dedicated decode-graph manager
  (`autoregressive/speculator.py` + `DecodeSpeculatorCudaGraphManager`), which
  captures forward+sample+update as one graph and rebuilds metadata per step with
  a step-invariant `draft_max_seq_len`; porting that into the V1 proposer's
  double-wrapped (`_decode_fwd_sample` + `self.model`) CUDAGraphWrapper path is a
  larger change (a naive per-step `build(fast_build=False)` refresh HANGS in V1).
