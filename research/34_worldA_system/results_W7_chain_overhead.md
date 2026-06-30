# W7 — chain-overhead reduction in the draft loop (V2-Lite, b64, K=2)

Goal: remove the remaining per-step overhead in the V1 `draft_model` draft loop
(`llm_base_proposer.py`) under the draft FULL cudagraph
(`VLLM_SELF_SPEC_DRAFT_FULL_CG`, default off). Starting point: draft FULL-CG
with correct attention (1-split) lands accept_len ~2.7–2.81 and a
**chain_overhead ~4.2 ms/step**, the gap to the V2-Lite forward-based ceiling.

Setup: DeepSeek-V2-Lite, DP=2 + EP, forced-PCIe
(`NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1`), emulated A2A 100µs,
bf16 full-replica comm-free draft, CUDA graphs on. Harness
`scripts/w7_microbench.py` (`MB_STEADY=100 MB_WARMUP=50`,
`VLLM_SELF_SPEC_DRAFT_FULL_CG=1 MB_FULL_REPLICA=1`). All timings are
CUDA-synced wall-clock means over the post-warmup steady tail.

## Step 1 — CONFIRMED per-step breakdown (instrumented)

Fine-grained timers (`VLLM_SELF_SPEC_PROFILE_FINE=1`, env-gated so the extra
syncs do not perturb the default `draft_chain` total) around each region of the
decode loop and the step-0 setup. Measured (b64 K2, FULL-CG on), per region
mean ms (each independently CUDA-synced):

| Region | measured ms | prior estimate |
|---|---|---|
| `step0_set_inputs` (input shift/copy kernels) | 1.05 | — |
| `step0_determine_batch` (CG dispatch + build_model_inputs) | 1.11 | — |
| `step0_build_attn_md` | 0.68 | (part of 1.8) |
| `step0_sample` (compute_logits+argmax) | 0.71 | (part of 0.9) |
| `chain_setup` (uniform determine_batch + qsl clone + seq_len adj) | 1.22 | — |
| `step_pos_slot_update` (per decode step) | 0.60 | ~0.8 |
| `step_build_attn_md` (per decode step) | 0.67 | ~1.8 |
| `step_sample` (per decode step) | 0.65 | ~0.9 |
| `step_input_buffering` (per decode step) | 0.48 | ~0.4 |

Key corrections to the prior estimate:
- **attn-metadata build is ~0.66–0.68 ms, NOT 1.8 ms.** For pure-decode MLA,
  `build_for_drafting`→`build()` early-returns from `split_decodes_and_prefills`
  (max_query_len=1 ≤ threshold, no D2H sync) and the chunked-prefill path is
  skipped, so it is just Python dataclass construction.
- Overhead is spread fairly evenly; the largest single *fixable* category is
  sampling (`step0_sample` 0.71 + `step_sample` 0.65 = 1.36 ms total), which the
  sampling-in-graph fix targets.
- The largest *serial* chunks are step-0 input-prep / orchestration
  (`set_inputs` 1.05, `determine_batch` 1.11, `chain_setup` 1.22 ≈ 3.4 ms),
  which neither proposed fix addresses (they are `propose()` input-prep, not the
  K-loop, and run once per propose() regardless of K).

## Step 2 — reductions implemented (both behind `VLLM_SELF_SPEC_DRAFT_FULL_CG`)

### #1 in-place attn-metadata → skip the per-step rebuild entirely

`vllm/v1/spec_decode/llm_base_proposer.py` decode loop (`step_build_attn_md`
region). Investigation finding: under FULL-CG replay the captured graph reads
the live attention tensors (`seq_lens`/`block_table`/`slot_mapping`) **through
the pointers captured at graph-capture time** — `CUDAGraphWrapper.__call__`
just `replay()`s and never re-reads the per-step `attn_metadata` object from the
forward context. And `_update_positions_dependent_metadata` already advances
those tensors **in place**: the loop's `common_attn_metadata.seq_lens` is a view
of `runner.seq_lens` (which `_build_draft_decode_capture_metadata` captured
against), and the slot mapping lands in the persistent `_slot_mapping_buffer`.

So the per-step `build_for_drafting` is **dead work for FULL-CG replay**. The
fix skips it (the step-0 metadata object is kept only as a placeholder for
`set_forward_context`; its contents are ignored on replay). This is the ultimate
"in-place" — zero rebuild — and it is **provably lossless**: accept_len is
unchanged (2.698) because the in-place buffers are exactly what the graph reads.
A/B knob: `W7_DISABLE_SKIP_REBUILD=1`.

### #2 sampling in the graph

`vllm/v1/spec_decode/llm_base_proposer.py` (`_DraftDecodeForwardSample` class,
decode loop, `dummy_run`) + `vllm/v1/worker/gpu_model_runner.py` (wrap pass).
Mirrors V2 `AutoRegressiveSpeculator._generate_draft`: a combined runnable
(forward + `compute_logits` + greedy `argmax`) FULL-wrapped as its own graph
(sibling to `drafter.model`, wrapping the RAW model to avoid nested graphs),
captured on the draft FULL (uniform-decode) `dummy_run` pass and replayed per
decode step. The draft tokens come straight out of the captured graph; the
separate eager `compute_logits` launch per step is gone. Only the greedy argmax
path is captured (probabilistic draft sampling stays eager). A/B knob:
`W7_DISABLE_SAMPLE_IN_CG=1`.

## Decisive numbers — before vs after (V2-Lite, b64, K=2, FULL-CG on)

Matched A/B on the **same binary** (`W7_DISABLE_*` knobs), 3 runs each, clean
`draft_chain` (fine timers OFF so the chain total is undisturbed):

| | accept_len | draft_chain | chain_overhead | verify | implied speedup |
|---|---|---|---|---|---|
| **before** (opts OFF) | 2.696 | 31.94 ms | 4.40 ms | 19.76 ms | 0.936 |
| **after** (opts ON) | 2.697 | 31.55 ms | 3.88 ms | 19.78 ms | 0.943 |
| Δ | +0.001 | −0.39 ms | **−0.52 ms** | — | +0.007 |

- chain_overhead 4.40 → **3.88 ms** (−0.52 ms), accept_len **unchanged**.
- implied speedup `accept_len·17.95/(chain+verify)`: 0.936 → **0.943**.
- losslessness: spec-ON and spec-OFF produce **byte-identical** token sequences
  for all spot-check prompts (`specOFF == specON: True`); both diverge from the
  dense no-spec path at the same positions (a pre-existing bf16 full-replica
  numerics property, not introduced here). Greedy output is coherent.

The instrumented sub-region timers confirm the work is gone: `step_build_attn_md`
0.66 → 0.39 ms and `step_sample` 0.65 → 0.42 ms (both now at the ~0.4 ms
profiler-sync floor, i.e. the real work is eliminated).

## Why the wall-clock win is only ~0.5 ms (and 1.1–1.14× is NOT reachable here)

The ~1.3 ms of CPU work removed (per-step build + sample) was **largely
overlapped with the GPU forwards** — it ran on the CPU concurrently with draft
attention/MLP kernels — so removing it only recovers the non-overlapped tail
(~0.5 ms). The chain is **GPU-bound by the forwards**: at b64 K2,
`draft_forward_first` ≈ 15.05 ms + `draft_forward` ≈ 12.63 ms = 27.68 ms of the
31.55 ms chain.

Consequently, with the measured forward costs the **forward-based ceiling is
only ~1.02×, not 1.14×**: to reach 1.14× the draft chain would have to be
≤ 22.7 ms, but the forwards alone are 27.68 ms. The step-0 forward (15.05 ms) is
notably more expensive than the decode-step forward (12.63 ms) because it
processes the verify query positions (~K+1/req, padded). **Cutting
chain_overhead cannot reach 1.1–1.14× here** — that requires making the draft
*forwards themselves* faster (the step-0 forward in particular), which is a
different lever than per-step orchestration overhead.

## Blocker / remaining approach

No correctness blocker — both reductions are lossless and shipped behind the
opt-in flag (default off → unchanged). The in-place attn-md was **not** risky
in the end: the draft FULL graph already reads in-place buffers, so the safe and
maximal version is to skip the rebuild outright (no partial in-place update of a
persistent `MLACommonMetadata` needed).

The remaining gap to the V2-Lite ceiling is **not** per-step overhead — it is
the draft forward cost itself, dominated by the step-0 forward. Remaining
approaches to actually move the speedup:
1. Shrink the step-0 draft forward (it processes K+1 padded verify positions;
   a tighter step-0 batch / removing padded positions would cut ~2–3 ms).
2. Reduce the serial step-0 input-prep (`set_inputs` + `determine_batch` +
   `chain_setup` ≈ 3.4 ms once per propose()) — e.g. cache the
   `query_start_loc_cpu` clone, fuse the CG dispatch.
3. These are larger structural changes to `propose()` input-prep / the step-0
   forward, outside the scope of the two per-step reductions landed here.
