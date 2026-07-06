# Phase 65 results — draft-path overhead optimization at 16k, 2-node EP16

Setup: identical to Phase 64 (h107+h106, DP16/EP16, ctx 16,384 distinct
per-request chat + on-dist prompts, max_model_len 20,480, MNB 8192,
gpu_mem 0.90, greedy, two-length slope 160/32, iters=2 warmup=1, harness
w7_2node.py). Self-spec W512 sinks=16 self-draft; b32 remains pool-blocked
(Phase 64: 2.3x over-pool preemption livelock) and was not run.

Stages (cumulative):

- **f1** = `VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1` — draft-path DP
  coordination on the gloo CPU group (zero GPU->CPU syncs in the propose
  path; result identical).
- **f12** = f1 + `VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1` — chain builds its
  FA metadata once, per-step updates are data-only in-place buffer writes +
  a max_seq_len scalar sync; persistent compaction gather buffer; cached
  slot-mapping dict.
- **f123** = f12 + the banked Phase-52 comm-free draft
  (`W7_DRAFT_QUANT=fp8 W7_DRAFT_FULL_REPLICA=1 W7_DRAFT_LOCAL_ROUTE=1`).
- **f123+skip** = f123 + `VLLM_SELF_SPEC_DRAFT_SKIP_DP_COORD=1` (the
  Phase-52 flag, safe only for the comm-free draft: skips the draft's DP
  coordination entirely instead of running it on the CPU group).

## Stage table — decode tok/s (accept_len), 16k

(scripts/analyze.py; no-spec refs: b6 287.6±3.5 measured this phase,
b8 386.8 / b12 395.7 from Phase 64)

| stage | K | b6 tok/s (accept) | x | b8 tok/s (accept) | x | b12 tok/s (accept) | x |
|---|---|---|---|---|---|---|---|
| base (P64) | K=2 | -- | -- | 131.1±0.4 (2.869) | 0.34x | -- | -- |
| base (P64) | K=4 | -- | -- | 148.2±0.1 (4.673) | 0.38x | 202.9±1.4 (4.695) | 0.51x |
| f1 | K=2 | -- | -- | 132.7±0.3 (2.869) | 0.34x | -- | -- |
| f1 | K=4 | -- | -- | 150.5±0.6 (4.670) | 0.39x | 204.4±2.9 (4.723) | 0.52x |
| f12 | K=2 | 104.1±0.2 (2.854) | 0.36x | 132.1±1.0 (2.869) | 0.34x | -- | -- |
| f12 | K=4 | 113.0±1.0 (4.594) | 0.39x | 148.7±0.2 (4.670) | 0.38x | 203.1±1.5 (4.723) | 0.51x |
| f123 | K=2 | 151.9±11.8 (2.841) | 0.53x | -- | -- | -- | -- |
| f123 | K=4 | 156.9±6.6 (4.477) | 0.55x | 43.7±4.8 (4.548) OVER-POOL | 0.11x | not run (pool) | -- |
| f123+skip | K=2 | 157.5±1.2 (2.843) | 0.55x | -- | -- | -- | -- |
| f123+skip | K=4 | **176.8±2.4 (4.465)** | **0.61x** | -- | -- | -- | -- |

Accept sanity: EP-routed accepts reproduce Phase 64 exactly (2.869 /
4.670-4.723); f123 accepts 4.477-4.548 bracket Phase 62 arm C's 4.547;
`[kv-window]` engaged everywhere (win_seq 530 of ~16.4k).

## Fixed/marginal split (cycle = F + K*D, from the K2/K4 pair)

| stage | solve batch | cycle K2 ms | cycle K4 ms | fixed F ms | marginal D ms/draft-step |
|---|---|---|---|---|---|
| base (P64) | b8 | 175.1 | 252.2 | 97.9 | 38.6 |
| f1 | b8 | 172.9 | 248.2 | 97.6 | 37.7 |
| f12 | b8 | 173.7 | 251.3 | 96.2 | 38.8 |
| f12 | b6 | 164.5 | 244.0 | 85.1 | 39.7 |
| f123 | b6 | 112.2 | 171.2 | 53.3 | 29.5 |
| **f123+skip** | b6 | 108.3 | 151.5 | **65.2** | **21.6** |

(f123 K2's ±11.8 noise puts ~±4 ms on its F/D split — the f123+skip pair
(±1.2/±2.4) is the trustworthy final-config split; no-spec step at
b6 = 20.9 ms, b8 = 20.7 ms — per-step flat, tok/s scales with batch in
this regime.)

## What each fix bought

**Fix 1 (sync storm) — verified removed, ~0 wall on the EP-routed draft.**
The flag provably works: the f123 trace shows ZERO cudaStreamSynchronize /
`aten::item` inside draft spans (the Phase-64 trace's #1 entry, 1891.9 ms
over the window, is gone entirely) and accept is bit-identical. But at
2-node the `.item()` block was ABSORBING skew, not adding it: the cycle's
critical path is the GPU pipeline of cross-node EP collectives in every
draft forward (~26 ms NCCL/forward), and parking the CPU cost nothing.
Same lesson as Phase 52's SKIP_DP_COORD A/B (+9.4% single-node, ~0 at
2-node). F 97.9->97.6, D 38.6->37.7.

**Fix 2 (light chain metadata) — numerically inert, masked for the same
reason.** Greedy A/B accept is EXACT (4.365 == 4.365 at the DP4 smoke;
4.670/4.723 == f1 at 16k). The metadata rebuild it removes is small
against the per-layer op dispatch it cannot remove (the chain still runs
96 eager custom ops + ~98 graph-piece launches per step). F 96.2, D 38.8
at b8 — noise-level vs f1.

**Fix 3 (comm-free fp8 full-replica draft) — the big lever: +39% tok/s at
the same batch; cycle -73 ms.** At b6 (the largest resident batch under
the replica): 113.0 -> 156.9 tok/s K=4 (0.39x -> 0.55x). Accept cost is
the known ~0.1 (4.594 -> 4.477).

**Bonus: +SKIP_DP_COORD on the comm-free draft — +12.7% more (156.9 ->
176.8, 0.61x; accept parity 4.465).** Once the draft has no collectives,
even the CPU-group coordination rendezvous costs ~10 ms/rendezvous of
rank-skew absorption (2 per cycle); skipping it entirely (safe only
comm-free — the EP-routed draft NEEDS the DP agreement for its collective
buffer sizes) moves all skew absorption to the verify's single
coordination. Final config cycle 151.5 ms = F 65.2 + 4 x 21.6.

## Where the remaining gap lives (f123 trace, K=4 b6, 54 steps)

Committed: `data/trace_f123_w512k4_b6_summary.txt` +
`data/trace_f123_w512k4_b6/profiler_out_0.txt` (raw gz on disk only).

- **Draft GPU is nearly optimal already: 21.1 ms CUDA per whole K=4
  propose = ~5.3 ms per draft forward** (vs the ~7 ms/step economic
  budget). NCCL in the trace is verify-only — the draft is truly
  comm-free.
- The draft SPAN (CPU wall) is ~53 ms/step traced: the marginal step
  D = 29.5 ms is ~5 GPU + ~24 ms of CPU dispatch — per step: 48x
  `unified_kv_cache_update` + 48x `unified_attention_with_output`
  (~15 ms CPU traced), ~98 cudaGraphLaunch piece launches (~10 ms),
  ~7 blocking cudaMemcpyAsync (~12 ms traced, ~1.7 ms each — H2D/D2H
  copies in the step glue, worth a targeted hunt), ~160 kernel launches.
- Fixed F = 53.3 decomposes as verify forward ~21 + verify-side NCCL
  DP-coordination all_reduce ~7.6 (32 calls in the window — now visible
  since the draft no longer hides it) + bookkeep ~12 + preprocess ~7 +
  step0 draft GPU ~5.

**Success bars vs measured: 1+2 target (D <= 15 ms) NOT met on the
EP-routed draft (masked by comm — structurally unreachable there);
1+2+3(+skip) target (D <= 10, E2E >= 1.3x) NOT met: D = 21.6, E2E 0.61x
at b6 (vs 0.38x before the phase at b8, 0.51x at b12).**
The residual is exactly the piecewise/eager chain dispatch — the one item
the mission pre-authorized as a possible rabbit hole:

### What chain FULL-CG needs (the D 29.5 -> ~6-8 ms lever)

1. The FA3 (GQA) decode kernel freezes its host-side work distribution at
   capture and does not re-derive it from the live `seqused_k` at replay
   (Phase 35: accept 4.9 -> 1.9; insensitive to scheduler_metadata /
   num_splits / capture seq_len). A CG-safe FULL chain needs either a
   kernel that re-reads its work distribution on-device, or a different
   draft-chain attention backend (FlashInfer persistent plan / Triton
   decode kernel with device-derived splits).
2. **Window scratchpad attention (proposed, Phase 66):** the W512 window
   bounds the draft's KV read to ~34 pages; gather sinks+window ONCE per
   cycle into a contiguous [bs, 544+K, kv_heads, d] scratch, run the K
   chain steps as fixed-shape dense decode attention (fully CG-capturable,
   no paging, no growing seq — the shape is cycle-constant), append chain
   KV to scratch + true cache. This sidesteps the FA3 replay bug entirely
   and fuses away the per-layer paged-attention glue.
3. Cheaper interim: batch the 48 per-layer kv-update + attention custom
   ops into fewer host calls (e.g. fold kv-update into the piece graphs —
   it is data-only and CG-compatible by construction, the same argument
   as the window write).

With D -> ~7 ms (GPU-bound chain), the b6 cycle becomes ~65 + 4x7 = 93 ms
-> ~288 tok/s = **~1.0x at b6**, and restoring b12 residency via the
Phase-66 KV cap (12 x 4.5 tokens/cycle at a ~95-100 ms cycle) puts
**~1.35-1.45x** in reach — i.e. the 1.3-1.6x target needs chain-CG (or
scratchpad attention) AND the Phase-66 KV work, not more micro-trimming.
The other F lever is the verify-side ~8 ms NCCL DP-coordination
(engine-level `--disable-nccl-for-dp-synchronization`, untested here —
the harness would need to pass it).

## Stage f123 — the fp8 replica hits the KV wall at protocol batches

Measured pools (tokens/rank, gpu_mem 0.90, 16k):

| engine | pool | b6 | b8 | b12 |
|---|---|---|---|---|
| no-spec (P64) | 567,136 | 17% | 23% | 35% |
| self-spec EP-routed (f1/f12) | 236,512 | 42% | 56% | 84% |
| self-spec fp8 replica (f123) | **115,968** | **85%** | **114%** | **171%** |

The fp8 full replica costs 120.5k tokens = 22.1 GiB at EP16 on top of the
draft-KV duplication, pushing BOTH protocol batches over-pool. Protocol
b8 (114%): scheduler waves — 43.7±4.8 tok/s (0.11x), accept_len 4.548
(== Phase 62 arm C's 4.547: accept sanity PASSES; the tok/s is pool
physics, not draft speed; 316 "Waiting:" scheduler lines, decode passes
23.6 s vs 6.9 s at f12). b12 (171%) not run to completion (same physics
as Phase 64's b32 livelock). b6 (85%) is the resident diagnostic point,
with f12 and no-spec re-measured at b6 for apples-to-apples.

## Phase-66 groundwork — capping / sharing the drafter's KV

Where the drafter's KV is sized: the draft model's 48 `Attention` layers
register in the static forward context; the runner's
`get_kv_cache_spec()` (vllm/v1/worker/gpu_model_runner.py:7679) collects
one spec per layer, and since draft layers have `sliding_window=None`
they return `FullAttentionSpec` (vllm/model_executor/layers/attention/
attention.py:581) — identical to the target's. All 96 layers therefore
land in ONE KV-cache group with shared block tables and a per-token page
cost of 192 KiB (the Phase-64 pool-halving discovery). There is no
"drafter pool" to cap: draft KV rides the same block allocation as
target KV, block for block.

A `VLLM_SELF_SPEC_DRAFT_KV_POOL_TOKENS` cap therefore needs the draft
layers in a SEPARATE kv-cache group with its own (smaller/wrapping)
allocation, i.e. the hybrid-KV-cache-manager path:

1. Draft `Attention.get_kv_cache_spec()` returns a windowed spec (e.g.
   `SlidingWindowSpec(sliding_window=cap)`) when the flag is set — the
   manager then allocates/frees draft blocks in a window automatically.
2. The propose path must consume the DRAFT group's block table + slot
   mappings (today it reuses the target-group `common_attn_metadata`;
   the per-layer `slot_mappings` dict argument already exists).
3. Sinks: `SlidingWindowSpec` frees the FIRST blocks too, killing the
   16 attention-sink tokens the Phase-62 window keeps. Needs either a
   sink-aware spec (keep first block + trailing window — the natural
   "StreamingLLM spec") or a measured accept at sinks=0.
4. `validate_same_kv_cache_group` (all draft layers in one group) still
   holds; scheduler-side hybrid-manager compatibility with the padded
   drafter batch is the main untested surface.

The full Phase-66 fix (true shared-KV self-draft: draft READS the
target's cache through the windowed view, allocates only its window
tail) removes the second allocation entirely and restores b32 residency
(522k of 567k tokens/rank) — and combined with the scratchpad-attention
chain above addresses both the memory wall and the dispatch wall.

## Smoke validation (single-node DP4/EP4, 2k ctx, b4)

| stage | K | accept_len | note |
|---|---|---|---|
| f1 | 2 | 2.850 | window engaged (win_seq 541 of ~2045) |
| f12 | 4 | 4.365 | light-MD active on chain steps 2-3 |
| f1 | 4 | **4.365** | == f12 EXACTLY (greedy A/B: light-MD is inert) |
| f123 | 4 | 4.273 | fp8 replica loaded (use_ep=False), small accept cost |

fp8-replica pool cost at DP4/EP4 gpu_mem 0.85: 142,896 -> 77,616
tokens/rank (delta 65.3k tokens = 12.0 GiB; the EP16 delta is 22.1 GiB —
the replica's footprint grows with EP degree since the per-rank shard it
duplicates shrinks).

## Data

Committed: `data/w72n_q30b_p65_*.json` (f1/f12 at b8/b12+K2; f12+f123+
f123-skip+nospec at b6; smokes; the f123 b8 over-pool row survives only
in `logs/w512k4_f123_try1.log` — its try was cut before the b12 point so
no JSON was written), trace summary + vLLM key-averages for f123
(`data/trace_f123_w512k4_b6*`), `scripts/` (stage runner, b6 runners,
smoke, trace runner + patched harness, analyzer). On disk only: raw trace
gz, engine logs under `logs/`.
