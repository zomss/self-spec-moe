# Phase 64 results — window-KV self-draft END-TO-END at 16k, 2-node EP16 (Qwen3-30B-A3B)

Setup: h107+h106, DP16/EP16 (W7_NODES=2 W7_LOCAL_WORLD=8), 8-rail fabric,
ctx 16,384 (measured prompts 16,312-16,315 tok, distinct per request), chat +
on-dist prompts, max_model_len 20,480, MNB 8192, gpu_mem 0.90, greedy,
two-length decode slope (160/32), iters=2 warmup=1. Harness: w7_2node.py
(phase 52, unchanged); runner `scripts/run_arm.sh`. Self-spec arms: EP-routed
bf16 self-draft (NO quant, NO replica, NO local/node routing), sinks=16.
SCOPE NOTE: after the resident points landed, the mission was re-scoped to
"before"-baseline + diagnosis for a draft-path optimization phase: W256 arm
and the EAGLE re-reference were dropped; a rank-0 torch-profiler trace of the
draft cycle was added (`scripts/w7_trace16k.py` / `run_trace16k.sh`).

## The KV-capacity discovery (changes what b32 means)

The draft_model self-spec path DUPLICATES per-token KV: the drafter registers
its own 48 KV layers, so the unified pool costs 192 KiB/token instead of the
model's 96 KiB. Measured pools at this exact config:

| engine | available KV | pool (tokens/rank) | b8 demand | b32 demand |
|---|---|---|---|---|
| no-spec | ~52 GiB | **567,136** | 130k = 23% | 522k = 92% (resident) |
| self-spec (any W) | 41.8 GiB | **228,256** | 130k = 57% (resident) | 522k = **229% (waves)** |

So the mission premise "EP-routed bf16 self-draft needs ZERO extra memory"
is FALSE in the current implementation: weights are shared, but draft KV is
a second full allocation (plus ~10 GiB fixed overhead vs no-spec). At 16k
this HALVES the servable resident batch: b32 fits the no-spec engine but is
2.3x over the self-spec pool -> scheduler waves + preemption/recompute. The
largest clean resident self-spec batch is b12-b13 (86-94% pool); b12 was
measured as the extra resident serving point.

## Main table — decode tok/s (accept_len), 16k

| arm | b8 tok/s (accept) | x | b12 tok/s (accept) | x | b32 tok/s (accept) | x |
|---|---|---|---|---|---|---|
| no-spec (fresh) | 386.8+-7.1 | 1.00 | 395.7+-2.6 | 1.00 | 379.0+-8.6 | 1.00 |
| W512 K=2 | 131.1+-0.4 (2.869) | **0.34x** | -- | -- | THRASH (below) | ~0.01-0.1x |
| W512 K=4 | 148.2+-0.1 (4.673) | **0.38x** | 202.9+-1.4 (4.695) | **0.51x** | not run (same physics) | -- |

(no-spec fresh denominator vs Phase 59 refs 348.5/367.9: +8-11%, same
regime — 16k stays batch-flat 387/396/379 at b8/b12/b32.)
W512 K=2 b8 reproduced across two independent engine launches: 132.7+-0.9
and 131.1+-0.4, accept_len 2.869 both times.

**b32 self-spec = preemption livelock, not a serving point.** At 229% of
the spec-engine KV pool the scheduler cycles evict -> re-prefill 16.3k ->
evict: the rank-0 engine log shows re-prefill bursts at ~1200 tok/s prompt
throughput alternating with short decode bursts (peak ~98 tok/s/rank,
blended ~2-4 tok/s/rank vs no-spec's 379); the run did not finish even ONE
warmup pass of the 6-pass two-length protocol in 2400s (no-spec b32 finishes
all 6 passes in ~4 min). W512-K4 b32 was dropped: identical pool, identical
physics. (Scope re-cut mid-phase: W256 arm and the EAGLE re-reference were
also dropped — the phase pivoted to before-baseline + diagnosis for the
draft-path optimization phase.)

## Headline verdict

**The window-KV self-draft LOSES end-to-end at 16k on the real 2-node
fabric: 0.34-0.51x of no-spec — nowhere near the ~1.9x model prediction and
below EAGLE's 1.3-1.8x band — while accept is exactly as predicted.**
The accept side reproduces Phase 62 on the fabric (sanity PASS):

- K=2: accept_len 2.869 (r=0.957 predicts 2.87) — delta 0.00.
- K=4: accept_len 4.673/4.695 vs Phase 62's 4.584 — delta +0.09/+0.11 < 0.2.
- Window engagement at 2-node: `[kv-window] call=1001 bs=12 drafted=2
  true_max_seq=16455 win_seq(min/max)=531/544` — the draft reads ~530-544 of
  ~16.4k KV tokens (30x reduction), sustained through decode.

So the failure is pure systems cost of the draft path, not speculation
quality, and not the window mechanism's accept cost.

## Where the time goes (cycle arithmetic)

cycle_ms = decode_s / (128 tokens / accept_len); no-spec "cycle" = 1 step.

| point | ms/cycle | tokens/cycle | note |
|---|---|---|---|
| no-spec b8 | 20.7 | 8 | full target step |
| no-spec b12 | 30.3 | 12 | |
| no-spec b32 | 84.5 | 32 | |
| W512 K=2 b8 | 175.1 | 8 x 2.87 | |
| W512 K=4 b8 | 252.2 | 8 x 4.67 | |
| W512 K=4 b12 | 277.7 | 12 x 4.70 | ~flat in batch -> fixed costs dominate |

Solving b8 cycle = F + K*D from K=2/K=4: **D ~ 38.6 ms per marginal draft
step, F ~ 98 ms fixed per cycle** (verify forward ~21-25 ms of it, leaving
~75 ms/cycle of propose-path fixed overhead). Against the economic model:
the windowed draft step was predicted at 0.35x a full step (~7 ms at b8);
measured marginal cost is ~1.9x a full step (38.6 ms), PLUS ~75 ms/cycle
fixed — i.e. the draft path costs ~12-25x its KV-optimal budget. The KV
read that the window removes (~65% of a full step at 16k) is real but tiny
against these overheads: window ON already saved ~17-20% tok/s in Phase 62's
single-node b8 vs full-KV drafting; the remaining cost is NOT KV.

## Torch-profiler trace (rank 0, W512 K=2 b8, 60-step window)

`scripts/w7_trace16k.py` via `run_trace16k.sh 2 8 60` (2-node, rank 0
traced, `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`); trace at
`data/trace_w512k2_b8/` (5.5 MB gz, NOT committed), committed summary
`data/trace_w512k2_b8_summary.txt` + vLLM key-averages
`data/trace_w512k2_b8/profiler_out_0.txt`.

Runner-span totals over the window (60 execute steps; spans re-enter per
graph piece so counts exceed steps — totals are what matter):

| span (per engine step) | ms/step | share |
|---|---|---|
| draft (K=2 propose) | **98.6** | ~4x the verify forward |
| forward (verify) | 24.7 | == no-spec step (20.7-24.7) |
| bookkeep | 9.8 | |
| preprocess/sample/postprocess | ~2.5 | |

Named suspects, in order:

1. **DP-coordination `.item()` sync storm** (the historical draft-path
   pathology, confirmed at 16k): `aten::item` -> `_local_scalar_dense` ->
   `cudaStreamSynchronize` INSIDE draft spans totals **31.6 ms/step**
   (~6 calls/step; worst single call 43 ms). Overall the trace spends
   2.05 s of its CPU wall (52%) in 538 `.item()` calls.
2. **Per-draft-step cross-node EP collectives**: every MoE layer of every
   forward pays AllGather 363 us + ReduceScatter 173 us on the 16-rank
   fabric — ~25.7 ms of NCCL per forward, and the K draft forwards each pay
   it. Window totals: AllGather 1.67 s (45% of all CUDA time), RS 0.80 s
   (21%). The EP-routed draft is comm-bound, not KV-bound.
3. **Eager/piecewise draft chain CPU dispatch**: per-layer
   `unified_kv_cache_update` 8.7 ms/step + eager `unified_attention` 3.6
   ms/step + H2D/D2H copies ~9.9 ms/step inside draft spans.

Verify stays healthy (24.7 ms/step ~ no-spec step), so the entire loss is
the propose path: ~99 ms/propose vs the ~14 ms (2 x 0.35 x 20.7) the
economic model allotted. The window itself works as designed (draft reads
530-544 of ~16.4k KV tokens) — KV read is simply not the binding term of
the 2-node draft step; sync + comm are.

(Method note: torch profiling inflates CPU-side costs somewhat; the traced
cycle is consistent with the untraced 175 ms K=2 cycle. The 60-step window
covers batch drain — steps = tokens of the slowest request — so per-step
means late in the window average over smaller batches; shares, not exact
ms, are the readout.)

## Systems caveats for the optimizer (Phase 65)

1. **Draft-KV duplication** halves the resident batch at long context
   (228k vs 567k tokens/rank). A true self-spec shared-KV drafter (the
   window only changes what draft attention READS — same layers, same
   cache) would restore b32 residency AND remove the allocation entirely.
2. **~75 ms/cycle fixed propose overhead + ~39 ms/marginal draft step**
   (vs 20.7 ms full target step at b8). Trace-confirmed order of attack:
   (a) kill the propose-path `.item()` syncs (~32 ms/step, 6 syncs/step);
   (b) get the draft off the cross-node EP collectives (~26 ms NCCL per
   draft forward) — node-local or device-local draft routing (Phase 53/54
   knobs) trades accept for comm and is measured infrastructure already;
   (c) shave the per-layer eager dispatch of the piecewise chain
   (~22 ms/step CPU in KV-update + attention + copies).
3. b32 self-spec is 2.3x over-pool -> preemption livelock (evict /
   re-prefill-16.3k cycles); there is no b32 spec number to report, only
   the thrash signature above. Fixing (1) restores b32 residency (522k of
   567k) and makes the headline batch measurable at all.

## Data

Committed: `data/w72n_q30b_p64_*.json` (the W512-K2 b8 JSON is from the
b8-only relaunch, 131.1; the first combined-8,32 attempt's 132.7 row was
killed pre-JSON by the b32 timeout and survives in
`logs/run_all_driver.log`), trace summaries
`data/trace_w512k2_b8_summary.txt` + `data/trace_w512k2_b8/profiler_out_0.txt`,
table generator `scripts/analyze.py`. On disk only (uncommitted): raw
trace gz under `data/trace_w512k2_b8/`, engine logs under `logs/`
(`w512k2_b32_try1.log` = the b32 thrash evidence; `w256k4_try1.log` = the
dropped W256 arm, killed pre-measurement).
