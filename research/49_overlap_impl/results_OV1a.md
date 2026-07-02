# OV1(a) results — free-running ahead-chain, validation mode: PASS

**TL;DR.** The free-running draft works end to end in validation mode: during each
verify, the chain continues K+1 steps on the side stream (bonus guess + the next
cycle's drafts) with real inputs, metadata, and KV writes; the measured per-request
**hit rate is 0.909–0.910** (19,200 requests / 300 cycles; hit = all-K accepted AND
bonus == guess) — above the beta^(K+1)≈0.85 prior; accept stays **2.91** (output
byte-identical, outputs discarded in this mode); and the 3-forward (~45 ms) ahead
chain hides **~60% at a2a=0 / ~72% at a2a=500** — the OV0a physics ceiling. The
overlap window is BOUNDED to the verify's GPU time (stream-level wait), which is
the proven-safe region; an intermittent illegal-access race in the unbounded
window (ahead ∥ post-verify epilogue) is documented and side-stepped, not root-caused.

## Validated numbers (Qwen3-30B FP8 replica, DP4/EP4 GPUs 0-3, b64, K=2, greedy)

| a2a µs | off tok/s | ahead-on tok/s | ratio | ahead exposure | hidden | accept | hit rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0   | 2478 | 1994.7 | 0.805 | ~18 ms of ~45 | ~60% | 2.91 | 0.909 |
| 500 | 1710 | 1530.0 | 0.895 | ~13 ms of ~45 | **~72%** | 2.91 | 0.910 |

Validation mode still pays BOTH the ahead chain and the normal propose; full OV1(b)
replaces the propose for the ~91% hit cohort, converting this overhead run into the
speedup run.

## The debugging arc (what it took)

The concurrent ahead chain crashed with a sticky CUDA illegal memory access. The
bisection matrix (each row one engine run):

| config | result |
|---|---|
| serialized, everything live (KV on / off) | clean |
| concurrent, 1 ahead step | clean |
| concurrent shadow (pure replay, no allocations) | clean |
| concurrent GEMMs (allocations, no replays) | clean |
| concurrent ≥2 steps: full / no-KV / no-sample / frozen-metadata / no-MemPool / mem-headroom 0.85 / boundary-fenced | **all crash**, consistently right after cycle ~50 (the first long generate's tail), AFTER logging hit rate 0.92 |
| concurrent + per-sub-op side-stream syncs (host-ordered epilogue) | **clean** |

Theories tested and killed: piecewise address determinism, cross-stream MemPool,
expandable segments, allocator reclaim under memory pressure, generate-boundary
races (the fence is kept — it is semantically required regardless). The decisive
observation: ahead ∥ verify is safe (shadow + sync-mode evidence); the race needs
the ahead chain in flight during the POST-VERIFY window (sampler / bookkeeping /
next-step prep / next verify). Root cause in that window not yet identified —
`W7_AHEAD_UNBOUNDED=1` re-opens it for a future compute-sanitizer hunt.

**The fix (bounded overlap):** after launching the ahead chain, the MAIN stream
waits (stream-level, not host) on the ahead-done event — the ahead fully overlaps
the verify and nothing else. Cost: max(0, ahead_end − verify_end) exposed, which
at a2a=500 is ~13 ms (~28% of the chain) and shrinks as f grows.

## New machinery (all env-gated, default off)

- `VLLM_SELF_SPEC_AHEAD_CHAIN` — the free-running ahead chain (validation mode).
- Scheduler lookahead K → 2K+2 under the flag (ahead KV always in-bounds).
- Metadata privatization (cloned seq_lens/block_table for the ahead loop) — OV1
  isolation rule #5: never mutate runner-shared views off the main stream.
- Batch-boundary fence (`fence_ahead_chain`) — rule #6: speculation never crosses
  a non-uniform scheduler step.
- Bounded-overlap stream wait — rule #7: side-stream work confined to the verify
  window until the epilogue race is root-caused.
- Debug knobs: `W7_AHEAD_STEPS/NO_KV/NO_SAMPLE/FROZEN_MD/SYNC`, `W7_AHEAD_UNBOUNDED`,
  `W7_SHADOW_MAIN_STREAM`.

## Next: OV1(b)

Consume the hits: for the ~91% hit cohort, skip the normal propose and use the
ahead tokens (they are exactly what the propose would produce — same chain, same
state); sub-batch re-propose for misses. Expected effect at a2a=500: remove the
~30 ms lockstep chain from ~91% of cycles at the cost of the ~13 ms bounded ahead
exposure → roughly +15-20% throughput over lockstep spec at f≈0.6, growing with f,
before any K-retune upside.
