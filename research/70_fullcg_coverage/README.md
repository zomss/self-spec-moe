# Phase 70 — draft-chain FULL cudagraph: capture/replay COVERAGE fix

Source: Phase 69. P69 landed the window-scratchpad dense attention that makes the
GQA draft chain a fixed-shape, CUDA-graph-capturable forward. It PROVED the win
where the FULL graph replays (DP4 / 2k / W64 / cap96: +93% tok/s at K4, accept
preserved). The 16k / DP8 / W512 / cap544 serving batch REGRESSED: the chain
"claims FULL" (skip_rebuild fires, accept stays correct at 4.666) but the draft
step goes 23 -> 52 ms — the forward runs EAGER instead of replaying a captured
graph. Everything behind `VLLM_SELF_SPEC_DRAFT_FULLCG` (default off) +
`VLLM_SELF_SPEC_DRAFT_KV_WINDOW`.

## Objective

Crack the ONE remaining bug: make the cap544 / 16k / W512 draft-chain FULL graph
actually REPLAY (drop the draft step to its ~6-8 ms floor), then measure the
2-node E2E win.

## Method

Step 1 (debugging): instrument the cudagraph dispatch/lookup path
(`W7_FULLCG_DBG=1`) to print, per draft-chain forward, the runtime batch
descriptor used to LOOK UP a captured graph vs the descriptors actually
CAPTURED at warmup, and the wrapper outcome (REPLAY / CAPTURE-MISS /
PASSTHROUGH). The captured-vs-requested diff IS the bug.

- `vllm/compilation/cuda_graph.py` `CUDAGraphWrapper.__call__`: one-shot
  per-(wrapper, outcome, descriptor) log of the outcome + ctx descriptor +
  captured keys.
- `vllm/v1/spec_decode/llm_base_proposer.py` propose chain-setup: one-shot dump
  of `batch_size`, dispatched `_last_batch_desc`, `_full_captured`, and the
  drafter.model FULL wrapper's captured descriptor keys.

Step 2: align the warmup capture so the cap544 scratchpad chain forward is
recorded under the SAME descriptor used at 16k replay (fix per Step-1 finding).

Step 3 (single-node hard gates): (a) log "REPLAYED" at 16k/W512/cap544;
(b) draft step 23.1 -> target <= ~8 ms; (c) accept preserved (16k W512 K4
4.66 +-0.03; canary W64 K2 bit-exact); (d) ~1 launch/step.

Step 4 (2-node E2E, only if Step 3 green): fresh no-spec denominator at
b12/b24, then arm-B + FULLCG. Target ~1.4x b12, >1.2x b24.

## Scripts

- `scripts/env_1node.sh` — arm-B single-node env (copied from P69).
- `scripts/run_1node.sh` — DP8 single-node self-spec runner (TAG K BATCH FINE
  [PORT] [EXTRA_ENV]).
- `scripts/probe.sh K` — lean 16k FULLCG run with `W7_FULLCG_DBG=1` for the
  descriptor diff.

Results: `results.md`. Data: `data/`, logs in `logs/`.
