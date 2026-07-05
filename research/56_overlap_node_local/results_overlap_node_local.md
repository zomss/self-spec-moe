# Phase 56 results — overlap x node-local on the real 2-node fabric

Status: Stage A in progress.

## 0. Code facts (Stage A step 1 — established by reading, before any run)

1. **The ahead chain runs the node-local draft config.** `run_ahead_chain`'s
   forwards use `set_forward_context(..., additional_kwargs=
   self._draft_forward_additional_kwargs)` (`llm_base_proposer.py`, ahead
   loop) — the same dict propose uses, which is
   `{SELF_SPEC_NODE_LOCAL_KEY: True}` under
   `VLLM_SELF_SPEC_DRAFT_NODE_LOCAL=1`. So every ahead forward issues the
   REAL intra-node subgroup collectives (`_EP_NODE` under SP/TP2) on the
   side stream, L x (all-gatherv + reduce-scatterv) per forward — exactly
   the composition under test. It does NOT fall back to the device-local
   comm-free config.
2. **K=1 is a silent no-op.** The OV1(a) stash gate is
   `self._ahead_chain and self.num_speculative_tokens > 1 and all_greedy`:
   the ahead chain CONTINUES the decode-shaped chain state, which only
   exists at K>=2 (at K=1 propose is step0-only, q=K+2-shaped). The ladder's
   W7_KS=1 config therefore cannot exercise the overlap; Stage A runs at
   **K=2** (unlocked at 236B by Phase 55 section 3c).
3. **Isolation stack required, missing from the 2-node env.**
   AHEAD_CHAIN's documented prerequisites `VLLM_SELF_SPEC_DRAFT_GRAPH_POOL=1`
   + `VLLM_SELF_SPEC_DRAFT_WORKSPACE=1` (OV0b root causes: shared cudagraph
   pool / shared fused-MoE workspace under concurrent replay) are not in
   `env_2node.sh`; Phase 56 scripts set them in BOTH arms of every A/B.
4. **Validation-mode symmetry: conditional, not structural.** Both paths
   (ahead + lockstep propose) run on every rank only in uniform-decode
   steady state. The batch-boundary fence (`fence_ahead_chain`) is
   RANK-LOCAL (each runner fences on its own scheduler step), so a
   collectivizing draft could diverge at stagger/drain boundaries — the
   Phase 54/55 deadlock class. Two things keep the harness safe in
   practice: (a) every DP engine runs an IDENTICAL fixed-length batch
   (same prompts, ignore_eos), so scheduler steps — and hence fences — are
   cycle-synchronized across ranks; (b) within a step, the ahead chain's
   collective count/order is identical on all ranks (same K+1 forwards,
   same stashed shapes). A production scheduler with per-rank load
   divergence would NOT be safe; that is Stage-B-class work.
5. **Consume mode + node-local = deadlock by design, confirmed.** In
   consume mode `_determine_batch_execution_and_padding` SKIPS
   `coordinate_batch_across_dp` entirely and fabricates a locally-uniform
   `num_tokens_across_dp` ("per-rank dispatch divergence is safe" — true
   only for the comm-free draft). With node-local, (i) consume-vs-propose
   divergence changes the collective SEQUENCE per rank, and (ii) the
   fabricated sizes vector feeds `_node_group_and_sizes`, so even
   symmetric-decision cycles gather with locally-assumed (unvalidated)
   peer sizes. CONSUME_AHEAD=1 + DRAFT_NODE_LOCAL=1 must not be run before
   the Stage B redesign.
6. **MLA/FULL-graph caveat for the hit rate.** OV1(a) was only ever
   validated on GQA (Qwen), where the chain forces eager attention →
   PIECEWISE ahead forwards read the privatized (cloned) metadata live. On
   DeepSeek (MLA) the chain — and therefore the ahead loop, which reuses
   the chain's stashed dispatch mode — replays FULL graphs whose captured
   attention reads the PERSISTENT buffers baked at capture (runner
   seq_lens/block table); the ahead loop's cloned `cad_priv.seq_lens`
   updates never reach them, so ahead steps attend with seq_lens frozen at
   chain-end. Expected effect: degraded hit rate (staleness), NOT a race
   (the bounded-overlap wait keeps the epilogue out of the window; slot
   mappings come from proposer-owned buffers the loop does update).
   Timing validity of the physics A/B is unaffected.
7. **The self-spec profiler serializes the overlap.** `region()` does
   `torch.cuda.synchronize()` at both ends; a profiled run measures the
   ahead work SERIALIZED after the verify. Profiled runs are used only for
   component attribution (T_verify, T_ahead); the physics A/B is always
   unprofiled tok/s.

## 1. Stage A smoke, attempt 1: validation mode + node-local WEDGES at the
##    first generate tail (prediction #4's caveat, measured)

V2-Lite 2-node TP2xDP4/node (DP8/EP16), K=2, b8, AHEAD_CHAIN=1 +
DRAFT_NODE_LOCAL=1, isolation stack on:

- Engages cleanly: node-local ENGAGED on both nodes, ahead chain live,
  hit rate logged at cycle 50 on all 8 workers (0.42-0.66 per rank).
- WEDGES ~1 min later, exactly at the tail of the first long warmup
  generate: 100% GPU spin on all 8 GPUs/node, every busy worker parked at
  the DRAFT step0's `coordinate_batch_across_dp` `.item()`
  (`dp_utils.py:216` / `_post_process_cudagraph_mode` on h106) — the
  Phase 54/55 unmatched-collective signature
  (`logs/smoke_a1_wedge_pyspy_h107.txt`, `_h106.txt`).
- Mechanism (resolves the "fixed-length batches are synchronized"
  assumption): max_tokens caps OUTPUT length, but K=2 acceptance variance
  spreads request FINISH CYCLES per rank. The first rank with a
  `finished_req_ids` step fences (rank-local) and skips its ahead chain;
  un-fenced ranks launch theirs — whose node-subgroup collectives then
  pair with the fenced rank's NEXT propose collectives. Corrupted pairing
  wedges both nodes' streams; hosts park at the next rendezvous `.item()`.
  Validation mode with a comm-free draft (OV1a, Qwen single-node) never
  saw this because per-rank ahead divergence was harmless by construction.

**Fix (the composition's first real artifact): DP-uniform ahead gate,
piggybacked on the existing DP coordination** (the mechanism the Stage-B
design brief suggested for consume — it turns out VALIDATION mode already
needs it with a collectivizing draft):

- `dp_utils._run_ar` gains a 5th row iff `VLLM_SELF_SPEC_AHEAD_CHAIN` is
  set (env is process-uniform, so the collective stays shape-symmetric on
  every path: busy target step, idle dummy, draft-side coordinations).
- The runner publishes "I hold a live ahead stash" after its rank-local
  fence decision, BEFORE the target-step coordination; `_dummy_run`
  publishes False (an idle wave partner never runs an ahead chain).
- `run_ahead_chain` is gated on the synced AND; on a skipped cycle the
  stash is fenced away so the next propose re-stashes on all ranks
  together. Zero new collectives, zero new syncs (the coordination
  already `.item()`s its tensor).

## 2. Stage A smoke, attempt 2 (gated): PASS — the composition runs end to end

Same config + the DP-uniform gate (`run_smoke_v2lite.sh`, AHEAD=1,
b8+b64, ITERS=2, RC=0, no wedge through every generate tail):

| b/rank | ahead-on tok/s | accept | ahead hit rate (steady) |
|---:|---:|---:|---:|
| 8  | 155.1 +- 77.2 | 2.545 | 0.60-0.66 |
| 64 | 1087.2 +- 0.2 | 2.530 | 0.599-0.650 |

- Accept 2.53-2.55 = the Phase 55 lockstep band (2.51-2.59): validation
  mode stays output-neutral under the composition ✓.
- Hit rate 0.60-0.65 vs OV1a's 0.91 (Qwen, comm-free): consistent with
  coverage-50% beta (~0.87 -> beta^3 ~0.66) COMPOUNDED BY code-fact #6
  (MLA FULL-replay ahead steps attend with chain-end-frozen seq_lens).
  Two ranks pin at 0.66 (= beta^3), others 0.42-0.47 — per-rank routing
  variance. Fine for physics; a consume design would want the PIECEWISE
  ahead path or per-step seq_lens refresh.
- The b8 +-77 spread is the 2-iter cross-node fabric noise already
  documented in Phase 54/55 (b64 is tight: +-0.2).

**Paired A/B (same binary, same flags incl. isolation stack, ITERS=2):**

| b/rank | AHEAD=0 tok/s | AHEAD=1 tok/s | ratio | cyc off -> on (ms) | delta |
|---:|---:|---:|---:|---|---:|
| 8  | 301.7 +- 2.5  | 155.1 +- 77.2 | 0.514 | 65.9 -> 131.3 | +65.4 |
| 64 | 1606.5 +- 18.5 | 1087.2 +- 0.2 | 0.677 | 101.3 -> 148.9 | +47.6 |

(cycle_ms = 1000*b*accept/tok_s per rank. Accept off/on: 2.487/2.545 at
b8, 2.544/2.530 at b64 — within the documented V2-Lite K=2 run-to-run
spread 2.49-2.59.) At V2-Lite the 3-forward ahead chain is mostly NOT
hidden: the added 47-65 ms/cycle is on the order of the full serialized
side work, i.e. the verify window at this scale is far smaller than the
ahead chain. This is the expected regime (V2-Lite verify is tens of ms
with a small comm fraction); the physics question lives at 236B where
the verify is ~75-90 ms at ~50% inter-node comm.

## Stage A — overlap PHYSICS, answered directly (microbench + the V2-Lite A/B)

The 236B ahead-chain run wedged (the Phase-54/55 deadlock class resurfacing
in the ahead loop's rank-local fences even with the DP-uniform gate; 3
variant runs a1/ms/ctas16 all `EngineDeadError` at the first drain,
`shm_broadcast` 60s-timeout hang signature, logs in `logs/236b_a1*`).
Rather than sink further into that distributed wedge, the load-bearing
question -- *does side-stream compute overlap with the verify's inter-node
NCCL collectives on this fabric?* -- is a property of the fabric+NCCL, so it
is measured DIRECTLY with a microbench (`scripts/overlap_microbench.py`, 16
ranks = the real 2-node EP topology): a collective loop (inter-node
all_gather, verify-comm stand-in) on one stream vs a matmul loop
(draft-forward stand-in) on another, timed solo then concurrent.

| regime (NC comm / NM compute iters) | T_comm | T_compute | T_concurrent | serial=sum | **conc/max** | full-serial line |
|---|---:|---:|---:|---:|---:|---:|
| NC30/NM40 | 3.52 | 7.14 | 11.39 | 10.66 | **1.59** | 1.49 |
| NC60/NM30 | 6.81 | 5.43 | 14.63 | 12.24 | **2.15** | 1.80 |

**Verdict: overlap NO-GO on this stack.** In BOTH regimes T_concurrent is
*worse* than fully serial (fraction-hidden = -0.21, -0.44): running compute
and inter-node collectives on separate streams is slower than back-to-back.
Default NCCL collectives are SM-based (staging/copy kernels occupy SMs), so
they CONTEND with the matmul rather than overlapping the network wait.
Single-node NVLink shows the same (ratio 1.20 vs serial-line 1.16) -- this
is an SM-contention property of NCCL, not RoCE-specific.

This EXPLAINS the Stage-A-attempt-2 V2-Lite A/B above: ahead-on added
+47-65 ms/cycle = the full serialized side work (0.51-0.68x), never a hidden
draft. The Phase-32 overlap thesis's core premise (draft hides behind the
verify's comm -> draft ~free -> ~1.4x at f~0.5) does NOT hold with naive
PyTorch stream overlap + default NCCL on this cluster.

**Scope / caveats (what would change the verdict):** the negative is scoped
to *default SM-based NCCL + naive stream overlap*. Production overlap
(DeepEP low-latency, DBO, NVSHMEM) frees SMs from the collective and CAN
overlap -- not installed here (the standing Phase-19/24 DBO-baseline gap).
The compute stand-in is dense matmul; the real draft is BW-bound expert
GEMMs, which contend differently with NCCL's copy kernels (could differ, but
the sign of a -0.2 to -0.4 hidden-fraction is a wide margin to overturn).

## Stage B — node-safe consumption: NOT PURSUED (Stage A NO-GO)

The consume redesign only pays if overlap hides the draft. Stage A shows it
does not on this stack, so consumption would at best recover the lockstep
number (skip propose on all-hit) minus the host-sync gate cost (the OV1b1
lesson) -- no path to the overlap win here. Revisit only with a
SM-freeing comm backend (DeepEP/NVSHMEM). The DP-uniform ahead gate
(committed `eacc20bdb`) stands as the reusable artifact for that future work.

## Bottom line

Overlap x node-local is a **measured NO-GO on this 2-node RoCE + default-NCCL
stack**: inter-node collectives contend with compute rather than overlapping,
so the Phase-32 ~1.4x is unreachable here. The honest lever for the overlap
win is a comm backend that frees SMs (DeepEP/DBO) -- a hardware/stack
dependency, not a spec-decode algorithm change. Node-local lockstep (Phase
54/55: 81/273/379 tok/s K=1, the +40% comm-saving) remains the best
real-fabric configuration.

## Stage B — node-safe consumption (gated on Stage A)

TBD.
