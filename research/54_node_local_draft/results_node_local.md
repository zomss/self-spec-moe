# Phase 54 results — node-local draft: acceptance validated, execution mode open

**Implementation** (branch working tree, env-gated, default off):
`VLLM_SELF_SPEC_DRAFT_NODE_LOCAL=1` -- draft routes to any expert resident on
its NODE (union residency mask, `determine_node_expert_mask`) and
dispatches/combines over a new intra-node `dp_node` subgroup (NVLink-only
AgRs, node-sliced sizes); verify unchanged (global EP). Files:
`envs.py, forward_context.py, parallel_state.py, expert_map_manager.py,
moe_runner.py, all2all.py, llm_base_proposer.py`.

## 1. The test ladder: acceptance side FULLY VALIDATED

| step | config | routed coverage | accept @K=1 | prediction |
|---|---|---:|---:|---:|
| 1 | V2-Lite, 1 node DP8 | 100% (node=all) | **1.982** | ~1.9+ (invariant) |
| 2 | V2-Lite, 2 nodes DP16 | 50% | **1.878** | 1.88 (measured EP2 point) |
| 3 | DeepSeek-V2 236B, 2 nodes | 50% | **1.905 / 1.847** (2 runs, b8) | 1.85-1.9 |

The coverage->acceptance curve is now measured in-engine end to end:
6.25% -> 1.36 (Phase 53 preview), 12.5% -> 1.36, 50% -> ~1.88, 100% -> 1.98,
and it TRANSFERS to 236B unchanged. The node-local draft breaks the
acceptance-vs-EP anti-correlation exactly as the Phase 11/19 design intended:
at N nodes the draft sees 1/N of experts by NODE, not 1/EP by device.

## 2. Open issue 1: draft cycle inflation -- DIAGNOSED (not eager; rendezvous drain)

Node-local wall-clock is far below the cycle model: 236B b8 = 68 tok/s vs
264 baseline (cycle ~217 ms vs ~55-60 predicted); V2-Lite cycles 32 (1-node)
/ 56 ms (2-node) vs device-local 23 ms.

**Torch-trace A/B (V2-Lite DP8 1-node, node-local vs device-local):**
launch profiles are comparable (332 vs 410 graph launches, 1.5k vs 1.9k
kernel launches -- NOT an eager storm) and the blocking-sync COUNT is
identical (460 vs 466 aten::item), but the sync WAITS inflate 786 vs 535 ms
(max 19 vs 12 ms). The draft's real intra-node collectives lengthen the
drain at every existing DP-coordination rendezvous; the per-chain-step
coordination (dp_utils min().item(), Phase 52's finding) is the multiplier.

**DRAFT_SKIP_DP_COORD x node-local: UNSTABLE** -- CUDA illegal instruction
during graph capture (subgroup PG): with coordination skipped, ranks can
capture divergent shape sequences around the node collectives (the exact
deadlock class the OV1 consume-mode comment warns about). Flag combination
unsupported until fixed.

**Amortization TESTED (Phase 55, `VLLM_SELF_SPEC_DRAFT_AMORTIZE_DP_COORD`):**
memoize coordinate_batch_across_dp per distinct shape per propose() (dummy
runs/capture always coordinate -- stable, unlike SKIP_DP_COORD). Verdict:
accept parity everywhere (1.973/1.885/1.916), **+10.7% single-node
(490->543 tok/s), ~0 cross-node** (270 vs 268 V2-Lite; 65 vs 68 @ 236B).
Together with the SKIP result this FALSIFIES the rendezvous-count hypothesis:
the wait relocates to the remaining sync points. **ROOT CAUSE CONFIRMED (profiled 236B b8 node-local):**
draft_forward_first (step0, the only draft forward at K=1) = 72.3 ms
(~all of draft_chain 75.9) vs ~10-15 ms compute; verify = 73.3 ms vs ~30 ms
no-spec step. Both are the SAME defect: q>1 / non-plain-decode shapes
dispatch PIECEWISE, so per-layer collectives + prepare/finalize glue run in
PYTHON between captured body pieces (~1 ms/MoE-layer x 59). The comm-free
device-local draft dodges it (its collectives are no-ops); any dispatching
draft AND the spec verify pay it. This unifies with the Phase 52
fabric-independent verify overhead -- one fix serves both:
**make spec forwards eligible for FULL cudagraph capture** so collectives
replay inside the graph like the plain decode step.

**Phase 55 attempt (STEP0_FULL_CG, landed but NOT ENGAGING):** the full
machinery is in (draft dispatcher keyed at q=K+1, q-aware capture-metadata
builder, step0 uniform dispatch + descriptor forwarding, runner capture-size
bypass; env-gated, accept parity everywhere, zero regressions). Per-shape
dispatch logging then showed WHY it cannot engage yet: steady-state step0 is
NOT per-request uniform -- shapes are (24 tok, 8 req)=3/req, (21,7), (15,5),
... The drafter pads the TOTAL to a bucket, not each request to a fixed q,
so no uniform-q FULL key ever matches. The real fix is one of:
(a) pad step0 per-request to a fixed q (EAGLE padded-drafter-batch style;
the landed capture machinery then engages as-is), or
(b) move the MoE dispatch/combine INSIDE the compiled piecewise body so
PIECEWISE replays the collectives on-graph (helps step0 AND all mixed
shapes, model-agnostic). (b) likely also fixes the Phase 52 verify anomaly
for chunked-prefill shapes. Both are drafter/compile surgery -- the next
session's task, now with exact shape evidence.

**Phase 55 v2 (STEP0_FULL_CG ENGAGED -- keyed at q=K+2, LOSSLESS):** the
shape evidence above was misread: (24,8)/(21,7)/(15,5) are exactly 3
tokens/request -- steady-state step0 IS per-request uniform, just at
q = K+2 = 3, not K+1. With the padded drafter batch, set_inputs_first_pass
already pads every decode request to exactly K+1 verify slots + 1 appended
sampled-token slot (rejected slots PAD-masked by compute_new_slot_mapping)
-- i.e. the "pad per-request to fixed q" work of option (a) was ALREADY
DONE by the existing kernel; only the FULL-CG keying was wrong (q=K+1=2).
Fix (all under VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG):
- key/capture/dispatch the draft's uniform FULL graphs at q=_step0_q=K+2
  (llm_base_proposer: dispatcher qlen, capture qlen, uniform condition
  `num_tokens == batch*q AND max_query_len == q`);
- route the captured graph's seq_lens through a proposer-owned buffer
  refreshed at each step-0 replay (extend_all_queries_by_N returns a FRESH
  +1 seq_lens tensor each cycle, so the runner's buffer cannot be the
  captured pointer; pad-request rows zeroed);
- inject q-divisible capture sizes round_up(3*2^k, 6) (compilation.py) and
  pad uniform dispatches up to the next q-divisible capture size
  (cudagraph_dispatcher bump; no-op for the runner whose sizes are already
  multiples of its q);
- drive the draft capture at 3*num_reqs per runner uniform desc
  (gpu_model_runner) + a captured-shape set guard in propose so a
  keyed-but-never-captured shape can never dispatch FULL.
Two latent bugs found and fixed on the way:
1. DEVICE DEADLOCK at batch drain (100% GPU spin, all workers parked at the
   next coordinate .item(); py-spy + per-dispatch logs): runtime idle-rank
   draft dummies replayed TWO FULL graphs (model + forward-sample) while a
   busy rank's propose replayed ONE -- with the node-local draft's
   collectives captured in-graph, the second collective sequence has no
   partner. Flag-off never hit it (busy step0 was PIECEWISE -> mode-min
   downgraded idle dummies too). At K=1 the fws graph is never replayed at
   all, so it is now skipped entirely under the flag.
2. The post-DP-sync re-dispatch could bump the agreed padded size when the
   synced mode was not FULL (assert); the uniform claim is now dropped
   unless the synced mode is FULL.
Numbers (V2-Lite DP8 1-node, b8, K=1, same binary same day):
- STEP0CG=1: accept_len 1.975 (ref 1.977 -> LOSSLESS), tok/s 448.8.
  [step0-cg] logs confirm engagement: `24/8 uniform=True -> FULL padded=24`
  steady state; ALL clean decode batches engage down to (3,1)->FULL(6);
  mixed prefill shapes stay PIECEWISE as designed.
- STEP0CG=0 control: accept_len 1.974, tok/s 477.4 -- flag-off unchanged.
- Single-node V2-Lite is ~6% SLOWER flag-on: at this scale the piecewise
  glue the FULL graph removes is cheap, while ramp/drain batches pad up to
  the 24-token graph. The fix targets the 236B 2-node node-local draft
  where the glue costs ~60 ms/step0 (below).

**236B 2-node profile (b8, K=1) -- ENGAGED but the win DOES NOT MATERIALIZE
and accept DEGRADES (do not enable at 236B):**
- Startup: flag-on inflates the CUDA-graph memory ESTIMATE (0.93 -> 4.56
  GiB even with a single injected size; the estimator's per-graph
  extrapolation now includes the drafter's FULL captures) -> KV sizing
  fails at 0.90 -> the flag-on arm ran at util 0.95 with
  W7_STEP0_CG_EXTRA_SIZES=24 (both knobs added this session).
- Engagement confirmed: `[step0-cg] 24/8 uniform=True -> FULL padded=24`
  on all ranks; profiler regions over 225 steps x 16 workers.
- draft_forward_first: 71.9 ms (flag-off control, re-measured today;
  matches yesterday's 72.3/74.7 reference) -> 65.9 ms flag-on. Only -6 ms,
  NOT the hoped 15-25 ms. draft_chain 75.6 -> 69.3 ms; verify unchanged
  (74.5 vs 75.8). CONCLUSION REVISION for section-2's root cause: the
  step-0 cost is NOT primarily per-layer Python glue -- a single captured
  FULL graph (collectives in-graph) still takes ~66 ms. The dominant term
  is in-graph peer-wait (DP/EP rendezvous skew absorbed by the first
  collective) which no dispatch-mode change removes.
- LOSSLESSNESS FAILS at 236B (TP2, 2-node; V2-Lite TP1/1-node is clean).
  Full accept bisect (same day, same binary):
    flag-off @0.90:            1.905   (control)
    flag-off @0.95:            1.924   (util is NOT the cause)
    flag-on, FORCE_PW @0.95:   1.824   (captures alone, FULL never
                                        dispatched -> capture-side effect)
    flag-on, FULL @0.95:       1.675   (FULL replay adds the rest)
  Suspect for the capture-side component: piecewise address-determinism
  disturbed by the extra FULL captures sharing the global pool (the same
  fragility class the OV0b shadow-pool comment documents). The FULL-replay
  component is unexplained (TP2-specific; V2-Lite TP1 replay is exactly
  lossless at 1.975 vs 1.974).
- tok/s at 236B is uninformative single-iter fabric noise: flag-off gave
  103.9 @0.90 and 66.0 @0.95 (identical config otherwise; reference 64.7).
  At MATCHED util 0.95, flag-on 72.1 vs flag-off 66.0 -- consistent with
  the -6 ms/cycle, but within noise.
- VERDICT: keep VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG OFF for 236B/TP2. The
  mechanism is validated lossless at V2-Lite (TP1); the 236B accept gap
  (both components) and the ~66 ms in-graph rendezvous floor are the open
  items -- the latter caps the best case at ~-6 ms/cycle even if accept is
  fixed, so this direction alone cannot recover the 236B draft cost.

## 3. Open issue 2 RESOLVED (Phase 55): b>=32 unlocked -- the "1" was the FP8 per-tensor scale, not stale metadata

**Root cause (repro'd at V2-Lite 1-node, `scripts/repro_mixed_step.py`,
per-dispatch `W7_A2A_DEBUG=1` logging).** The Phase 53/54 diagnosis was
wrong on both counts: the drafter's dp_metadata is CORRECT (the sizes
vector [90, 64, ...] matches its own coordinated step-0 inputs exactly)
and the draft's forward context DOES carry the node-local key in the
crashing forward. The "1" in `1 != 203/90/64` is the on-the-fly FP8
draft's per-TENSOR activation scale -- `dynamic_scaled_fp8_quant` returns
a `(1,)` scale, and `naive_dp_ep._quantize_and_setup_dispatch` appended it
to the token-sized all-gatherv (`skip_gather_scales` only exempted
ndim==0 scalars). Uniform per-rank sizes MASK the bug: equal sizes
collapse to a plain all_gather (no size assert) -- that is why b8 was
clean and why the device-local draft is immune (it skips the gather
entirely). The first step with NON-uniform per-rank token counts (mixed
prefill/decode from scheduler stagger at b>=32) keeps the sizes vector and
the communicator asserts `1 != num_tokens` on every rank. Debug capture of
the crashing dispatch (V2-Lite, node branch, key present):
`sizes=[147, 129, ...] shapes=[(129, 2048), (129, 8), (129, 8), (1,)]` ->
`AssertionError: 1 != 129`.

Side finding from the same per-dispatch logging (resolves the "why did the
236B traceback show the DEFAULT branch" puzzle): at 236B/TP2 EVERY naive
dispatch runs with `is_sequence_parallel=True` (SP-sliced tokens, EP-wide
sizes vector of len DP*TP), so the Phase 54 node-local gather branch --
gated on `... and not is_sequence_parallel` -- NEVER engages at TP2. The
draft's node-local key only masks the ROUTER; its dispatch/combine are
EP-wide (cross-node). The measured 236B "node-local" numbers are therefore
node-masked ROUTING + full-EP comm; making the intra-node gather work
under SP is an open Phase 54 item, not part of this fix.

**Fix (not env-gated -- a genuine bug on any per-tensor-quant gathering
path):** per-tensor scales are gathered ONE-PER-RANK
(`extra_tensors_per_rank` marker: naive_dp_ep -> GroupCoordinator ->
CudaCommunicator -> AgRs `_dispatch_gather`, which gathers marked extras
with `sizes=[1] * group_world` in whichever branch is active -- node-local
subgroup, full-DP default, comm-free passthrough, SKIP_A2A tile). On
uniform steps this reproduces the previous gathered `(group_world,)` scale
tensor exactly; on non-uniform steps it no longer trips the size assert.
The marker is quant-CONFIG-driven, never shape-driven, so every rank makes
the same decision every step: collective count and order stay
rank-symmetric (the graph-replay deadlock class stays impossible), and a
rank with 1 actual token under per-act-token quant still joins the
token-sized gather.

**Failed first attempt (recorded because the numerics are load-bearing):**
keeping the per-tensor scale rank-LOCAL (skip the gather) is crash-free
and lossless at V2-Lite TP1 (1.978), but collapses 236B/TP2 accept
1.92 -> 1.22: dequanting the gathered fp8 tokens with rank-inconsistent
scales distorts each token's cross-rank expert sum, while the old gathered
vector (Triton per-tensor path reads element [0] = DP-rank0's scale on
every rank) is a rank-CONSISTENT per-token scaling error that downstream
norms largely absorb. The per-rank gather restores the old bytes exactly.

**A/B on the V2-Lite mixed-step repro** (DP8, fp8 draft, node-local,
rank-0 batch skew + 256-token prefill budget -> non-uniform mixed steps
every cycle): unfixed = `1 != 129` on all ranks at the first mixed step-0;
fixed = clean completion, same mixed step logs the scale in the per-rank
gather (`per_rank=[(1,)]`). Step-1 ladder regression: accept 1.973
(ref 1.974-1.982), 513 tok/s.

**Second blocker uncovered once the crash was gone (SEPARATE, pre-existing,
memory-regime): timing-dependent engine hang at b>=32 under KV thrash at
gpu_mem 0.90.** On current HEAD the memory estimator leaves only 2.2-3.5k
KV tokens/rank at 0.90 (rank-NONuniform; vs 34.4k at 0.95 and the Phase 53
recipe's expectation) -- b32 needs ~6k, so the engine runs perpetual
preemption/re-prefill. In that regime the run hangs (all 16 GPUs spinning
in NCCL, CPUs in enqueue backpressure inside the TARGET forward --
`logs/step3_hang_dump.txt`): reproduced twice at 0.90 without debug
logging, did NOT reproduce under per-dispatch logging (b32 completed,
accept 1.922, `dbg32` logs: every mixed step's 59x16 dispatch lines
rank-CONSISTENT sizes) -- i.e. a race, not a deterministic collective
mismatch, and the same class Phase 53 recorded at b64 with the COMM-FREE
device-local draft ("engine hang, dequeue timeout" -- no draft collectives
involved). Benchmarked below at gpu_mem 0.95 (the prof236b-validated
recipe, KV 34.4k: no thrash); the 0.90-thrash hang is tracked as its own
open issue.

### THE benchmark unlocked: 2-node 236B b8/32/64 (K=1, node-local draft, gpu_mem 0.95)

| b/rank | no-spec tok/s (P53, 0.90) | spec tok/s | accept @K=1 |
|---:|---:|---:|---:|
| 8  | 264  | 65.9 +- 1.0  | **1.904** |
| 32 | 756  | 189.1 +- 0.4 | **1.931** |
| 64 | 1130 | 264.3 +- 13.3 | **1.868** |

b32 and b64 COMPLETE for the first time (previously: crash at b32 warmup),
and accept is FLAT across batch at 1.87-1.93 -- the Phase 54 coverage
curve holds at scale and at batch. Wall-clock stays ~0.23-0.25x of
no-spec: that is the Phase 54 section-2 draft-cost problem (PIECEWISE
per-layer collectives + the ~66 ms in-graph rendezvous floor), unchanged
by this fix and still the binding constraint. At b8 the fix run matches
the pre-fix reference exactly (65.9 vs 64.7-66.0 tok/s, 1.904 vs
1.905/1.924 accept -- same binary semantics on uniform steps, as
designed). The b32 debug run (per-dispatch logging, gpu_mem 0.90 thrash
regime) independently measured accept 1.922 at b32. Reminder per the side
finding above: at TP2 these numbers are node-masked ROUTING + EP-wide
dispatch.

**K=2 at 236B: BLOCKED by a separate pre-existing wedge (first-ever K=2
attempt at this scale).** Acceptance itself confirms the coverage curve:
live windows during b8 warmup read 2.41-2.48 (predicted 2.3-2.6). But the
run wedges in the b8 warmup drain: all 16 workers park in
`_dummy_run -> coordinate_batch_across_dp ->
_post_process_cudagraph_mode` blocking on `.item()` of the device-side
sync tensor -- the GPU streams are stuck on an earlier unmatched
collective (`logs/step3_k2_hang_dump.txt`). At K=2 the draft CHAIN
replays its captured FULL graph (collectives in-graph) on busy ranks
while drained ranks run idle dummies -- the exact graph-replay-count
asymmetry class this phase already documented (section 2, latent bug 1;
only the K=1/fws case was fixed). Chain-vs-dummy replay symmetry at K>=2
is follow-up drafter work, independent of the scale-gather fix.

## 4. Where this leaves the thesis

With f ~0.5 measured at 236B (Phase 53) and accept 1.9 measured for the
node-local draft at the same scale, the cycle model gives
`accept / ((1-f)*q_draft + verify_ratio)` ~= 1.9 / (0.5*0.8 + 1.5) ~= 1.0-1.15
at K=1 -- break-even to modest win IF the draft executes at its modeled cost.
Both open issues are execution-path engineering, not physics or acceptance.
The measured pieces (f at scale, coverage curve, node-local mechanism) are
the paper's load-bearing numbers either way.
