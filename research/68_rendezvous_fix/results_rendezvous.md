# Phase 68 results — DP-rendezvous attribution (the decisive 2-node trace)

Fabric: h107 (rank 0, trace) + **h108** (rank 1). Master IP 192.168.0.17.
Arm B = fp8 full-replica comm-free draft + shared-KV + W512 sinks16 + Phase-65
flag stack (DP_COORD_CPU + CHAIN_LIGHT_MD + **SKIP_DP_COORD**). 16k, b12, K=4.
Rank-0 torch-profiler trace; collectives attributed to the draft-step path vs
the verify path by CORRELATION (each GPU NCCL kernel matched to its CPU-side
launch, whose ts nests in the runner's `draft` / `forward` / `preprocess`
record_function spans). Attributing by the GPU kernel's own timestamp is wrong
here — async CPU-ahead slides verify kernels into the neighbouring draft window.

## VERDICT: hypothesis REFUTED — the draft fires ZERO cross-node collectives

The per-draft-step DP-rendezvous hypothesis is **refuted**. In arm B the entire
K=4 draft chain (step-0 + 3 chain forwards) issues **0 cross-node NCCL kernels
and 0 gloo/nccl coordinate all-reduces per cycle**. There is no per-draft-step
rendezvous to remove. The ~56 ms "unexplained" propose-fixed remainder is NOT
cross-node coordination — it is the **local, comm-free draft-chain GPU compute**
(plus a small idle stall waiting on the verify forward's GPU drain). E2E parity
is blocked by (a) the inherent DP-EP verify-forward comm that no-spec pays too,
and (b) the added local draft chain — neither is a draft-side rendezvous.

Per the mission's Stop/honesty clause, Step 1 refutes => STOP, document, no fix
(a default-off `NO_DP_RENDEZVOUS` flag would be a no-op: SKIP_DP_COORD already
removes the only draft-side rendezvous). Deliverable = the inventory below.

## Per-cycle cross-node collective inventory (arm B, b12 K4)

Clean len=80 trace `data/trace_armb_w512k4_b12/` — 32 cycles (= 32 coordinate
AllReduce kernels), trace wall 4207 ms => **cycle 131.5 ms** (in-trace;
reconciles P66's measured 159.7 ms modulo profiler overhead + steady-state
window). **92% GPU utilisation** — the cycle is GPU-bound, not comm-stalled.
Full output: `data/inventory_armb_w512k4_b12.txt`.

| path | collective | n/cycle | ms/cycle |
|---|---|---|---|
| **draft (step-0 + 3 chain)** | **any NCCL kernel** | **0.00** | **0.00** |
| **draft** | **gloo/nccl coordinate all-reduce** | **0.00** | **0.00** |
| verify | MoE AllGather (dispatch, 1/layer x48) | 48 | 32.0 |
| verify | MoE ReduceScatter (combine, 1/layer x48) | 48 | 14.8 |
| verify | coordinate_batch_across_dp AllReduce (GPU grp) | 1 | 14.0 |
| **verify total** | | **97** | **60.9** |

The coordinate AllReduce is a cross-node DP barrier: per-call p50 14.1 ms,
max 45.0 ms — it absorbs cross-rank straggle (16 DP ranks, 2 nodes). It is
verify-side (1/cycle) and no-spec pays it too.

Draft `draft` span (comm-free): the K=4 local 48-layer forwards run back-to-back
on the GPU with ~22 ms/cyc idle inside the span — a stall waiting on the verify
forward's GPU drain to produce target hidden states, NOT a collective. **Zero
NCCL and zero gloo anywhere in the draft span.** Of the 131.5 ms cycle,
~61 ms is verify-side cross-node NCCL; the remaining ~70 ms is local GEMM /
attention (verify compute + the entire comm-free draft chain).

## Answers to the Step-1 questions

1. **How many cross-node collectives fire per cycle, draft vs verify?**
   Verify path: 48 MoE AllGather + 48 MoE ReduceScatter + 1 coordinate
   AllReduce = **97 GPU NCCL kernels/cycle, 60.9 ms/cycle**. Draft path:
   **0 kernels/cycle, 0 ms**. The MoE dispatch/combine is 1 AllGather + 1
   ReduceScatter per MoE layer (Qwen3-30B-A3B = 48 layers), fired once by the
   single verify forward. A **no-spec decode step fires the same 96 A2A
   collectives** (structural, 1/layer; only the payload — 12 vs 60 tokens at
   b12 — differs, and at decode batch these are latency-bound not
   bandwidth-bound). So the verify comm is NOT extra speculation cost.

2. **ms per draft-step rendezvous / total draft-side rendezvous ms/cycle;
   does it reconcile ~56 ms?** There are **no draft-step rendezvous** — the
   draft-side rendezvous total is **0 ms/cycle**. It does NOT reconcile the
   ~56 ms because the ~56 ms is not rendezvous: it is the draft chain's local
   GPU compute + a ~22 ms/cyc verify-drain idle stall inside the `draft` span
   (of the 131.5 ms cycle, only ~61 ms is verify-side NCCL; the other ~70 ms is
   local GEMM/attention, 92% GPU-busy overall). The comm-free replica draft
   routes to resident experts only
   (`all2all.py`: `self_spec_local_route_enabled()` returns the input tensors
   without a gather — verified in code and trace), so it has nothing to
   all-to-all.

3. **Is SKIP_DP_COORD engaged; what does it skip vs leave running?**
   **Engaged.** Proof by contrast with Phase-65 f123 (identical arm minus
   SKIP_DP_COORD): f123's draft fires **2 `gloo:all_reduce`/cycle (~6 ms)** —
   the DP_COORD_CPU coordinate for step-0 and for chain-setup
   (`_determine_batch_execution_and_padding`, one per group). Arm B shows
   **0 `gloo:all_reduce` anywhere** — SKIP_DP_COORD takes the branch at
   `llm_base_proposer.py:3722` (`SKIP_DP_COORD and local_route and dp>1`) and
   builds a local `num_tokens_across_dp` with no collective; the downstream
   `set_forward_context` then also fires none (it only coordinates when
   `num_tokens_across_dp is None`). SKIP **skips** the draft's
   coordinate_batch_across_dp entirely; it **leaves running** the verify
   forward's coordinate AllReduce (GPU group, `gpu_model_runner.py:3914`,
   1/cycle) and the verify MoE A2A — correctly, since those are target-side.

## Why the mission's arithmetic mis-pointed

The Phase-66/67 F/D linear solve labelled cycle 159.7 = verify 29.4 +
chain 4x10.6 + propose-fixed 87.7. The "verify 29.4" is the *no-spec* step; the
*spec* verify forward is bigger (60 vs 12 tokens through the same 96 A2A) and
its coordinate AllReduce absorbs cross-rank straggle (per-call p50 14.1 ms,
max 45.0 ms). So "propose-fixed" was a residual bucket that lumped
(a) the local draft-chain compute and (b) the spec-vs-nospec verify
inflation — NOT a per-draft-step rendezvous. SKIP_DP_COORD already removed the
only true draft-side rendezvous (~6 ms/cyc gloo), which is why P65 saw a partial
(+12.7%) gain that "did not fully eliminate" the residual: the residual was
never coordination.

## Honest verdict: blocked-inherent (write-up path)

E2E parity for the comm-free self-draft is **blocked by inherent vLLM DP-EP
coordination + the local draft cost**, not by a removable draft rendezvous:

- The no-spec decode step is already comm-bound: 96 cross-node A2A collectives
  (48 MoE layers x dispatch+combine), latency-bound at decode batch.
- The spec cycle keeps that verify comm (~61 ms/cyc NCCL) AND adds a full local
  48-layer x K draft chain (comm-free GEMM/attention). Speculation buys ~4.66
  accepted tokens/cycle but the cycle costs ~5.4x a no-spec step, so at b12 the
  best arm is 0.86x (P66) — consistent with this decomposition.
- The remaining lever is NOT DP coordination (already skipped for the draft).
  It is either (i) shrinking the local draft chain (chain-FULL-CG, ~0.95x
  ceiling per P66 — parity, not a win) or (ii) overlapping the draft with the
  verify tail (the Phase-32/49 overlap program) — out of scope here, and a
  general vLLM DP-EP A2A rewrite is explicitly out of scope.

The decisive deliverable is the inventory above: **draft-path cross-node
collectives = 0/cycle**. The hypothesis is refuted; no code change is warranted.

## Data / repro

- `data/trace_armb_w512k4_b12/` (clean len=80) + `data/trace_smoke_w512k4_b12/`
  (len=24) — rank-0 torch traces.
- `scripts/analyze_collectives.py TRACE` — the correlation-attributed inventory.
- Contrast: `research/65_draft_overhead_opt/data/trace_f123_w512k4_b6/`
  (arm minus SKIP_DP_COORD) — shows the 2 draft gloo:all_reduce/cyc that
  SKIP_DP_COORD removes.
