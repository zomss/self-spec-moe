# Phase 68 — DP-rendezvous attribution + fix (the >1.0x E2E gate)

Source: research/67_propose_fixed_opt (propose-fixed is ~18 ms single-node but
~88 ms 2-node; the ~70 ms gap is cross-node and invisible single-node) and
research/66_shared_kv (best arm B = fp8 comm-free replica + shared-KV, 0.86x
at b12; cycle 159.7 = verify 29.4 + chain 4x10.6 + propose-fixed ~88).

Fabric: h107 (rank 0, writes trace/results) + **h108** (rank 1, peer). NOT
h106 (foreign user). Master IP 192.168.0.17. All runners parameterize the peer
via `${W7_PEER:-h108}`; shared-node guards (foreign-GPU wait, own-pattern
pkill) apply to both nodes.

## Objective

Confirm-or-refute the per-draft-step DP-rendezvous hypothesis via a 2-node
rank-0 torch trace, then either fix it or document the true attribution.

HYPOTHESIS (arithmetic-reconciled, unconfirmed): the ~56 ms unexplained
propose-fixed remainder = the K=4 comm-free draft-step forwards EACH firing a
cross-node DP rendezvous (~14 ms each). SKIP_DP_COORD helped +12.7% but did not
fully eliminate it.

Code note (pre-trace): with `VLLM_SELF_SPEC_DRAFT_SKIP_DP_COORD=1` +
`W7_DRAFT_LOCAL_ROUTE=1` (arm B), `_determine_batch_execution_and_padding`
(llm_base_proposer.py:3722) takes the SKIP branch for BOTH step-0 and the chain
(the chain coordinates once before the loop, line 1217), building a local
`num_tokens_across_dp` with NO `coordinate_batch_across_dp` collective. The
verify/target forward (gpu_model_runner.py:3914) always coordinates on the GPU
group. So the trace must show whether any draft-side cross-node collective
survives.

## Runners (scripts/)

- `env_selfspec_2node.sh` — base 2-node self-spec stack (FULL_CG,
  COMPILE_CONSISTENT, CHAIN_PIECEWISE, RoCEv2 rails). Master IP 192.168.0.17.
- `w7_trace16k.py` — rank-0 torch-profiler trace harness (from P65; honors
  W7_DRAFT_QUANT).
- `run_trace16k.sh [K b trace_len to tag]` — arm-B trace, peer `${W7_PEER}`.
- `run_arm.sh {nospec|b_rep} K [batches]` — fresh 2-node measurement.
- `analyze_collectives.py TRACE` — per-cycle cross-node collective inventory
  (GPU NCCL AllReduce/AllGather/A2A + CPU gloo coord), draft-path vs verify.
- `analyze_trace16k.py TRACE` — runner-span decomposition (from P64).

## Decision criteria

- Step 1 CONFIRM => draft-side collective ms/cycle reconciles the ~56 ms =>
  implement `VLLM_SELF_SPEC_DRAFT_NO_DP_RENDEZVOUS` (default off), canary
  (W0 K2 accept must stay 3.000; arm-B accept ~4.6 +/-0.05), re-measure.
- Step 1 REFUTE => the ~70 ms is verify-side coordination no-spec also pays =>
  STOP, document true attribution (blocked-inherent). Either way the per-cycle
  collective inventory is the deliverable.

## Deliverables

results_rendezvous.md (the inventory + verdict), data/ (trace + measurement
JSON), scripts/, code flag (default off) committed prefix "[W7][68] rendezvous:".
