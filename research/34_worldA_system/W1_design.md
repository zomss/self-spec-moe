# W1 design: correct comm-free local-routing MoE forward

Scoped the vLLM AgRs MoE path (the comm-bound one): `MoEPrepareAndFinalizeNaiveDPEPModular`
(`fused_moe/prepare_finalize/naive_dp_ep.py`) + `AgRsAll2AllManager`
(`distributed/device_communicators/all2all.py`).

## How the AgRs (comm-bound) path works today
- `prepare()` -> `get_ep_group().dispatch(a1q, topk_weights, topk_ids)` = **all_gatherv**:
  each rank gathers ALL ranks' tokens.
- the MoE kernel computes the rank's EP-shard experts for all gathered tokens (others via
  `expert_map`).
- `finalize()` -> `get_ep_group().combine(out)` = **reduce_scatterv**: sum expert
  contributions across ranks + scatter each rank's tokens back.
- So: gather-all -> compute-shard -> reduce-scatter. Two collectives per MoE layer.

The existing `VLLM_SELF_SPEC_SKIP_A2A` patches the all2all manager to skip the collectives,
but it **tiles the local chunk** to keep the gather shape -> values are garbage (timing
only), and the router is **not masked**, so topk_ids reference non-resident experts.

## Correct comm-free local routing (what W1 must do)
Per rank, process ONLY the local tokens, route them to ONLY the RESIDENT experts, compute
locally, no gather/scatter. Two pieces:

1. **Router masking (skip-cold) -- the correct-routing core.** Before top-k, mask the gate
   probabilities to the resident expert set and renormalize over the survivors. Tokens whose
   true top-k include non-resident experts fall back to their top RESIDENT experts (skip-cold).
   This is the harness `local_forward` logic; reference + tests in `local_routing_ref.py`.
   At resident = all experts this is identical to full routing (sanity).

2. **Skip dispatch/combine, local-only shapes.** A local-routing prepare/finalize: `prepare`
   does NOT gather -- it uses the local tokens as-is with the masked topk_ids; the kernel
   computes the resident experts on the local tokens; `finalize` does NOT combine -- the
   output is already local. (This is the real shape change vs SKIP_A2A's tile-to-total.)

## Resident set
- **W1 MVP:** resident = the rank's EP shard (`expert_map`), so no extra memory; coverage is
  low (E/num_devices) -> low beta, but it validates the correct comm-free path.
- **W2:** resident = a larger FP4 cache (globally-hot top-C / replica) for coverage.

## Signal
The local-routing flag rides on `ForwardContext.additional_kwargs` (set by the lockstep
driver's draft forward), read in the routing + prepare/finalize selection. Gates BOTH the
router masking AND the dispatch/combine skip.

## Intervention points
- routing: where topk_ids/weights are produced (the GPU select_experts / topk path) -> apply
  `local_route` masking when the flag is set.
- `naive_dp_ep.py`: a local-routing prepare/finalize (skip dispatch/combine, local shapes),
  OR a flag in the existing class.
- `all2all.py`: the SKIP_A2A tiling is replaced by the correct local path (no tiling).

## Testing ladder
1. `local_route` unit tests (this commit): resident=all -> full routing; subset -> masked +
   renormed; skip-cold drops non-resident. NO model needed.
2. Single-GPU: draft forward with local routing vs the harness local_forward -> match.
3. Distributed forced-PCIe: draft step is comm-free (no all-to-all) + correct (matches the
   single-GPU local-routing output).
4. End-to-end via the custom driver (W0): lossless + tokens/s (W7).

## Status
- **Increment 1 (done):** the correct-routing core (`local_route`) + unit tests.
- **Increment 2 (done, validated, merged):** the vLLM integration, built in worktree
  `ssm-w1` and merged to `research/self-spec-moe`. Additive + flag-gated; OFF by default.
  - Flag: `ForwardContext.additional_kwargs["self_spec_local_route"]` (driver sets it per
    forward) with env `VLLM_SELF_SPEC_LOCAL_ROUTE` as the default; reader
    `forward_context.self_spec_local_route_enabled()`.
  - Router masking: `fused_moe/local_route.py::mask_router_logits_to_resident` (non-resident
    logits -> dtype min; equivalent to the reference's prob-masking for both selection and
    renormed weights -- the normalizer cancels), applied in `runner/moe_runner.py`. No-op
    when `expert_map is None` -> resident=all is byte-identical to baseline.
  - Comm-skip: `AgRsAll2AllManager.{dispatch,dispatch_router_logits,combine}` return local
    tensors as-is / `pass` when the flag is set; `real=` collective counter proves 0.
  - **Validated:** resident=all -> byte-identical to baseline (sanity invariant);
    DP=2+EP flag-ON -> `real=0` (comm-free) + coherent non-garbage output; forced-PCIe ->
    `real=0`. (Losslessness is a driver-level property, validated later at W7.)
  - Resident set = the rank's EP shard (`expert_map`) -> low coverage/beta; W2 expands it
    with the FP4 cache.
- **Next:** W0 (driver toggles the flag per phase) + W2 (FP4 resident cache).
