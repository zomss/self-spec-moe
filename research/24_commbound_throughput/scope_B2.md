# Scope: Stage B2 -- Integrated Comm-Bound Throughput of the Lockstep Cycle

Status: **scoped / not yet built**. Stage A measured the comm-bound step times
(real transports), Stage B1 validated acceptance + losslessness (the algorithm). B2
closes the loop: run the **actual lockstep draft/verify cycle on the multi-GPU EP
engine over the comm-bound (socket) fabric** and measure real end-to-end tokens/s.

## What B2 uniquely adds (vs Stage A x B1)

Stage A composed speedup = `T(k,beta) * S_verify / (k*S_draft + S_verify)` using
`S_draft ~= S_nvlink` (a *proxy* for the comm-free draft step) and `beta` from B1.
B2 replaces the two idealizations with measured reality:

1. **Real draft step** on the comm-bound engine (the actual comm-free MoE step, not
   the NVLink proxy) -- confirms `S_draft_skip ~= compute-only` on the socket fabric.
2. **Integration overheads** the composition ignores: per-phase config switch,
   distributed rejection sampling, verify-warmed cache update, KV management across
   draft/verify. The real tokens/s captures all of it.
3. **System-level losslessness**: the running system's output matches full-EP bf16
   (modulo the B1 GPU-numerics caveat under greedy; exact under sampling by theorem).

## Architecture

```
per DP rank (attention-DP: each rank owns its sequences), over the socket fabric:
  loop:
    draft phase  (k steps): MoE in SKIP-A2A mode (router masked to local shard;
                            dispatch/combine are local, no cross-rank collective)
    verify phase (1 step) : MoE in full-EP mode (normal all_gatherv/reduce_scatterv)
    rejection-sample (rank-local) the draft prefix vs verify; commit + bonus token
    update the per-rank verify-warmed expert cache from the verify routing
```

## Two sub-stages (B2a de-risks B2b)

### B2a -- real per-phase step times + speedup (achievable, low risk)

The one real code change: a **`VLLM_SELF_SPEC_SKIP_A2A`** flag in
`AgRsAll2AllManager` (`dispatch`, `dispatch_router_logits`, `combine`). When set, the
collective is replaced by a **shape-preserving local op** (allocate the gathered-shape
tensor locally / slice on combine) so the step runs compute-without-comm. Dummy
weights -> this is a *timing* measurement (garbage values are fine), exactly as
Stage A/Phase 20.

- Reuse `bench_commbound.py` with a `--config {verify, skip}` switch over the socket
  fabric (and NVLink as a sanity check). Measure real `S_verify` (full EP) and
  `S_draft_skip` (comm-free) at matched batch.
- Recompute the Stage A speedup with the **measured** `S_draft_skip` (not the NVLink
  proxy) and B1's `beta`. Expectation: `S_draft_skip ~= S_nvlink` (confirming comm is
  what sockets slowed), so the Stage A number is validated on the real engine.

Deliverable: measured-draft-step speedup table; confirms the comm-free draft really
is ~compute-only on the comm-bound engine.

### B2b -- full integrated lockstep run (the systems artifact, larger build)

Drive a real cycle and count real tokens. Path of least vLLM surgery: a custom
multi-process harness that drives `GPUModelRunner.execute_model` directly (bypassing
the continuous-batching scheduler), toggling SKIP-A2A per phase, with rank-local
rejection sampling and the verify-warmed cache. Real weights for losslessness.

- For correctness (not just timing) the SKIP-A2A draft needs a true **local-mode MoE**
  (process the rank's own tokens through its local expert shard), not the
  shape-preserving timing cheat -- the main correctness engineering.
- Metrics: real tokens/s (spec vs full-EP baseline, both over sockets); accepted
  length; end-to-end losslessness (greedy match to full-EP bf16 modulo B1 caveat;
  distributional under sampling).

Deliverable: integrated tokens/s + system losslessness; the headline systems number.

## Metrics and baselines

- tokens/s: spec cycle vs plain full-EP decode, both on the socket (comm-bound) fabric.
- accepted length / realized beta in the running system.
- losslessness: output equivalence to full-EP bf16.
- (desired, gap) DBO-overlapped EP baseline -- needs DeepEP, still out of scope here.

## Decision criteria

- **B2a GO->B2b:** measured `S_draft_skip <= ~1.3x S_nvlink` (draft really is
  comm-free on the engine) -> the Stage A composition stands on real measurements.
- **B2b success:** integrated lossless tokens/s `>= ~1.3x` the socket baseline at the
  target batch (net of integration overhead). Below that -> profile and attribute the
  overhead (sampling, cache, config switch) before claiming the systems result.

## Risks and honest constraints

- **Socket is a pessimistic proxy** (Stage A caveat): B2's "comm-bound tokens/s" is
  over sockets, not real PCIe-P2P. The integration-overhead and losslessness findings
  are fabric-independent; the absolute speedup is pessimistic-fabric-specific. The
  production PCIe/multi-node number still needs a real box.
- **SKIP-A2A correctness** (B2b): the shape-preserving timing cheat (B2a) does not
  give correct outputs; a true local-mode MoE is required for losslessness -- the main
  engineering risk.
- **Lockstep driver bypasses continuous batching:** measures cycle throughput, not
  production serving throughput; note this. Full vLLM spec-framework integration is a
  further (B2c) step, not required for the core result.
- Largest build in the project so far (vLLM MoE path + distributed driver +
  rejection sampling). B2a is hours; B2b is days.

## Expected artifacts

- B2a: `VLLM_SELF_SPEC_SKIP_A2A` hook (envs.py + all2all.py, default-off,
  behavior-preserving), `bench_commbound.py --config skip`, results table.
- B2b: the lockstep driver, integrated tokens/s + losslessness results.
