# Phase 49 — draft-behind-verify-comm overlap (implementation)

**Source phases** 32 (analytic overlap model: the draft cost-vs-quality tradeoff flips
at R=comm/compute≈2 — at high f the verify's A2A hides the comm-free draft for free
and only beta matters), 47/48 (measured win curve; K=2 chain ≈33 ms, verify comm
window M_v ≈ f·T_v), 45 (self-spec CPU ~1.7 ms; DP-sync barrier identified).

**Objective.** Turn the lockstep cycle `K·T_d + T_v` into the overlapped schedule
`max(K·T_d, M_v) + C_v` by running the comm-free draft chain on a second CUDA stream
while the verify's all-to-all is in flight — the free-running-draft design (see
`OV1_design.md`). Expected: native single-node PCIe mid-batch ~1.0-1.03x → ~1.4x;
emulated/multi-node latency-bound regime → toward accept_len (2.89 at K=2, more with
K retuned upward).

## Rungs (each is a go/no-go gate for the next)

- **OV0a — physics probe (standalone, no vLLM).** `ov0a_overlap_bench.py` via
  torchrun x8 on forced-PCIe NCCL: AgRs-style collectives (all_gather_into_tensor +
  reduce_scatter_tensor, 96/step = 48 layers x 2) on the main stream vs a calibrated
  GEMM load (~1 draft chain, ~32 ms) on a side stream; also the emulated-latency
  variant (`torch.cuda._sleep` bursts). Metric: hidden fraction
  `(T_comm + T_comp - T_both)/min(T_comm, T_comp)`. GO if ≥~0.7 on the sleep variant
  and materially >0 on real PCIe-SHM comm.
- **OV0b — in-engine shadow probe (env-gated vLLM hook).**
  `VLLM_SELF_SPEC_SHADOW_CHAIN=N`: right after the verify forward is *enqueued*,
  replay N draft-chain decode-step forwards on a side stream using the previous
  cycle's cached dispatch state, with the slot-mapping buffer filled with
  PADDING_SLOT_ID so all shadow KV writes are discarded; the main stream waits on a
  shadow-done event before the real propose (correct max() semantics, no buffer
  races). Timing-only, output-neutral by construction. A/B with `w7_fp8_timing`
  (spec, b64, a2a ∈ {0, 500}): GO if (i) accept_len unchanged (no cudagraph
  memory-pool aliasing between the concurrently replaying draft/verify graphs) and
  (ii) tok/s(shadow-on) ≈ tok/s(off) at a2a=500 (the chain hides in the comm
  window). At a2a=0 some slowdown is expected (little idle to hide in).
- **OV1 — free-running draft (the real thing).** Only after OV0a+OV0b pass. Design
  in `OV1_design.md`: the draft chain runs K+1 steps ahead during the verify
  (bonus-guess + next-cycle tokens), per-request hit (~beta^(K+1)≈0.86 greedy) uses
  the precomputed tokens, misses rewind via the existing rejected-token machinery
  (sub-batch re-propose). Greedy-only v0.

## Decision criteria

OV0a hidden fraction and OV0b tok/s-delta + accept-delta decide whether OV1's
engineering (cohort sub-batching, host-loop restructure) is justified on this box or
should be deferred to the multi-node rental where the window is larger.

## Commands

```
bash research/49_overlap_impl/scripts/run_ov0a.sh          # physics probe
bash research/49_overlap_impl/scripts/run_ov0b.sh          # shadow A/B (4 engine runs)
```

## Artifacts

`data/ov0a_*.json`, `data/w7fp8_ov0b_*.json`, `results_OV0.md`, `OV1_design.md`.
