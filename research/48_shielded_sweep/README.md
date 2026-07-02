# Phase 48 — delay-shielding gate fix + shielded sweep2 rerun

**Source phase** 47 (sweep2). Phase 47's measured win curve was a *conservative
lower bound*: the benchmark's emulated A2A sleep in
`vllm/distributed/device_communicators/all2all.py::_emulate_exposed_a2a_delay`
fired unconditionally — including on the comm-free local-route branch — so the
self-spec DRAFT was charged emulated communication it provably never performs
(collective counter: draft `real=0`). Phase 47 / RESULTS.md future-work item 2.

## Change

`_emulate_exposed_a2a_delay()` now returns before sleeping when
`self_spec_local_route_enabled()` is true (the draft's forward-context flag),
counting the skip in a new `shielded` counter (logged at engine destroy next to
total/active/real). The branch resolves at CUDA-graph capture time per model:
draft and verify capture separate graphs, so the draft's graphs contain no
sleep while the verify's keep it. A/B revert knob `W7_CHARGE_DRAFT_A2A=1`
reproduces the pre-fix (Phase 42/47) charging on the same binary. Default
paths (no local-route flag) are unchanged.

## Objective

Rerun the Phase-47 sweep unchanged (Qwen3-30B-A3B, DP8/EP8, forced-PCIe, FP8
full-replica comm-free draft, fully-optimized stack, K=2, greedy;
`VLLM_SELF_SPEC_EMULATE_A2A_DELAY_US in {0,100,250,500,1000}` x batch
{32,64,128}) and confirm the MEASURED curve now matches the analyzer's
model-shielded curve (Phase 47 section 4: b32/b64/b128 ~2.18/1.95/1.46 at
1000us) instead of the charged lower bound (1.30/1.24/1.14).

## Commands

```
bash research/48_shielded_sweep/scripts/run_sweep_shielded.sh   # spec sweep
SPEC_DATA=research/48_shielded_sweep/data LOGD=research/48_shielded_sweep/logs \
  .venv/bin/python research/47_sweep2/scripts/analyze_sweep2.py # analysis
```

No-spec baselines REUSED from Phase 42 (`research/42_comm_sweep/data`): the
no-spec engine runs no draft and never sets the local-route flag, so the gate
cannot affect it. Pre-fix spec curve for the delta comes from Phase 47 data.

## Decision criteria

1. a2a=0 points reproduce Phase 47 within noise (no delay -> gate is a no-op).
2. Delay-slope of the spec engine drops to ~ s_ns/accept_len (verify-only).
3. Measured curve ~= Phase-47 model-shielded curve; crossover moves to
   "everywhere >=1.0" for b32/b64 and well below ~600us for b128.

## Artifacts

`data/w7fp8_sweep2_a2a*_fp8_spec_cg*_K2.json` (same tag as Phase 47 so the
analyzer globs work via SPEC_DATA), `logs/*.log`, `results_W7_shielded.md`.
