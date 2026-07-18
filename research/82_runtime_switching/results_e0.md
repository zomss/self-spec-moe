# E0 — toggle-cost anatomy (MEASURED, 2026-07-18; DRAFT)

Engine: Qwen3-8B + W4 draft, win512 K6, b8 steady decode, piecewise-CG
chain (per-step full CG; whole-chain FULLCG caveat below). Driver:
scripts/e0_toggle.py (worker-side attribute mutation via collective_rpc,
probes confirm the mutations land on the live DraftModelProposer).
Microbenches: scripts/e0_swap_requant.py.

## The toggle-cost table

| toggle | mechanism | switch latency | throughput blip | mem delta | accept perturbation | class |
|---|---|---|---|---|---|---|
| OFF <-> on | scheduler per-step primitive (stock `disable_by_batch_size`) | 0 (per-step decision) | none | 0 | none | **FREE** |
| K (gamma) | scheduler per-step `dynamic_sd_lookup[batch]` (fork's `num_speculative_tokens_per_batch_size`, scheduler.py:1063) | 0 (per-step) | none | 0 | tau change only | **FREE** (CG widths for the K set captured at init) |
| window size (<= init cap) | `drafter._kv_window` mutation | ~2 ms RPC | <1% first round | 0 | beta only: win64 control accept 5.564->5.479, restore EXACT | **HOT-SWITCHABLE** |
| draft ckpt swap | weight upload (5.65 GiB W4 8B) | **0.11 s** pinned-staged (50.4 GiB/s roofline); 1.03 s pageable | n/a | both-resident rent 5.65 GiB | new draft beta | **CHEAP if staged** or pre-resident |
| kvq pool entry (C9) | CPU requant bf16->e4m3 + upload of resident KV | **~8.0 s** at b8/16k (18 GiB; CPU-quant-bound 0.19 s/layer, upload 0.04 s/layer) | stalls serving | +9 GiB pool | kvq beta | **EXPENSIVE** -> deploy-time choice, not a runtime toggle |

## Findings

1. Three of four switching axes (OFF, K, window) are FREE or ~2 ms:
   the strategy map can be consulted per regime shift at no cost.
   Regime-driven switching (E1/E2) is latency-bound by DETECTION, not
   by the switch itself.
2. Draft swap is bandwidth-cheap (0.11 s staged) -- the real cost is the
   HBM rent question (5.65 GiB both-resident vs swap latency), which is
   a residency-feasibility input, not a latency one.
3. C9 answers DIRECTION 3.2: KV requant does NOT belong in the runtime
   switch set. ~8 s for one b8/16k cell, linear in resident KV, CPU-
   bound. kvq is a deploy-time (or session-boundary) choice.
4. NEGATIVE (measured): mutating `drafter.num_speculative_tokens` is a
   NO-OP for chain length (accept pinned at 5.564 across "K2/K4" --
   chain length is scheduler-driven). The per-batch-size K schedule is
   the correct primitive; worker-side K mutation must not be used.

## Caveats

- Whole-chain FULLCG (VLLM_SELF_SPEC_DRAFT_FULLCG) bakes the chain:
  window/K toggles under it need the capture-set cost (unmeasured here;
  the piecewise chain used for this table is the fixed-stack default for
  toggling deployments).
- Window toggles are legal only <= the init cap (scratchpad sized at
  init); growing past cap = engine restart (deploy-time).
- Single model/hardware column (Qwen3-8B, 1xH100); b8 steady load.
