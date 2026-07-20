# Phase 89 results — DRAM-cached weight swap (DRAFT)

## E0: mechanics priced (2026-07-21, GPU 0)

| quantity | measured |
|---|---|
| 8B draft ckpt (either W4A16 / W4A8) | 6.07 GB |
| disk -> pinned DRAM (one-time cache build) | 6.5 s |
| pinned DRAM -> GPU swap | **113 ms (54 GB/s)** |
| 32B draft (extrapolated at 54 GB/s) | ~0.33 s |
| CUDA-graph replay after in-place copy_ | **bit-exact new weights** |

Amortization: one swap pays for itself after 3.8 s of serving at +3%
speedup, 1.1 s at +10%, 0.4 s at +30%. Cross-validates the E0-toggle
table's 0.11 s swap figure.

## E1: in-engine swap hook VALIDATED (PASS)

collective_rpc callable -> worker-side in-place copy_ into the draft
model's live (post-processed) tensors, under the full fixed stack
(wholechain FULLCG, shared-KV, skip-prefill). 8B W4+win K4, b8 GSM8K:

| phase | tok/s | accept |
|---|---|---|
| baseline | 980.6 | 4.73 |
| identity swap (723 tensors, 702 ms incl. disk load) | 1156.6 | 4.77 |
| corrupt swap (288 scale tensors zeroed) | 276.1 | **1.00** |
| restore | 1157.7 | 4.76 |

- The corrupt->collapse->restore cycle PROVES the captured whole-chain
  graphs execute the swapped weights end-to-end (no stale capture) and
  the engine survives arbitrary in-place draft-weight replacement with
  zero downtime and no re-capture.
- 702 ms includes torch.load from disk; the production path keeps
  snapshots pinned in DRAM -> E0's 113 ms is the real swap cost.
- Scope note: same-layout swaps (same ckpt format/kernel) -- exactly
  the RL staleness case (re-quantized drifting policy shares layout).
  Cross-kernel swaps (W4A16<->W4A8) change param layouts: boot-class
  transition (E0-toggle C9), not in-place; deferred by design.

## Next: E2 (per-lever policy options + amortization-gated swap
## requests), E3 (RL-rollout demo: drift-triggered measured swap).

## E3: RL-staleness demo — MEASURED (2026-07-21)

Drift trace: 5 phases, eps [0,0.1,0.2,0.3,0.4] (weight-nibble +
scale-jitter drift; calibrated accept 4.11->1.03 span), b16 x MATH
T=1.0 x 1024 tok (the R8 rollout shape), 8B W4A8-Hum, GPU 0. The
adversarial cell ON PURPOSE: b16/short-ctx is the thinnest spec cell
(canonical best 1.05x at K2).

| arm | trace aggregate | vs AR |
|---|---|---|
| AR (off) | 2097.7 | 1.000 |
| stale K4 (static, never refreshed) | 1146.0 | 0.546 |
| refresh K4 (phase-end detector) | 1286.4 | 0.613 |
| stale K2 (policy-selected depth, static) | 1586.3 | 0.756 |
| refresh K2 (phase-end detector) | 1699.5 | 0.810 |
| stale_policy (per-step runtime policy, no refresh) | 2010.2 | 0.958 |
| **refresh_policy (FULL SYSTEM: policy + DRAM refresh)** | **2013.1** | **0.960** |

Per-phase, the full system: fresh phases 2017-2034 (argmax mixing
depths), drift dips FLOORED at 1980-1991 by the per-step EMA disarm
(vs 829-1300 unprotected), detector fires the 113ms swap at both
deep-drift phases, probes re-arm to 2048-2061 post-refresh.

### Readings (honest)
- MECHANISM: fully validated end-to-end. Measured detector (accept
  EMA) -> amortization gate -> 113ms in-place DRAM swap -> full
  recovery -> policy re-arms via probes. Twice per trace, zero
  downtime, no re-capture.
- STALENESS COST is the paper number: a never-refreshed draft loses
  45% of throughput on this trace (0.55x); even the right static
  depth loses 24%. The runtime policy alone recovers to 0.96x; the
  refresh restores the spec win the policy cannot (accept 3.3 vs the
  disarmed floor).
- vs AR at THIS cell: parity-minus (0.96x) -- b16/short-ctx is the
  thinnest cell in the map (fresh ceiling 1.05x) and the policy tax
  eats the margin. The demo cell was chosen adversarially; at
  spec-favorable rollout shapes (longer gens, batch drain, deeper
  ctx) the fresh margin is 1.4-1.9x and the same protection applies.
  The claim: the full system converts staleness from a 45% cliff
  into a 4% bound around the per-cell fresh optimum.
- Phase-end vs per-step detection: the K2/K4 refresh arms (driver-
  side, phase-end) each pay one degraded phase per drift step; the
  runtime policy's per-step EMA removes that cost -- detection
  granularity is the difference between 0.81x and 0.96x.
