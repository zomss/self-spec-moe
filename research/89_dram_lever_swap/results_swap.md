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
