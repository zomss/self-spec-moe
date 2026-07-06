# Phase 66 results — shared-KV self-draft (eliminate draft KV duplication)

Code: `VLLM_SELF_SPEC_SHARED_KV=1` (default off) binds the drafter's 48
attention layers to the TARGET layers' KV tensors via the existing
cross-layer-sharing plumbing (runner `shared_kv_cache_layers`), with the
draft's KV writes kept ENABLED and PAD-masked down to not-yet-verified
slots (appended sampled-token slot + chain-drafted slots). See commit
"[W7][66] shared kv: draft binds to target KV tensors".

## Validation ladder

| rung | check | ref (duplicated KV) | shared-KV | verdict |
|---|---|---|---|---|
| 1a | canary 2k, W=0, K=2 accept | 3.000 (P62, bit-exact) | **3.000** (378.4 tok/s) | PASS |
| 1b | canary 2k, W=64, K=2 accept | 2.540 (P62) | **2.689** (200.1 tok/s) | PASS (better) |
| 2 | pool tokens/rank (canary cfg) | 217,696 | 435,408 (**2.00x**) | PASS |
| 2b | pool tokens/rank (16k 1-node) | 221,664 (P62 B2) | 443,344 (**2.00x**) | PASS |
| 3 | 16k W512 K=4 accept (1-node DP8) | 4.584 (P62 B2) | **4.676** (206.2 tok/s) | PASS |

## Pool restoration (16k protocol configs)

Measured pools, tokens/rank at gpu_mem 0.90 (refs from P64/P65):

| engine | dup (192 KiB/tok) | shared (96 KiB/tok) | b12 | b24 | b32 |
|---|---|---|---|---|---|
| no-spec (ref) | -- | 567,136 | 35% | 70% | 92% |
| EP-routed self-spec | 236,512 | TBD | TBD | TBD | TBD |
| fp8-replica self-spec | 115,968 | TBD | TBD | -- | -- |

The duplication removal doubles the pool exactly (canary: 2.000x), but the
self-spec engine's ~8.6 GiB profiling/persistent overhead vs no-spec
remains, so the shared pool is expected ~473k (EP-routed), not the full
567k: b32 (530k demand) is projected ~112% = near-pool.

## 2-node 16k, batches 12,32 (Phase 64/65 protocol)

no-spec refs (P64): b12 395.7, b32 379.0 tok/s.

TBD

## Headline verdict

TBD
