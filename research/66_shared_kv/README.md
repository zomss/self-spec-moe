# Phase 66 — shared-KV self-draft: eliminate the draft KV duplication

Mission: the self-spec drafter registered its own 48 `FullAttentionSpec`
layers in the target's KV-cache group => 192 KiB/token vs 96 target-only,
halving the pool (Phase 64: 228k vs 567k tokens/rank) and pinning every
measurement at b6-b12 while the KV-bound-regime arithmetic says the E2E
win lives at b32. Fix: the draft IS the target model — bind its attention
layers to the TARGET layers' KV tensors (`VLLM_SELF_SPEC_SHARED_KV=1`,
default off), restore serving-batch residency, and measure the window-KV
self-draft at b32 with the Phase-65 flag stack.

Correctness discipline (see the commit message of the code change):
draft-written KV is provisional — the verify pass overwrites drafted
slots with target-exact KV before any non-chain read; the draft's step-0
rewrites of already-verified slots are PAD-masked (they would clobber
target KV with draft-computed values — fatal for the fp8-replica draft).

## Validation ladder

1. `scripts/run_canary.sh` — single-node 2k canary: shared ON, window OFF,
   K=2 => accept EXACTLY 3.000; W=64 => 2.540 (Phase 62 refs).
2. Pool check (from the same logs): tokens/rank must be ~target-only.
3. `scripts/run_accept16k.sh` — single-node 16k W512 K4: accept ~4.58.
4. `scripts/run_arm.sh` — 2-node EP16 16k, batches 12,32 (Phase 64/65
   protocol): arms A (EP-routed bf16, K4+K2), B (fp8 comm-free replica,
   K4), C (node-local, stretch). DONE — b24 replaced b32 as the largest
   resident point (b32 = 106% of the shared pool, still livelocks);
   fresh same-day no-spec denominators (P64's b32 ref drifted +31%).
   Verdict in `results_shared_kv.md`: best 0.86x (B, b12); no arm
   crosses 1.0x.

Results: `results_shared_kv.md`. Data: `data/` (w7_2node.py JSONs),
`logs/` (engine logs, on disk only).
