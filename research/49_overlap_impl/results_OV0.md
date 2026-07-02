# OV0 results — overlap feasibility probes

**TL;DR.** OV0a (physics): 72–81% of a chain-sized load hides inside forced-PCIe
comm windows — GO. OV0b (engine): concurrent side-stream compute is *stable and
correct* per se (GEMM shadow: accept 2.91), and a chain-sized side load at the
a2a=500 point costs only 17.5% of throughput; but concurrently replaying the DRAFT
MODEL corrupts the **verify** itself (served tokens diverge from position ~0) —
a shared attention-kernel workspace is the prime suspect and is the one open
correctness bug gating OV1. Three reusable isolation requirements were found and
implemented en route (event ordering, dedicated draft cudagraph pool, private
MemPool).

## OV0a — physics probe: PASS

Standalone torchrun x8, forced-PCIe NCCL (`ov0a_overlap_bench.py`). Side-stream GEMMs
overlap real PCIe-SHM collectives and the emulated `torch.cuda._sleep` delay:

| shape | T_comm | T_comp | T_both | hidden fraction |
|---|---:|---:|---:|---:|
| b64 (24 tok/rank, 96 coll) | 12.8 ms | 35.4 ms | 38.7 ms | **0.74** |
| b512 (192 tok/rank) | 43.2 ms | 60.0 ms | 68.1 ms | **0.81** |
| b64, sleep 500 µs x96 | 48.5 ms | 35.4 ms | 58.5 ms | **0.72** |

SMs are genuinely available during the comm window (also during `_sleep`, though the
spin kernel contends ~28% — note for benchmark honesty). **GO.**

## OV0b — in-engine shadow probe: the landmine ledger

`VLLM_SELF_SPEC_SHADOW_CHAIN=N` replays N draft-chain decode-step forwards on a side
stream concurrently with the enqueued verify; KV writes discarded via a
PADDING_SLOT_ID fill (alias to the metadata slot_mapping verified True at runtime);
the real propose waits on a shadow-done event. GPUs 0-3, DP4/EP4, b64 global, K=2,
Qwen3-30B FP8 full-replica stack. Baselines (shadow off): a2a=0 **2478 tok/s**,
a2a=500 **1710 tok/s**, accept 2.90-2.91.

| # | condition | result | conclusion |
|---|---|---|---|
| 1 | naive side-stream replay (shared graph pool) | **CRASH** sticky illegal-instruction | two races (below) |
| 2 | + propose-done event (shadow waits for the previous chain's device work) | still crashes | ordering necessary, not sufficient |
| 3 | + `VLLM_SELF_SPEC_DRAFT_GRAPH_POOL=1` (dedicated pool for draft-tagged graphs; upstream TODO in `cuda_graph.py` warns exactly this) | **no crash**, but accept 2.91 → **1.02** (silent) | concurrent replay stable; something still corrupts values |
| 4 | serialized shadow (`W7_SHADOW_MAIN_STREAM=1`, same replay/fill/metadata) | accept **2.91** clean, tok/s 2478→1754 (the serialized shadow cost) | corruption REQUIRES concurrency; the fill/-1/metadata/replay machinery itself is correct |
| 5 | + private `torch.cuda.MemPool` for shadow transients (cross-stream allocator-reuse hypothesis) | accept still **1.01** | not the allocator |
| 6 | bisect A: fill+context only, NO model call (`W7_SHADOW_NO_FORWARD=1`) | accept **2.91**, tok/s 2538 ≈ baseline | machinery innocent |
| 7 | bisect B: private GEMMs only, no vLLM state (`W7_SHADOW_GEMM_ONLY=1`) | accept **2.91**, tok/s 1821 (partial hide of the ~30 ms load at a2a=0, as expected) | ANY concurrent compute is fine |
| 8 | token-dump pair (`W7_DUMP_TOKENS`): diff generated ids, shadow off vs on | **DIVERGES at position 0–1** on all 4 sequences (off accept 2.92, on 1.02) | the **VERIFY is corrupted** — served output garbage, not just draft quality |

A + B clean pins the corruption to **concurrently executing the draft model
specifically** (its graph replays and/or its eager FA3 attention), not to streams,
allocators, pools, or generic contention. Row 8 adds: the corrupted party is the
verify. Prime suspect: a workspace shared between the draft's EAGER FA3 attention
call and the verify's CAPTURED FA3 kernels (e.g. a cached scheduler-semaphore /
accumulator buffer in the flash-attn wrapper) — the draft's captured pieces write
only the dedicated pool, so the eager attention is the main writer left standing.
Next step for OV1: audit `vllm_flash_attn` / the FA backend for per-device cached
workspaces and give the draft's attention a private copy.

## OV0b timing — engine-level overlap headroom (GEMM shadow, correctness clean)

The GEMM-only shadow is a valid overlap probe (chain-shaped ~31 ms side load, no
vLLM state): accept stays 2.91 and the throughput cost measures how much the
engine's real schedule can hide:

| a2a µs | off tok/s | gemm-shadow tok/s | ratio | side-load hidden |
|---:|---:|---:|---:|---:|
| 0   | 2478 | 1821 | 0.735 | ~13% |
| 500 | 1710 | 1411 | **0.825** | ~26% |

Engine-level hiding is far below OV0a's 72–81% because the verify is comm+compute
*interleaved per layer* — the side load contends for SMs during the verify's compute
phases, and the probe's coarse 4096² GEMMs (~0.17 ms each, whole-GPU) cannot slot
into sub-millisecond comm gaps. Two known levers for the real draft: (1) the real
chain's decode kernels at per-rank B=16 are far smaller-grained → finer
interleaving; (2) CUDA stream priority (verify high / draft low) makes the draft
yield SMs during verify compute. The headroom grows with the comm window (13% →
26% from a2a 0 → 500), i.e., with f — consistent with the thesis that overlap pays
in the comm-bound regime.

## Verdict

- **OV0a: GO** (physics).
- **OV0b: stability GO** (isolation stack works), **correctness BLOCKED** on one
  identified bug class (verify corruption via the draft's concurrent eager
  attention — workspace audit is the next action), **timing: partial** (13–26%
  naive hiding with a coarse probe; finer-grained draft kernels + stream priority
  expected to raise it; grows with f).
- **OV1 go/no-go: proceed, in this order** — (1) FA3 workspace audit + private
  draft workspace; (2) re-run the model-replay shadow expecting accept 2.91;
  (3) then the free-running implementation per `OV1_design.md` (which also avoids
  the probe's double-replay artifact entirely).

## Standing conclusions for OV1 (independent of the remaining bisect)

1. Draft-behind-verify overlap needs, at minimum: propose-done event ordering, a
   dedicated cudagraph pool for draft graphs, and (cheap, keep anyway) a private
   MemPool for draft transients. All three are implemented and env-gated.
2. The physics is favorable (OV0a: 72-81% hidden) and the engine is *stable* under
   concurrent replay with the isolation above; the remaining blocker is one
   values-correctness bug being bisected.
3. The serialized-shadow run (row 4) doubles as a clean measurement of the shadow's
   full cost (2478→1754 = +29% cycle at a2a=0): the overlap upside to recover once
   correctness holds.
