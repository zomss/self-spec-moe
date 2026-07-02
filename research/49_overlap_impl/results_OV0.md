# OV0 results — overlap feasibility probes

**TL;DR — full GO.** OV0a (physics): 72–81% of a chain-sized load hides inside
forced-PCIe comm windows. OV0b (engine): after root-causing a verify-corruption
bug to the **shared fused-MoE workspace** (`workspace13/workspace2` from the
global WorkspaceManager, whose addresses both models' captured graphs bake — fix:
`VLLM_SELF_SPEC_DRAFT_WORKSPACE=1`, a dedicated draft workspace), the REAL draft
chain replayed concurrently with the verify is **correct (accept 2.91–2.92) and
~60% hidden at a2a=0 / ~71% hidden at a2a=500 — at the physics ceiling**. Four
reusable isolation requirements for OV1, all implemented and env-gated: propose-
done event ordering, dedicated draft cudagraph pool, private MemPool for shadow
transients, dedicated draft MoE workspace.

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

| 9 | ROOT CAUSE: the fused-MoE modular kernel draws `workspace13/workspace2` (gemm scratch; ws13 doubles as the MoE OUTPUT) from the **global `WorkspaceManager`** (`vllm/v1/worker/workspace.py`) — draft and verify captured graphs bake pointers into the SAME buffer; concurrent replay = concurrent writes to shared MoE scratch → verify output garbage | audit: flash-attn wrapper clean (per-call allocs); `current_workspace_manager` users → `modular_kernel.py:1094` | explains all rows: serial clean (no concurrency), GEMM shadow clean (no workspace use), pools irrelevant (workspace is a persistent tensor outside graph pools) |
| 10 | fix: `VLLM_SELF_SPEC_DRAFT_WORKSPACE=1` — `current_workspace_manager()` returns a DEDICATED draft manager when the draft forward-context flag is active (flag present during draft profiling/capture → draft graphs bake draft-workspace addresses) | a2a=0: accept **2.91**, 2144.7 tok/s (0.865×); a2a=500: accept **2.92**, 1586.2 tok/s (**0.928×**) | **CORRECT + OVERLAPPED** |

A + B clean pinned the corruption to concurrently executing the draft model
specifically; row 9 found the shared writer (MoE workspace, not FA3 — the
flash-attn wrapper allocates per call), and row 10 fixes it for one extra
workspace buffer.

## OV0b timing — REAL-chain overlap (correct, the number OV0b exists for)

Side load = 2 draft chain-step replays (~29 ms, the real kernels):

| a2a µs | off tok/s | shadow-on tok/s | ratio | added ms/cycle | **hidden** |
|---:|---:|---:|---:|---:|---:|
| 0   | 2478 | 2144.7 | 0.865 | 11.6 of ~29 | **~60%** |
| 500 | 1710 | 1586.2 | 0.928 | 8.5 of ~29 | **~71%** |

The real chain's fine-grained decode kernels (per-rank B=16) interleave into the
verify's comm gaps at essentially the OV0a physics ceiling (72–74%) — the coarse
GEMM proxy's 13–26% was a granularity artifact, as suspected. Stream priority is
now optional polish, not a requirement.

*(Historical: the GEMM-only proxy measured 13–26% hiding — a granularity
artifact of whole-GPU 4096² blocks; superseded by the real-chain numbers above.)*

## Verdict

- **OV0a: GO** (physics, 72–81% hideable).
- **OV0b: GO** — correctness restored (workspace fix, accept 2.91–2.92) AND the
  real draft chain hides at the physics ceiling (~60% @ a2a=0, ~71% @ a2a=500).
- **Required isolation stack for OV1** (all implemented, env-gated, default-off):
  1. propose-done event ordering (side stream vs previous chain),
  2. `VLLM_SELF_SPEC_DRAFT_GRAPH_POOL=1` (dedicated draft cudagraph pool),
  3. private MemPool for side-stream transients,
  4. `VLLM_SELF_SPEC_DRAFT_WORKSPACE=1` (dedicated draft MoE workspace — the
     root-cause fix).
- **Next: build OV1 per `OV1_design.md`** — the free-running chain consumes this
  stack directly; expected recovery ≈ 71% of the chain cost at f≈0.6, more at
  higher f, compounding with upward K-retune.

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
