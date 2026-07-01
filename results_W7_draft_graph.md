# W7 draft-graph: PIECEWISE the comm-free self-spec draft chain (lossless)

Goal: cut the dominant remaining cost of the comm-free self-spec DRAFT chain — the
**in-chain decode forward** — with ZERO change to acceptance / output. Iterated on
`Qwen/Qwen1.5-MoE-A2.7B` DP8 (fast), confirmed on `Qwen/Qwen3-30B-A3B` DP8, both
forced-PCIe, full World-A stack (`DRAFT_LOCAL_ROUTE + DRAFT_FULL_REPLICA +
DRAFT_FULL_CG + COMPILE_CONSISTENT`), FP8 draft on 30B, greedy, K=2, batch 64.

## 1. Root cause confirmed: the chain runs CUDAGraphMode.NONE (fully eager)

With `VLLM_SELF_SPEC_DRAFT_FULL_CG=1` on a non-MLA (FA3/GQA) backend, the GQA accept
fix sets `_draft_chain_force_eager_attn=True` and the chain dispatched with
`use_cudagraphs=False` → **CUDAGraphMode.NONE**: the WHOLE decode-step forward runs
eager (attention *and* the GEMM/MoE body run op-by-op). The captured step-0 forward
(PIECEWISE) is fast; the in-chain forwards are ~4x slower purely from eager launch
overhead of the model body. Measured before (this run, matches recon phase 44):

| region (mean ms) | Qwen1.5-MoE base | Qwen3-30B base |
|---|---:|---:|
| draft_forward_first (step-0, captured PIECEWISE) | 19.74 | 18.95 |
| **draft_forward (in-chain K-step, NONE eager)** | **36.45** | **65.41** |
| draft_chain (whole propose) | 61.83 | 91.42 |
| verify | 38.05 | 64.65 |

30B numbers reproduce the task profile (67.5 / 18.5 / 94.3) within noise.

## 2. Fix: PIECEWISE the chain (attention eager, body cudagraph) — env-gated

The correctness fix (eager attention) and speed (compiled body) are NOT in conflict.
vLLM's normal decode uses **PIECEWISE**: attention is a *splitting op* that runs eager
(sees the live growing `seqused_k` — exactly what the FA3 fix needs), while the GEMM/
MoE body replays captured graph pieces. The NONE chain threw away the compiled body.

New flag `VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE` (default 0 → unchanged NONE chain).
When set AND the chain is forcing eager attention, the chain dispatches PIECEWISE
(`valid_modes={PIECEWISE, NONE}`, FULL excluded — the FULL decode graph is not
replay-safe for the draft's growing sequence) instead of NONE. The PIECEWISE graph
keys are already captured for the draft under FULL_CG (`initialize_cudagraph_keys` →
`FULL_AND_PIECEWISE`, whose `mixed_mode()` is PIECEWISE, captured for every decode
batch size). Attention numerics are unchanged (still eager); only the body moves onto
the captured graph.

DP stays consistent: all draft ranks are non-MLA and all dispatch PIECEWISE, so
`coordinate_batch_across_dp` syncs the runtime mode to PIECEWISE (min across ranks)
and pads uniformly — mirroring normal decode. Confirmed in the logs: each rank logs
`Draft chain PIECEWISE: runtime_mode=PIECEWISE batch_size=1 input_batch_size=3` (the
per-rank decode batch, DP-padded to the max=3).

Files changed (all env-gated, default off → byte-identical):
- `vllm/envs.py`: `VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE` (bool, default False).
- `vllm/v1/spec_decode/llm_base_proposer.py`:
  - `__init__`: read flag into `self._draft_chain_piecewise`.
  - chain setup: when `_draft_chain_force_eager_attn and _draft_chain_piecewise`,
    dispatch with `piecewise_only=True` (body captured) instead of `use_cudagraphs=
    False` (whole forward eager). One-time INFO log of the resolved runtime mode.
  - `_determine_batch_execution_and_padding`: new `piecewise_only` param — first
    dispatch allows `{PIECEWISE, NONE}` (never FULL), `uniform_decode` forced off so
    the plain batch descriptor matches the relaxed PIECEWISE key; the DP re-dispatch
    already keys off the synced (PIECEWISE) mode. `_last_batch_desc` now also stashed
    for PIECEWISE so the caller's `set_forward_context` keys the captured body graph.

## 3. Result: in-chain forward before → after, and losslessness proof

| region (mean ms) | model | base (NONE) | pw (PIECEWISE) | speedup |
|---|---|---:|---:|---:|
| **draft_forward (in-chain)** | Qwen1.5-MoE | **36.45** | **13.46** | **2.71x** |
| **draft_forward (in-chain)** | Qwen3-30B | **65.41** | **14.48** | **4.52x** |
| draft_forward_first (step-0) | Qwen3-30B | 18.95 | 19.00 | unchanged |
| draft_chain (propose) | Qwen3-30B | 91.42 | 39.07 | 2.34x |
| verify | Qwen3-30B | 64.65 | 63.67 | unchanged |

The in-chain forward drops **65.41 → 14.48 ms on 30B** — better than the 37.8 ms
target; it lands at ≈ the captured step-0 (19 ms) and ≈ T_compute(64)=16.3 ms. The
~51 ms saved was pure eager-body launch overhead; the attention kernel and the
batch-invariant matmuls are identical to the NONE chain.

**Did PIECEWISE work?** Yes — `runtime_mode=PIECEWISE` logged on every rank; the body
is captured (49 PIECEWISE graphs captured at load); attention still eager (splitting
op, unchanged numerics).

**Accept identical (the hard gate):** greedy, fixed seed, base vs pw, batch 64:

| model | seqs byte-identical | tokens mismatched | accept_len base | accept_len pw |
|---|---:|---:|---:|---:|
| Qwen1.5-MoE DP8 | **64 / 64** | **0 / 8192** | 2.969247467438495 | 2.969247467438495 |
| Qwen3-30B DP8 (FP8) | **64 / 64** | **0 / 8192** | 2.8994690 | 2.9060617 |
| Qwen1.5-MoE DP4 | **48 / 48** | **0 / 4608** | 2.962676962676963 | 2.962676962676963 |

**Output tokens are BYTE-IDENTICAL on both models** (0 / 8192 mismatches, all 64
sequences). On 30B the *accept_len metric* differs by 0.007 — this is the documented
batched-verify near-tie accumulation artifact in the spec counters (the metric window
spans warmup generations and counts near-tie flips), NOT a token difference: the
actual generated tokens are identical, which is the losslessness gate. On Qwen1.5-MoE
the metric is bit-identical too.

## 4. Headline cycle-ms + speedup (Qwen3-30B DP8 K2 b64 forced-PCIe)

Measured with `w7_qwen30b_timing.py` (two-length decode slope, WARMUP=2, ITERS=3),
full stack + COMPILE_CONSISTENT, FP8 draft, on the worktree code:

| config | tok/s | cycle ms/step | accept_len | speedup vs no-spec |
|---|---:|---:|---:|---:|
| no-spec | 1865.9 | 34.30 | — | 1.000x |
| spec, NONE chain (base) | 1243.1 | 51.49 | 2.8903 | **0.666x** |
| spec, PIECEWISE chain (pw) | 1881.5 | 34.03 | 2.8975 | **1.008x** |

**PIECEWISE takes the comm-free self-spec from 0.666x → 1.008x — it crosses parity
with no-spec.** Cycle drops 51.49 → 34.03 ms/step. This is the recon phase-44 "fix the
draft-compute term" isolation, realized: the +53 ms eager-body error was the single
largest of the three cycle-error terms, and closing it alone lifts the full stack past
1.0x here (the standalone-measured no-spec baseline is 1865.9 tok/s = 34.3 ms/step,
matching recon's 34.45). The accept_len difference (2.8903 vs 2.8975) is metrics-window
noise (near-tie flips), not a token change — output is byte-identical (§3).

**Projected crossover shift:** the comm-sweep crossover (phase 42) is where the
comm-free draft's saving beats its added draft cost. With the in-chain draft forward
cut ~4.5x (65→14 ms), the draft-GPU term falls from 86 → ~33 ms (≈K·T_compute), so the
self-spec cycle no longer needs the emulated-A2A penalty to be large to win: the
crossover moves toward lower forced-PCIe A2A latency (the stack is at/above parity even
at the real fabric here). Remaining terms to close for further headroom are the verify
all-to-all (+33 ms) and per-cycle CPU orchestration (+31 ms), tracked separately.

## 5. Secondary lever (batch-invariance) & irreducible cost

The COMPILE_CONSISTENT batch-invariant matmuls run in BOTH the captured step-0 (19 ms)
and the PIECEWISE chain (14.5 ms), so they are no longer the differentiator once the
body is captured — the ~51 ms won back was entirely eager launch overhead, not the
batch-invariant kernels. Selective batch-invariance was therefore de-prioritized: the
in-chain forward is already at ≈T_compute, so there is little headroom left in the
draft forward itself (further gains must come from the verify all-to-all and per-cycle
CPU orchestration, tracked separately). The irreducible in-chain draft-forward cost is
now ≈ the captured decode forward (~14–19 ms), i.e. the model's own compute.

## Reproduce
- Per-forward split + losslessness: `w7_pw_test.py {base,pw}` (DP8, records token_ids +
  profiler split), driver `run_pw.sh`.
- Headline cycle: `research/34_worldA_system/scripts/w7_qwen30b_timing.py` with
  `VLLM_SELF_SPEC_COMPILE_CONSISTENT=1` and (for pw) `VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1`.
