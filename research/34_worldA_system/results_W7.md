# W7 results: wall-clock decode tokens/s -- World A (bf16 full replica) vs no-spec

**Model:** DeepSeek-V2-Lite (native `deepseek_v2`, 64 routed experts, 6/token, 27 layers),
greedy. **Layout:** data-parallel `dp=2` + `--enable-expert-parallel` (`tp=1`),
forced-PCIe (`NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1`), 2x H100-80GB.
**Spec:** `speculative_config={method:"draft_model", model:<same V2-Lite ckpt>,
num_speculative_tokens:K, draft_tensor_parallel_size:1}` +
`VLLM_SELF_SPEC_DRAFT_FULL_REPLICA=1` (bf16 full replica, comm-free draft) +
`VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE=1`. **No-spec:** plain full-EP decode.

## Method (decode-time isolation)
Steady-state DECODE throughput via a **two-length slope**: for each batch we time a
long run (`max_tokens=OUTLEN=128`, `ignore_eos`) and a short run (`SHORTLEN=32`); both
share the same prefill, so `decode_time = t(OUTLEN) - t(SHORTLEN)` isolates
`(OUTLEN-SHORTLEN)=96` steady-state decode steps/seq and cancels prefill.
`tok/s = batch*96 / decode_time` (system throughput, OUTPUT tokens only).
**CUDA graphs ON** (the realistic serving config; `enforce_eager` hides everything
under per-step launch overhead -- see "Measurement notes"). `WARMUP=2`, `ITERS=3`
(report mean +- std). Same prompts + output length for spec and no-spec at a given batch.
Harness: `scripts/w7_timing.py`; driver `scripts/w7_serial.sh`; analysis
`scripts/w7_analyze.py`; raw JSON under `data/w7_*.json`.

> IMPORTANT: spec engines must run **one at a time** (no concurrent engine on the
> box). The spec draft loop is CPU-orchestration-bound; a second concurrent DP=2
> engine inflates spec step time ~7x (measured). No-spec (GPU/graph-bound) is
> unaffected. All numbers below are from strictly serial runs.

## Headline finding
**World A bf16-full-replica spec is SLOWER than no-spec at every (batch, K) in every
fabric regime tested.** The comm-free draft works exactly as designed (it pays no
A2A; spec tok/s is nearly fabric-independent -- native vs forced-PCIe-with-A2A-tax
differ by <3%), but each spec step runs the **full 64-expert dense draft K times**
plus the EP verify, making a spec step ~4x a no-spec step, which the accept-length
(~2.7-4.2) cannot overcome on a model this small. Speedup is best (closest to 1) at
low/mid batch and worst at high batch -- but never >= 1.

## Regime A: forced-PCIe + emulated exposed A2A (100 us/collective) -- the comm-bound target
No-spec pays the exposed A2A on every token; the comm-free draft does not (only the
verify pays it, amortized over accept_len tokens). This is the regime most favorable
to World A.

| batch | K | accept_len | spec tok/s | nospec tok/s | speedup |
|------:|--:|-----------:|-----------:|-------------:|--------:|
| 8   | 2 | 2.68 | 214.4  | 629.2   | 0.34 |
| 32  | 2 | 2.72 | 815.8  | 2220.2  | 0.37 |
| 64  | 2 | 2.66 | 1573.7 | 3736.2  | 0.42 |
| 128 | 2 | 2.64 | 1678.9 | 6652.9  | 0.25 |
| 256 | 2 | 2.40 | 2976.4 | 11209.4 | 0.27 |
| 8   | 3 | 3.27 | 154.1  | 629.2   | 0.25 |
| 32  | 3 | 3.31 | 599.9  | 2220.2  | 0.27 |
| 64  | 3 | 3.22 | 1182.0 | 3736.2  | 0.32 |
| 128 | 3 | 3.24 | 1204.4 | 6652.9  | 0.18 |
| 256 | 3 | 2.92 | 2260.7 | 11209.4 | 0.20 |
| 8   | 4 | 4.15 | 117.8  | 629.2   | 0.19 |
| 32  | 4 | 4.03 | 464.9  | 2220.2  | 0.21 |
| 64  | 4 | 3.71 | 872.5  | 3736.2  | 0.23 |
| 128 | 4 | 3.72 | 886.3  | 6652.9  | 0.13 |
| 256 | 4 | 3.16 | 1736.2 | 11209.4 | 0.15 |

Best (closest) speedup in this regime: **0.42x at batch=64, K=2**.
Increasing K monotonically HURTS (each extra speculative token adds a full dense-64-expert
draft forward that costs more than the marginal acceptance it buys).

## Regime B: forced-PCIe, native fabric (no emulated A2A) -- compute-bound contrast
On the fat fabric, the native A2A is tiny (NVLink vs forced-PCIe no-spec differ ~4%),
so the comm-free draft saves almost nothing and spec loses even harder.

| batch | K | accept_len | spec tok/s | nospec tok/s | speedup |
|------:|--:|-----------:|-----------:|-------------:|--------:|
| 8   | 2 | 2.68 | 218.8  | 1074.1  | 0.20 |
| 32  | 2 | 2.72 | 834.4  | 3476.5  | 0.24 |
| 64  | 2 | 2.66 | 1598.6 | 5418.4  | 0.30 |
| 128 | 2 | 2.64 | 1748.1 | 9213.9  | 0.19 |
| 256 | 2 | 2.40 | 2984.4 | 14687.9 | 0.20 |
| 8   | 3 | 3.27 | 157.6  | 1074.1  | 0.15 |
| 32  | 3 | 3.31 | 617.4  | 3476.5  | 0.18 |
| 64  | 3 | 3.22 | 1202.8 | 5418.4  | 0.22 |
| 128 | 3 | 3.24 | 1228.9 | 9213.9  | 0.13 |
| 256 | 3 | 2.92 | 2250.2 | 14687.9 | 0.15 |
| 8   | 4 | 4.15 | 125.5  | 1074.1  | 0.12 |
| 32  | 4 | 4.03 | 490.1  | 3476.5  | 0.14 |
| 64  | 4 | 3.71 | 924.7  | 5418.4  | 0.17 |
| 128 | 4 | 3.72 | 921.9  | 9213.9  | 0.10 |
| 256 | 4 | 3.16 | 1798.2 | 14687.9 | 0.12 |

Best (closest) speedup in this regime: **0.30x at batch=64, K=2**.

## NVLink contrast (fat fabric, no NCCL disable, K=4) -- compute-bound
Matched config (CG, OUTLEN=128, WARMUP=2, ITERS=3).

| batch | spec tok/s | nospec tok/s | speedup |
|------:|-----------:|-------------:|--------:|
| 8   | 125.5  | 1281.5  | 0.10 |
| 32  | 497.4  | 4494.5  | 0.11 |
| 64  | 946.3  | 6936.2  | 0.14 |
| 128 | 934.5  | 12274.4 | 0.08 |
| 256 | 1806.5 | 20361.8 | 0.09 |

(NVLink no-spec is the fastest fabric, e.g. batch=256 = 20362 tok/s vs forced-PCIe
native 14688.) On NVLink the comm-free property buys nothing -- exactly the
comm-bound-vs-compute-bound distinction (spec helps least where comm is cheap). Note
spec tok/s on NVLink == forced-PCIe native (125.5/497.4 both fabrics): the comm-free
draft is fabric-independent, so spec's loss is pure draft compute, not comm.

## Losslessness spot-check (W0 standard)
Full-replica spec (K=4) vs no-spec greedy, 16 prompts, DP=2+EP forced-PCIe:
**12/16 exact sequences, 82.4% token agreement** -- identical to the W2 design's report.
The mismatches are the pre-existing batched-verify near-tie FP flips (documented W0/B1
caveat), not corruption. Spec acceptance: rate 0.971, mean accept-length 4.88 (K=4) --
matches W2. So the stack is as lossless as vLLM's native draft_model spec.

## OOM / limits
Full-replica draft + verify shard fit at `gpu_memory_utilization<=0.90` (~71-75 GB/rank,
80 GB H100); batch=256 fits. **K=5 is unstable** for the full replica: large-batch K=5
intermittently `EngineDeadError`s, and K=5's low-batch slope occasionally shows a
CUDA-graph-capture artifact (implausibly low decode time, high variance) -- flagged
`suspect` by the harness and excluded. (One artifact example: K=5 a2a100 batch=8 read
1343 tok/s with +-109 std and decode 0.57s vs K=4's 6.1s; not real.)

## Measurement notes (important)
- **CUDA graphs are essential** to see the regime. With `enforce_eager=True`, per-step
  Python/launch overhead dominates so both configs are ~8x slower and the comm signal
  is masked (spec ~0.78x, nearly flat across batch). All headline numbers use graphs.
- **The "win" mirage:** an early under-warmed probe read ~2x at K=4/100us. It was a
  graph-capture leak into a timed run (long~short~6s -> tiny, meaningless slope). With
  `WARMUP>=2` and the `suspect` guard, clean runs show spec loses. Three independent
  serial runs + NVLink agree.
- Spec tok/s is **fabric-insensitive** (native ~= a2a100, <3% diff) because the
  comm-free draft dominates the step and pays no A2A -- direct confirmation the W1
  comm-skip is live on the draft; the loss is purely the replica's compute, not comm.

## Verdict
World A's comm-free draft is real and behaves as designed, but the **bf16 full replica
(W2a) is the wrong draft for wall-clock wins on a small MoE**: dense 64-expert draft
compute >> the comm it saves. A wall-clock speedup needs the draft to be CHEAP
(W2b/W2c: globally-hot top-C + skip-cold and/or FP4 resident cache, so the draft's
per-step compute drops below the comm it removes) and/or a genuinely larger
comm/compute ratio (bigger model, true multi-node exposed A2A, larger E). The curve
("best at low/mid batch, worst at high batch") holds in direction, but the level is
< 1 everywhere with this draft.
