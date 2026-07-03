# Phase 52 results — first REAL 2-node run: comm-free self-spec on h107+h106

**Setup.** Qwen3-30B-A3B, attention-DP16 + EP16 (tp=1) across 2x 8xH100
(h107/h106), real fabric: NVLink intra-node + 8x400G ConnectX-7 RoCEv2 rails
inter-node, GPUDirect RDMA. Phase-47 spec stack (FP8 full-replica comm-free
draft, DRAFT_LOCAL_ROUTE, FULL_CG, COMPILE_CONSISTENT, PIECEWISE, K=2).
Harness: `scripts/w7_2node.py` (node-rank-aware copy of the unchanged W7
harness), two-length slope, b=32/64/128 per rank, greedy, ignore_eos.
Branch `research/self-spec-moe` @ 94cb8e33c. Data in `data/`, logs in `logs/`.

**System changes made during the phase (persist on both nodes):**
- `ninja` installed into the shared venv (torch.compile on h106 failed without it).
- CPU governor powersave -> **performance** on both nodes (all 224 cores).
- gloo/bootstrap/DP-RPC moved from the frontend NIC to rail 0
  (`NCCL_SOCKET_IFNAME=GLOO_SOCKET_IFNAME=enmlx0`, master IP 192.168.0.17).

## 1. Headline numbers

| batch/rank | 1n-nospec (NVLink ref) | 2n-nospec | 2n-spec K2 | accept | speedup | f (lower bound) |
|---:|---:|---:|---:|---:|---:|---:|
| 32  | 2730 | 1769 | 861  | 2.91 | **0.49x** | 0.35 |
| 64  | 3830 | 2855 | 1075 | 2.90 | **0.38x** | 0.25 |
| 128 | 4225 | 3880 | 1350 | 2.87 | **0.35x** | ~0.1 (noisy ref) |

(powersave-era sweep; governor fix moved b64 to nospec 3023 / spec 1109 =
0.37x, rail-gloo to 1233 = 0.41x — the regression is NOT an environment
artifact.) Single-node DP8 NVLink spec (same stack): 1656 vs 3830 = **0.43x**.

**Acceptance transfers perfectly cross-node** (2.87-2.91 = Phase 47's
2.88-2.91, flat in batch). The loss is entirely wall-clock.

**f falls with batch on this fabric** (0.35 -> 0.25 -> ~0.1): NCCL collectives
here are latency-bound (~70-115 us/coll at decode payloads, Phase 51), so comm
cost is ~fixed while compute grows — the OPPOSITE slope vs the forced-PCIe
emulation (f rose 0.37->0.65). The high-batch comm-bound regime the single-node
emulation predicted does not exist on a rail-optimized fabric at 30B scale.

## 2. Where the cycle actually goes (fine profile, b64, perf+rail)

Cycle from throughput: 172 ms (2-node), 108 ms (1-node NVLink). Decomposition
(rank-0 profiler, decode-only cycle arithmetic):

| term | 1-node DP8 | 2-node DP16 | token-equivalent honest cost |
|---|---:|---:|---:|
| draft_forward_first + draft_forward | 18.3 + 13.6 | 18.5 + 13.9 | (as modeled ~16/fwd) |
| draft_chain total (incl. glue) | 40.7 ± 7.3 | 61.8 ± 8.1 | forwards = 32 |
| verify (spec, q=3, 192 tok) | **69.4 ± 30** | **~107 (slope) / 116 ± 106 (region)** | nospec@192tok step = **37** |
| nospec step (b64) | 16.8 | 22.4 | — |

Diagnostic ladder (each step measured, none sufficient):
1. **CPU governor powersave** -> performance: chain jitter collapsed
   (std 44->8 ms, max 450->88) but speedup +3% only.
2. **gloo/TCP on frontend** (200 ms-quantized RTO-like spikes): moved to rail
   -> +11% (1109->1233 tok/s). Real, small.
3. **AgRs token-scaling** (verify carries Bx(K+1) tokens): b192-nospec
   discriminator = 37 ms vs b64's 22 ms -> explains only ~15 ms.
4. **Cross-rank skew**: with chains now tight (±8 ms), max skew ~20 ms — cannot
   explain the verify excess.
5. **The dominant term is fabric-INDEPENDENT**: the spec verify forward costs
   ~65-70 ms at b64 on NVLink DP8 (69.4), forced-PCIe DP8 (Phase 44: 65.4),
   and ~107 ms on 2-node — i.e. **~2-3x its token-equivalent captured step
   everywhere**, plus ~+40 ms cross-node. The excess is an execution-path
   cost of the q=3 verify, not communication.

**Candidate mechanisms for the verify excess (next to check, in order):**
(a) the q=3 uniform batch missing the FULL-cudagraph dispatch (48 layers of
piecewise/eager launches ~ the right magnitude; capture log shows q3 FULL
graphs captured, largest=498, but runtime dispatch unverified);
(b) COMPILE_CONSISTENT batch-invariant kernels taxing the (large) verify
forward — the nospec baseline may not pay this;
(c) FA3 spec-decode (q=3) attention kernels vs q=1 decode kernels.
A torch-profiler/nsys trace of ~5 cycles on rank 0 settles it.

## 3. Honest ceiling on this (model, fabric) even with plumbing fixed

With measured verify token-scaling (verify_ratio ~1.65x step) the correct
cycle model is `K(1-f) + verify_ratio` steps per accept_len tokens:
at f=0.25-0.35, K=2, accept 2.9 -> **~0.9-1.0x best case**. The Phase-48
mapping (1.2-1.4x) used the Phase-47 shielded curve, which under-counted
verify scaling (the Phase-27 correction, reconfirmed here on real hardware).
Conclusion: **on a rail-optimized 400G fabric, Qwen3-30B decode is not
comm-bound enough for World A to win at any batch — independent of the
implementation issues.** The win regime requires f >= ~0.5: bigger/deeper MoE
(DeepSeek-scale: 160-360 us/coll x more layers), thinner fabric (1-2 NICs/node
deployments), or wider EP.

## 4. What Phase 52 establishes (positive contributions)

1. First real inter-node validation: **acceptance (2.9) and losslessness
   machinery transfer to multi-node unchanged** — the algorithm side is done.
2. Real-fabric f measurement: 0.35/0.25/~0.1 at b32/64/128 — f FALLS with
   batch on latency-bound fabrics, inverting the single-node-emulation slope.
   The comm-bound target regime must be produced by model scale, not batch.
3. The verify execution-path overhead (~2-3x token-equivalent, fabric-
   independent) is the top systems blocker for spec-decode on MoE-EP — it was
   invisible in single-node forced-PCIe work because slow comm masked it
   (Phase 44 saw +33 ms but attributed it to the same bucket as comm).
4. Deployment hygiene for multi-node spec cycles: performance governor,
   coordination traffic off the frontend NIC, ninja in the venv. Each is
   small alone; spec cycles (3 sync-sensitive phases/cycle) are far more
   sensitive to them than plain decode (1 phase/step).

## 5. Next (Phase 53 proposal)

1. **Trace the verify** (torch profiler, 5 cycles, rank 0): pin (a)/(b)/(c);
   if (a), fix dispatch -> verify ~40 ms -> 2-node cycle ~110 ms (0.55x);
   with chain-glue parity (~41 ms) -> ~90 ms (~0.7x). Still <1 here — but
   these fixes transfer to the winning regime.
2. **DeepSeek-scale shared-expert MoE on these 2 nodes** (the f >= 0.5 test):
   e.g. DeepSeek-V2/V3-family at bf16-doesn't-fit-per-node scale. This is the
   go/no-go for the method itself; predicted 1.4-2x if f lands 0.5-0.7.
3. **DeepEP install** (nvshmem present) — both as the strong baseline
   (Phase 19 requirement) and as the low-latency A2A that shrinks the AgRs
   comm term; note RoCE support is "partially tested" upstream.

## 6. Item-storm root cause + fix A/B (post-merge a6d3996f8)

Stack-enabled torch trace pinned the spec cycle's host syncs to ONE site:
`dp_utils.py _post_process_cudagraph_mode` <- `coordinate_batch_across_dp` —
96 calls / 1078 ms of a 1559 ms window (66%): every draft-chain forward runs
the cross-DP batch-coordination all_reduce and blocks on `.item()`.
The comm-free draft needs no DP agreement (no collectives inside), so the
OV1(b) consume-mode skip was generalized to the lockstep chain:
**`VLLM_SELF_SPEC_DRAFT_SKIP_DP_COORD=1`** (env-gated, default off;
`llm_base_proposer.py` + `envs.py`).

A/B (b64, K=2, performance governor, rail gloo):

| config | off | on | accept off/on |
|---|---:|---:|---|
| 1-node DP8 | 1541 | 1686 (**+9.4%**) | 2.893 / 2.893 (bit-parity) |
| 2-node DP16 | 1232 | 1205 (**~0**, within noise) | 2.895 / 2.912 |

**Reading:** the `.item()` block was mostly ABSORBING skew, not adding it.
Single-node, removing it lets the CPU run ahead of the GPU chain (+9%).
Cross-node, the wait just moves to the next rendezvous (the verify's EP
collective) — net zero. Cycle accounting after all fixes (2-node, 150 ms):
chain 62 ms (32 ms forwards + ~30 ms glue/skew) + verify ~88 ms (37 ms real
work at 192 tokens + ~20 ms skew + ~30 ms residual). The remaining residual
is bounded and does not change the Section 3 conclusion: best case ~0.9-1.0x
on this (model, fabric); the win test remains model scale (Phase 53).
Flag kept default-off; worth enabling single-node and in comm-bound regimes
(it can only help when the verify wait does not dominate).
