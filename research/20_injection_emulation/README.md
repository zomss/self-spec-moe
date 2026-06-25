# Phase 20: Single-Node Emulation of Inter-Node A2A via Latency Injection

Source: Phase 19 (multi-node plan), Phase 02 (the A2A hook), Phase 15 (the model
this measures). This phase makes the inter-node go/no-go **measurable on one node**
by injecting the inter-node all-to-all latency instead of paying it on real IB.

## Why not real IB on this node (tested, negative)

The box has the hardware (8x ConnectX-7, all ports ACTIVE at 400G NDR, `perftest`
installed), so we tried to force real RDMA loopback to fake a second node. It does
not work here:

- **NIC -> switch -> NIC hairpin** (`ib_write_lat` mlx5_1 -> mlx5_0):
  `transport retry counter exceeded (syndrom 0x81)` -- the fabric will not hairpin
  RDMA between two HCAs on the same host.
- **NIC self-loopback** (rdma_cm to localhost): `ADDR_ERROR` -- no IPoIB address to
  resolve against.

This is the same wall Phase 02 hit with `NCCL_P2P_DISABLE=1`. The fabric is
provisioned for genuine inter-node traffic only, and we have no admin to change the
SM/IPoIB config. Even if it worked it would be the wrong instrument: the real path
is **DeepEP/NVSHMEM/IBGDA** (not installed here, not NCCL), a single-node loop has
no cross-node switch geometry, and it cannot reproduce **DBO overlap** -- the actual
risk. So: **no IB self-looping. Inject the latency instead.**

## Why injection is the right instrument

The inter-node go/no-go is fundamentally a **sweep over exposed-A2A latency per
collective**: "at what exposed `d` does a collective-free local draft net a TPOT
win?" Injection sweeps that parameter directly, reproducibly, on the engine we have.
Real hardware only pins *where on that x-axis* the operating point sits -- a single
number that DeepEP's published figures (dispatch ~163us, combine ~320us) already
give.

## The hook (Phase 02, upgraded here)

`VLLM_SELF_SPEC_EMULATE_A2A_DELAY_US` injects a per-collective delay on the **GPU
stream** via `torch.cuda._sleep`, calibrated once at all2all-manager init (eager,
1979 cycles/us on this H100). Unlike the old host `time.sleep(ms)`, a stream delay
is **captured into the CUDA graph** and replays every decode step -- so it is
visible under `enforce_eager=False` (representative decode), not just at capture.
Validated: requested 163/320 us realize to within 5%/3%. Default off,
behavior-preserving.

## Measurement

`bench_injection_sweep.py`: real full-EP decode (Qwen3-30B-A3B, EP=4, dummy
weights), injecting `d` on every MoE AgRs dispatch/combine. Per (batch B, delay d)
we record the decode step time `S(B,d)`. Linear by construction:

```
S(B,d) = S_base(B) + N * d      N = injected collectives per step (~2 per MoE layer)
f(B,d) = N*d / S(B,d)           exposed-A2A fraction (step level)
```

`N` and `S_base` come from the slope/intercept across d. The speedup ceiling of a
collective-free draft is `1/(1-f)`; combined with measured acceptance `beta`
(Phases 09/13/18) it gives the end-to-end envelope -- now grounded in the **real
per-engine collective count**, not the Phase 15 assumption.

### Important caveat: this is the FULLY-EXPOSED upper bound

The naive AgRs path serializes the injected `_sleep` after each collective with **no
DBO**, so `f` here is the optimistic maximum. A real DeepEP+DBO system hides some
fraction; that residual is the **one quantity that still needs real multi-node
hardware** (Phase 19 go/no-go gate). We report `f` as the ceiling and flag the DBO
discount explicitly.

## Results

Full results and derivation in `results_injection.md`. Headlines (dp=4 attention-DP
+ EP, the path that actually exercises the hook -- a TP-only EP run does NOT, it
combines experts via an EP-group all-reduce):

- **N ~= 145 exposed collectives per decode step** (flat across batch; ~3 AgRs
  collectives x 48 MoE layers). This is the real per-engine count Phase 15 assumed.
  GPT-OSS-20B (24 layers, MXFP4) gives **N ~= 70 = ~3 x 24** -- N is structural
  (~3/layer), so the win scales with MoE depth (deeper real targets => bigger win).
- **Exposed-A2A fraction f = 0.58-0.84** at DeepEP's 163-320 us/collective, highest
  at low batch.
- **Break-even exposed latency d* ~= 11-18 us/collective** (3.3 us for FP8-beta).
  Because a step pays ~145 collectives, the bar to net a win is tiny -- far below
  DeepEP's 163-320 us collective cost.
- **Lossless speedup 1.1-3.3x** across the plausible exposed-after-DBO range
  (1.6-3.3x at full DeepEP latency / zero overlap; 1.1-1.8x at a conservative
  d=50-100 us residual).

The one quantity this node cannot produce is how much DBO hides at the target batch
(the naive path has no DBO, so these f are the fully-exposed ceiling). That gate now
has a precise threshold -- `d* ~ 15 us` -- to test on a 2-node IB rental.

`python analyze.py` regenerates all tables from `data/dp4_d*.json`.
