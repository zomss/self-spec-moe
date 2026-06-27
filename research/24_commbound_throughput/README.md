# Phase 24: Comm-Bound Throughput of the Quantized-Draft / bf16-Verify Cycle

Status: **complete on this hardware** (Stage A + B1 + B2a + B2b-pragmatic);
**B2b-full deferred** to a real PCIe/multi-node rental. This is the make-or-break
*benefit*-side phase -- prior phases measured the **cost** (acceptance) and *modeled*
the benefit; this phase measures the **real benefit** on a comm-bound testbed.

Results: `results_commbound_throughput.md` (Stage A regime + B2a comm-free draft +
B2b-pragmatic integrated speedup), `results_stageB.md` (B1 losslessness + acceptance),
`scope_B2.md` (B2b-full plan, deferred). Headline: off NVLink EP decode is 84-89%
communication; the comm-free draft step is ~1.3x the compute floor; the lockstep cycle
is lossless; overhead-accounted integrated speedup is **~2.5-3.1x (beta 0.92) /
1.9-2.2x (beta 0.82)** at the (pessimistic) socket operating point, growing with batch.
Remaining gaps (PCIe-exact point, DBO baseline, full distributed driver) are
hardware-gated.

The original scope follows.

Source: Phases 18/22/23 (measured acceptance of weight + activation quant), Phase 20
(collective counts, the A2A hook + emulation env, speedup model), Phase 21
(verify-warmed cache/locality), Phases 12/16 (NVLink = compute-bound negative), the
"Current Positioning" machine-balance argument in `../status.md`.

## Objective

Produce the one number the top-tier story is missing:

```
real, lossless throughput speedup of a quantized / comm-light speculative draft
+ bf16 full-EP verify, on a COMMUNICATION-BOUND testbed, vs plain EP serving.
```

Concretely: force the EP all-to-all off NVLink (`NCCL_P2P_DISABLE=1`) to create a
comm-bound regime on the available 8xH100 box, confirm it matches the machine-balance
prediction (f large at serving batch), then measure the draft/verify cycle throughput.

## Assumptions and hardware constraints (state honestly)

- **No real PCIe-only or multi-node box.** We emulate comm-boundedness with
  `NCCL_P2P_DISABLE=1` (GPU<->GPU goes through host staging over PCIe). This is a
  **pessimistic** PCIe proxy (host bounce is worse than real PCIe-P2P), so it
  upper-bounds comm cost; we calibrate the operating point with a message-size ->
  latency microbenchmark rather than trusting the single emulated point.
- **No native FP4 tensor cores on H100.** FP4 *compute* speedup is NOT measurable
  here (needs Blackwell). This phase targets the **communication** benefit (measurable
  as all-to-all latency) plus **FP8 compute** (measurable). FP4 remains the
  memory/acceptance lever (already measured, Phases 22/23); flag the compute gap.
- **DeepEP / DBO not installed.** The DBO-overlap baseline -- the single most
  important reviewer comparison -- is **not directly measurable here**. Treated as a
  known gap (needs a DeepEP build or a separate testbed); we instead bound it via the
  Phase 20 exposed-after-overlap argument and flag it explicitly.
- Acceptance `beta` is taken from Phases 18/22/23 (real, weight+activation quant).
- EP path = attention-DP + EP (the AgRs dispatch/combine path, per Phase 20); a
  TP-only EP run does not exercise the hooked all-to-all.

## Design -- two stages (A de-risks before B's engineering investment)

### Stage A -- component-measured speedup on the comm-bound testbed (do first)

Real per-step latency, no integrated scheduler, speedup composed from measured parts.

1. **Confirm the regime (key figure).** Decode-step time and the comm fraction
   `f = (S_full - S_localskip) / S_full` vs batch, with `NCCL_P2P_DISABLE` **on vs
   off**. Prediction (machine balance): f ~small on NVLink, ~large (>=0.4 at serving
   batch) with P2P off. This is the empirical version of the positioning table.
2. **Per-step latency of each draft config** (P2P off, matched batch):
   - `verify`  : bf16, full EP, normal all-to-all      -> S_verify (baseline)
   - `local`   : all-to-all skipped (local routing)     -> S_draft, comm eliminated
   - `act8/act4`: all-to-all with FP8/FP4-sized payload   -> S_draft, comm x0.5 / x0.25
   - `fp8c`    : FP8 experts, full EP                    -> isolates compute benefit
   (comm payload size swept via the Phase 20 AgRs hook; local-skip via the Phase 01/18
   local-routing path; FP8 via vLLM native FP8 MoE.)
3. **Compose the speedup** with measured `S` and measured `beta`:
   `speedup(k) = (E[acc](k,beta) + 1) / (k * S_draft + S_verify)`, swept over k, batch,
   and draft config. Report tokens/s vs the `verify` baseline. Entirely measured
   inputs -- no injection, no assumed f.

Deliverable: the f-vs-fabric figure + a real-hardware speedup table. **Decision gate
for Stage B.**

### Stage B -- integrated lockstep scheduler (only if Stage A is GO)

Build the real cycle in vLLM and measure true end-to-end throughput + losslessness.

- **Cycle:** draft runs the comm-light config for `k` steps; verify runs one bf16
  full-EP step; rejection-sample to accept the longest valid prefix; the verify step's
  routing warms the draft's expert cache (Phase 21 / verify-as-oracle prefetch).
- **Where in vLLM:** spec-decode framework for the loop; the Phase 02/20
  `AgRsAll2AllManager` hook for the per-phase comm config (full vs skipped vs
  quantized payload); the Phase 18 router patch for draft local routing.
- **Metrics:** real tokens/s vs baseline; real multi-token acceptance; and
  **losslessness** -- exact-match to bf16 greedy decode (and matched distribution
  under sampling) over a fixed prompt set.

Deliverable: the scheduler patch, end-to-end tokens/s, and the losslessness proof.

## Commands (Stage A skeleton)

```bash
V=/data/smcho/self-spec-moe/.venv/bin/python
# regime check: P2P on vs off, sweep batch (attention-DP + EP)
for P2P in 0 1; do
  NCCL_P2P_DISABLE=$P2P NCCL_SHM_DISABLE=0 VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
  VLLM_SELF_SPEC_LOG_A2A_COUNTS=1 \
  $V bench_commbound.py --model Qwen/Qwen3-30B-A3B --data-parallel-size 4 \
     --batch-sizes 1,8,32,64,128 --config verify --tag p2p$P2P
done
# draft configs (P2P off): local-skip, act-quant payload, fp8 compute
NCCL_P2P_DISABLE=1 $V bench_commbound.py --config local   --tag local
NCCL_P2P_DISABLE=1 $V bench_commbound.py --config act8    --tag act8
NCCL_P2P_DISABLE=1 $V bench_commbound.py --config fp8c    --tag fp8c
$V compose_speedup.py   # measured S + measured beta -> speedup table
```

(Env notes: `VLLM_DISABLE_CUSTOM_ALL_REDUCE=1` so the custom all-reduce does not mask
the NCCL path; keep `NCCL_SHM_DISABLE=0` so P2P-off falls back to host SHM/PCIe, not
sockets. `bench_commbound.py` extends the Phase 20 `bench_dp_injection.py` harness with
a `--config` switch and the P2P toggle.)

## Decision criteria

- **GO (proceed to Stage B):** comm-bound regime confirmed (`f >= ~0.4` at serving
  batch with P2P off, ~small with P2P on -> validates machine balance) **and**
  component-measured lossless speedup `>= ~1.3-1.5x` over the comm-bound baseline at
  the target batch.
- **WEAK:** f modest or speedup `< ~1.2x` -> reposition (lossless-vs-lossy-quant on
  commodity HW, or inter-node-only) before building the integrated system.
- Either way the central *benefit* claim is, for the first time, measured rather than
  modeled.

## Risks and caveats

- `NCCL_P2P_DISABLE` is a pessimistic, host-staged PCIe proxy -> calibrate with the
  message-size microbenchmark; report speedup as a function of the comm operating
  point, not one emulated point.
- DBO-overlap baseline absent here -> the key reviewer comparison is deferred/bounded,
  not closed. Decide early whether a DeepEP build is in scope.
- FP4 compute speedup unmeasurable on H100 -> scope to comm + FP8 compute; be explicit.
- Stage B is real vLLM spec-decode engineering; Stage A exists precisely to avoid
  investing in it before the benefit is shown.

## Expected next artifact

`results_commbound_throughput.md`: the f-vs-fabric (P2P on/off) figure validating the
machine-balance prediction, and the component-measured lossless speedup table -- the
GO/WEAK decision for the integrated scheduler.
