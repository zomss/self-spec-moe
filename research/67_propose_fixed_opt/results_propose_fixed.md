# Phase 67 results — propose-fixed decomposition + step-0 decode compaction

Single-node h107 DP8/EP8, 16k (CTX 16,384 / max_model_len 20,480), arm-B
config (fp8 comm-free replica + shared KV + W512 sinks16 + P65 flag stack:
DP_COORD_CPU + CHAIN_LIGHT_MD + SKIP_DP_COORD). Greedy, two-length slope
160/32, iters=2 warmup=1. Pool 248,448 tokens/rank (b8 = 52%, b12 = 79%;
both resident).

## Baseline F/D (clean tok/s, no profiler)

| point | tok/s (accept) | cycle ms |
|---|---|---|
| K2 b8 | 261.9 (2.888) | 88.2 |
| K4 b8 | 301.1 (4.594) | 122.0 |

-> **F = 54.4 ms, D = 16.9 ms/step** (K4 tps ±24.8 => split ±~few ms).

## Propose-fixed decomposition (FINE profiler, K4 b8, CUDA-synced)

Regions are CUDA-synchronized (serialized) so absolute values are upper
bounds on the exposed cost; the point is the attribution.

| component | ms | note |
|---|---|---|
| verify (target fwd) | 41.6 | out of scope (fundamental) |
| **step-0 draft forward** | **14.6** | q=(K+2)=48 tok; the dominant fixed propose term |
| step-0 CPU glue (set_inputs/build_md/determine/sample/chain_setup) | 2.6 | |
| gap cpu (rejection/parse/bookkeep/next-input/prepare) | 1.18 | already crushed by P65 |
| marginal chain step | 11.8 | draft_forward 10.0 + glue 1.8 (out of scope: chain) |

**propose-fixed single-node ~= 18.4 ms** (step0 fwd 14.6 + glue 2.6 + gap 1.2).

Key inference: the CPU/sync propose-fixed is only ~3.8 ms single-node (P65
optimizations already crushed it — there is nothing left to trim there). The
~88 ms 2-node propose-fixed is therefore NOT step-0/glue/bookkeep-dominated;
its bulk is CROSS-NODE (DP-coordination rendezvous + cross-node gather +
verify-side NCCL) that does not appear single-node.

## Fix: shared-KV step-0 decode compaction

`VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1` -- step-0 becomes a q=1 decode of
the appended token(s); the K+1 re-ingested verify tokens (PAD-masked writes,
discarded hidden states) are skipped.

### Bit-exact gate (2k canary, bf16 self-draft, compaction ON)

| config | accept | tok/s | ref (P66 shared-KV) |
|---|---|---|---|
| W=0 K=2 | **3.000** (EXACT) | 412.0 | 3.000 / 378.4 (+8.9% tok/s) |
| W=64 K=2 | 2.675 | 211.5 | 2.689 / 200.1 (+5.7% tok/s) |

W=0 K=2 = 3.000 exactly => the compaction is structurally bit-exact (the
appended token reads verify's cached KV regardless of q). W=64 2.675 vs the
P66 shared-KV ref 2.689 is within cross-session variance; the authoritative
same-session A/B is the arm-B 16k accept below (baseline_clean vs
armb_dec_clean, identical stack). +5.7-8.9% tok/s at 2k.

### Arm-B 16k A/B (same-session, W512, fp8 replica)

| point | no compaction | + compaction | delta |
|---|---|---|---|
| K2 accept | 2.888 | 2.848 | **-0.040** |
| K2 tok/s | 261.9 | 265.6 | +1.4% |
| K2 cycle ms | 88.2 | 85.8 | -2.4 |
| K4 accept | 4.594 | 4.524 | **-0.070** |
| K4 tok/s | 301.1 | 308.5 | +2.5% |
| K4 cycle ms | 122.0 | 117.3 | -4.7 |

F/D solve (noisy K4 tps ±24-28): baseline F 54.4 / D 16.9; compaction
F ~54.3 / D ~15.8. The clean K2 point is authoritative: cycle -2.4 ms is the
EXPOSED step-0 forward saving single-node (the 14.6 ms serialized FINE value
mostly hides behind the pipeline). ~2.4 ms/cycle at the cost of -0.04..-0.07
accept.


**The compaction is NOT accept-preserving in the windowed regime.** Bit-exact
only at W=0 (canary 3.000). With a window it perturbs accept (W64 bf16 -0.014;
W512 fp8 -0.040) because it switches draft-1's FA3 kernel from the varlen
(q=K+2) path to the decode (q=1) path -- draft-1 sits at ~93% per-position
acceptance, so a kernel-path ULP shift flips ~1-4% of draft-1 tokens. (vLLM
writes query K/V to cache before attention and the step-0 non-appended slots
are PAD-masked, so the appended token reads the SAME bf16 cached KV in both
paths -- the change is the kernel, not the key set.) The accept drop nearly
cancels the forward saving: net +1.4% tok/s at K2. Per the mission's
"accept unchanged to 3 decimals" bar, the step-0 forward is therefore NOT
accept-losslessly compactable in the operating config.

### FINE re-decomposition (compaction ON, K4 b8)

TODO(fill draft_forward_first after).

## 2-node projection (arm-B b12; 2-node DEFERRED, h106 off-limits)

Established (P66): 2-node arm-B b12 F = 117.1 ms, propose-fixed ~87.7 ms
(cycle 159.7 = verify 29.4 + chain 4x10.6 + propose-fixed 87.7).

Single-node this phase pins the CODE-LOCAL propose-fixed at ~18 ms
(step-0 fwd 14.6 + step0 glue 2.6 + gap cpu 1.2). The step-0 forward is
comm-free (fp8 replica) so it is present at 2-node unchanged (~15 ms). The
CPU/sync glue is ~4 ms single-node AND at 2-node (P65 crushed it). Therefore
the 2-node propose-fixed's remaining **~70 ms is CROSS-NODE** and invisible
single-node:

- verify-side NCCL DP-coordination all_reduce (~8 ms, P65 trace);
- the cycle-boundary DP rendezvous the propose path still runs at DP16
  (arm B skips the DRAFT coord, but the target/verify DP coordination and
  the cross-node num_tokens_across_dp agreement remain);
- cross-node all-gather in the verify input prep at DP16.

Best-case (were the step-0 compaction lossless): -2.4..-4.6 ms ->
2-node F 117 -> ~113, propose-fixed 88 -> ~84 ms. Still > the 78 ms bar.
And it is NOT lossless (accept -0.04..-0.07), so the honest accept-preserving
single-node delta is ~0 ms.

## Verdict: IRREDUCIBLE single-node -> write-up warranted

- The single-node propose-fixed CODE is ~18 ms and near its floor: the CPU/
  sync glue (~4 ms) was already crushed by P65, and the one sizable piece --
  the step-0 draft forward (14.6 ms serialized, ~2.4 ms exposed) -- is NOT
  accept-losslessly compactable (any compaction switches draft-1's FA3 kernel
  and perturbs accept beyond the 0.003 bar).
- Projected 2-node F stays ~113-117 ms (propose-fixed ~84-88 ms), well above
  the ~78 ms crossing bar. The ~70 ms cross-node remainder is the real
  target, and it lives in engine-level DP coordination / cross-node gather --
  NOT the propose path this phase can touch, and only measurable at 2-node.

### 2-node arms to run when h106 frees

1. torch-profiler trace of the 2-node arm-B b12 cycle (rank-0, ~15 cycles):
   attribute the ~70 ms across verify-NCCL / DP-rendezvous / cross-node
   gather (reuse `research/64_window_e2e` trace harness at W7_NODES=2).
2. Engine `--disable-nccl-for-dp-synchronization` A/B on the verify DP
   coordination (untested; the harness must pass it through).
3. If (1) confirms the DP rendezvous dominates: an amortized / one-shot
   cross-node coordination per cycle (the P65 AMORTIZE_DP_COORD idea, applied
   to the verify side).
