# Phase 67 — shrink the per-cycle propose-fixed block

Source: Phase 66 (`results_shared_kv.md`). Established config: shared-KV +
fp8 comm-free replica draft + the Phase-65 flag stack + W512. At 2-node 16k
b12 the self-spec cycle = 159.7 ms = verify 29.4 + chain 4x10.6 +
**propose-fixed ~87.7 ms**. The chain marginal D is at its GPU floor; the E2E
win is now gated by this once-per-cycle fixed block.

Objective: find what composes the ~88 ms propose-fixed block and cut it
(CODE optimization; single-node profiling + validation; 2-node confirmation
DEFERRED). Target: fixed block <= ~75 ms (2-node projection crosses 1.0x);
stretch <= ~65 ms.

## Method (single-node, h107 DP8/EP8, 16k)

1. Reproduce arm-B single-node (`W7_NODES=1 W7_LOCAL_WORLD=8`), solve
   `cycle = F + K*D` from a K=2/K=4 pair (`scripts/run_prof.sh TAG K B off`).
2. Decompose F with the `SelfSpecProfiler` FINE regions
   (`VLLM_SELF_SPEC_PROFILE=1 VLLM_SELF_SPEC_PROFILE_FINE=1`,
   `scripts/analyze_prof.py`): verify / step0_* / cpu_* gap / marginal.
3. Attack the dominant component, keep accept bit-exact.

## Finding

Single-node propose-fixed = ~18 ms: **step-0 draft forward 14.6 ms**
(the K+1 re-ingested verify tokens are wasted FLOPs under shared KV) + CPU
glue/gap ~3.8 ms (already crushed by P65). The ~88 ms 2-node propose-fixed is
therefore dominated by CROSS-NODE effects (DP-coordination rendezvous +
cross-node gather + verify-side NCCL) that do NOT appear single-node.

## Fix

`VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1` (default off): compact the step-0
draft forward to a q=1 decode of only the appended sampled token per request.
Bit-exact (appended token reads verify's cached target-exact KV whether the
draft is bf16 or fp8; the other K+1 tokens' writes are PAD-masked and their
hidden states discarded). Gate: canary W=0 K=2 accept must stay 3.000.

## Validation ladder

1. `scripts/run_canary.sh` — 2k canary, step0-decode ON: W=0 K=2 -> 3.000
   (bit-exact gate), W=64 -> ~2.69 (P66 shared-KV ref).
2. `scripts/run_prof.sh armb_dec_clean 2,4 8 off ... STEP0_DECODE=1` — arm-B
   16k F/D with the compaction; accept must match baseline (2.888 / 4.594).
3. `scripts/run_prof.sh armb_dec_fine 4 8 1 ... STEP0_DECODE=1` — confirm
   `draft_forward_first` dropped toward a chain-step cost.

Results: `results_propose_fixed.md`. Data: `data/` (harness JSONs +
`prof_*/` profiler dumps); `logs/` on disk only.
