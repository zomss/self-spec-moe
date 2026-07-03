# Phase 39: TEMPERATURE recovery of the comm-free draft at large EP

**Source phase:** 38 (EP-isolation) and 34 (World A system). Phase 38 showed the
comm-free FULL-REPLICA draft accept "collapses" at DP=8/EP=8 under GREEDY (proxy
A=2.18; Qwen3-30B ~1.0 in phase 34) because the comm-free local sum of the MoE
experts diverges in FP-reduction order from the verify's EP all-to-all, flipping
near-tie ARGMAXES. **Every prior measurement was greedy — the worst case for a
tiny tie-flipping perturbation.**

## Hypothesis (the recovery)
Under TEMPERATURE sampling the lossless distributional rejection sampler accepts
at rate `min(1, p_target/p_draft)` and is `1 − TV(draft, verify)` in expectation.
The full-replica draft computes the SAME experts (just summed locally), so
`p_draft ≈ p_target` (small TV) → the FP divergence should be ABSORBED and accept
should RISE toward K+1, while staying LOSSLESS (standard rejection-sampling
guarantee). Greedy is the pathological case (an arbitrarily small logit gap flips
the argmax → reject); temperature smooths it.

## The load-bearing config
The rejection sampler's lossless ratio test needs the DRAFT's probabilities. The
draft only returns them when BOTH:
- `speculative_config.draft_sample_method = "probabilistic"`  (default `"greedy"`)
- `speculative_config.rejection_sample_method = "standard"`   (default; lossless)

Verified in `vllm/v1/spec_decode/llm_base_proposer.py:337-340`
(`_enable_probabilistic_draft_probs`) and `vllm/v1/sample/rejection_sampler.py`
(`rejection_random_sample_kernel`, accept `target_prob/draft_prob >= uniform`).
At `temperature=0` the draft falls back to argmax (`all_greedy` gate) → reproduces
the greedy collapse baseline. No source changes; only the spec config + the
SamplingParams temperature change.

## Setup
Proxy `Qwen/Qwen1.5-MoE-A2.7B` (60 experts top-4), **DP=8 → EP=8**, forced-PCIe
(`NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1`), **bf16** (no FP8
confound), K=4, batch 16, OUTLEN 128. Full-replica comm-free draft
(`DRAFT_FULL_REPLICA=1 DRAFT_LOCAL_ROUTE=1 DRAFT_FULL_CG=1 COMPILE_CONSISTENT=1`),
verify full-EP (`LOCAL_ROUTE=0`). `VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0`.

## Measurements
1. **Temperature sweep** — mean accept_len at temp ∈ {0.0, 0.5, 0.7, 1.0}, same
   prompts/seed. Hypothesis: accept_len rises toward ~4–5.
2. **Losslessness under temperature** — spec vs no-spec at temp 0.7, fixed seed:
   probabilistic-draft path active? standard rejection sampler? per-token output
   agreement (seed-coupled).
3. **If recovered → Qwen3-30B-A3B** DP=8 EP=8 FP8 full-replica: accept_len at temp
   {0, 0.7, 1.0} + **decode tok/s speedup vs no-spec at temp 0.7** + comm-free
   confirmation. Also run PIECEWISE (FULL_CG off) since prior W7 found FULL_CG is
   lossy on Qwen3's FA3 attention.

## Commands
```
bash research/39_temperature_recovery/scripts/run_proxy.sh     # proxy sweep
bash research/39_temperature_recovery/scripts/run_lossless.sh  # losslessness
bash research/39_temperature_recovery/scripts/run_qwen30b.sh   # headline (if recovered)
python research/39_temperature_recovery/scripts/analyze.py {sweep|lossless|q30}
```
Each harness tears down its OWN DP workers between runs (idle GPUs only).

## Output
`results_W7_temperature.md` — the sweep table, losslessness, the verdict.
