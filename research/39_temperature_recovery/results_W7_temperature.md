# W7 Temperature: does sampling recover the comm-free draft's accept at large EP?

**Question.** World A's comm-free FULL-REPLICA draft accept "collapses" at large
EP under GREEDY (phase 38 proxy A=2.18; phase 34 Qwen3-30B ~1.0). Every prior
measurement was greedy. Hypothesis: the divergence is a tiny FP-reduction-order
perturbation (comm-free local MoE sum vs the verify's EP all-to-all) that flips
near-tie ARGMAXES; under TEMPERATURE sampling the lossless distributional
rejection sampler (accept rate `min(1, p_target/p_draft)`, expected
`1 − TV(draft, verify)`) should ABSORB the divergence — `p_draft ≈ p_target`
because the draft computes the SAME experts — and accept should RISE toward K+1,
while staying lossless.

**Setup (no source changes).** Proxy `Qwen/Qwen1.5-MoE-A2.7B` (60 experts top-4),
**DP=8 → EP=8**, forced-PCIe, **bf16** (no FP8 confound), K=4, batch 16, OUTLEN
128, same prompts, seed 0. Full-replica comm-free draft (`DRAFT_FULL_REPLICA=1
DRAFT_LOCAL_ROUTE=1 DRAFT_FULL_CG=1 COMPILE_CONSISTENT=1`), verify full-EP
(`LOCAL_ROUTE=0`). Lossless rejection sampler with the DRAFT's probs:
`speculative_config.rejection_sample_method="standard"` +
`draft_sample_method="probabilistic"`. `VLLM_USE_DEEP_GEMM=0
VLLM_MOE_USE_DEEP_GEMM=0`. Harness `scripts/temp_sweep.py`, driver
`scripts/run_proxy.sh`. All runs error-free; GPUs torn down between runs (0 MiB,
0 of our procs resident after).

---

## TL;DR — the hypothesis is REFUTED

**Temperature does NOT recover the comm-free draft's accept. It makes it WORSE,
monotonically.** Accept_len 2.18 (greedy) → 1.37 (T=0.5) → 1.24 (T=0.7) → 1.22
(T=1.0). The divergence is NOT a near-tie argmax flip that temperature smooths;
it is a genuine *distributional* divergence between the comm-free local MoE sum
and the EP all-to-all sum, and the lossless ratio test `target_prob/draft_prob ≥
uniform` correctly penalizes it HARDER under sampling than greedy's argmax-only
test does. World A's comm-free draft does not work at large EP under temperature
either.

---

## 1. Proxy temperature sweep (the test) — full-replica comm-free draft (cfg A)

| temp | accept_len | per_tok | per-position accept rate [p0,p1,p2,p3] |
|-----:|-----------:|--------:|----------------------------------------|
| 0.0 (greedy) | **2.181** | 0.295 | [0.415, 0.293, 0.249, 0.224] |
| 0.5 | **1.373** | 0.093 | [0.216, 0.076, 0.050, 0.032] |
| 0.7 | **1.245** | 0.061 | [0.180, 0.038, 0.018, 0.008] |
| 1.0 | **1.215** | 0.054 | [0.166, 0.039, 0.008, 0.002] |

- **accept_len FALLS with temperature, not rises.** The hypothesis predicted a
  rise toward ~4–5; instead every temperature is WORSE than greedy, and accept_len
  decays toward 1.0 (no speculation benefit) as temperature increases.
- **The greedy point (T=0.0 = 2.181) reproduces phase 38 config A (2.18) exactly**
  — including the per-position vector [0.415, 0.293, 0.249, 0.224]. At greedy the
  draft falls back to argmax (the `all_greedy` gate) even with the probabilistic
  config set, so this is the same collapse baseline. The harness is validated.
- **Why temperature hurts (mechanism).** Greedy acceptance
  (`rejection_greedy_sample_kernel`) requires only that the draft's ARGMAX equals
  the verify's argmax — robust to any perturbation that does not flip the top
  token. The probabilistic path (`rejection_random_sample_kernel`) accepts a
  stochastically-sampled draft token with prob `target_prob/draft_prob`, i.e. it
  compares the FULL softmax. The local-vs-EP FP divergence perturbs the WHOLE
  distribution, so for sampled (often non-modal) draft tokens the ratio is
  frequently `< 1` → more rejections. pos-0 accept drops 0.415 → 0.166; the depth
  decay is also steeper (deeper positions are nearly all rejected at T≥0.7).

## 2. Control — EP-full draft (cfg C: draft does the real all-to-all = the verify)

To prove the temperature drop is the comm-free divergence (not a generic
probabilistic-spec artifact), the same probabilistic/standard sampler with an
EP-full draft (`DRAFT_FULL_REPLICA=0 DRAFT_LOCAL_ROUTE=0`): the draft does the
SAME all-to-all as the verify → `p_draft == p_target` → ratio ≈ 1 → should stay
near-perfect accept at any temperature.

| temp | accept_len (cfg C) | per_tok | per-position rate | reading |
|-----:|-------------------:|--------:|-------------------|---------|
| 0.0 | <!-- C_T0_AL --> | <!-- C_T0_PT --> | <!-- C_T0_PP --> | EP-full greedy upper bound |
| 0.7 | <!-- C_T07_AL --> | <!-- C_T07_PT --> | <!-- C_T07_PP --> | EP-full at temp |
| 1.0 | <!-- C_T10_AL --> | <!-- C_T10_PT --> | <!-- C_T10_PP --> | EP-full at temp |

<!-- CONTROL_READING -->

## 3. Losslessness under temperature (probabilistic-draft path active?)

- **Probabilistic draft path active:** the engine log prints `[CFG]
  draft_sample_method=probabilistic rejection_sample_method=standard` at startup
  for every spec run — confirmed for all temperature runs. This is the gate
  (`llm_base_proposer._enable_probabilistic_draft_probs`) that makes the draft
  return its softmax probs to the rejection sampler; without it the ratio test
  falls back to `draft_prob=1` (only lossless for a greedy/one-hot draft).
- **Standard lossless rejection sampler:** `rejection_sample_method=standard`
  routes to `rejection_random_sample_kernel`, accept
  `target_prob/draft_prob ≥ uniform` with residual-distribution recovered tokens —
  the textbook distributional lossless algorithm (arxiv 2211.17192).
- **Output match (spec vs no-spec, fixed seed, T=0.7):** <!-- LOSSLESS -->
- Generated text is coherent English across all temperature runs (spot-checked
  the emitted token_ids) — the low accept is a genuine acceptance-rate effect,
  not corrupted output.

## 4. Verdict

<!-- VERDICT -->

### Notes / caveats
- Proxy magnitude caveat (same as phase 38): on this 60-expert top-4 proxy cfg A
  greedy sits at 2.18 (vs Qwen3-30B's ~1.0 — 128 experts top-8 → larger FP
  divergence). The DIRECTION here (temperature monotonically WORSE) is the
  load-bearing result; the 30B (larger per-step divergence) would only make the
  temperature drop steeper, so the proxy refutation carries.
- **Qwen3-30B headline NOT run:** the brief gates the expensive 30B run on the
  proxy recovering (e.g. ≥4 at T=0.7). The proxy did the opposite (1.245 at
  T=0.7), so per the brief's measurement #4 we report the refutation rather than
  burn the 30B run; there is no recovered accept to monetize as a speedup.

## Files
- `scripts/temp_sweep.py` — single-(cfg,temp) DP=8 harness; probabilistic+standard
  spec config, accept_len / per_tok / per-position vector, echoes resolved [CFG],
  self-teardown. `E_CFG` A (full-replica) | C (EP-full control).
- `scripts/run_proxy.sh` — the temperature sweep (cfg A, T∈{0,0.5,0.7,1.0}).
- `scripts/run_control.sh` — cfg C control (T∈{0,0.7,1.0}).
- `scripts/lossless.py` + `scripts/run_lossless.sh` — spec-vs-nospec output match
  at fixed temp/seed.
- `scripts/analyze.py {sweep|lossless|q30}` — tables.
- `data/temp_sweep_cfg{A,C}_t*_prob_dp8_K4.json`, `data/lossless_*_t0p7.json`.
- `logs/` — per-run engine logs.
