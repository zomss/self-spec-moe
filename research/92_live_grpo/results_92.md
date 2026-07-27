# Phase 92 results — REAL GRPO drift vs the stale drafter (DRAFT)

Date: 2026-07-27. Stage 1: EfficientRollout release stack (veRL 0.7 fork +
vendored vLLM 0.11.2) run REAL on 2xH100 (GPUs 4-5, one-time grant), their
exact Qwen2.5-7B GRPO recipe scaled to 2 GPUs: batch 32, group n=8, 8k
response, T=1.0, MATH lv.3-5 (simplerl-8k-hard), lr 5e-7, 16 steps, full HF
policy dump every step (16 x 29 GB). Training completed in 46 min
(150 s/step); the policy genuinely learned (final mean reward 0.359,
max 1.0; KL 4e-4 -> 1.4e-3; grad-norm spikes to 37 at step 8).

Stage 2: OUR stack (fork @ research/self-spec-moe), accept of the fixed
step-0 drafter (RTN W4-sym of base Qwen2.5-7B, the EfficientRollout
quantization set: Linear-only, lm_head ignored) against each policy dump.
64 prompts from the REAL training distribution (train.parquet), K=4,
T=1.0, 2 seeds, target served bf16 (as veRL serves it). Pipeline
validation rows (16-prompt E1) superseded by the standardized E2 rows.

## E2 — the real staleness curve (accept, mean of 2 seeds)

| policy @ step | drafter@0 (stale) | drafter@step (fresh) |
|---|---|---|
| 0  | 3.931 | = (same weights) |
| 1  | 3.929 | - |
| 2  | 3.930 | - |
| 4  | 3.837 | - |
| 8  | 3.951 | 3.834 |
| 12 | 3.863 | - |
| 16 | 3.977 | 3.883 |

Seed spread: paired seeds differ by 0.03-0.15 (seed is the sampling seed;
both seeds agree on every trend below). Per-point sampling noise ~±0.05.

## Findings

1. **Over 16 real GRPO steps, staleness is ZERO within noise.** The stale
   drafter's accept is flat: 3.93 at step 0, 3.98 at step 16 (range
   3.84-3.98, no monotone trend). At the reference recipe's real drift
   rate (KL ~1e-3/step at lr 5e-7), a training-free W4 self-drafter does
   NOT go stale on this horizon.
2. **Per-step re-quantization buys nothing here — it is pure overhead.**
   The reference system re-quantizes the drafter EVERY step (their booked
   cost: 1.3-2.6 s/step). The fresh drafter@8/@16 measures slightly BELOW
   the stale drafter@0 (-0.08 +- 0.05; both seeds, both steps — possibly
   the fp32-dump RTN path vs the bf16-hub RTN path, or genuinely neutral
   requant). There is no horizon-16 regime where the refresh pays.
3. **The detector-fired design is validated by silence.** Our phase-89
   controller fires refresh at accept < gate (3.6 in E3). The real curve
   never drops below 3.83 -> the detector correctly never fires -> zero
   refresh cost, correctly. The reference design pays ~1.3-2.6 s x 16
   steps for the same accept.
4. **Calibration of the phase-89 emulation.** Real 16-step GRPO drift maps
   to eps ~<= 0.04 on the phase-89 noise knob (the smallest calibrated
   level, itself near-lossless). The phase-89/E6 drift arms (eps
   0.25/0.35, accept cliffs down to -45%) are therefore ADVERSARIAL
   drift — far-off-policy regimes (long horizons, higher lr, epoch-scale
   distribution shift), not per-step reality at this recipe. The paper's
   staleness-bound claim (§9.1) should be framed as robustness against
   the adversarial regime, with the REAL rate measured here as the base
   case.
5. Sanity anchors: accept(drafter@0, base) = 4.10 at K=4 on 16 prompts /
   3.93 on 64 prompts — consistent with EfficientRollout's published
   tau band (3.59 @ gamma=3, 5.18 @ gamma=5, their Tab. 4, A100, b1 2k).

## Scope / caveats (honest ledger)

- 16 steps = 512 prompts ~ 0.06 epoch of their 2-epoch recipe. Long-
  horizon drift (100s of steps) NOT yet measured — their Table-2 gains
  are over full training. Extension = rerun stage 1 with
  TOTAL_STEPS=128+ (~5.5 h on 2 GPUs) if the long-horizon tail is wanted.
- 2 sampling seeds, one training run, one box (H100; their hardware is
  A100). Drafter is RTN W4-sym (their Tier-0; not our production
  W4A8-GPTQ column).
- Stage-1 recipe deltas from theirs: 2 GPUs (FSDP offload + dynamic
  token-budget batching added — memory-shape only), global batch 32 vs
  128. Per-GPU rollout shape identical.
- Their release repo bugs found: verl/trainer/config/data/ missing
  (gitignore `data/` swallowed it; restored from upstream v0.7.0).

## Artifacts

- Curve: data/e1_curve.jsonl (E1 pipeline rows + E2 standardized rows)
- Policy dumps: /data/smcho/ckpts/92_grpo_no-sd/global_step_{1..16}
- Drafters: ~/ckpts/Qwen2.5-7B-W4A16-INT4-{sym,asym}, W8A16-INT8-sym,
  ~/ckpts/92-drafter-step{8,16}-W4A16-INT4-sym
- Logs: logs/stage1_no-sd.log (training), logs/e2_sweep.log
- Scripts: scripts/run_stage1_ckpt_dump.sh, e1_staleness_curve.py,
  run_e2_final_sweep.sh, make_fresh_drafter.sh
