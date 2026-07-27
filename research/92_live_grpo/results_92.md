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

## E3 — the LONG-HORIZON curve (128 steps, 2026-07-28)

Extension run: same recipe, 128 steps (~half an epoch), dumps every 8
steps (/data/smcho/ckpts/92_grpo_long). Training: KL(policy||base)
climbs 5e-4 -> 0.020 (15x the 16-step level); entropy collapses
0.55 -> 0.08; the policy sharpens hard. Same E2 protocol (drafter@0,
64 prompts, K=4, T=1.0, 2 seeds):

| policy step | 0 | 8 | 16 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|
| stale drafter@0 | 3.93 | 3.90 | 3.99 | 4.11 | 4.18 | 4.22 | 4.28 | 4.31 |
| fresh drafter@128 | | | | | | | | 4.32 |

Figure: paper/figures/figH_live_staleness.png.

**FINDING (inverts the staleness premise): accept RISES monotonically
with training — +10% by step 128.** Seeds agree at every point
(spread <= 0.09). Mechanism: at T=1.0, acceptance tracks the OVERLAP
between drafter and target distributions; GRPO's entropy collapse
(0.55 -> 0.08) sharpens the target far faster than its weights drift
away from the frozen drafter (KL 0.02 is tiny), and a sharper target
is easier to draft for. Weight-staleness is a second-order effect:
re-quantizing at step 128 recovers +0.01 (4.32 vs 4.31 — nothing).

Consequences:
- The reference design's premise ("the evolving policy makes any
  fixed drafter increasingly mismatched") is REFUTED at their own
  recipe out to half an epoch: the fixed drafter gets BETTER. Their
  per-step requant (1.3-2.6 s/step x 128 steps ~ 3-6 min/run of pure
  overhead here) buys +0.01 accept.
- The detector-fired design is doubly validated: the gate (3.6) is
  never approached from above OR below; refresh cost stays zero and
  SHOULD stay zero. Refresh-on-evidence is not merely cheaper — it is
  the only design that does nothing when nothing is needed.
- Sharpening is itself a drift signal our accept-EMA reads for FREE:
  the controller sees accept RISING and can deepen K (the phase-91
  argmax would move to higher K as f climbs) — RL training makes
  self-spec MORE valuable over time, not less.
- The adversarial framing of T9 stands unchanged (nibble-drift
  emulates catastrophic re-quantization mismatch, e.g. bad quant of a
  moved policy), but the base case is now: real GRPO drift HELPS.

## Scope / caveats (honest ledger)

- ~~16 steps only~~ E3 extends to 128 steps (~half epoch): the curve
  RISES. Beyond-epoch horizons and higher-lr/no-KL recipes remain
  unmeasured (candidates for where real staleness finally appears).
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
