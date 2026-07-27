# Phase 92 — live GRPO wiring: real policy drift for the staleness claim

Source phases: 89 (DRAM refresh + emulated-drift RL demo), 91 E6 (GRPO-shaped
rollout benchmark, 1.300x), 75 (EfficientRollout reproduction + decomposition).
Paper anchor: §9 (research/79_paper/paper_draft.md), T9/T9b.

## Objective (the WHY)

C4's one disclosed weakness: our drift is EMULATED (calibrated scale-jitter +
int4 nibble-flip on the drafter). The reference system (EfficientRollout,
arXiv 2606.18967) runs a REAL GRPO loop. This phase closes the gap the cheap
way first: run their released training stack to produce **real per-step policy
checkpoints**, then measure the **real staleness curve** on our stack and
calibrate the phase-89 emulation against it. Stretch: our controller inside
their trainer.

Deliverable sentence for §9: "the drift emulation is calibrated against
measured accept decay over N live GRPO steps (veRL, Qwen2.5-7B, MATH lv.3-5)."

## Their repo, mapped (studied 2026-07-27)

Repo: https://github.com/furiosa-ai/EfficientRollout — veRL v0.7.0 fork +
vendored patched vLLM 0.11.2 (`third_party/vllm/`), torch 2.9.0/cu128,
python 3.10, target 8xA100. Local clone: `/data/smcho/efficientrollout`.

Integration points (the parts that matter for us):

| where | what |
|---|---|
| `verl/workers/rollout/vllm_rollout/vllm_rollout.py:206-242` | builds `SpeculativeConfig(method="quant_self", quant_method=rtn/awq, sd_toggle_mode=roofline, gamma_ladder=[5,7,9,11])` from hydra flags |
| `vllm_rollout.py:312-320` | **the hook**: after each trainer→rollout weight sync (`model.load_weights(weights)`), calls `model_runner.on_weights_updated()` → drafter requantize; `_last_requantize_sec` flows to trainer timing (`fsdp_workers.py:1096-1100`) |
| `third_party/vllm/vllm/v1/spec_decode/quant_self_proposer.py` (1213 loc) | AWQ-native drafter induced from target, **KV sharing with target** (their SHARED_KV analog), `_requantize_from_target()` = FP16 → RTN W4 → in-place `copy_()` into AWQ buffers — same in-place-copy-under-CUDA-graphs trick as our 113 ms DRAM refresh, but sourced GPU-side from the live target |
| `third_party/vllm/vllm/v1/spec_decode/online_quantizer.py` | 3 requant tiers: RTN (sub-second, `quant_interval=1` = every step), fixed-β activation-aware, replay-AWQ every K steps; `ActivationStatsCollector` hooks on q/o/gate/down proj inputs FSDP-side (`fsdp_workers.py:917`) |
| `sd_toggle/` | roofline SD on/off toggle: per-GPU F_eff bench + per-model sweep/fit → `configs/a100_tp1_*.json`; their §4.2 batch gate. A100 configs shipped; H100 would need recalibration (`scripts/calibrate_*.sh`, ~25 min) |
| `scripts/run_qwen2.5_7b_sd.sh` | the E2E run: GRPO, `train_batch_size=128`, group `n=8`, response 8192, T=1 (rollout default), Qwen2.5-7B, TP1, `gpu_memory_utilization=0.6` (colocated FSDP actor+ref+vLLM per GPU), simplerl-8k-hard, 4 arms: no-sd / rtn / toggle / adaptive |

Notable convergences with our stack (worth a paper sentence): in-place weight
copy for graph-safe refresh; drafter/target KV sharing; accept-rate-driven γ
adaptation (`VLLM_GAMMA_AR_THRESHOLD` 0.94/0.85 hysteresis vs our accept-EMA +
asymmetric hysteresis). Divergences: their refresh is UNCONDITIONAL per step
(requant_interval=1, cost 1.3-2.6 s/step booked in trainer timing); ours is
detector-fired (113 ms, only when accept breaks the gate) — that contrast is
now measurable on real drift.

## Design decision (the wiring path)

Their veRL v0.7.0 pins vLLM 0.11.2 internals; our fork is a much newer dev
tree with our scheduler/bandit/skip/refresh work. Porting either direction
wholesale is a multi-week lift. So: staged.

- **Stage 1 (this phase, core)** — run THEIR stack unmodified on GPUs 0,1
  (2xH100, co-tenant box), `no-sd` arm, short horizon (~12-16 GRPO steps,
  `trainer.save_freq=1`, reduced batch to fit 2 GPUs), dump per-step actor
  checkpoints. Optional second arm `rtn` for their live accept/requant-cost
  telemetry on H100 (also feeds the §9.3 hardware-inversion story: their
  toggle should keep SD mostly OFF on H100 per our ridge analysis).
- **Stage 2 (core)** — OUR stack, offline over the checkpoint ladder:
  for step k = 0..N: accept(drafter@step0, policy@stepk) = the REAL staleness
  curve; accept(drafter@stepk, policy@stepk) = the refresh ceiling. Map the
  curve onto the phase-89 eps knob (which eps reproduces step-k's accept drop)
  → the calibration table. Then re-run the E3/E6 refresh demo with
  real-delta drafts instead of nibble-flip.
- **Stage 3 (stretch, assess after stage 2)** — controller-in-trainer: add a
  `quant_interval=∞ + accept-gate` mode to their `OnlineQuantizer`/proposer
  (detector-fired requant, our policy) and compare against their
  unconditional per-step requant on real training steps. This is the
  apples-to-apples controller comparison ON THEIR STACK; their hook
  structure makes it a small patch (`should_requantize()` is already the
  seam).

Why not veRL-on-our-fork first: version drift (0.11.2 APIs vs our dev tree)
is the highest-risk/lowest-information path; stages 1-2 deliver the paper
sentence without it, and stage 3 gets the controller comparison with a
~50-line patch to THEIR seam instead.

## Run plan / budget

- Env: `uv venv --python 3.10` at `/data/smcho/efficientrollout/.venv`;
  torch 2.9.0 cu128; vendored vLLM source build (nvcc 12.8 on box,
  `TORCH_CUDA_ARCH_LIST=9.0`, 192 cores); their pinned flash-attn cp310
  wheel; `pip install -e .` (veRL). All caches on /data.
- Data: `prepare_data_simplerl_8k_hard.sh` (MATH lv.3-5 → parquet).
- Stage-1 run: `n_gpus=2`, `train_batch_size=32`, `ppo_mini_batch_size=32`,
  micro 4/GPU, response 8192, save_freq=1, total ~12-16 steps. Rough cost:
  rollout 32 prompts x8 samples @8k on 2xH100 ≈ tens of minutes/step →
  budget 1-2 days wall on GPUs 0,1 (interruptible; checkpoints land as they
  come; even 6-8 steps is enough for the curve — their Fig accept decay is
  near-linear early).
- Disk: 7B bf16 ckpt ≈ 15 GB x 16 steps ≈ 240 GB → fine on /data (2.9T free);
  prune optimizer state from dumps if veRL allows (model-only save).

## Risks / guards

- Co-tenant box: jhchoi active on 2-5; stale smcho engines hold GPUs 6,7
  (kill blocked by permission mode — user to clear). Stage 1 pinned to 0,1
  via CUDA_VISIBLE_DEVICES.
- veRL colocated FSDP+vLLM at gpu_util 0.6 on 80 GB with 8k responses: their
  config is per-GPU-identical, so 2 GPUs = fewer workers, same memory shape.
- wandb: run with `trainer.logger='["console"]'` (no external logging).
- Their sd_toggle A100 configs do NOT block stage 1 (no-sd / rtn-always-on
  need no toggle config); H100 recalibration only if we want the toggle arm.
- DRAFT flag: all numbers here are draft until multi-seed.
