#!/bin/bash
# Phase 92 stage 1: real GRPO on the EfficientRollout stack (2xH100),
# dumping per-step HF policy checkpoints for the staleness calibration.
#
# Usage: bash run_stage1_ckpt_dump.sh [no-sd|rtn] [extra hydra overrides...]
#   no-sd  AR rollout baseline (the checkpoint producer; default)
#   rtn    their always-on W4-RTN self-SD arm (H100 telemetry, optional)
set -euo pipefail

MODE=${1:-no-sd}
shift || true

ER=/data/smcho/efficientrollout
PHASE=/data/smcho/self-spec-moe/research/92_live_grpo
CKPT_DIR=${CKPT_DIR:-/data/smcho/ckpts/92_grpo_${MODE}}
mkdir -p "$CKPT_DIR" "$PHASE/logs"

# GPU 4-5: one-time co-tenant grant for this run (user, 2026-07-27)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
export HF_HOME=/data/smcho/huggingface
export TMPDIR=/data/smcho/tmp
export VLLM_LOGGING_LEVEL=INFO
export NGPUS_PER_NODE=2

# 2-GPU downscale of their 8xA100 recipe: same shape per GPU
# (group n=8, 8k response, TP1, colocated FSDP+vLLM), smaller global batch.
# save_freq=1 + hf_model contents -> a full HF policy dump every GRPO step.
cd "$ER"
source .venv/bin/activate
# crashed prior attempts leave ray::WorkerDict zombies pinning the GPUs
# (observed 2026-07-27: 74 GiB held after a wake_up OOM); force-clean first.
ray stop --force >/dev/null 2>&1 || true
sleep 5
# 2-GPU FSDP = 4x per-GPU optimizer shard vs their 8-GPU recipe -> CPU
# offload is mandatory or vLLM wake_up OOMs after the first train step.
# resume_mode=disable: hf_model-only dumps cannot seed a veRL resume.
# dynamic token-budget batching: post-step-1 responses lengthen and a
# fixed micro=8 x ~9k-tok update OOMs (observed at step 2's update).
# NB: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is INCOMPATIBLE with
# vLLM sleep-mode's CuMem pool (asserts at engine boot) -- do not set it here.
bash scripts/run_qwen2.5_7b_sd.sh "$MODE" \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    trainer.resume_mode=disable \
    data.train_batch_size=32 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.checkpoint.save_contents='["hf_model"]' \
    trainer.logger='["console"]' \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.save_freq=1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${TOTAL_STEPS:-16} \
    trainer.n_gpus_per_node=2 \
    "$@" 2>&1 | tee "$PHASE/logs/stage1_${MODE}.log"
