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

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export HF_HOME=/data/smcho/huggingface
export TMPDIR=/data/smcho/tmp
export VLLM_LOGGING_LEVEL=INFO
export NGPUS_PER_NODE=2

# 2-GPU downscale of their 8xA100 recipe: same shape per GPU
# (group n=8, 8k response, TP1, colocated FSDP+vLLM), smaller global batch.
# save_freq=1 + hf_model contents -> a full HF policy dump every GRPO step.
cd "$ER"
source .venv/bin/activate
bash scripts/run_qwen2.5_7b_sd.sh "$MODE" \
    data.train_batch_size=32 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.checkpoint.save_contents='["hf_model"]' \
    trainer.logger='["console"]' \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.save_freq=1 \
    trainer.max_ckpt_to_keep=null \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${TOTAL_STEPS:-16} \
    trainer.n_gpus_per_node=2 \
    "$@" 2>&1 | tee "$PHASE/logs/stage1_${MODE}.log"
