#!/bin/bash
# A800 冒烟测试：batch=16, n=8, response=4096（单轮优化版）
set -e

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_MODULE_LOADING=LAZY
export TORCHDYNAMO_DISABLE=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export WANDB_MODE=online

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# 清除旧 checkpoint 防止 resume
rm -rf "$PROJECT_DIR/checkpoints/agentic-rl-cloud/a800-smoke-r4096"

python3 scripts/patch_vllm.py

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$PROJECT_DIR/data/grpo_train_a800.parquet" \
  data.val_files="$PROJECT_DIR/data/grpo_train_a800.parquet" \
  data.train_batch_size=16 \
  data.max_prompt_length=1024 \
  data.max_response_length=4096 \
  data.filter_overlong_prompts=True \
  data.truncation=left \
  actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
  actor_rollout_ref.model.lora.rank=16 \
  actor_rollout_ref.model.lora.alpha=32 \
  actor_rollout_ref.model.lora.target_modules=all-linear \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.003 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.enable_activation_offload=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.max_model_len=6144 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path="$PROJECT_DIR/src/reward.py" \
  custom_reward_function.name=compute_score \
  reward.num_workers=4 \
  data.seed=42 \
  trainer.critic_warmup=0 \
  trainer.logger=["console","wandb"] \
  trainer.project_name=agentic-rl-cloud \
  trainer.experiment_name=a800-smoke-r4096 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=999 \
  trainer.test_freq=-1 \
  trainer.total_training_steps=3 \
  trainer.val_before_train=False \
  trainer.resume_mode=disable \
  '+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_P2P_DISABLE="1"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE="1"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_WORKER_MULTIPROC_METHOD="spawn"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_MODULE_LOADING="LAZY"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.HF_ENDPOINT="https://hf-mirror.com"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE="1"'
