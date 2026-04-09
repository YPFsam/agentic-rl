#!/bin/bash
# ============================================================
# 本地单卡 GRPO 训练（Qwen3-1.7B，12GB GPU）
# veRL 0.8+ 配置：vLLM rollout + LoRA + sleep mode + fp8
# WSL2 兼容：NCCL 禁用 + Ray env 透传
# ============================================================
set -e

# ---- WSL2 兼容性修复 ----
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_MODULE_LOADING=LAZY
export TORCHDYNAMO_DISABLE=1

# ---- HuggingFace 镜像 + 离线模式（AutoDL 网络受限）----
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

# ---- wandb 在线模式（实时查看训练曲线）----
export WANDB_MODE=online

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

TRAIN_DATA="$PROJECT_DIR/data/grpo_train.parquet"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "错误：训练数据不存在，请先运行 python src/data_prepare.py"
    exit 1
fi

echo "=========================================="
echo "本地单轮 GRPO 训练"
echo "模型: Qwen/Qwen3-1.7B"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
echo "数据: $TRAIN_DATA"
echo "=========================================="

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$TRAIN_DATA" \
  data.train_batch_size=16 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=left \
  actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
  actor_rollout_ref.model.lora.rank=8 \
  actor_rollout_ref.model.lora.alpha=16 \
  actor_rollout_ref.model.lora.target_modules=all-linear \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.max_model_len=1024 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path="$PROJECT_DIR/src/reward.py" \
  custom_reward_function.name=compute_score \
  reward.num_workers=2 \
  data.seed=42 \
  trainer.critic_warmup=0 \
  trainer.logger=["console","wandb"] \
  trainer.project_name=agentic-rl-local \
  trainer.experiment_name=qwen3-1.7b-grpo-single \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.test_freq=-1 \
  trainer.total_training_steps=100 \
  trainer.resume_mode=auto \
  trainer.val_before_train=False \
  '+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_P2P_DISABLE="1"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE="1"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_WORKER_MULTIPROC_METHOD="spawn"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_MODULE_LOADING="LAZY"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.HF_ENDPOINT="https://hf-mirror.com"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE="1"'
