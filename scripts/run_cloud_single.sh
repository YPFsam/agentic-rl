#!/bin/bash
# ============================================================
# 云端单卡 GRPO 训练（Qwen3-4B，80GB GPU）
# veRL 0.8+ 配置：vLLM rollout + LoRA r=16
# ============================================================
# !! OOM 风险提醒 (运行前必须修复) !!
# 1. grad_ckpt=False + 4B + seq=6656 → 激活 ~39 GiB，直接 OOM
#    修复: enable_gradient_checkpointing=True
# 2. gpu_memory_utilization=0.6 → KV pool ~39.5 GiB，sleep mode 下残留 + actor 爆
#    修复: gpu_memory_utilization=0.4 + free_cache_engine=True
# 修复后峰值预估: ~60 GiB / 80 GiB (75%)
set -e

# ---- wandb 在线模式（实时查看训练曲线）----
export WANDB_MODE=online

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

TRAIN_DATA="$PROJECT_DIR/data/grpo_train_full.parquet"
if [ ! -f "$TRAIN_DATA" ]; then
    echo "扩充数据不存在，生成中..."
    python src/data_prepare.py --output "$TRAIN_DATA" --full
fi

echo "=========================================="
echo "云端单轮 GRPO 训练"
echo "模型: Qwen/Qwen3-4B"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
echo "数据: $TRAIN_DATA"
echo "=========================================="

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$TRAIN_DATA" \
  data.train_batch_size=128 \
  data.max_prompt_length=512 \
  data.max_response_length=6144 \
  data.filter_overlong_prompts=True \
  data.truncation=left \
  actor_rollout_ref.model.path=Qwen/Qwen3-4B \
  actor_rollout_ref.model.lora.rank=16 \
  actor_rollout_ref.model.lora.alpha=32 \
  actor_rollout_ref.model.lora.target_modules=all-linear \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=False \
  actor_rollout_ref.model.enable_activation_offload=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.n=16 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path="$PROJECT_DIR/src/reward.py" \
  custom_reward_function.name=compute_score \
  data.seed=42 \
  trainer.critic_warmup=0 \
  trainer.logger=["console","wandb"] \
  trainer.project_name=agentic-rl-cloud \
  trainer.experiment_name=qwen3-4b-grpo-single \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=-1 \
  trainer.total_training_steps=200 \
  trainer.val_before_train=False
