#!/bin/bash
# ============================================================
# 本地多轮 AgentLoop GRPO（Qwen3-1.7B，24GB GPU）
# veRL 0.8+ 配置：vLLM rollout + multi_turn
# ============================================================
set -e

# WSL2 兼容性修复：禁用 NCCL P2P/IB 避免 CUDA 内存分配器崩溃
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

TRAIN_DATA="$PROJECT_DIR/data/grpo_train.parquet"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "错误：训练数据不存在，请先运行 python src/data_prepare.py"
    exit 1
fi

echo "=========================================="
echo "本地多轮 AgentLoop GRPO 训练"
echo "模型: Qwen/Qwen3-1.7B"
echo "max_turns: 2"
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
  data.return_raw_chat=True \
  actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
  actor_rollout_ref.model.lora.rank=8 \
  actor_rollout_ref.model.lora.alpha=16 \
  actor_rollout_ref.model.lora.target_modules=all-linear \
  actor_rollout_ref.actor.optim.lr=3e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
  actor_rollout_ref.rollout.max_model_len=1024 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path="$PROJECT_DIR/src/reward_multiturn.py" \
  custom_reward_function.name=compute_score_multiturn \
  data.seed=42 \
  trainer.critic_warmup=0 \
  trainer.logger=["console","wandb"] \
  trainer.project_name=agentic-rl-local \
  trainer.experiment_name=qwen3-1.7b-grpo-multi \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=10 \
  trainer.total_training_steps=100 \
  trainer.val_before_train=True
