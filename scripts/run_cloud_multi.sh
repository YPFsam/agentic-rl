#!/bin/bash
# ============================================================
# 云端多轮 AgentLoop GRPO（Qwen3-4B，80GB GPU）
# veRL 0.8+ 配置：vLLM rollout + LoRA r=16 + multi_turn
# ============================================================
# !! OOM 风险提醒 (运行前必须修复) !!
# 1. free_cache_engine 未设置 → KV pool 残留 ~40 + actor ~38 = 78 GiB，极度危险
#    修复: 添加 rollout.free_cache_engine=True
# 2. gpu_memory_utilization=0.6 对 3 轮 20K seq 偏大
#    修复: 可降到 0.5
# 修复后峰值预估 (free_cache=True): ~46 GiB / 80 GiB (57%)
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
echo "云端多轮 AgentLoop GRPO 训练"
echo "模型: Qwen/Qwen3-4B"
echo "max_assistant_turns: 3"
echo "max_response_length: 3072"
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
  data.return_raw_chat=True \
  actor_rollout_ref.model.path=Qwen/Qwen3-4B \
  actor_rollout_ref.model.lora.rank=16 \
  actor_rollout_ref.model.lora.alpha=32 \
  actor_rollout_ref.model.lora.target_modules=all-linear \
  actor_rollout_ref.actor.optim.lr=2e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
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
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.max_model_len=20480 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.n=16 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path="$PROJECT_DIR/src/reward_multiturn.py" \
  custom_reward_function.name=compute_score_multiturn \
  data.seed=42 \
  trainer.critic_warmup=0 \
  trainer.logger=["console","wandb"] \
  trainer.project_name=agentic-rl-cloud \
  trainer.experiment_name=qwen3-4b-grpo-multi \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=-1 \
  trainer.total_training_steps=200 \
  trainer.val_before_train=False
