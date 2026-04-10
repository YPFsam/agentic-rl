#!/bin/bash
# ============================================================
# 云端多轮 AgentLoop GRPO（Qwen3-1.7B，A800 80GB）
# veRL 0.8+ 配置：vLLM rollout + LoRA r=16 + 3 turns + KL loss
# ============================================================
# 从 4B 方案降级到 1.7B 的升级版（含金量对齐）
# 升级点 vs 旧 1.7B 多轮(48GB):
#   数据 464→1185(MBPP+APPS-easy), batch 16→48, n 4→8, response 2048→8192
#   LoRA r=8→16, 轮次 2→3, 新增 KL loss, steps 80→200
# 显存预估: ~60 GiB / 80 GiB (75%)
# 时间预估: ~7 min/step × 200步 ≈ 23小时 ≈ 140元
# Epochs: 200 × 48 / 1185 ≈ 8.1
set -e

# ---- 运行环境兼容性修复 ----
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_MODULE_LOADING=LAZY
export TORCHDYNAMO_DISABLE=1

# ---- HuggingFace 镜像（AutoDL 网络受限）----
export HF_ENDPOINT=https://hf-mirror.com

# ---- wandb 在线模式 ----
export WANDB_MODE=online

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ---- vLLM + numpy 补丁（每次克隆实例后必须运行）----
python3 scripts/patch_vllm.py

# ---- Checkpoint 保存到数据盘 ----
CKPT_DIR="/root/autodl-tmp/checkpoints"
mkdir -p "$CKPT_DIR"
ln -sf "$CKPT_DIR" "$PROJECT_DIR/checkpoints"

TRAIN_DATA="$PROJECT_DIR/data/grpo_train_multi_full.parquet"
if [ ! -f "$TRAIN_DATA" ]; then
    echo "多轮扩充数据不存在，生成中..."
    python src/data_prepare.py --output "$TRAIN_DATA" --full --apps-easy --multiturn
fi

echo "=========================================="
echo "云端多轮 AgentLoop GRPO 训练（1.7B 升级版）"
echo "模型: Qwen/Qwen3-1.7B"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
echo "max_assistant_turns: 3"
echo "max_response_length: 8192 (2730/轮)"
echo "数据: $TRAIN_DATA (1185 MBPP+APPS-easy)"
echo "=========================================="

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$TRAIN_DATA" \
  data.train_batch_size=48 \
  data.max_prompt_length=1024 \
  data.max_response_length=8192 \
  data.filter_overlong_prompts=True \
  data.truncation=left \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
  actor_rollout_ref.model.lora.rank=16 \
  actor_rollout_ref.model.lora.alpha=32 \
  actor_rollout_ref.model.lora.target_modules=all-linear \
  actor_rollout_ref.actor.optim.lr=2e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=24 \
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
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.max_model_len=12288 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path="$PROJECT_DIR/src/reward_multiturn.py" \
  custom_reward_function.name=compute_score_multiturn \
  reward.num_workers=4 \
  data.seed=42 \
  trainer.critic_warmup=0 \
  trainer.logger=["console","wandb"] \
  trainer.project_name=agentic-rl-cloud \
  trainer.experiment_name=qwen3-1.7b-grpo-multi-upgraded \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.max_actor_ckpt_to_keep=4 \
  trainer.test_freq=-1 \
  trainer.total_training_steps=200 \
  trainer.resume_mode=auto \
  trainer.val_before_train=False \
  '+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_P2P_DISABLE="1"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE="1"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_WORKER_MULTIPROC_METHOD="spawn"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_MODULE_LOADING="LAZY"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.HF_ENDPOINT="https://hf-mirror.com"'
