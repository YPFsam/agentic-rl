#!/bin/bash
# ============================================================
# H20 96GB 正式训练：多轮 AgentLoop GRPO + 过程奖励（Qwen3-1.7B）
# veRL 0.8+ 配置：vLLM rollout + LoRA r=16 + 2 turns + KL loss + process reward
# ============================================================
# 基于 run_cloud_multi.sh，修改：
#   algorithm.adv_estimator=grpo_process_outcome
#   custom_reward_function.name=compute_score_process_multiturn
# 降低显存压力配置：
#   gpu_util=0.2（进一步降低 KV cache 预留）
#   response=4800, max_model_len=6400
#   batch=8, n=8, 2 turns
#   300 steps, 4 epochs (602 samples / batch=8 ≈ 75 steps/epoch)
#   每 50 步保存 checkpoint，只保留最近 3 个
#   支持 resume_mode=auto 断点续训
set -e

# ---- 运行环境兼容性修复 ----
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_MODULE_LOADING=LAZY
export TORCHDYNAMO_DISABLE=1

# ---- HuggingFace 镜像（AutoDL 网络受限）----
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

# ---- wandb 在线模式 ----
export WANDB_MODE=online

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# 确保 Ray worker 能导入 src.agent_loop.CodeAgentLoop
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# 清理 __pycache__：Ray worker 会缓存旧 .pyc，必须每次启动前清理
rm -rf "$PROJECT_DIR/src/__pycache__"

# ---- vLLM + numpy 补丁（每次克隆实例后必须运行）----
python3 scripts/patch_vllm.py

# ---- 注册自定义 advantage estimator ----
# 在 reward_multiturn.py 顶部自动导入注册，无需额外操作

# ---- Checkpoint 保存到数据盘 ----
CKPT_DIR="/root/autodl-tmp/checkpoints"
mkdir -p "$CKPT_DIR"
ln -sf "$CKPT_DIR" "$PROJECT_DIR/checkpoints"

TRAIN_DATA="$PROJECT_DIR/data/grpo_train_multi_full.parquet"
if [ ! -f "$TRAIN_DATA" ]; then
    echo "多轮扩充数据不存在，生成中..."
    python src/data_prepare.py --output "$TRAIN_DATA" --full --apps-easy --multiturn
fi

# ---- GPU 显存监控 ----
mkdir -p "$PROJECT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 5 > "$PROJECT_DIR/logs/h20_formal_gpu_${TIMESTAMP}.csv" &
GPU_MONITOR_PID=$!

echo "=========================================="
echo "H20 正式训练：多轮 AgentLoop GRPO + 过程奖励"
echo "模型: Qwen/Qwen3-1.7B"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
echo "max_assistant_turns: 2"
echo "max_response_length: 4800"
echo "max_model_len: 6400"
echo "gpu_memory_utilization: 0.2"
echo "数据: $TRAIN_DATA (605 samples)"
echo "steps: 300, epochs: 4"
echo "save_freq: 50, max_ckpt_keep: 3"
echo "resume_mode: auto"
echo "=========================================="

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo_process_outcome \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$TRAIN_DATA" \
  data.train_batch_size=8 \
  data.max_prompt_length=1024 \
  data.max_response_length=4800 \
  data.filter_overlong_prompts=True \
  data.truncation=left \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
  actor_rollout_ref.model.lora.rank=16 \
  actor_rollout_ref.model.lora.alpha=32 \
  actor_rollout_ref.model.lora.target_modules=all-linear \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
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
  actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.max_model_len=6400 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$PROJECT_DIR/configs/agent_loop.yaml" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path="$PROJECT_DIR/src/reward_multiturn.py" \
  custom_reward_function.name=compute_score_process_multiturn \
  reward.num_workers=4 \
  data.seed=42 \
  trainer.critic_warmup=0 \
  trainer.logger=["console","wandb"] \
  trainer.project_name=agentic-rl-cloud \
  trainer.experiment_name=h20-process-reward-v1 \
  trainer.default_local_dir="$PROJECT_DIR/checkpoints/agentic-rl-cloud/h20-process-reward" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.max_actor_ckpt_to_keep=3 \
  trainer.max_critic_ckpt_to_keep=3 \
  trainer.test_freq=-1 \
  trainer.total_training_steps=300 \
  trainer.total_epochs=4 \
  trainer.val_before_train=False \
  trainer.resume_mode=auto \
  '+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_P2P_DISABLE="1"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE="1"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_WORKER_MULTIPROC_METHOD="spawn"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_MODULE_LOADING="LAZY"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.HF_ENDPOINT="https://hf-mirror.com"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE="1"'

# 训练结束后停止 GPU 监控
kill $GPU_MONITOR_PID 2>/dev/null || true
echo "=== H20 过程奖励训练完成 ==="
