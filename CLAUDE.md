# Agentic-RL 项目指南

## 项目概述
GRPO + 执行反馈的代码生成 RL 训练项目。使用 veRL 0.8 框架，从单轮 GRPO 训练开始，逐步进阶到多轮 AgentLoop 错误纠正。

## 当前阶段
- 基模型 Qwen3-1.7B baseline 评估已完成（HumanEval pass@1: 26.2%）
- veRL 0.8 API 已迁移完成，4 个训练脚本已验证
- **当前任务**：在云端 GPU 上运行 GRPO 训练，验证 loss 下降、reward 上升

## 关键文件
- `scripts/run_local_single.sh` — 本地/云端 24GB 单轮训练（1.7B 模型）
- `scripts/run_cloud_single.sh` — 云端 80GB 单轮训练（4B 模型）
- `scripts/run_cloud_multi.sh` — 云端 80GB 多轮训练（4B 模型）
- `src/reward.py` — 单轮门控奖励函数
- `src/reward_multiturn.py` — 多轮渐进奖励函数
- `changes.md` — 所有修改记录
- `progress.md` — 进度追踪

## veRL 0.8 关键 API（已踩过的坑）
- LoRA: `model.lora.rank` / `model.lora.alpha` / `model.lora.target_modules`（不是 `peft_config.*`）
- Rollout: 只支持 `vllm`/`sglang`/`trtllm`，HF rollout 已移除
- Seed: `data.seed=42`（不是顶层 `seed=42`）
- Rollout mode: sync 已移除，async 是唯一模式
- Ray env 透传: `+ray_kwargs.ray_init.runtime_env.env_vars.KEY="VALUE"`

## 训练脚本显存预算
- 本地单轮 (1.7B, 24GB): 需开启 param_offload + fp8
- 云端单轮 (4B, 80GB): micro_batch=2, ckpt关, offload关 → 48.2GB (60%)
- 云端多轮 (4B, 80GB): micro_batch=2, ckpt开, offload关 → 34.7GB (43%)
  - 多轮 seq_len=20,144（3轮累积），不开 ckpt 会 OOM

## 常用命令
```bash
# 环境搭建
bash scripts/setup_cloud.sh

# 准备训练数据
python src/data_prepare.py

# 训练
bash scripts/run_local_single.sh    # 1.7B 单轮
bash scripts/run_cloud_single.sh    # 4B 单轮

# 训练可视化（需先 wandb login）
# 脚本已配置 wandb，自动上传

# Baseline 评估
python src/evaluate.py --model Qwen/Qwen3-1.7B --benchmark humaneval
```

## 注意事项
- 代码注释用中文，commit message 用中文
- veRL 安装：`git clone https://github.com/volcengine/verl.git /tmp/verl && cd /tmp/verl && pip install -e .`
- 数据文件 `data/grpo_train.parquet` 不在 git 中，需运行 `data_prepare.py` 生成
- WSL2 本地无法训练（vLLM CUDA bug），只能在云端原生 Linux 上跑
