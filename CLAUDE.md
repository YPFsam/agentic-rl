# Agentic-RL 项目指南

## 项目概述
GRPO + 执行反馈的代码生成 RL 训练项目。使用 veRL 0.8 框架，从单轮 GRPO 训练开始，逐步进阶到多轮 AgentLoop 错误纠正。

## 当前阶段
- 基模型 Qwen3-1.7B baseline 评估已完成（HumanEval pass@1: 26.2%）
- **单轮 GRPO 训练完成**（100 步，失败：max_response_length=512 导致全截断，无正 reward）
- **多轮 GRPO 训练完成**（80 步，成功：score/mean=+0.046，62.5% 步有样本通过测试）
- **当前任务**：用多轮 checkpoint 跑 HumanEval 评估，量化训练效果

## 关键文件
- `scripts/run_local_single.sh` — 本地/云端 48GB 单轮训练（1.7B 模型）
- `scripts/run_local_multi.sh` — 本地/云端 48GB 多轮训练（1.7B 模型）
- `scripts/run_cloud_single.sh` — 云端 80GB 单轮训练（4B 模型，⚠️ OOM 风险）
- `scripts/run_cloud_multi.sh` — 云端 80GB 多轮训练（4B 模型，⚠️ OOM 风险）
- `scripts/patch_vllm.py` — vLLM LoRA 权重名 patch（每次切换镜像需重跑）
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

## vLLM 补丁（必须！）
每次切换 Docker 镜像或重新安装 vLLM 后必须运行：
```bash
python3 scripts/patch_vllm.py
```
修复两个问题：
1. **QKV 权重名**：LoRA wrapper 重命名参数，vLLM load_weights 找不到
2. **numpy.int64 索引**：`np.cumsum` 返回 int64，PyTorch 2.8 不支持直接索引 Tensor

## 训练脚本显存预算（实测）
- 本地单轮 (1.7B, 48GB): 峰值 ~36 GiB (75%)，param_offload=True + optimizer_offload=True
- 本地多轮 (1.7B, 48GB): 峰值 **43.4 GiB (90%)**，param_offload=False，余量仅 4.6 GiB
- 云端单轮 (4B, 80GB): ⚠️ 预估 OOM，需先修复（开 gradient_checkpointing + free_cache_engine）
- 云端多轮 (4B, 80GB): ⚠️ 预估 OOM，需先修复

## 训练结果（1.7B 模型，RTX 4090 48GB）
| 指标 | 单轮 100 步 | 多轮 80 步 |
|------|-----------|-----------|
| score/mean | -0.02 | **+0.046** |
| 正 reward 步占比 | 0% | **62.5%** |
| response_length/mean | 512（全截断） | **1489** |
| clip_ratio | 1.00 | **0.40** |
| entropy 均值 | 0.22 | **0.083** |

## 常用命令
```bash
# 环境搭建（云端）
export HF_HOME=/root/autodl-tmp/hf_cache
export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache
export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
export TORCH_HOME=/root/autodl-tmp/torch_cache
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
export HF_ENDPOINT=https://hf-mirror.com

# vLLM 补丁（每次切换镜像后必须运行）
python3 scripts/patch_vllm.py

# 准备训练数据
python src/data_prepare.py

# 单元测试
python -m pytest tests/test_sandbox.py -v
python -m pytest tests/test_reward.py -v
python -m pytest tests/test_reward_multiturn.py -v

# 训练（需 GPU + wandb login）
bash scripts/run_local_single.sh    # 1.7B 单轮（48GB GPU）
bash scripts/run_local_multi.sh     # 1.7B 多轮（48GB GPU）
bash scripts/run_cloud_single.sh    # 4B 单轮（80GB GPU，⚠️ OOM 风险）
bash scripts/run_cloud_multi.sh     # 4B 多轮（80GB GPU，⚠️ OOM 风险）

# GPU 显存监控（后台运行）
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 5 > logs/gpu_monitor.csv &

# 训练可视化
# wandb 已配置 online 模式，训练时自动上传
# 训练后评估
python src/evaluate.py --model <checkpoint_path> --benchmark humaneval
```

## 注意事项
- 代码注释用中文，commit message 用中文
- veRL 安装：`git clone https://github.com/volcengine/verl.git /tmp/verl && cd /tmp/verl && pip install -e .`
- 数据文件 `data/grpo_train.parquet` 不在 git 中，需运行 `data_prepare.py` 生成
- WSL2 本地无法训练（vLLM CUDA bug），只能在云端原生 Linux 上跑
- **不要在训练环境安装调试工具**（如 uncompyle6），会污染 numpy
- **numpy 必须保持 1.26.4**（veRL 官方版本）
- **max_response_length 必须 >= 2048**，512 会导致全截断无法训练
- 验证已关闭（`test_freq=-1`, `val_before_train=False`），训练后单独评估

## 云端环境（AutoDL）特别说明
- **当前配置**：官方 Docker 镜像 `verlai/verl:vllm011.latest`, Python 3.12.11, PyTorch 2.8.0+cu128, RTX 4090 48GB
- **缓存重定向必须**：系统盘仅 30GB，必须重定向到数据盘
- **HuggingFace 镜像**：`HF_ENDPOINT=https://hf-mirror.com`
- **数据盘 100GB**：模型缓存 ~8GB，checkpoint ~42GB（保留 2 个），剩余 ~50GB
- **Checkpoint 路径**：`/root/autodl-tmp/checkpoints/agentic-rl-local/`
