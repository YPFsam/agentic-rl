# Agentic-RL 项目指南

## 项目概述
GRPO + 执行反馈的代码生成 RL 训练项目。使用 veRL 0.8 框架，从单轮 GRPO 训练开始，逐步进阶到多轮 AgentLoop 错误纠正。

## 当前阶段
- 基模型 Qwen3-1.7B baseline 评估已完成（HumanEval pass@1: 26.2%）
- **单轮 GRPO 训练完成**（100 步，失败：max_response_length=512 导致全截断，无正 reward）
- **多轮 GRPO 训练完成**（80 步，成功：score/mean=+0.046，62.5% 步有样本通过测试）
- **当前任务**：云端 A800 80GB 训练 Qwen3-1.7B 升级版（多轮，max_response=4800，3轮）
- **KL 修复完成**：FSDP + LoRA 下 `disable_adapter()` 失效导致 kl_loss=0，已强制使用独立 ref 模型

## 关键文件
- `scripts/run_local_single.sh` — 本地/云端 48GB 单轮训练（1.7B 模型）
- `scripts/run_local_multi.sh` — 本地/云端 48GB 多轮训练（1.7B 模型）
- `scripts/run_cloud_single.sh` — 云端 A800 80GB 单轮训练（1.7B 升级版，response=8192）
- `scripts/run_cloud_multi.sh` — 云端 A800 80GB 多轮训练（1.7B 升级版，response=8192，3轮）
- `scripts/run_smoke_test_h20.sh` — H20 96GB 冒烟测试（2轮，response=7200）
- `scripts/strip_optimizer.py` — Checkpoint 优化器剥离（训练后删除旧 ckpt 的 optim）
- `scripts/patch_vllm.py` — vLLM LoRA 权重名 patch（每次切换镜像需重跑）
- `configs/agent_loop.yaml` — 自定义 CodeAgentLoop 注册配置（多轮训练必须）
- `src/reward.py` — 单轮门控奖励函数
- `src/reward_multiturn.py` — 多轮渐进奖励函数（含纠错率 + thinking 指标日志）
- `src/agent_loop.py` — 多轮代码纠错 AgentLoop（含异常样本日志）
- `logs/multiturn_metrics_*.jsonl` — 每样本的多轮指标（两个 Ray worker 各一个文件，需合并分析）
- `logs/anomalous_samples.jsonl` — 异常样本详情（单轮 gen>200s 或 tool>10s）
- `logs/thinking_metrics.jsonl` — 单轮训练的 thinking token 分析
- `changes.md` — 所有修改记录
- `progress.md` — 进度追踪

## veRL 0.8 关键 API（已踩过的坑）
- LoRA: `model.lora.rank` / `model.lora.alpha` / `model.lora.target_modules`（不是 `peft_config.*`）
- Rollout: 只支持 `vllm`/`sglang`/`trtllm`，HF rollout 已移除
- Seed: `data.seed=42`（不是顶层 `seed=42`）
- Rollout mode: sync 已移除，async 是唯一模式
- Ray env 透传: `+ray_kwargs.ray_init.runtime_env.env_vars.KEY="VALUE"`

## vLLM 补丁（必须！）
每次切换 Docker 镜像、重新安装 vLLM、或**克隆 AutoDL 实例**后必须运行：
```bash
python3 scripts/patch_vllm.py
```
（训练脚本已内置自动运行，手动跑时才需要单独执行）
修复四个问题：
1. **QKV 权重名**：LoRA wrapper 重命名参数，vLLM load_weights 找不到
2. **numpy.int64 索引**：`np.cumsum` 返回 int64，PyTorch 2.8 不支持直接索引 Tensor
3. **disable_adapter 方法名**：veRL 调用单数形式，Qwen3 LoRA 接口是复数
4. **numpy _core/ 残留**：Docker 镜像构建时 numpy 2.x 降级到 1.26.4，_core/ 残留 2.x 文件导致 vLLM spawn 子进程崩溃（**每次克隆实例必现**）
5. **显存泄漏补丁**：Patch 5（sleep 后清理 prefix cache）+ Patch 6（步尾 gc）+ Patch 7（_update_actor/update_weights 后 gc.collect）+ Patch 8（每 bucket 后 empty_cache），详见 `changes.md`
   - **禁止在 ray_trainer.py 中调用 torch.cuda.ipc_collect() 或 empty_cache()**——fit() 运行在 CPU-only TaskRunner 进程（`ray.remote(num_cpus=1)`），IPC handles 在 WorkerDict/vLLM 进程中。详见踩坑 23

## 环境变量禁忌
- **绝对不能设置** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`——与 vLLM sleep mode 的 `CuMemAllocator` 不兼容，会导致启动直接崩溃（AssertionError）。详见 `changes.md` 踩坑 21

## 自定义 AgentLoop 注册（多轮训练必须！）
多轮训练使用自定义 `CodeAgentLoop`（`src/agent_loop.py`），必须通过 veRL 的 `agent_loop_config_path` 机制注册：
- 配置文件：`configs/agent_loop.yaml`
- 训练脚本参数：`actor_rollout_ref.rollout.agent.agent_loop_config_path=$PROJECT_DIR/configs/agent_loop.yaml`
- **必须**设置 `PYTHONPATH=$PROJECT_DIR`，否则 Ray worker 无法导入 `src.agent_loop.CodeAgentLoop`
- veRL 内置只有 3 个 agent loop：`single_turn_agent`、`diffusion_single_turn_agent`、`tool_agent`
- **AgentLoopOutput 必须包含 `metrics` 字段**（veRL 新增必填），对照 `single_turn_agent_loop.py` 的返回值
- **自定义 AgentLoop 必须截断 response_ids 到 response_length**：多轮交互的 LLM tokens + feedback tokens 累计可能超出 `max_response_length`，veRL 的 `tokenizer.pad` 只填充不截断，超出会导致 `_postprocess` 中 `torch.cat` tensor size mismatch

## vLLM 预抢占（Preemption）分析（多轮训练必读）

### 根因：KV cache 容量不足 + 多轮放大
- Qwen3-1.7B KV cache: `2 × 8(kv_heads) × 128(head_dim) × 2(fp16) × 28(layers)` = **112 KB/token**
- H20 96GB, gpu_util=0.25 → vLLM 可用 ~19.7 GB → max_model_len=8816 → **~21 个并发序列**
- 实际需求: **64 个并发序列**（8 prompts × 8 GRPO samples）
- 缺口 43 个序列必须排队等待或被抢占（preemption）

### 多轮放大效应
```
单轮: 64 请求分 3 批 (21+21+22) → 正常 ~130s
多轮: Turn 1 完成 → KV cache 保留(prefix caching)
      Turn 2 到达 → 需要新槽位但 Turn 1 KV cache 未释放
      → 可用槽位从 21 进一步下降 → 抢占级联
      → 极端情况: 所有样本互相等待 → 死锁 → 993s（step 15 实测）
```

### 实测数据（H20 冒烟 16 步）
- 正常 gen_max: ~130s（vLLM 分批处理）
- 异常 step 15: gen_max=993s, tool_max=861s（抢占级联 + asyncio 事件循环积压导致沙盒超时回调延迟）
- `num_preempted` 所有步骤报 -1（veRL 未正确追踪此指标）
- 发生概率: 16 步中 1 次严重（6.25%），2 次轻微（step 5/14, tool_max=15s）

### agent_loop.py 动态 max_new_tokens
每轮调整 `max_new_tokens = min(remaining_budget, original_max)`，**不防止抢占**，但：
- 防止生成超出 response_length 的无用 token（省时间）
- 防止多轮累积序列超过 max_model_len（避免 vLLM 直接报错）

### gpu_util 调参空间（H20 96GB）
| gpu_util | KV cache | 并发序列 | 预估峰值 | 风险 |
|----------|---------|---------|---------|------|
| 0.25（当前）| 19.7 GB | 21 | 86.6 GB | 抢占频繁但可控 |
| 0.35 | 29.3 GB | 31 | ~92 GB | 余量 4GB，有 OOM 风险 |
| 0.45 | 38.9 GB | 41 | ~99 GB | 必 OOM |

## 训练脚本显存预算（实测）
- 本地单轮 (1.7B, 48GB): 峰值 ~36 GiB (75%)，param_offload=True + optimizer_offload=True
- 本地多轮 (1.7B, 48GB): 峰值 **43.4 GiB (90%)**，param_offload=False，余量仅 4.6 GiB
- 云端单轮 (1.7B 升级版, A800 80GB): 峰值 **79.6 GiB (97%)** OOM，gpu_util=0.55，需降到 0.4
- 云端多轮 (1.7B 升级版, A800 80GB): 运行中，gpu_util=0.2，response=4800, max_model_len=6400，含独立 ref 模型
- H20 冒烟 (1.7B, 96GB): 峰值 **36.7 GiB (37.5%)**，gpu_util=0.35，余量 55 GiB，可大幅提参
- H20 正式 (1.7B, 96GB): 峰值 **86.6 GiB (88.5%)**，gpu_util=0.25，余量 9 GiB，多轮抢占频繁

## 训练数据（当前）
- **训练集**: `data/grpo_train_multi_full.parquet`（602 条 APPS-easy + MBPP 混合）
- **batch_size=8, n=8**（每步 64 个 GRPO 样本）
- **每 epoch 75 步**，总 250 步 = 3.3 epoch
- **resume_mode=auto**，checkpoint 在 `checkpoints/agentic-rl-cloud/h20-formal-2turn-r4800/`，重启自动续训

## KL 散度修复（踩坑 25）
- **问题**：FSDP + LoRA 下 `disable_adapter()` 无法工作（FSDP flatten 参数后隐藏 LoRA 层）
- **修复**：强制 `ref_in_actor=False`，使用独立 ref 模型
- **修改文件**：`/tmp/verl/verl/trainer/main_ppo.py` + `/tmp/verl/verl/trainer/ppo/ray_trainer.py`
- **显存影响**：额外 ~3.4 GiB ref 模型
- **这是 veRL 框架的通用问题**，FSDP + LoRA + KL loss 用户都会遇到；Megatron 后端不受影响

## 多轮训练指标（step 1-20 实测）
| 指标 | 数值 |
|------|------|
| 首轮成功率 | 12.0% |
| 第2轮纠错率 | 2.1% |
| 第3轮纠错率 | 0.9% |
| NO_CODE（≈截断） | 26.9% |
| kl_loss | ~0.001（稳定非零） |
| 截断率 clip_ratio | 25-39% |
| 负 reward = NO_CODE | 完全一一对应 |

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
bash scripts/run_cloud_single.sh    # 1.7B 单轮升级版（A800 80GB）
bash scripts/run_cloud_multi.sh     # 1.7B 多轮升级版（A800 80GB）

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
- **numpy 降级必须彻底删除**：`rm -rf numpy*` + `pip install`，`--force-reinstall` 不可靠（会残留 2.x 文件）
- **Qwen3-4B 模型路径用本地绝对路径**：ModelScope 下载的 `latest/` 目录和 HF 缓存格式不兼容，用 `HF_HUB_OFFLINE` 会导致 tokenizer 加载失败
- **max_response_length 必须 >= 8192**（APPS-easy thinking 可达 4000+ tokens），之前 512 导致全截断
- 验证已关闭（`test_freq=-1`, `val_before_train=False`），训练后单独评估

## 云端环境（AutoDL）特别说明
- **当前配置**：官方 Docker 镜像 `verlai/verl:vllm011.latest`, Python 3.12.11, PyTorch 2.8.0+cu128, A800 80GB
- **缓存重定向必须**：系统盘仅 30GB，必须重定向到数据盘
- **HuggingFace 镜像**：`HF_ENDPOINT=https://hf-mirror.com`
- **数据盘 100GB**：模型缓存 ~8GB，checkpoint 按需清理，数据 ~1184 条
- **Checkpoint 路径**：`/root/autodl-tmp/checkpoints/agentic-rl-local/`
