# Agentic-RL: 基于执行反馈的代码生成 GRPO 强化学习训练

> 使用 veRL 0.8 框架，对 Qwen3-1.7B 进行单轮/多轮 GRPO 训练，通过代码执行反馈实现自纠错能力的学习。

## 项目动机

大语言模型在代码生成任务中经常产生看似合理但实际有 bug 的代码。本项目探索一条路径：让模型在训练中**自动执行自己生成的代码**，根据执行结果获得 reward，通过 GRPO 强化学习逐步提升代码正确率，并进一步引入**多轮纠错机制**（AgentLoop），让模型学会根据错误反馈修正代码。

## 技术架构

### 训练框架

```
                    ┌─────────────────────────────────┐
                    │        GRPO Trainer (veRL)        │
                    │   PPO loss + KL penalty (0.003)   │
                    └──────────┬──────────┬─────────────┘
                               │          │
                    ┌──────────▼──┐  ┌────▼──────────┐
                    │ Actor + LoRA │  │  Ref Model    │
                    │  ( trainable)│  │  ( frozen)    │
                    │   rank=16    │  │               │
                    └──────┬──────┘  └───────────────┘
                           │
              ┌────────────▼────────────┐
              │     vLLM Rollout        │
              │  (async, sleep mode)    │
              │  gpu_util=0.20          │
              │  max_model_len=6400     │
              └────────────┬────────────┘
                           │
            ┌──────────────▼───────────────┐
            │   CodeAgentLoop (自定义)      │
            │   多轮交互: generate → exec   │
            │   → feedback → regenerate    │
            │   max_turns=3, budget=4800   │
            └──────┬───────────────────────┘
                   │                    │
          ┌────────▼────────┐  ┌────────▼────────┐
          │  evalplus 沙盒   │  │  reward 函数     │
          │  untrusted_check │  │  稀疏: +1/-1/0   │
          └─────────────────┘  └─────────────────┘
```

### 核心组件

| 组件 | 文件 | 功能 |
|------|------|------|
| 多轮 AgentLoop | `src/agent_loop.py` | 自定义 veRL AgentLoop，实现代码生成→执行→反馈→纠错的循环 |
| 单轮奖励函数 | `src/reward.py` | 代码执行 pass=+1, fail/syntax=0, no_code=-1 |
| 多轮奖励函数 | `src/reward_multiturn.py` | 稀疏奖励：T1通过=1.0, T2通过=0.85, T3通过=0.70, 失败=0, NO_CODE=-1 |
| evalplus 沙盒 | `src/evalplus_sandbox.py` | 直接调用 evalplus 的 `untrusted_check()`，100% 对齐 `evalplus.evaluate` 判卷 |
| 旧沙盒 | `src/sandbox.py` | 训练时用的同步沙盒（subprocess 隔离） |
| 数据准备 | `src/data_prepare.py` | MBPP(train+val) + APPS-easy 混合，去重后 602 条 |
| vLLM 补丁 | `scripts/patch_vllm.py` | 修复 LoRA 权重名、numpy int64、显存泄漏等 8 个问题 |

### 关键设计决策

1. **独立 Ref 模型解决 KL 失效**：FSDP + LoRA 下 `disable_adapter()` 无法工作（FSDP flatten 参数后隐藏 LoRA 层），强制使用独立 ref 模型计算 KL 散度，额外占用 ~3.4 GiB 显存
2. **evalplus 判卷对齐**：评估沙盒直接导入 evalplus 的 `untrusted_check()`，确保与 `python -m evalplus.evaluate` 100% 一致
3. **错误反馈脱敏**：traceback 中的 assert 行包含完整测试输入和期望输出，替换为 `assert <test_case>`，防止模型记忆测试用例
4. **response_mask 机制**：LLM 生成的 token mask=1（参与梯度），环境反馈 mask=0（不参与），模型不应学习生成错误信息

## 训练配置

### 超参数

| 参数 | 单轮 GRPO | 多轮 GRPO |
|------|----------|----------|
| 基模型 | Qwen3-1.7B | Qwen3-1.7B |
| LoRA rank / alpha | 16 / 32 | 16 / 32 |
| 学习率 | 1e-6 | 1e-6 |
| batch_size / GRPO n | 8 / 8 | 8 / 8 |
| response_length | 4800 | 4800 |
| max_model_len | 6400 | 6400 |
| KL coef | 0.003 | 0.003 |
| max_turns | 1 | 3 |
| gpu_util (vLLM) | 0.20 | 0.20 |
| 硬件 | H20 96GB × 1 | H20 96GB × 1 |
| 总步数 | 350 | 300 |
| 训练数据 | MBPP+APPS (602条) | MBPP+APPS (602条) |

### 训练数据

- **MBPP**（train+val split，464 条）：排除 test split 防止与 MBPP+ 评估数据泄露
- **APPS-easy**（cleaned，~140 条）：按 prompt 长度过滤，保留 <600 字符的简单题
- **去重**：跨数据源 Jaccard 相似度 >0.7 去重，MBPP 优先
- **最终**：602 条混合数据，每 epoch 75 步（batch=8, n=8），共训练 ~4 epochs

## 训练过程分析

### 训练曲线对比

**累积平均 reward（衡量整体训练趋势，过滤步间振荡）：**

| step | 多轮 GRPO | 单轮 GRPO | 差值 |
|------|----------|----------|------|
| 50 | -0.074 | -0.075 | +0.001 |
| 100 | +0.001 | -0.041 | **+0.043** |
| 150 | +0.084 | -0.018 | **+0.101** |
| 200 | +0.148 | +0.003 | **+0.145** |
| 225 | +0.174 | +0.010 | **+0.164** |
| 300 | — | +0.031 | — |

多轮 GRPO 在 step 100 后累积 reward 显著领先单轮。多轮 GRPO 在 step 36 即首次突破 reward=0.3，而单轮需要到 step 79。

**关键收敛里程碑：**

| 里程碑 | 多轮 GRPO | 单轮 GRPO |
|--------|----------|----------|
| 首次 reward>0 | step 4 | step 3 |
| 首次 reward>0.3 | **step 36** | step 79 |

多轮 GRPO 的 reward 上升更快，因为多轮纠错提供了更多正 reward 信号（T2 纠错贡献了 66% 的正 reward），帮助模型更快摆脱初期负 reward 区域。

### KL 散度对比

| step | 多轮 KL | 单轮 KL |
|------|--------|--------|
| 50 | 0.0050 | 0.0036 |
| 100 | 0.0147 | 0.0112 |
| 200 | 0.0334 | 0.0157 |
| 300 | 0.0547 | 0.0309 |

多轮 GRPO 的 KL 散度始终高于单轮（约 1.5-2x），说明多轮训练中策略偏离参考模型更快。这可能是因为多轮交互的 reward signal 更丰富（成功/失败/纠错多种情况），驱动模型做更大的策略更新。KL 散度较高也意味着过拟合风险更大——与多轮训练后期 reward 震荡加剧的现象一致。

### 截断率趋势

| step | 多轮 clip% | 单轮 clip% |
|------|----------|----------|
| 1 | 26.6% | 25.0% |
| 100 | 21.9% | 18.8% |
| 150 | 7.8% | 1.6% |
| 200 | 21.9% | 32.8% |
| 300 | 1.6% | 0.0% |

两种训练方式的截断率都呈现下降趋势，但波动较大（GRPO 的典型特征）。多轮训练的截断率波动更剧烈，因为多轮交互的 token 消耗更不稳定。

### 多轮训练中的纠错能力演化

**step 200-300 训练数据（6400 samples = 100 steps × 64）：**

| 区间 | T1 成功率 | T2 纠错率 | T3 纠错率 | NO_CODE% | 总通过率 | reward |
|------|---------|---------|---------|---------|---------|--------|
| step 201-210 | 15.3% | 31.7% | 0.7% | 11.2% | 42.8% | 0.273 |
| step 211-220 | 15.3% | 41.7% | 1.1% | 5.8% | 51.6% | 0.402 |
| step 221-230 | 16.4% | 47.1% | 1.1% | 4.2% | 56.7% | 0.463 |
| step 231-240 | 10.6% | 33.0% | 3.1% | 5.5% | 43.0% | 0.322 |
| step 241-250 | 14.7% | 35.5% | 1.6% | 7.0% | 46.4% | 0.344 |
| step 251-260 | 17.5% | 39.2% | 0.8% | 3.9% | 50.5% | 0.415 |
| step 261-270 | 17.2% | 43.2% | 3.2% | 6.1% | 55.6% | 0.434 |
| step 271-280 | 15.2% | 37.8% | 0.9% | 5.9% | 48.0% | 0.370 |
| step 281-290 | 16.1% | 48.4% | 2.6% | 5.2% | 58.9% | 0.470 |
| step 291-300 | 19.4% | 39.0% | 1.2% | 4.2% | 51.7% | 0.425 |

T1 成功率在训练全程保持稳定（10-19%），说明模型在训练中的**首次代码生成能力提升缓慢**。但 T2 纠错率从初期的 ~8% 逐步提升到 40-48%，**纠错能力确实在训练中不断改善**。

**reward 组成分解：**

| reward 来源 | 数量 | 占比 |
|------------|------|------|
| T1 成功 (reward=1.0) | 1009 | 15.8% |
| T2 纠错成功 (reward=0.85) | 2135 | 33.4% |
| T3 纠错成功 (reward=0.70) | 89 | 1.4% |
| 执行失败 (reward=0) | 2789 | 43.6% |
| NO_CODE (reward=-1) | 378 | 5.9% |

**多轮纠错贡献了 66% 的正 reward（T2 55.4% + T3 1.4% + T1 的部分间接收益）**，是多轮训练 reward 的主要来源。

### 训练中的 NO_CODE 分析

| 轮次 | NO_CODE 数量 | 占比 |
|------|-------------|------|
| Turn 1 | 376 | **97.7%** |
| Turn 2 | 9 | 2.3% |
| Turn 3 | 0 | 0% |

**训练中 97.7% 的 NO_CODE 发生在首轮**，后续轮次几乎没有 NO_CODE（T2 仅 0.2%）。这说明：

1. 首轮 NO_CODE 是因为模型对某些题目的 prompt 无法生成有效代码（模型能力问题），不是 budget 问题
2. 后续轮次因为已有代码作为上下文 + 错误反馈引导，模型几乎总能输出代码块
3. **NO_CODE 的 -1 惩罚主要落在首轮失败上**，不应该被简单归因于多轮 budget 不足

### 训练 vs 评估的巨大差异

| 指标 | 训练 (sampling) | 评估 (greedy) |
|------|----------------|---------------|
| T1 NO_CODE | 5.9% | 13-32% |
| T2 NO_CODE | **0.2%** | **~60%** |
| T2 纠错率 | **42.8%** | **9-35%** |

这个差异的核心原因是 **temperature**：

- **训练**（temperature ~0.7）：模型生成更多样化，thinking tokens 较短，T2 几乎总能输出代码，纠错率 42.8%
- **评估**（temperature=0，greedy）：模型进入确定但极长的 thinking 模式，T2 的 thinking 还没完成就耗尽了 4800 token budget，60% 被截断无法输出代码

这意味着**模型确实在训练中学到了纠错能力**（42.8% T2 纠错率），但评估时 greedy decoding 的 thinking 过长掩盖了这个能力。如果用 temperature>0 评估或增大 response_length，纠错率可能显著提升。

## 评估结果

### 评估方法

- **单轮评估**：greedy（temperature=0），max_new_tokens=4800，evalplus 判卷
- **多轮评估**：max_turns=3，每轮生成后用 evalplus base tests 判定 pass/fail，失败则发送错误反馈继续，最终用 `evalplus.evaluate` 获取 HumanEval 和 HumanEval+ 分数
- **评估基准**：HumanEval（164 题，base test cases）+ HumanEval+（base + extra test cases，更严苛）

### 主要结果

#### 多轮 GRPO 训练 checkpoints

| checkpoint | 单轮 Base | 单轮 HEval+ | 多轮 Base | 多轮 HEval+ | 多轮提升 | T1 | T2纠错 | T2纠错率 |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.610 | 0.561 | 0.622 | 0.573 | +0.012 | 100 | 2 | 2/12=17% |
| mt_s100 | 0.726 | 0.677 | 0.738 | 0.683 | +0.006 | 119 | 2 | 2/21=10% |
| **mt_s150** | **0.787** | **0.732** | **0.805** | **0.750** | **+0.018** | **129** | **3** | **3/14=21%** |
| mt_s200 | 0.750 | 0.695 | 0.768 | 0.707 | +0.012 | 123 | 2 | 2/22=9% |
| mt_s250 | 0.780 | 0.720 | 0.799 | 0.732 | +0.012 | 128 | 3 | 3/15=20% |
| mt_s300 | 0.720 | 0.677 | 0.750 | 0.701 | +0.024 | 118 | 4 | 4/23=17% |

#### 单轮 GRPO 训练 checkpoints

| checkpoint | 单轮 Base | 单轮 HEval+ | 多轮 Base | 多轮 HEval+ | 多轮提升 | T1 | T2纠错 | T2纠错率 |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.610 | 0.561 | 0.622 | 0.573 | +0.012 | 100 | 2 | 2/12=17% |
| st_s100 | 0.726 | 0.665 | 0.750 | 0.689 | +0.024 | 119 | 4 | 4/18=22% |
| st_s150 | 0.738 | 0.665 | 0.762 | 0.683 | +0.018 | 121 | 4 | 4/18=22% |
| st_s200 | 0.744 | 0.683 | 0.756 | 0.695 | +0.012 | 122 | 2 | 2/16=12% |
| st_s250 | 0.744 | 0.689 | 0.756 | 0.695 | +0.006 | 122 | 2 | 2/16=12% |
| **st_s300** | **0.768** | **0.732** | **0.805** | **0.762** | **+0.030** | **126** | **6** | **6/17=35%** |

> T2纠错率 = T2纠错数 / T1失败(有代码)数；多轮提升 = 多轮HEval+ - 单轮HEval+

### 关键发现

#### 1. GRPO 训练大幅提升代码生成能力

无论单轮还是多轮训练，GRPO 都带来了显著提升：

- **基模型**：HumanEval 0.610 / HumanEval+ 0.561
- **最佳单轮训练**（st_s300）：HumanEval 0.768 / HumanEval+ 0.732（**+25.8% / +30.5%**）
- **最佳多轮训练**（mt_s150）：HumanEval 0.787 / HumanEval+ 0.732（**+29.0% / +30.5%**）

#### 2. 多轮 GRPO 收敛更快，但最终性能相当

多轮 GRPO 的累积 reward 始终高于单轮（step 200 时差距达 +0.145），主要因为多轮纠错提供了额外的正 reward 信号（T2 贡献 66% 的正 reward）。但最终评估分数持平（最佳 HEval+ 都是 0.732），说明多轮训练的更快收敛并没有转化为更好的最终性能。

原因分析：多轮训练的 KL 散度增长更快（step 300 时 0.055 vs 单轮 0.031），策略偏离参考模型更多，后期 reward 震荡也更剧烈。多轮的丰富 reward signal 是双刃剑——加速了早期学习，但也加速了后期的策略坍缩。

#### 3. 单轮训练模型的纠错率反而更高

| 指标 | 单轮训练最佳 (st_s300) | 多轮训练最佳 (mt_s150) |
|------|---------------------|---------------------|
| 单轮 HEval+ | **0.732** | **0.732** |
| 多轮 HEval+ | **0.762** | 0.750 |
| T2 纠错率 | **35% (6/17)** | 21% (3/14) |
| 多轮纠错增益 | **+0.030** | +0.018 |

单轮训练的 st_s300 在多轮评估中表现更好：纠错率 35%（多轮训练最佳仅 21%），多轮 HEval+ 0.762（多轮训练最佳仅 0.750）。

这说明**多轮训练并没有让模型学到更好的纠错能力**，反而可能因为优化目标更复杂（同时优化生成+纠错），导致两方面都没有达到最优。单轮训练专注于提升首次生成质量，间接也提升了模型对代码的理解能力，使其在纠错任务上也表现更好。

#### 4. 评估中纠错率低的主因是 greedy thinking 截断，不是模型能力不足

评估中 T2 NO_CODE 率高达 ~60%，但训练中只有 0.2%。这不是模型不会纠错，而是 temperature=0 下 Qwen3 的 thinking tokens 过长（T2 平均 2000+ tokens），耗尽了 4800 token 预算。如果用 temperature>0 评估（与训练一致），或增大 response_length 到 8192+，纠错率可能会显著提升。

#### 5. 训练后期出现过拟合

多轮训练 mt_s300 单轮分数（0.677）比 mt_s150（0.732）低 5.5%。单轮训练续训到 350 步后 reward 也从 0.609 降到 0.344。300 步（~4 epochs）可能是 602 条训练数据下的最佳训练窗口。

#### 6. 纠错题目的性质：只对"偶尔失误"有效

所有 checkpoint 总共只有 10 道题曾被纠错成功，且都是"大部分 checkpoint T1 能做对"的中等题。纠错本质上是对 T1 偶尔失误的保险，而不是解决难题的能力。不同 checkpoint 纠错的题目几乎没有重叠（仅 HumanEval/101 被纠错 3 次），说明纠错成功率高度依赖模型在特定题目上的生成随机性。

## 工程经验总结

### 踩过的坑（25+ 个）

| # | 问题 | 影响 | 解决方案 |
|---|------|------|---------|
| 1 | max_response_length=512 导致全截断 | 100步无正reward | 增大到 4800+ |
| 5 | numpy 2.x 降级残留 `_core/` | vLLM spawn 崩溃 | `rm -rf numpy*` + 重装 |
| 13 | LoRA QKV 权重名 `base_layer.` 前缀 | vLLM load_weights 失败 | patch_vllm.py 正则替换 |
| 21 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | vLLM sleep mode 启动崩溃 | 禁止设置此变量 |
| 23 | ray_trainer 中调用 `torch.cuda.ipc_collect()` | CPU-only 进程调用 GPU API 报错 | 移除，在 vLLM 进程中 gc |
| 25 | FSDP + LoRA 下 `disable_adapter()` 失效 | kl_loss=0，无正则化 | 独立 ref 模型 |

### 显存预算（H20 96GB）

| 组件 | 显存占用 |
|------|---------|
| Actor (LoRA, bf16) | ~6.8 GiB |
| Ref Model (bf16) | ~3.4 GiB |
| vLLM Rollout (gpu_util=0.20) | ~19.7 GiB |
| FSDP 优化器状态 | ~15 GiB |
| KV cache + 梯度 | ~40+ GiB |
| **峰值总计** | **~86.6 GiB (88.5%)** |

### vLLM 多轮抢占问题

H20 96GB、gpu_util=0.25 下 KV cache 仅支持 ~21 个并发序列，但实际需求 64 个（8 prompts × 8 GRPO samples）。多轮训练时 Turn 1 的 KV cache 未释放，Turn 2 到达后触发抢占级联，实测极端情况单步耗时从 130s 飙升到 993s（16 步中出现 1 次严重、2 次轻微）。

## 项目文件结构

```
agentic-rl/
├── src/
│   ├── agent_loop.py           # 多轮代码纠错 AgentLoop
│   ├── reward.py               # 单轮门控奖励函数
│   ├── reward_multiturn.py     # 多轮渐进奖励函数
│   ├── evalplus_sandbox.py     # evalplus 对齐评估沙盒
│   ├── sandbox.py              # 训练用同步沙盒
│   ├── evaluate.py             # 单轮串行评估脚本
│   ├── data_prepare.py         # 训练数据准备（MBPP+APPS）
│   └── utils.py                # 工具函数
├── scripts/
│   ├── run_cloud_multi.sh      # 云端多轮训练脚本
│   ├── run_cloud_single.sh     # 云端单轮训练脚本
│   ├── eval_all_checkpoints.sh          # 单轮评估所有 checkpoint
│   ├── eval_all_checkpoints_multiturn.sh # 多轮评估所有 checkpoint
│   ├── evaluate_multiturn.py   # 多轮评估脚本
│   ├── patch_vllm.py           # vLLM 8 个补丁
│   └── strip_optimizer.py      # Checkpoint 瘦身工具
├── configs/
│   └── agent_loop.yaml         # 自定义 AgentLoop 注册配置
├── data/                       # 训练数据（.gitignore）
├── logs/                       # 训练指标日志
├── eval_results_v2/            # 评估结果
└── wandb/                      # W&B 训练日志
```

## WandB 训练日志

- [多轮 GRPO 训练 (step 1-225)](https://wandb.ai/17872439283ypf-tongji-university/agentic-rl-cloud/runs/myjwtuks)
- [多轮 GRPO 续训 (step 200-300)](https://wandb.ai/17872439283ypf-tongji-university/agentic-rl-cloud/runs/li01rf8t)
- [单轮 GRPO 训练 (step 1-300)](https://wandb.ai/17872439283ypf-tongji-university/agentic-rl-cloud/runs/rnplbvm0)
- [单轮 GRPO 续训 (step 300-350)](https://wandb.ai/17872439283ypf-tongji-university/agentic-rl-cloud/runs/dmsv1uo0)

## 复现指南

```bash
# 1. 环境搭建（云端 AutoDL）
bash scripts/setup_cloud.sh

# 2. 准备训练数据
python src/data_prepare.py --expanded

# 3. vLLM 补丁（每次切换镜像后必须运行）
python3 scripts/patch_vllm.py

# 4. 多轮 GRPO 训练
bash scripts/run_cloud_multi.sh

# 5. 单轮 GRPO 训练（消融对比）
bash scripts/run_cloud_single.sh

# 6. 评估
bash scripts/eval_all_checkpoints.sh          # 单轮评估
bash scripts/eval_all_checkpoints_multiturn.sh # 多轮评估
```

## 结论与未来改进

### 核心结论

1. **GRPO 有效**：从 Qwen3-1.7B 基线 56.1% 提升到 73.2%（HumanEval+），增幅 30.5%
2. **多轮训练收敛更快**：step 100 后累积 reward 显著领先单轮（+0.04 → +0.16），因为纠错提供了额外的正 reward signal
3. **多轮训练未带来更好的最终性能**：最佳 HEval+ 持平（0.732），且单轮训练模型在多轮评估中反而更好（0.762 vs 0.750）
4. **纠错能力被 greedy decoding 掩盖**：训练中 T2 纠错率 42.8%，评估中仅 9-35%，主因是 temperature=0 下 thinking 过长导致 60% T2 被截断
5. **多轮训练是双刃剑**：更丰富的 reward signal 加速早期学习，但 KL 散度增长更快，后期过拟合风险更高

### 改进方向

1. **增大 response_length 至 8192+**：当前 4800 token 是评估纠错率的硬瓶颈
2. **评估时使用 temperature>0**：与训练条件一致，可能显著提升评估纠错率
3. **限制 Qwen3 thinking**：thinking 消耗大量 budget，可加 thinking budget 上限
4. **更大基模型**：Qwen3-4B 或 8B，代码生成基线更强
5. **多轮 reward shaping**：当前纯稀疏奖励（+1/0/-1），可考虑渐进式奖励（部分通过给部分分）
6. **降低 KL coef**：多轮训练 KL 散度增长过快，降低 coef 可能缓解后期过拟合
