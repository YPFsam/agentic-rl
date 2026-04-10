# Agentic-RL 项目进度日志

## 项目概述
基于 GRPO + 执行反馈的代码生成强化学习项目，从单轮到多轮纠错。

---

## 阶段零：基础设施搭建

### [1] 本地环境搭建
- **状态**：✅ 完成
- **日期**：2026-04-06
- **内容**：
  - [x] 确认 WSL2 环境（Linux 6.6.87.2-microsoft-standard-WSL2）
  - [x] 确认 GPU：RTX 5070 Ti 12GB，CUDA 12.8
  - [x] 创建项目目录结构
  - [x] 创建所有核心源代码文件
  - [x] 启用 Docker Desktop WSL 集成
  - [x] 修复用户不在 docker 组的问题
  - [x] 修复 WSL2 DNS 问题（chattr -i 修改 resolv.conf）
  - [x] 迁移 WSL 和 Docker 数据到 D 盘（C 盘空间不足）
  - [x] 拉取 veRL 官方镜像 `verlai/verl:vllm011.latest`（16GB 压缩/48.9GB 解压）
  - [x] 验证容器内 GPU 直通（PyTorch 2.8.0+cu128，CUDA OK，VRAM 11.9GB）
  - [x] 在容器内安装 veRL（v0.8.0.dev0，从源码）+ evalplus
  - [x] 运行 verify_env.py **全部 8 项通过**

### [1.5] AgentLoop 早期验证
- **状态**：✅ 完成
- **日期**：2026-04-06
- **内容**：
  - [x] 验证 `verl.experimental.agent_loop.agent_loop.AgentLoopBase` import 成功
  - [x] 确认 API 签名：`run(sampling_params, **kwargs) -> AgentLoopOutput`
  - [x] 确认 `server_manager.generate()` 参数为 `prompt_ids`（非文档中的 `input_ids`）
  - [x] 更新 `src/agent_loop.py`：正确继承 AgentLoopBase，使用实际 API
  - **结论**：AgentLoop 可用，继续走主方案

### [2] 数据准备
- **状态**：✅ 完成
- **日期**：2026-04-06
- **内容**：
  - [x] 运行 `python src/data_prepare.py --output data/grpo_train.parquet`
  - [x] 生成 464 条 MBPP train+val 训练数据（不含 test split，避免数据泄露）
  - [x] 验证 parquet 格式正确（5 列：data_source, prompt, ability, reward_model, extra_info）

### [3] 沙盒模块单元测试
- **状态**：✅ 完成
- **日期**：2026-04-06
- **内容**：
  - [x] 测试 1：正常代码执行成功
  - [x] 测试 2：语法错误检测
  - [x] 测试 3：运行时错误检测（含 assert 失败）
  - [x] 测试 4：超时检测
  - [x] 测试 5：带测试用例的代码执行
  - [x] 测试 6：批量并发执行
  - **结果**：全部 6 个测试通过

### [4] 奖励函数单元测试
- **状态**：✅ 完成
- **日期**：2026-04-06
- **内容**：
  - [x] 测试：代码块提取（```python 格式）
  - [x] 测试：无代码时返回 None
  - [x] 测试 1：完美输出 → reward = 1.0
  - [x] 测试 2：无代码输出 → reward = -1.0（门控触发）
  - [x] 测试 3：代码错误 → reward = 0.0
  - [x] 测试 4：乱输出 → reward = -1.0（门控触发）
  - [x] 测试 5：语法错误 → reward = 0.0
  - [x] 测试 6：超时代码 → reward = 0.0
  - **结果**：全部 7 个测试通过

**阶段零验收标准**：13 个测试全部通过 ✅，`data/grpo_train.parquet` 生成成功 ✅

### [4.5] 端到端验证（阶段零点五）
- **状态**：✅ 完成
- **日期**：2026-04-06
- **内容**：
  - [x] 在 Docker 容器 `agentic-rl` 内运行 `scripts/verify_e2e.py`
  - [x] Qwen3-1.7B 模型下载 + 加载成功（约 3.4GB，含网络重试共耗时约 20 分钟）
  - [x] 测试 1：模型生成 + 代码提取 PASS（模型输出含 `<think` 思考过程，成功提取 39 字符 Python 代码）
  - [x] 测试 2：沙盒执行模型代码 SKIP（第二个 prompt 模型未生成代码块，基座模型不完全稳定的正常现象）
  - [x] 测试 3：完整 reward 管道 PASS（正确代码→1.0，错误代码→0.0，无代码→-1.0）
  - **结论**：端到端全链路验证通过，可以安全进入阶段一

---

## 阶段一：单轮 GRPO 训练
- **状态**：🔄 进行中

### [5.0] 训练前准备
- **状态**：✅ 完成
- **日期**：2026-04-07
- **内容**：
  - [x] 添加 thinking metrics 日志到 `src/reward.py`
  - [x] 优化 4 个训练脚本参数（lr/kl/truncation）
  - [x] 修复 `evaluate.py` evalplus dataset_flag 大小写 bug（"HumanEval" → "humaneval"）

### [5.0.1] veRL 0.8 API 迁移
- **状态**：✅ 完成
- **日期**：2026-04-07
- **内容**：
  - [x] 识别 veRL 0.8 破坏性变更（LoRA命名空间、Rollout后端、seed位置、sync模式移除）
  - [x] 重写 `scripts/run_local_single.sh` 适配 veRL 0.8 API
  - [x] 重写 `scripts/run_local_multi.sh` 适配 veRL 0.8 API
  - [x] 重写 `scripts/run_cloud_single.sh` 适配 veRL 0.8 API
  - [x] 重写 `scripts/run_cloud_multi.sh` 适配 veRL 0.8 API（含补加 max_model_len）
  - [x] 确认本地 WSL2 无法训练：vLLM V1 引擎 CUDA 不兼容
  - **结论**：需在原生 Linux 环境（AutoDL 24GB GPU）运行训练

### [5.0.2] WSL2 本地训练可行性验证
- **状态**：✅ 完成（结论：不可行）
- **日期**：2026-04-07
- **内容**：
  - [x] 测试 NCCL 禁用修复（`NCCL_P2P_DISABLE=1` + `NCCL_IB_DISABLE=1`）
  - [x] 单进程 vLLM 测试通过（gpu_util=0.4，模型加载+生成成功）
  - [x] veRL 多进程训练测试 1：param_offload=False → GPU OOM（actor 9.57GB + vLLM 无法加载）
  - [x] WSL2 RAM 增加到 24GB + reward workers=2 + fp8 + param_offload=True → 仍失败
  - [x] 双重阻塞：①GPU 显存（actor 9.57GB + vLLM 模型 > 12GB）②WSL2 CUDA 指针释放 bug
  - **两个独立问题**：即使解决显存，WSL2 的 CUDA PluggableAllocator 在多进程清理时崩溃
  - **结论**：12GB GPU + WSL2 无法运行 veRL 训练，需原生 Linux + 24GB GPU（AutoDL）

### [5.1] Baseline 评估
- **状态**：✅ 完成
- **日期**：2026-04-07
- **内容**：
  - [x] Qwen3-1.7B HumanEval 基线评估（164 题，约 1.5 小时）
  - **结果**：
    - **HumanEval pass@1: 26.2%**
    - **HumanEval+ pass@1: 24.4%**

### [5.2] 云端环境迁移（官方 Docker 镜像）
- **状态**：✅ 完成
- **日期**：2026-04-08
- **内容**：
  - [x] 从手工组装环境切换到 veRL 官方 Docker 镜像（verlai/verl:vllm011.latest）
  - [x] 新环境：Python 3.12.11, PyTorch 2.8.0+cu128, flash-attn 2.8.1, vLLM 0.11.0
  - [x] GPU 从 RTX 4090 24GB 升级到 RTX 4090 **48GB**（24GB 显存不足以跑 GRPO）
  - [x] 数据盘从 50GB 扩容到 100GB

### [5.3] 云端单轮 GRPO 训练（1.7B 模型，48GB GPU）
- **状态**：✅ 完成
- **日期**：2026-04-08
- **环境**：AutoDL, RTX 4090 48GB
- **训练配置**：
  - 模型：Qwen3-1.7B + LoRA (rank=8, alpha=16)
  - max_response_length=512, n=2, batch_size=16
  - param_offload=True, optimizer_offload=True, gradient_checkpointing=True
  - 无 fp8 量化（RTX 4090 不支持 fp8，会触发 numpy dtype 错误）
- **结果（100 步）**：
  - **score/mean: -0.02**（几乎全部为 0 或负分）
  - **正 reward 步数: 0/100**（没有一个样本通过测试）
  - **response_length: 512（100% 截断）** — 所有序列都被 max_new_tokens=512 截断
  - **entropy: 0.22 → 0.26**（策略基本没学到东西）
  - **val_acc@1: -0.02 ~ -0.03**（验证集同样惨淡）
- **分析**：
  - 根本原因是 `max_response_length=512` 太短，Qwen3 thinking 模式消耗了大量 token（平均 1335），剩余空间不够输出完整代码
  - 所有输出都在 512 token 处被截断，无法生成可执行的代码
  - 训练曲线平坦，reward 从未转正
- **checkpoint**：已删除（无保留价值，模型未学到有效知识）

### [5.4] 云端多轮 GRPO 训练（1.7B 模型，48GB GPU）
- **状态**：✅ 完成
- **日期**：2026-04-08
- **环境**：AutoDL, RTX 4090 48GB
- **训练配置**：
  - 模型：Qwen3-1.7B + LoRA (rank=8, alpha=16)
  - max_response_length=2048, n=4, batch_size=16
  - multi_turn.enable=True, max_assistant_turns=2
  - max_model_len=5120, gpu_memory_utilization=0.45
  - param_offload=False, optimizer_offload=False（48GB 足够）
  - gradient_checkpointing=True
  - lr=3e-6
- **结果（step 21→100，共 80 步）**：
  - **score/mean: +0.046**（正 reward！远优于单轮的 -0.02）
  - **正 reward 步数占比: 62.5%**（50/80 步有样本通过测试）
  - **response_length/mean: 1489**（能输出完整代码，不再被截断）
  - **clip_ratio: 0.40**（只有 40% 的序列被截断，远优于单轮的 100%）
  - **entropy: 0.14 → 0.06**（下降 57%，策略越来越确定）
  - **最高单步 score: 0.2188**（step 29，约 14/64 样本通过）
- **分阶段趋势**：
  | 阶段 | avg_score | 正reward占比 | entropy | clip_ratio |
  |------|-----------|-------------|---------|------------|
  | 21-40 | 0.052 | 60% | 0.112 | 44% |
  | 41-60 | 0.034 | 45% | 0.080 | 42% |
  | 61-80 | 0.049 | 70% | 0.070 | 40% |
  | 81-100 | 0.047 | 75% | 0.068 | 34% |
- **显存分析**：
  - 峰值：43.4 GiB / 48 GiB (90%)
  - 稳态均值：36 GiB (75%)
  - sleep mode 最低：~21 GiB
  - 典型锯齿波形：actor 训练时 ~37 GiB，sleep 时 ~21 GiB
- **checkpoint**：保存在 `/root/autodl-tmp/checkpoints/agentic-rl-local/qwen3-1.7b-grpo-multi/`
  - 每个 ~21GB（model 7.6G + optimizer 13G），保留最近 2 个

### [5.5] 单轮 vs 多轮训练对比

| 指标 | 单轮 100 步 | 多轮 80 步 |
|------|-----------|-----------|
| score/mean | -0.02 | **+0.046** |
| score/max | 0.0 | **1.0** |
| 正 reward 步占比 | 0% | **62.5%** |
| response_length/mean | 512（全截断） | **1489** |
| clip_ratio | 1.00 | **0.40** |
| entropy 均值 | 0.22 | **0.083** |

**关键发现**：max_response_length 是单轮训练失败的根本原因。512 token 对 Qwen3 thinking 模式远远不够。

---

## 阶段二：评估与分析
- **状态**：⏳ 待开始
- **下一步**：
  - 用多轮 checkpoint 跑 HumanEval 评估（量化 pass@1 提升幅度）
  - 考虑加长训练到 200-300 步

## 阶段三：4B 模型训练（已放弃）
- **状态**：❌ 放弃（OOM + 经济不可行）
- **日期**：2026-04-09 ~ 2026-04-10
- **已完成**：
  - [x] 云端脚本显存修复（free_cache_engine + grad_ckpt + gpu_util 降级）
  - [x] 云端脚本全面审计（AutoDL 环境变量、Ray 透传、resume、enforce_eager 等）
  - [x] Checkpoint 优化器剥离脚本编写并本地验证（释放 25.6 GiB）
  - [x] 4B 模型路径改用本地绝对路径（避免 HF 缓存格式不兼容）
  - [x] numpy 2.x 残留文件彻底清理
- **尝试结果**：4B 单轮训练第 1 步 OOM（79,647/81,920 MiB = 97.3%），10 分钟未完成一步
- **放弃原因**：
  - 显存：cumem allocator 导致 actor + vLLM 双份权重常驻，80GB 不够
  - 速度：即使不 OOM，预估 20-30 min/步 × 100 步 = 33-50 小时
  - 经济：200-300 元训练费，且成功率不确定
- **踩坑**：
  - HF 离线模式 + ModelScope 下载格式 → tokenizer 加载失败（踩坑 12）
  - numpy 2.x `_core/` 残留 → vLLM repr 崩溃（踩坑 13）
  - 4B 单卡 GRPO OOM（踩坑 14）：cumem 双份权重 + KV cache 全量分配被系统性低估

---

## 阶段三（替代）：1.7B 升级版训练（A800 80GB）
- **状态**：⏳ 待开始
- **日期**：2026-04-10
- **策略**：放弃 4B 模型，将 4B 方案含金量（数据多样性、GRPO 采样数、多轮纠错、KL 正则化）全部迁移到 1.7B + A800
- **训练配置（单轮）**：
  - 模型：Qwen3-1.7B + LoRA (rank=16, alpha=32)
  - 数据：1184 MBPP+APPS-easy（grpo_train_full.parquet，APPS 过滤 prompt<600字）
  - batch=48, n=8, max_response=**8192**, max_prompt=1024
  - KL loss=True (coef=0.003), lr=1e-6
  - gpu_util=0.55, max_model_len=**10240**
  - 200 步 (~8.1 epochs)
  - 预估：~10 小时，~60 元
- **训练配置（多轮）**：
  - 模型：Qwen3-1.7B + LoRA (rank=16, alpha=32)
  - 数据：1184 MBPP+APPS-easy（grpo_train_multi_full.parquet，APPS 过滤 prompt<600字）
  - batch=48, n=8, max_response=**8192** (2730/轮), max_assistant_turns=3
  - KL loss=True (coef=0.003), lr=2e-6
  - gpu_util=0.55, max_model_len=**12288**
  - 200 步 (~8.1 epochs)
  - 预估：~23 小时，~140 元
- **Prompt 优化**：新增 "Provide only ONE complete Python code block" 指令，防止多代码块拼接错误
- **待完成**：
  - [ ] 上云验证 A800 环境可用
  - [ ] 运行 `python3 scripts/patch_vllm.py`
  - [ ] 运行单轮训练 `bash scripts/run_cloud_single.sh`
  - [ ] 运行多轮训练 `bash scripts/run_cloud_multi.sh`
  - [ ] 评估 checkpoint（HumanEval/MBPP pass@1）
- **脚本**：
  - `scripts/run_cloud_single.sh` — 1.7B 单轮升级版（A800）
  - `scripts/run_cloud_multi.sh` — 1.7B 多轮升级版（A800）

## 阶段四：评估与消融
- **状态**：⏳ 待开始
- **计划**：
  - 单轮 vs 多轮对比（相同数据/参数）
  - MBPP vs APPS 分层分析
  - Turn 分析（第 2/3 轮纠错成功率）
  - Thinking token 分析

## 分析与可视化
- **状态**：⏳ 待开始
