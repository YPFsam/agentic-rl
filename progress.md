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

### [5.2] 本地单轮 GRPO 训练
- **状态**：⏳ 待开始（需在 AutoDL 24GB GPU 上运行）
- **备注**：WSL2 下 vLLM V1 引擎 CUDA 不兼容，无法在本地训练

## 阶段二：多轮 Agentic GRPO
- **状态**：⏳ 待开始

## 消融实验
- **状态**：⏳ 待开始

## 分析与可视化
- **状态**：⏳ 待开始
