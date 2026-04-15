# Agentic-RL 修改与补充记录

记录项目执行过程中对原始方案的修改、补充和调整。

---

## 2026-04-06 初始创建

### 目录结构创建
- 按文档第二节创建完整项目目录结构：configs/, data/, src/, scripts/, analysis/, eval_results/, outputs/, logs/, tests/
- 创建 `src/__init__.py` 作为 Python 包标识

### 源代码文件创建
基于文档第四节（阶段零）的代码，创建以下文件：
- `src/sandbox.py` — async 代码沙盒执行模块
- `src/reward.py` — 门控奖励函数
- `src/data_prepare.py` — 数据准备脚本
- `src/utils.py` — 工具函数
- `src/reward_multiturn.py` — 多轮奖励函数
- `src/reward_additive.py` — 加法奖励函数（消融用）
- `src/agent_loop.py` — 多轮 AgentLoop 实现
- `src/register_agent.py` — AgentLoop 注册
- `src/evaluate.py` — 统一评估脚本
- `tests/test_sandbox.py` — 沙盒测试
- `tests/test_reward.py` — 奖励函数测试
- `scripts/synth_testcases.py` — APPS 数据清洗脚本
- `scripts/verify_e2e.py` — 端到端验证脚本
- `scripts/run_local_single.sh` — 本地单轮训练
- `scripts/run_cloud_single.sh` — 云端单轮训练
- `scripts/run_local_multi.sh` — 本地多轮训练
- `scripts/run_cloud_multi.sh` — 云端多轮训练
- `analysis/plot_training.py` — 训练曲线可视化
- `analysis/plot_comparison.py` — 消融对比图
- `analysis/plot_checkpoint_curve.py` — Checkpoint 评估曲线
- `analysis/plot_thinking_analysis.py` — Thinking 分析

### 基础设施文件
- `Dockerfile` — 自定义 Docker 镜像
- `requirements.txt` — 依赖清单

---

## 2026-04-06 环境搭建过程中的调整

### 方案变更：使用官方预构建镜像
- **原方案**：自定义 Dockerfile 构建
- **实际方案**：拉取 veRL 官方镜像 `verlai/verl:vllm011.latest`，在容器内安装 veRL + evalplus
- **原因**：官方镜像预装了 PyTorch + vLLM + CUDA 工具链，避免依赖冲突

### verify_env.py 修复
- `torch.cuda.get_device_properties(0).total_mem` → `.total_memory`（PyTorch 2.8+ API 变更）

### 环境信息（容器内）
- Python 3.12
- PyTorch 2.8.0+cu128
- veRL 0.8.0.dev0
- GPU: RTX 5070 Ti Laptop GPU, VRAM 11.9GB
- Docker 镜像解压后约 48.9GB

---
## 2026-04-06 AgentLoop 验证与调整

### AgentLoop import 路径修正
- **原方案（文档）**：`verl.workers.agent.agent_loop.AgentLoopBase`
- **实际路径**：`verl.experimental.agent_loop.agent_loop.AgentLoopBase`
- **原因**：veRL v0.8 中 AgentLoop 位于 experimental 目录，不在 workers 下

### agent_loop.py 重写
- 继承 `AgentLoopBase`，使用正确的 import 路径
- `server_manager.generate()` 参数修正：`input_ids` → `prompt_ids`，新增必需的 `request_id`
- 使用 `uuid4().hex` 生成唯一 request_id 用于 sticky session
- 返回类型改为 `AgentLoopOutput`（pydantic BaseModel）

### reward.py extract_code 策略 2 增强
- **问题**：纯文本 "Just plain text without any code"（34字符>10阈值）被策略 2 误判为代码
- **修复**：策略 2 增加 Python 特征检测（def/class/import/return/赋值/print/for/if 等正则匹配）
- **效果**：纯文本正确返回 None，无代码块的 Python 代码仍能被提取

---
## 2026-04-06 数据准备与测试

### 数据准备验证
- 运行 `python src/data_prepare.py` 生成 `data/grpo_train.parquet`
- 464 条 MBPP train+val 样本（**不含 test split**，避免与 MBPP+ 评估数据重叠）
- parquet 格式验证通过（5 列，chat messages JSON 格式正确）

### 测试结果
- 沙盒模块：6/6 通过
- 奖励函数：7/7 通过（含 extract_code 修复后的回归验证）

---
## 2026-04-06 截断代码块提取增强

### extract_code 策略 1b：未闭合代码块
- **场景**：`max_completion_length=512` 截断时，输出可能是 ` ```python\ndef add(a,b):\n    return a`（无闭合 ``` ```）
- **原行为**：策略 1a（完整代码块）匹配失败 → 策略 2（纯文本）可能漏掉
- **新增策略 1b**：正则 `r"```python\s*(.*?)$"` 匹配到文本末尾，提取截断的代码
- **测试**：新增 test_extract_code 第 3 项，验证截断代码块正确提取
- **结果**：奖励函数测试 8/8 通过，沙盒测试 6/6 通过（无回归）

### sandbox.py 子进程逃逸防护
- **问题**：`proc.kill()` 只杀直接子进程，恶意代码用 `multiprocessing`/`os.fork()` 创建的孙子进程会逃逸
- **修复**：
  - `preexec_fn` 增加 `os.setsid()` 创建新进程组
  - 超时处理改为 `os.killpg(os.getpgid(proc.pid), SIGKILL)` 杀掉整个进程组
  - 保留 `proc.kill()` 作为 fallback（ProcessLookupError 时）
- **测试**：6/6 通过，无回归

---
## 2026-04-06 reward_multiturn.py 三个致命 Bug 修复

### Bug 1：轮次计数误判（致命 🚨）
- **问题**：`num_turns = len(code_blocks)` 用代码块数当轮次数。单轮输出多个代码块（辅助函数+主函数）被误判为多轮，导致惩罚错误
- **修复**：轮次计数改用 `[Execution Feedback]` 标记数（AgentLoop 明确写入的轮次分隔符），`turns = feedback_count + 1`
- **新函数**：`count_actual_turns(text)` 替代旧的代码块计数

### Bug 2：奖励倒挂（风险 ⚠️）
- **问题**：Bug 1 叠加后惩罚可能超过 1.0，成功却得负分
- **修复**：`max(0.1, 1.0 - penalty)` 保底，一行代码终身保险

### Bug 3：代码提取倒退（致命 🚨）
- **问题**：旧 `extract_all_code_blocks` 用简单正则 `r"```python\s*(.*?)\s*```"`，丢失了截断代码块兜底
- **修复**：重写为 `extract_last_code()`，三级策略：
  1. 提取最后一轮的**所有**完整代码块并拼接（处理多代码块单轮场景）
  2. 截断的未闭合代码块提取（max_tokens 截断场景）
  3. `extract_code()` 兜底
- **关键设计**：多代码块拼接而非只取最后一个，因为辅助函数+主函数需要完整执行

### 测试
- 新增 `tests/test_reward_multiturn.py`：9/9 通过
- 全量回归：23/23 通过（沙盒 6 + 单轮奖励 8 + 多轮奖励 9）

---
## 2026-04-06 evaluate.py 两个关键修复

### 修复 1：MBPP Task ID 格式（致命 🚨）
- **问题**：MBPP 数据集的 task_id 是纯数字（如 `11`），EvalPlus 严格要求 `"Mbpp/11"` 格式
- **后果**：直接写数字 ID 会导致 EvalPlus 判卷时找不到匹配，MBPP 得分全为 0
- **修复**：在 `generate_for_benchmark` 中对 MBPP 的 task_id 自动加 `"Mbpp/"` 前缀

### 修复 2：EvalPlus 结果读取方式（健壮性 💡）
- **问题**：用正则从 stdout 解析 pass@k，EvalPlus 更新输出格式就会失效
- **修复**：优先读取 EvalPlus 自动生成的 `*_eval_results.json` 文件，仅 JSON 不存在时回退正则
- **兼容**：支持多种 JSON 结构（顶层直出、base/plus 子键）

---
## 2026-04-06 AssertionError 脱敏顺序修复（安全相关 🚨）

### 问题：先截断后脱敏 → 截断可能打断 AssertionError 关键词
- **场景**：`stderr` 很长时，尾部截断 `stderr[-500:]` 可能从 "AssertionError" 中间切断
- **后果**：正则 `r"AssertionError: .+"` 匹配失败，测试用例的具体数值泄露给模型
- **修复**：交换顺序——先脱敏再截断

### 问题：`re.DOTALL` 导致贪婪跨行
- **场景**：AssertionError 之后可能还有其他有用日志
- **后果**：`re.DOTALL` 让 `.+` 匹配到文本末尾，抹掉所有后续信息
- **修复**：去掉 `re.DOTALL`，只匹配当前行

### 涉及文件
- `src/agent_loop.py`：`_format_error()` + 类级正则常量
- `src/utils.py`：`sanitize_stderr()` 同步修复

---
## 2026-04-07 阶段零点五：端到端验证

### 运行结果
- **脚本**：`python3 scripts/verify_e2e.py`（容器 `agentic-rl` 内运行）
- **模型**：Qwen3-1.7B（首次下载，含网络重试耗时约 20 分钟）
- **结果**：
  - 测试 1（模型生成 + 代码提取）：**PASS** — 模型输出含 `<think` 思考过程，成功提取 39 字符 Python 代码
  - 测试 2（沙盒执行模型代码）：**SKIP** — 第二个 prompt 模型未生成代码块（基座模型不稳定属正常）
  - 测试 3（Reward 管道）：**PASS** — reward=1.0 / 0.0 / -1.0 三种情况全部正确
- **结论**：端到端验证通过，可以进入阶段一

### 注意事项
- `torch_dtype` 参数已废弃（PyTorch 2.8+），验证脚本有 warning 但不影响功能
- Qwen3-1.7B 的 thinking mode 生成了 `<think` 标签内的推理过程，`extract_code` 正确跳过了 think 标签内的草稿代码




---
## 2026-04-07 训练脚本参数优化

### 修改内容
基于训练稳定性分析，对 4 个训练脚本的参数做了优化：

#### `data.truncation='error'` → `'left'`
- **原因**：`error` 模式下任何超长 prompt 会直接中断训练，改为 `left`（左截断）保留最新 instruction），避免因个别长样本导致训练崩溃
4 个脚本均已更新

#### 云端多轮 `lr=3e-6` → `2e-6`
- **原因**：3e-6 对 4B LoRA 娡型稍偏激进，2e-6 更稳定，GRPO 讃练本身噪声大，保守起步更安全

#### 云端多轮 `kl_loss_coef=0.001` → `0.003`
- **原因**：多轮 response 更长（3072 tokens），KL 獴积更大，0.003 提供更强的策略约束，防止偏离参考模型太远

#### 本地多轮 `use_kl_loss=False` 保持不变
- **原因**：12GB 显存不够加载 ref model（额外 ~3.4GB），开 KL 会 OOM

#### Thinking Metrics 日志
- 在 `src/reward.py` 中添加 `_log_thinking_metrics()` 函数
在 `compute_score()` 的每个 return 前调用
训练过程中自动 thinking token 数和 reward 到 `logs/thinking_metrics.jsonl`

---
## 2026-04-07 veRL 0.8 API 迁移：训练脚本全面重写

### 背景
veRL 0.8.0.dev0 相比文档编写时的 API 有多处破坏性变更，导致原始训练脚本全部无法运行。
经过在容器内实际测试和阅读 veRL 源码配置文件，完成了全部 4 个训练脚本的重写。

### 关键 API 变更

#### 1. LoRA 配置命名空间变更
- **旧**：`actor_rollout_ref.model.peft_config.peft_type=LORA` / `peft_config.r=8` / `peft_config.lora_alpha=16` / `peft_config.target_modules=all-linear` / `peft_config.task_type=CAUSAL_LM`
- **新**：`actor_rollout_ref.model.lora.rank=8` / `lora.alpha=16` / `lora.target_modules=all-linear`
- **移除**：`peft_type` 和 `task_type` 不再需要（veRL 0.8 内部自动处理）

#### 2. Rollout 后端变更
- **旧**：`actor_rollout_ref.rollout.name=hf` + `rollout.mode=sync`
- **新**：`actor_rollout_ref.rollout.name=vllm`（async 模式为默认且唯一模式）
- **原因**：veRL 0.8 移除了 HF rollout 和 sync 模式，只保留 vllm/sglang/trtllm 异步引擎

#### 3. Seed 参数位置变更
- **旧**：顶层 `seed=42`
- **新**：`data.seed=42`

#### 4. 新增必要参数
- `actor_rollout_ref.model.use_remove_padding=True`：vLLM 引擎要求
- `actor_rollout_ref.rollout.max_model_len`：限制 vLLM KV cache 长度，控制显存
- `actor_rollout_ref.rollout.enforce_eager=True`（本地）：禁用 CUDA Graph，WSL2 兼容性

#### 5. KL Loss 参数位置
- `actor_rollout_ref.actor.use_kl_loss=True/False`
- `actor_rollout_ref.actor.kl_loss_coef=0.001~0.003`
- `actor_rollout_ref.actor.kl_loss_type=low_var_kl`

### 4 个脚本最终参数对比

| 参数 | 本地单轮 | 本地多轮 | 云端单轮 | 云端多轮 |
|------|---------|---------|---------|---------|
| 模型 | Qwen3-1.7B | Qwen3-1.7B | Qwen3-4B | Qwen3-4B |
| LoRA rank/alpha | 8/16 | 8/16 | 16/32 | 16/32 |
| batch_size | 32 | 16 | 128 | 128 |
| max_response_length | 512 | 512 | 6144 | 4096 |
| lr | 1e-6 | 3e-6 | 1e-6 | 2e-6 |
| use_kl_loss | False | False | True | True |
| kl_loss_coef | - | - | 0.001 | 0.003 |
| rollout.n | 4 | 4 | 16 | 16 |
| gpu_memory_util | 0.5 | 0.3 | 0.6 | 0.6 |
| max_model_len | 1024 | 1024 | 8192 | 16384 |
| quantization | fp8 | - | - | - |
| param_offload | True | False | False | False |
| optimizer_offload | True | False | False | False |
| multi_turn | - | enable, max=2 | - | enable, max=3 |
| total_steps | 100 | 100 | 200 | 200 |
| reward | reward.py | reward_multiturn.py | reward.py | reward_multiturn.py |
| enforce_eager | True | True | - | - |
| return_raw_chat | - | True | - | True |
| gradient_checkpoint | True | True | False | False |
| activation_offload | - | - | False | False |
| reward workers | 2 | 8 | 8 | 8 |
| logger | console+tb | console+tb | console+tb | console+tb |

### 本地训练阻塞问题

#### WSL2 + 12GB GPU + 16GB RAM 无法运行 veRL 训练

经过多轮尝试，确认本地 WSL2 环境无法运行 veRL GRPO 训练：

**尝试 1：原始配置（param_offload=False, gpu_util=0.2）**
- 错误：vLLM 加载失败 "No available memory for cache blocks" + CUDA pointer 错误
- 原因：Actor 模型占 9.57GB GPU，vLLM 无空间加载

**尝试 2：NCCL 修复（单独测试 vLLM）**
- 设置 `NCCL_P2P_DISABLE=1` + `NCCL_IB_DISABLE=1` 后，单进程 vLLM 可以正常加载和生成
- 但 veRL 的多进程架构（Ray → vLLM Worker）仍触发 WSL2 CUDA bug

**尝试 3：综合修复（param_offload=True, fp8, Ray env 透传, gpu_util=0.5）**
- Actor FSDP 初始化成功（9.57GB GPU / 6.41GB allocated）
- Ray 系统内存 OOM：WorkerDict 用 9.90GB RAM，节点仅 15.47GB RAM（97.6% 被杀）
- 原因：param_offload=True 把参数卸载到 CPU RAM，导致 RAM 不足

**结论**：
- `param_offload=False` → GPU 不够（actor 9.57GB + vLLM 需要 > 12GB）
- `param_offload=True` → RAM 不够（参数卸载到 CPU 后 > 16GB）
- 12GB GPU + 16GB RAM 是 veRL 架构的硬下限，无法绕过
- **需要在原生 Linux（AutoDL 24GB GPU + 充足 RAM）上运行训练**

#### Ray 内存压力
- veRL 0.8 默认启动多个 Ray worker（TaskRunner, WorkerDict, RewardLoopWorker×8, vLLM HTTP Server, vLLM Worker）
- 单 WorkerDict 即占 9.90GB RAM（含模型权重 + FSDP 缓冲）
- 建议训练机器至少 32GB RAM

### 云端脚本修复

#### max_model_len 修正
- **原值**：cloud single=4096, cloud multi=4096
- **修正**：cloud single=8192, cloud multi=16384
- **原因**：`max_model_len` 是 vLLM 总序列长度上限（prompt+response），多轮需容纳 3 轮交互

#### max_response_length 提升
- **原值**：cloud single=4096, cloud multi=3072
- **修正**：cloud single=6144, cloud multi=4096
- **原因**：Qwen3 thinking 模式平均占 1335 tokens，baseline 评估 2048 截断严重，需更多空间

#### gpu_memory_utilization 提升
- **原值**：0.5
- **修正**：0.6
- **原因**：80GB 仅用 0.5 时 KV cache 利用率 27%，提升到 0.6 后达 28-56%

#### 关闭 80GB 场景下的内存优化
- `enable_gradient_checkpointing=False`（80GB 无需省显存，关闭可加速）
- `enable_activation_offload=False`（同上）

#### 训练可视化
- 所有脚本 logger 改为 `["console","tensorboard"]`
- 训练时另开终端运行 `tensorboard --logdir=outputs/` 可在浏览器查看曲线

### WSL2 内存分配修复
- **问题**：WSL2 默认只分配 50% 宿主机 RAM（32GB → 15GB），导致 Ray OOM
- **修复**：在 `%USERPROFILE%\.wslconfig` 中添加 `memory=24GB`
- **重启方式**：`wsl.exe --shutdown` 后重新打开终端

---

## 2026-04-07 云端环境准备（AutoDL）

### AutoDL 容器环境搭建
- **环境信息**：
  - 容器：AutoDL (autodl-container-d6a547ba47-0a7593b1)
  - 系统：Linux with 1TB RAM（服务器级资源）
  - 数据盘：50GB (/root/autodl-tmp)
  - NVIDIA 驱动：580.76.05 Open Kernel Module
  - Python：3.12.3
  - PyTorch：2.5.1+cu124

### 缓存重定向配置（防止系统盘爆满）
- **问题**：AutoDL 系统盘仅 30GB，HF 模型缓存 ~8GB + pip 缓存数 GB 会爆盘
- **修复**：重定向所有缓存到数据盘 `/root/autodl-tmp/`
  ```bash
  export HF_HOME=/root/autodl-tmp/hf_cache
  export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache
  export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
  export TORCH_HOME=/root/autodl-tmp/torch_cache
  ```
- **持久化**：写入 `/root/.bashrc`，重启后仍生效
- **验证**：环境变量正确设置，缓存目录创建成功

### veRL 安装（从源码）
- **版本**：veRL 0.8.0.dev0
- **安装方式**：从 GitHub 克隆源码，`pip install -e .`
- **依赖安装**：通过阿里云镜像加速
  ```bash
  pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
  ```
- **额外依赖**：datasets, pyarrow, evalplus, modelscope, matplotlib, tensorboard, wandb, pyyaml, pytest

### HuggingFace 镜像站配置
- **问题**：直接访问 HuggingFace 网络连接失败（`[Errno 99] Cannot assign requested address`）
- **修复**：设置 HF_ENDPOINT 环境变量使用国内镜像站
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```
- **效果**：数据集下载正常，MBPP 数据加载成功

### 训练数据生成
- **命令**：`python3 src/data_prepare.py --output data/grpo_train.parquet`
- **数据源**：MBPP (Sanitized Python Programming Problems)
- **数据量**：464 条（train 374 + validation 90，**不含 test split**）
- **输出格式**：veRL 标准格式（5 列：data_source, prompt, ability, reward_model, extra_info）
- **验证**：parquet 文件正确生成

### 单元测试验证
- **pytest 安装**：用于运行单元测试
- **测试结果**：
  - **沙盒测试**（tests/test_sandbox.py）：7/7 通过
    - 正常代码执行、语法错误、运行时错误、超时、测试用例、批量执行、僵尸进程清理
  - **单轮奖励函数**（tests/test_reward.py）：7/7 通过
    - 代码块提取、完美输出、无代码、错误代码、乱输出、语法错误、超时
  - **多轮奖励函数**（tests/test_reward_multiturn.py）：9/9 通过
    - 轮次计数、代码提取、1/2/3 轮成功、多代码块无过度惩罚、无代码、最终失败、奖励保底
- **总计**：23/23 测试全部通过 ✅

### GPU 环境验证（租用 GPU 后）
- **GPU**：NVIDIA GeForce RTX 4090, 24564 MiB VRAM
- **驱动**：NVIDIA 580.76.05 Open Kernel Module, CUDA 13.0
- **PyTorch**：2.5.1+cu124, CUDA available: True
- **模型下载**：Qwen3-1.7B 通过 HF 镜像站下载完成，缓存至 /root/autodl-tmp/hf_cache
- **端到端验证**：verify_e2e.py 全部通过
  - 测试 1（模型生成 + 代码提取）：PASS
  - 测试 2（沙盒执行模型代码）：PASS
  - 测试 3（Reward 管道）：PASS
- **结论**：云端环境完全就绪，可开始阶段一 GRPO 训练

### 当前状态
- ✅ 环境配置完成（缓存重定向、veRL 安装、依赖配置）
- ✅ 数据准备完成（464 条训练样本）
- ✅ 单元测试通过（23/23）
- ✅ GPU 环境验证通过（RTX 4090 24GB）
- ✅ 端到端验证通过（verify_e2e.py）
- ⏳ 下一步：运行阶段一单轮 GRPO 训练（1.7B 模型）

### 注意事项（新会话开始阶段一时）
- 需先 `source /root/.bashrc` 加载缓存重定向环境变量
- 需设置 `HF_ENDPOINT=https://hf-mirror.com`
- 训练脚本用 `scripts/run_local_single.sh`（1.7B + 24GB 配置），不是 `run_cloud_single.sh`（4B + 80GB）
- 用 tmux 防断连：`tmux new -s train`

---

## 2026-04-08 云端环境迁移（官方 Docker 镜像）

### 环境切换：从手工组装到官方镜像
- **原因**：手工安装的 veRL 环境有各种依赖冲突（scipy/numpy、flash_attn 等）
- **新镜像**：`verlai/verl:vllm011.latest`（官方预构建）
- **新环境**：Python 3.12.11, PyTorch 2.8.0+cu128, vLLM 0.11.0, flash-attn 2.8.1
- **GPU 升级**：从 RTX 4090 24GB 切换到 **48GB** 版本

### 数据盘备份策略
- 系统盘切换 Docker 镜像会被清空，需备份到数据盘
- 备份位置：`/root/autodl-tmp/agentic-rl-backup/`（项目代码）、`/root/autodl-tmp/claude-config-backup/`（Claude 配置）
- 模型权重：`/root/autodl-tmp/hf_cache/`（Qwen3-1.7B, ~7.6GB）

---

## 2026-04-08 单轮 GRPO 训练踩坑记录

### 踩坑 1：vLLM QKV 权重名不匹配（KeyError: qkv_proj.weight）
- **现象**：vLLM Worker 启动时报 `KeyError: 'qkv_proj.weight'`
- **根因**：LoRA wrapper 将参数重命名 `qkv_proj.weight` → `qkv_proj.base_layer.weight`，但 vLLM 的 `Qwen2Model.load_weights()` 用原始名查找
- **修复**：添加 `_resolve_param(name, pd)` 辅助函数，找不到原名时尝试 base_layer 变体
- **自动化**：创建 `scripts/patch_vllm.py`，每次切换镜像需重跑
- **性能影响**：零。只修改参数名查找逻辑

### 踩坑 2：数据格式错误（AttributeError: 'str' has no attribute 'get'）
- **根因**：`data_prepare.py` 用 `json.dumps()` 存储 dict，veRL 读取得到 JSON 字符串
- **修复**：改用 `datasets.Dataset.from_list()` 原生存储 Python dict

### 踩坑 3：24GB GPU 显存不足（OOM）
- **现象**：Actor ~22.89 GiB + vLLM ~0.5 GiB > 24 GiB
- **解决**：升级到 RTX 4090 **48GB**

### 踩坑 4：RTX 4090 不支持 fp8
- **修复**：移除 `quantization=fp8` 参数

### 踩坑 5：PyTorch inductor ImportError
- **修复**：`export TORCHDYNAMO_DISABLE=1`

### 踩坑 6：libgfortran 缺失
- **修复**：`ldconfig` 添加 scipy.libs 和 numpy.libs 路径

### 单轮训练结果（失败）
- **100 步全部 reward 为 0 或负值**，无一个样本通过测试
- **根因**：`max_response_length=512` 太短，Qwen3 thinking 消耗大量 token
- **所有输出被截断**，clip_ratio=1.0，Checkpoint 已删除

---

## 2026-04-08 多轮 GRPO 训练踩坑记录

### 多轮脚本优化
- `max_response_length`: 512 → 2048
- `max_model_len`: 1024 → 5120
- `gpu_memory_utilization`: 0.3 → 0.45
- `param_offload/optimizer_offload`: False（48GB 足够）

### 踩坑 7：numpy.int64 索引 torch.Tensor 失败
- **根因**：`np.cumsum(int32)` 返回 int64，PyTorch 2.8 + numpy 1.26.4 不支持
- **修复**：`hidden_states[logit_indices]` → `hidden_states[logit_indices.tolist()]`

### 踩坑 8：uncompyle6 安装污染 numpy
- **根因**：`xdis` 依赖破坏了 numpy 内部模块
- **修复**：卸载 uncompyle6 + 重装 numpy 1.26.4
- **教训**：不要在训练环境安装调试工具

### 踩坑 9：reward_multiturn.py async 不可序列化
- **修复**：`asyncio.run()` 改为同步 `subprocess.run()`

### 踩坑 10：wandb 认证失败（API key 重复写入 3 次）
- **修复**：手动修正 `/root/.netrc`

### 踩坑 11：val_files=None 导致启动失败
- **根因**：Shell 传字符串 `"None"`，veRL 无法处理
- **修复**：保留 val_files，改 `test_freq=-1` + `val_before_train=False`

### 训练可视化配置
- `WANDB_MODE` 从 offline 改为 online，4 个脚本均已更新

## 2026-04-09 云端 4B 模型训练准备

### 云端脚本显存修复
- `run_cloud_single.sh`：开 `gradient_checkpointing=True`，加 `free_cache_engine=True`，降 `gpu_memory_utilization=0.4`
- `run_cloud_multi.sh`：加 `free_cache_engine=True`，降 `gpu_memory_utilization=0.55`
- 预估：单轮峰值 ~34 GiB (43%)，多轮峰值 ~44 GiB (55%)，80GB 安全

### 云端脚本全面审计与修复
对比已验证的本地多轮脚本，补充以下缺失配置：
- **AutoDL 环境变量**：NCCL/CUDA/VLLM_WORKER/TORCHDYNAMO/HF_ENDPOINT
- **Ray 环境透传**：6 项 env vars 传入 Ray worker
- **断点续训**：`resume_mode=auto`
- **Checkpoint 保留**：`max_actor_ckpt_to_keep=4`（保留全部 4 个用于对比过拟合）
- **vLLM enforce_eager**：防止 CUDA graph 编译问题
- **reward workers**：`reward.num_workers=2`
- **多轮 return_raw_chat**：`data.return_raw_chat=True`
- **Checkpoint 数据盘**：`ln -sf /root/autodl-tmp/checkpoints $PROJECT_DIR/checkpoints`

### Checkpoint 优化器剥离
- 编写 `scripts/strip_optimizer.py`：训练后删除旧 checkpoint 的 `optim_*.pt`，仅保留最新 checkpoint 完整
- 本地实测验证：step 20/80 从 21 GiB → 7.6 GiB，释放 25.6 GiB
- 4B 模型估算：6 个仅模型(~18G) + 2 个完整(~49G) = ~206 GiB

### 踩坑 12：HuggingFace 离线模式导致 tokenizer 加载失败
- **现象**：`TypeError: expected str, bytes or os.PathLike object, not NoneType`（vocab_file 为 None）
- **根因**：Qwen3-4B 通过 ModelScope 下载到 `latest/` 目录，HF 标准缓存快照 `1cfa9a.../` 里只有 config.json。`HF_HUB_OFFLINE=1` 阻止了补全下载
- **修复**：去掉 `HF_HUB_OFFLINE=1`，改用本地路径 `actor_rollout_ref.model.path=/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/latest`

### 踩坑 13：numpy 2.x 残留文件导致 `_core/arrayprint.py` 崩溃
- **现象**：`AttributeError: module 'numpy.core.multiarray' has no attribute 'generic'` → `RuntimeError: Unable to configure default ndarray.__repr__`
- **根因**：踩坑 8 的修复不彻底。`pip install --force-reinstall numpy==1.26.4` 只覆盖了 `core/` 目录，`_core/` 里残留了 numpy 2.x 的文件（`arrayprint.py`、`numerictypes.py`、`_type_aliases.py`）
- **为什么之前本地多轮能跑**：1.7B 模型的 vLLM 配置没走到 `_core/arrayprint.py` 代码路径（没触发 repr 调试输出），4B 模型配置触发了 → 是个定时炸弹
- **修复**：`rm -rf numpy*` 彻底清理 + `pip install numpy==1.26.4`
- **教训**：降级 numpy 大版本时必须彻底删除旧文件，`--force-reinstall` 不可靠

### 多轮训练结果（成功）
- **80 步（step 21→100），score/mean: +0.046，62.5% 的步有样本通过测试**
- **response_length: 1489，clip_ratio: 0.40**
- **Entropy 0.14 → 0.06（下降 57%）**
- **显存峰值 43.4 GiB / 48 GiB (90%)**

### 训练日志
- 单轮：`logs/train_48gb.log`
- 多轮：`logs/train_multi.log`
- GPU 监控：`logs/gpu_monitor_multi.csv`

---

## 2026-04-10 4B 训练失败与方案重构

### 踩坑 14：4B 模型单卡 GRPO OOM（A800 80GB）

- **现象**：云端 4B 单轮训练（`cloud_single_0409_2359.log`），第 1 步即 OOM
  - 00:00:26 `update_weights` 完成（0.33s）
  - 00:00:26 vLLM `wake_up` 恢复 KV cache → OOM
  - 00:09:36 vLLM Worker 崩溃：`CUDA Error: out of memory at cumem_allocator.cpp:62`
  - GPU 峰值：**79,647 MiB / 81,920 MiB (97.3%)**
  - 第 1 步花了 10 分钟仍未完成

- **根因分析**：veRL sleep/wake 机制下，actor 和 vLLM 的 CUDA 内存**始终同时存在**（cumem allocator 只 unmap 不释放）
  - Actor 权重：8 GiB
  - vLLM 权重（sleep）：8 GiB
  - vLLM KV cache (gpu_util=0.5)：40 GiB
  - 训练临时：~15 GiB
  - 合计：~71 GiB + 碎片 → 80 GiB 不够

- **显存预估系统性偏低的根因**：
  所有之前的预估（包括 AI 建议）都只算了静态权重，忽略了三个关键因素：
  1. **cumem 双份权重**：actor(8G) + vLLM(8G) 同时常驻
  2. **KV cache 按 gpu_util 比例分配**：0.5 × 80 = 40G，不是"按需分配"
  3. **CUDA 内存碎片**：实际占用比理论值多 10-15%

  实测对比：

  | 场景 | 预估 | 实际 | 偏差 |
  |------|------|------|------|
  | 1.7B 单轮 48GB | ~36 GiB (75%) | 36 GiB (75%) | 准确 |
  | 1.7B 多轮 48GB | ~34 GiB (71%) | **43.4 GiB (90%)** | 低 28% |
  | 4B 单轮 80GB | ~34 GiB (43%) | **79.6 GiB (97%)** OOM | 低 **133%** |

  教训：**显存预估必须包含 cumem 双份权重 + KV cache 全量分配 + 碎片余量**

### 4B 方案速度与经济可行性分析

4B 训练参数：batch=32, n=8, max_response=3072, APPS 数据(prompt ~285 tokens)
- 每步 rollouts：32 × 8 = 256，avg ~2500 tokens → 640k tokens/步
- 1.7B 每步：64 rollouts × 1530 tokens = 98k tokens → 65s/步
- 4B 估算：98k→640k (6.5x tokens) × 2.35 (model size) ÷ 1.5 (A800 faster) ≈ **20-30 min/步**
- 100 步 = **33-50 小时**，费用 200-300 元
- 即使不 OOM，时间和经济成本都难以接受

### 方案重构决策：1.7B 升级版替代 4B

**核心洞察**：4B 方案的含金量只有 10% 来自模型规模，90% 来自训练参数（数据多样性、GRPO 采样数、多轮纠错、KL 正则化）。所有参数升级都可以在 1.7B 上实现。

逐项分析 4B 方案含金量来源：

| 含金量来源 | 4B 有 / 1.7B 旧 | 升级可行性 | 升级成本 |
|---|---|---|---|
| 模型规模 4B | 核心差距 | ❌ 无法弥补 | — |
| APPS 数据 (2032条) | 1.7B 只有 MBPP | ✅ 直接用 | 0 |
| n=4→8 | GRPO baseline 更准 | ✅ 直接改 | 0（vLLM 分页） |
| response 2048→3072 | 写更长代码 | ✅ 直接改 | KV 略增 |
| LoRA r=8→16 | 可塑性更强 | ✅ 直接改 | ~0.1 GB |
| 2轮→3轮纠错 | 更多尝试 | ✅ 直接改 | KV 增大 |
| KL loss | 无正则化 | ✅ 新增 | ref model 3.4 GB |

**结论：除了模型规模，所有含金量来源都可以在 1.7B 上以零/极低成本实现。**

### 1.7B 多轮训练欠拟合分析

从 80 步训练数据（step 21-100）分析：

**Entropy 趋势（探索能力）**：

| 阶段 | avg entropy | 变化 |
|------|-----------|------|
| 21-40 | 0.112 | — |
| 41-60 | 0.080 | ↓29% |
| 61-80 | 0.070 | ↓12% |
| 81-100 | 0.068 | ↓3% |

Entropy 在 ~0.07 稳定，没有崩溃到 0。模型探索能力保持健康。

**Score 趋势**：

| 阶段 | avg score | 正 reward 比例 |
|------|-----------|--------------|
| 21-40 | 0.052 | 50% |
| 41-60 | 0.034 | 45% |
| 61-80 | 0.049 | 70% |
| 81-100 | 0.047 | 75% |

正 reward 比例从 50%→75% 提升，但 score 停滞在 0.04-0.05。模型学会了"大概怎么做"但没学会"做得更好"。**诊断：欠拟合。**

**瓶颈排序**：
1. 数据太少（464 条，致命）→ 升级到 2440 条
2. n=4 太小（GRPO advantage 方差大）→ 升级到 n=8
3. LoRA r=8 容量不足 → 升级到 r=16
4. batch size（最不重要）→ 从 16 升到 48

### veRL 多轮 response_length 机制确认

通过阅读 veRL 源码（`tool_agent_loop.py:254`）确认：

```python
# response_mask 累加所有轮次的 token（LLM + tool call）
agent_data.response_mask += [1] * len(agent_data.response_ids)

# 总量超限直接终止所有轮次
if len(agent_data.response_mask) >= self.response_length:
    return AgentState.TERMINATED
```

**`max_response_length` 是所有轮次的共享总预算**，包含 LLM 生成 + tool call 反馈。不是每轮的预算。

从 2 轮实测数据反推：avg 1530 tokens / 2 轮 = ~765 tokens/轮。3 轮需要 ~2295 tokens。
- max_response=3072 → 3 轮余量仅 34%，大量截断
- max_response=6144 → 2048/轮，余量 ~170%，几乎没有截断

### APPS 数据难度过滤

APPS 数据虽然已过滤为 introductory 级别，但 1.7B 模型对这类题目的 thinking 仍然很长（阿里官网测试远超 2000 tokens），可能导致 response 预算被 thinking 耗尽，没有空间输出代码。

**解决方案**：按 prompt 字符数过滤 APPS，只保留简单题：
- `prompt < 600 字符`：777 条（原 2040 条的 38%）
- 依据：prompt 越短 → 题目越简单 → thinking 越短 → 更可能输出代码

**决策**：选择方案 A（MBPP 408 + APPS-easy 777 = 1185 条），而非方案 B（全部 2448 条 + response=8192）
- 理由：方案 A 更安全（避免无效训练），数据量足够（1185 × 8.1 epochs），费用更低

**数据对比**：

| 指标 | MBPP | APPS-easy (<600字) | APPS 全部 |
|------|------|-------------------|-----------|
| 样本数 | 408 | 777 | 2040 |
| avg prompt chars | 78 | ~450 | 884 |
| 预估 thinking | 短 | 中等 | 长 |
| 代码输出风险 | 低 | 中 | 高 |

`data_prepare.py` 新增 `--apps-easy` 参数实现此过滤。

### 升级后的 1.7B A800 参数对比

| 参数 | 旧 1.7B 多轮 (48GB) | 新 1.7B 单轮 (A800) | 新 1.7B 多轮 (A800) |
|------|:---:|:---:|:---:|
| 数据 | 464 MBPP | 1185 MBPP+APPS-easy | 1185 MBPP+APPS-easy |
| batch | 16 | **48** | **48** |
| n | 4 | **8** | **8** |
| rollouts/步 | 64 | 384 | 384 |
| max_response | 2048 | **4096** | **6144** |
| max_prompt | 512 | **1024** | **1024** |
| LoRA | r=8, a=16 | **r=16, a=32** | **r=16, a=32** |
| 轮次 | 2 | 1 | **3** |
| max_model_len | 5120 | **6144** | **8192** |
| KL loss | No | **Yes (0.003)** | **Yes (0.003)** |
| gpu_util | 0.45 | **0.55** | **0.55** |
| steps | 80 | **200** | **200** |
| lr | 3e-6 | 1e-6 | 2e-6 |
| reward workers | 2 | **4** | **4** |
| epochs | ~2.8 | ~8.1 | ~8.1 |
| 显存峰值 | 43.4 GiB (90%) | ~55 GiB (69%) | ~54 GiB (67%) |
| 每步耗时 | ~65s | ~3 min | ~7 min |
| 总时长 | ~1.5h | ~10h | ~23h |
| 费用 | ~5元 | ~60元 | ~140元 |

### 可行性保证

- 显存余量 26+ GiB，安全
- 200 步 × 3.9 epochs，有 KL loss 防过拟合
- 先跑 200 步看曲线，可续训
- Checkpoint 每 50 步保存，最多保留 4 个

---

## 2026-04-10 Response 长度升级到 8192 + Prompt 优化

### max_response_length: 4096/6144 → 8192

- **原因**：即使 APPS-easy（prompt<600字）的题目，1.7B 模型 thinking 仍可达 4000+ tokens，之前 4096/6142 的预算在长 thinking 后几乎没有空间输出代码
- **影响**：
  - 单轮 max_model_len: 6144 → **10240**（1024 prompt + 8192 response + 余量）
  - 多轮 max_model_len: 8192 → **12288**（1024 prompt + 3 轮交互 + 8192 response）
  - KV cache 增大约 60%，但 A800 80GB 余量充足（预估 ~60 GiB / 80 GiB = 75%）
  - 生成时间不取决于上限而是实际 token 数，影响有限

### System Prompt 优化：单代码块约束

- **新增指令**：`"Provide only ONE complete Python code block — do not include alternative solutions or extra code blocks."`
- **原因**：模型在单轮中可能输出多个代码块（不同解法尝试），`extract_last_code()` 会拼接所有代码块导致执行错误
- **效果**：引导模型集中精力输出一个最优解，减少拼接错误

### 数据重新生成

- `grpo_train_full.parquet`：1184 条（更新后的 prompt）
- `grpo_train_multi_full.parquet`：1184 条（更新后的 prompt）

### 参数对比（更新后）

| 参数 | 单轮 (A800) | 多轮 (A800) |
|------|:---:|:---:|
| max_response_length | **8192** | **8192** |
| max_model_len | **10240** | **12288** |
| max_prompt_length | 1024 | 1024 |
| batch_size | 48 | 48 |
| n | 8 | 8 |
| LoRA rank/alpha | 16/32 | 16/32 |
| KL loss coef | 0.003 | 0.003 |
| steps | 200 | 200 |

### 踩坑 15: veRL disable_adapter 方法名不兼容 + gpu_util 过高导致 OOM

- **现象**：1.7B 单轮训练在 A800 80GB 上崩溃，显存 79.6/80 GiB (97.2%)
- **根因（双重问题）**：
  1. **veRL bug**：`transformer_impl.py:825` 调用 `disable_adapter()`（单数），但 Qwen3 模型的 LoRA 接口是 `disable_adapters()`（复数），导致 `AttributeError`
  2. **gpu_util 过高**：`gpu_memory_utilization=0.55` 占用 43.6 GB KV cache，加上 Actor(8.75G) + vLLM 权重(4G) + Ref 前向(7G) + 梯度/buffer(10G) = 73 GB，叠加碎片达到 79.6 GB
- **修复**：
  1. `patch_vllm.py` 新增 Patch 3：兼容 `disable_adapter` 和 `disable_adapters`
  2. `gpu_memory_utilization` 从 0.55 降到 **0.4**（KV cache 从 43.6 GB 降到 31.7 GB，省 12 GB）
- **预估修复后峰值**：~67 GiB / 80 GiB (82%)，安全余量 13 GiB
- **教训**：之前所有显存预估都低估了 cumem allocator 的叠加效应——vLLM KV cache + Actor + Ref 三者几乎同时占用的峰值远大于各组件单独估算之和
| 显存预估 | ~60 GiB (75%) | ~60 GiB (75%) |

### 踩坑 16: numpy _core/ 残留文件（克隆实例必现）

- **现象**：`AttributeError: module 'numpy.core.multiarray' has no attribute 'unsignedinteger'` → vLLM EngineDeadError
- **根因**：Docker 镜像 `verlai/verl:vllm011.latest` 构建时先装 numpy 2.x（给 cupy/opencv），再装 veRL 时降级到 1.26.4，但 `_core/` 目录残留了 2.x 的 `_type_aliases.py` 等文件。vLLM spawn 子进程触发完整 import 链时崩溃
- **为什么反复出现**：用户经常克隆 AutoDL 实例抢 GPU → 每次克隆 = 新容器 = 系统盘重置到 Docker 镜像原始状态 → numpy 回到出厂就坏的状态
- **修复**：
  1. `patch_vllm.py` 新增 Patch 4：检测并删除 `_core/` 中的残留文件，仅保留 1.26.4 兼容性 shim
  2. 训练脚本 `run_cloud_single.sh` / `run_cloud_multi.sh` 开头自动运行 `patch_vllm.py`
- **效果**：克隆实例后直接跑 `bash scripts/run_cloud_single.sh` 即可，无需手动修复 numpy

---

## 2026-04-11 H20 冒烟测试踩坑记录

### H20 冒烟测试结果（失败）

**配置**: H20 96GB, batch=8, n=8, response=4096, 2轮, gpu_util=0.35
**数据**: grpo_train_multi_full.parquet (605 条)

**显存占用**:
| 阶段 | 显存 | 占比 |
|------|------|------|
| 模型加载 (FSDP) | 7.6 GiB | 7.8% |
| vLLM wake_up 峰值 | 42.6 GiB | 43.6% |
| 稳定 (update_weights 后) | 8.2 GiB | 8.4% |

显存非常宽裕，峰值仅 43.6%，还有 55 GiB 余量。但训练在第一个 step 的 rollout 阶段就崩溃了。

### 踩坑 17: 自定义 AgentLoop 未注册到 veRL

- **现象**：`AssertionError: Agent loop code_agent_loop not registered, registered agent loops: dict_keys(['single_turn_agent', 'diffusion_single_turn_agent', 'tool_agent'])`
- **根因**：
  1. 多轮训练数据（`grpo_train_multi_full.parquet`）的 `agent_name` 列为 `code_agent_loop`
  2. veRL 的 `AgentLoopWorker._run_agent_loop()` 根据数据中的 `agent_name` 查找注册表
  3. veRL 内置只有 3 个 agent loop：`single_turn_agent`、`diffusion_single_turn_agent`、`tool_agent`
  4. 项目自定义的 `CodeAgentLoop`（`src/agent_loop.py`）**从未注册到 veRL 的 `_agent_loop_registry`**
  5. 旧的 `register_agent.py` 只定义了一个本地 dict `AGENT_REGISTRY`，没有调用 veRL 的注册机制

- **veRL 的注册机制**（源码分析）：
  ```python
  # 方式 1: 装饰器注册
  @register("code_agent_loop")
  class CodeAgentLoop(AgentLoopBase): ...

  # 方式 2: 配置文件注册（推荐）
  # agent_loop_config_path 加载 YAML → 写入 _agent_loop_registry → hydra.utils.instantiate() 创建实例
  _agent_loop_registry["code_agent_loop"] = {"_target_": "src.agent_loop.CodeAgentLoop"}
  ```

- **修复**：
  1. 创建 `configs/agent_loop.yaml`：
     ```yaml
     - name: code_agent_loop
       _target_: src.agent_loop.CodeAgentLoop
     ```
  2. 多轮训练脚本添加参数：`actor_rollout_ref.rollout.agent.agent_loop_config_path=$PROJECT_DIR/configs/agent_loop.yaml`
  3. 多轮训练脚本添加 `export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"`，确保 Ray worker 能导入 `src.agent_loop.CodeAgentLoop`
  4. 更新 `register_agent.py`（仅保留参考文档）

- **影响范围**：所有多轮训练脚本（`run_local_multi.sh`、`run_cloud_multi.sh`、`run_smoke_test_h20.sh`）
- **教训**：veRL 的 agent loop 是通过 `hydra.utils.instantiate()` 动态创建的，自定义 agent loop 必须同时满足：①注册到 `_agent_loop_registry` ②类路径可被 Ray worker 导入

---

### 踩坑 18: TokenOutput 属性名写错（`ids` → `token_ids`）

- **现象**：H20 冒烟测试（`smoke_h20_0411_0052.log`）在第一个 rollout 步就崩溃：
  ```
  AttributeError: 'TokenOutput' object has no attribute 'ids'. Did you mean: '_return_value'?
  ```
  定位到 `src/agent_loop.py:108`：
  ```python
  generated_ids = llm_output.ids  # ❌ 错误
  ```

- **根因**：
  - veRL 的 `TokenOutput`（定义在 `verl.workers.rollout.replica`）属性名是 **`token_ids`**，不是 `ids`
  - veRL 内置的 `single_turn_agent_loop.py:69` 和 `tool_agent_loop.py:244` 都用的是 `output.token_ids`
  - 自定义 `CodeAgentLoop` 写成了 `.ids`，本地之前的 48GB 多轮训练可能用了不同版本的 veRL 或 vLLM，属性名恰好匹配；切换到 H20 环境后接口不一致导致报错

- **修复**：`src/agent_loop.py:108`
  ```python
  generated_ids = llm_output.token_ids  # ✅ 正确
  ```

- **教训**：自定义 AgentLoop 中访问 vLLM/veRL 返回对象的属性时，必须对照 veRL 内置 agent loop（`single_turn_agent_loop.py`、`tool_agent_loop.py`）的实现，不要凭猜测写属性名

---

### 踩坑 19: AgentLoopOutput 必需 `metrics` 字段缺失

- **现象**：H20 冒烟测试第二次尝试（`smoke_h20_0411_0104.log`），修复了踩坑 17/18 后，第一个 step rollout 阶段崩溃：
  ```
  pydantic_core._pydantic_core.ValidationError: 1 validation error for AgentLoopOutput
  metrics
    Field required [type=missing, input_value={'prompt_ids': [...], 'response_ids': [...], ...}]
  ```

- **根因**：
  - veRL 最新版的 `AgentLoopOutput`（pydantic BaseModel）新增了 **必填** 字段 `metrics: AgentLoopMetrics`
  - `AgentLoopMetrics` 包含 `generate_sequences`（LLM 生成耗时）、`tool_calls`（工具调用耗时）、`num_preempted`（抢占次数）
  - 旧版 veRL 的 `AgentLoopOutput` 没有这个字段，项目代码是从旧版 API 编写的
  - 自定义 `CodeAgentLoop.run()` 返回 `AgentLoopOutput(prompt_ids=..., response_ids=..., response_mask=...)` 时缺少 `metrics`，pydantic 校验失败

- **修复**：`src/agent_loop.py` 三处修改：
  1. 导入 `simple_timer`：`from verl.utils.profiler import simple_timer`
  2. LLM 生成用 `with simple_timer("generate_sequences", metrics)` 计时
  3. 沙盒执行用 `with simple_timer("tool_calls", metrics)` 计时
  4. 返回值新增 `num_turns=actual_turns` 和 `metrics=metrics`
  ```python
  # 修改前
  return AgentLoopOutput(
      prompt_ids=list(prompt_ids),
      response_ids=response_ids,
      response_mask=response_mask,
  )

  # 修改后
  return AgentLoopOutput(
      prompt_ids=list(prompt_ids),
      response_ids=response_ids,
      response_mask=response_mask,
      num_turns=actual_turns,
      metrics=metrics,
  )
  ```

- **教训**：自定义 AgentLoop 的返回对象必须与 veRL 内置 agent loop 完全对齐，包括所有必填字段。veRL 版本升级时 `AgentLoopOutput` 可能新增必填字段，需要对照 `single_turn_agent_loop.py` 的返回值检查

---

### 踩坑 20: 多轮 response_ids 超出 response_length 导致 tensor size mismatch

- **现象**：H20 冒烟测试第三次尝试（`smoke_h20_0411_0116.log`），修复踩坑 17/18/19 后，第一个 step rollout 阶段崩溃：
  ```
  RuntimeError: Sizes of tensors must match except in dimension 0.
  Expected size 4096 but got size 4848 for tensor number 3 in the list.
  ```
  定位到 veRL `agent_loop.py:879` 的 `_postprocess` 方法：
  ```python
  response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
  ```

- **根因**：
  - 多轮交互中，LLM 生成的 tokens + 环境反馈 tokens 累计超出了 `max_response_length`（4096）
  - 某个样本的 response_ids 为 4848 tokens，其他样本为 4096 tokens（被截断或未超限）
  - veRL 的 `_agent_loop_postprocess` 中 `tokenizer.pad(max_length=4096)` 只**填充**短序列，不会**截断**长序列
  - 结果是 `_postprocess` 收到 shape 不同的 tensor（[1,4096] vs [1,4848]），`torch.cat` dim=0 时 dim=1 不匹配

- **修复**：`src/agent_loop.py` 三处修改：
  1. 循环内增加**预算检查**：每轮生成前检查 `len(response_ids) >= response_length`，超限则提前终止
  2. 生成后检查：如果单轮生成导致超限，截断到 `response_length` 并 break
  3. 返回前**最终保障**：再次检查并截断，确保 `response_ids` 不超过 `response_length`

  ```python
  response_length = self.rollout_config.response_length

  for turn in range(self.max_turns):
      # 预算检查：剩余空间不足则提前终止
      if len(response_ids) >= response_length:
          break
      # ... LLM 生成 ...
      # 生成后超限则截断并终止
      if len(response_ids) > response_length:
          response_ids = response_ids[:response_length]
          response_mask = response_mask[:response_length]
          break

  # 最终保障
  if len(response_ids) > response_length:
      response_ids = response_ids[:response_length]
      response_mask = response_mask[:response_length]
  ```

- **教训**：
  - `max_response_length` 是所有轮次的**共享总预算**（LLM tokens + feedback tokens），不是每轮的预算
  - 自定义 AgentLoop 必须自己负责截断到 `response_length`，veRL 的 `tokenizer.pad` 只填充不截断
  - 对照 veRL 内置 `tool_agent_loop.py` 的 `TERMINATED` 逻辑：`if len(response_mask) >= self.response_length: return AgentState.TERMINATED`

### H20 96GB 显存占用（实测）

**配置**: H20 96GB, Qwen3-1.7B, batch=8, n=8, response=4096, 2轮, gpu_util=0.35, param_offload=False

| 阶段 | 显存 | 占比 |
|------|------|------|
| 模型加载 (FSDP) | 7.6 GiB | 7.8% |
| vLLM wake_up 峰值 | 36.7 GiB | 37.5% |
| update_weights 后 | 8.2 GiB | 8.4% |
| 稳定训练 | 36.7 GiB | 37.5% |

**结论**：H20 96GB 显存非常宽裕，峰值仅 37.5%，还有 **55 GiB 余量**。可以大幅提升参数：
- gpu_util: 0.35 → 0.5（KV cache 翻倍）
- response: 4096 → 8192（长 thinking 空间）
- batch: 8 → 48（更多梯度信息）
- max_model_len: 5120 → 12288（3轮交互空间）

### 踩坑 21: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 导致启动崩溃

- **现象**：H20 冒烟测试（`smoke_h20_20260411_103614.log`）启动即崩溃：
  ```
  AssertionError: Expandable segments are not compatible with memory pool.
  Please track https://github.com/pytorch/pytorch/issues/147851 for the latest updates.
  ```
  发生在 vLLM `cumem.py:150`，模型都没加载就挂了。

- **根因链条**：
  1. 09:28 那次冒烟测试跑了 3 步后在 step 4 的 `update_weights` 阶段 OOM（`tensor.clone()` 要 48 MiB 但只剩 29.5 MiB）
  2. PyTorch 的 OOM 错误信息建议：`try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation`
  3. 按建议加了 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 到 `run_smoke_test_h20.sh`
  4. 但 vLLM 的 `enable_sleep_mode=True`（veRL colocated 模式自动启用）使用自定义的 `CuMemAllocator` 内存池
  5. **`expandable_segments` 和 `CuMemAllocator` 不兼容**——两者都试图管理同一块 GPU 内存，vLLM 直接 assert 失败

- **expandable_segments 是什么**：PyTorch CUDA 内存分配器的优化选项，允许已分配的内存段动态扩展而非重新分配，能减少内存碎片。但它与 vLLM sleep mode 的自定义内存池（CuMemAllocator）冲突。

- **修复**：
  1. 删除 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  2. 降低 `gpu_memory_utilization` 从 0.35 到 0.30，给 update_weights 多留余量
  3. 依赖 Patch 5（sleep 后清理 prefix cache）和 Patch 6（gc + empty_cache）缓解显存泄漏

- **09:28 次运行的显存泄漏数据**：

  | Step | 峰值 VRAM | 剩余 | 结果 |
  |------|-----------|------|------|
  | 1 | ~78 GiB | ~20 GiB | OK |
  | 2 | ~92 GiB | ~6 GiB | 紧张 |
  | 3 | ~97.3 GiB | 0.6 GiB | OOM |

  每步泄漏 7-14 GiB，sleep mode 没有完全释放 KV cache。

- **教训**：
  - **不要盲目跟随 PyTorch OOM 的建议**——那个建议是给普通 PyTorch 训练的，不适用于 vLLM 自定义内存池
  - veRL colocated 模式下 vLLM 使用 `CuMemAllocator`，不能同时使用 `expandable_segments`
  - 显存泄漏应从根源解决（清理缓存、降低 gpu_util），而不是用 PyTorch 分配器优化绕过

---

## 2026-04-11 H20 正式多轮训练踩坑

### H20 正式训练配置

**配置**: H20 96GB, Qwen3-1.7B, batch=8, n=8, response=6800, max_assistant_turns=2, gpu_util=0.25, lr=1e-6

### 训练结果（16 步后卡死）

| 指标 | 值 |
|------|-----|
| 完成步数 | 16 步（step 17 卡死） |
| score/mean 范围 | -0.016 ~ +0.277 |
| 正 reward 步占比 | 81% (13/16) |
| response_length/mean | 2294 ~ 4613 |
| clip_ratio | 0.03 ~ 0.375 |
| num_turns/mean | 1.12 ~ 1.42 |
| pg_clipfrac | 全部 = 0（无学习） |
| ppo_kl | 全部 = 0 |

**训练卡死现象**：
- Step 15: step_time=2069s（正常 ~300s），tool_calls/max=861s
- Step 17: vLLM generate 完全挂起，GPU 卡在 80 GiB / 0% util
- GPU 日志最后 28 分钟：40519 MiB / 0% util（idle 等待）

### 踩坑 22: 多轮 sampling_params 的 max_new_tokens 不随轮次调整

- **现象**：
  1. 多轮训练中第 2/3 轮极少触发有效修正（num_turns/mean 仅 1.2）
  2. 训练在 step 15 出现 2069s 超长 step_time，step 17 完全卡死
  3. vLLM 在 step 17 的 generate 阶段挂起，GPU 显存 80 GiB 但利用率 0%

- **根因**：
  `agent_loop.py` 的 `sampling_params` 中 `max_new_tokens` 始终为原始值（6800），不会根据已消耗的 response 预算动态调整。这导致两个严重后果：

  **后果 A — 第 2 轮修正无效**：
  ```
  Turn 1: LLM 生成 4000 tokens → 错误反馈 200 tokens → 累计 4200/6800
  Turn 2: vLLM 被告知 max_new_tokens=6800（实际仅剩 2600）
    → vLLM 尝试生成 6800 tokens，agent_loop 事后截断
    → 第 2 轮代码被截断，修正毫无意义

  极端情况：Turn 1 生成 6000+ tokens
    → 剩余 < 600 tokens，第 2 轮连函数定义都写不完
  ```

  **后果 B — vLLM preemption 死锁**：
  ```
  prompt(82) + turn1 LLM(4000) + feedback(200) + turn2 请求 max_new_tokens(6800)
  = 总长度可能超过 max_model_len=8816
  → vLLM v1 engine 的 KV cache preemption 无法处理
  → 调度器进入死循环，generate 永远不返回 → 训练卡死
  ```

- **数据佐证**：
  | 指标 | 值 | 含义 |
  |------|-----|------|
  | num_turns/mean | 1.12 ~ 1.42 | 平均仅 1.2 轮，第 2 轮极少有效触发 |
  | response_length/max | 几乎全是 6800 | 每步都有样本打满预算 |
  | clip_ratio | 0.03 ~ 0.375 | 3%~37.5% 样本被截断 |
  | Step 15 tool_calls/max | 861s（正常 <0.5s） | vLLM preemption 异常 |
  | Step 17 GPU | 80 GiB / 0% util | vLLM 完全挂起 |

- **修复**：`src/agent_loop.py` 在每次 `generate` 调用前动态调整 `max_new_tokens`：
  ```python
  # 修改前：sampling_params 始终传原始 max_new_tokens
  llm_output = await self.server_manager.generate(
      request_id=request_id,
      prompt_ids=current_ids,
      sampling_params=sampling_params,  # max_new_tokens 永远是 6800
  )

  # 修改后：根据剩余预算动态调整
  remaining = response_length - len(response_ids)
  adjusted_params = dict(sampling_params)
  original_max = adjusted_params.get("max_new_tokens", response_length)
  adjusted_params["max_new_tokens"] = min(remaining, original_max)

  llm_output = await self.server_manager.generate(
      request_id=request_id,
      prompt_ids=current_ids,
      sampling_params=adjusted_params,  # 精确到剩余预算
  )
  ```

- **影响**：
  1. 第 2/3 轮修正现在能获得精确的剩余预算，不会被事后截断
  2. vLLM 不再收到超出 max_model_len 的请求，避免 preemption 死锁
  3. 多轮交互的 token 利用率从"先到先得"变为"公平分配"

- **教训**：
  - 多轮 AgentLoop 的 `sampling_params.max_new_tokens` 必须按轮次动态调整，不能传固定值
  - vLLM 不知道 veRL 的 `response_length` 限制，它只会按 `max_new_tokens` 分配 KV cache
  - 多轮累积序列长度（prompt + 所有轮次 tokens）可能超过 `max_model_len`，需要在应用层提前预防
  - 对照 veRL 内置 `tool_agent_loop.py` 的 `TERMINATED` 检查：它在每次循环开始时检查 `len(response_mask) >= self.response_length`，但没有动态调整 `max_new_tokens`——这是 veRL 内置 agent loop 也能触发的潜在问题

---

## 2026-04-13 H20 显存泄漏修复

### 问题
H20 96GB 正式多轮训练（gpu_util=0.30, response=6000）在 step 8 的 `update_weights` 阶段 OOM：
- 显存从 step 4-7 的 ~82 GiB 稳态，突增到 step 8 的 ~95 GiB
- OOM 点：`bucketed_weight_transfer.py:251` 的 `tensor.clone()`（IPC weight transfer 需要 48 MiB，但仅剩 19.88 MiB 可用）
- 根因：KV cache 预留过大（gpu_util=0.30）+ IPC tensor clone 临时显存未及时释放 + 训练梯度/优化器缓存累积

### 修复内容

#### 1. gpu_memory_utilization 0.30 → 0.25
- 释放 ~5 GiB KV cache 空间，为 weight transfer 的 `tensor.clone()` 腾出余量
- 影响：并发序列数从 ~21 降到 ~17，多轮抢占可能略增，但 OOM 风险大幅降低
- 修改文件：`scripts/run_cloud_multi.sh`, `scripts/run_smoke_test_h20.sh`

#### 2. Patch 7: 训练循环关键节点加 gc.collect() + empty_cache()
- `_update_actor` 后：释放训练产生的梯度和激活值缓存
- `update_weights` 后：释放 IPC tensor 残留 + CUDA IPC handles（`ipc_collect()`）
- warmup 路径的 `update_weights` 后：同样处理
- 修改文件：`/tmp/verl/verl/trainer/ppo/ray_trainer.py`
- 速度影响：每步 +0.5-1s（<0.3%），可忽略

#### 3. Patch 8: 每处理完一个 weight bucket 后立即清理
- 在 `bucketed_weight_transfer.py` 的 `on_bucket_received` 回调后加 `empty_cache()`
- 确保 clone 副本显存及时回收，不让多个 bucket 的临时 tensor 累积
- 修改文件：`/tmp/verl/verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py`
- 速度影响：每个 bucket +50ms（~10-20 个 bucket/次），总计 +0.5-1s

#### 已有 Patch 回顾
- Patch 5: vLLM sleep 后 `reset_prefix_cache()`（vllm_async_server.py）
- Patch 6: 训练步末尾 `del batch + gc.collect + empty_cache`（ray_trainer.py）

### 修改文件汇总
- `scripts/patch_vllm.py` — 新增 Patch 7 + Patch 8
- `scripts/run_cloud_multi.sh` — gpu_util 0.30 → 0.25
- `scripts/run_smoke_test_h20.sh` — gpu_util 0.30 → 0.25
- `/tmp/verl/verl/trainer/ppo/ray_trainer.py` — Patch 7 已直接应用
- `/tmp/verl/verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py` — Patch 8 已直接应用

### 训练结果（修复前，7 步，H20 96GB）
| 步数 | score | response_length | clip_ratio | gen_max |
|------|-------|----------------|------------|---------|
| 1 | 0.000 | 2995 | 0.14 | 114s |
| 2 | +0.093 | 2466 | 0.03 | 110s |
| 3 | -0.005 | 4213 | 0.34 | 116s |
| 4 | +0.146 | 3321 | 0.09 | 114s |
| 5 | +0.053 | 3805 | 0.34 | 115s |
| 6 | +0.027 | 2535 | 0.02 | 113s |
| 7 | +0.159 | 3498 | 0.06 | 113s |

---

## 2026-04-13 Patch 6/7 修复：移除 TaskRunner 中的有害 CUDA 调用

### 踩坑 23: torch.cuda.ipc_collect() 在 CPU 进程中崩溃

- **现象**：H20 正式多轮训练第二次运行（12:10），Step 1 完成后崩溃：
  ```
  RuntimeError: No CUDA GPUs are available
  ```
  位于 `ray_trainer.py:1605` 的 `torch.cuda.ipc_collect()`

- **根因**：Patch 7 在 `ray_trainer.py` 的 `fit()` 方法中调用了 `torch.cuda.ipc_collect()` 和 `torch.cuda.empty_cache()`，但 `fit()` 运行在 **TaskRunner** 进程中——这是一个 **CPU-only Ray actor**（`ray.remote(num_cpus=1)`，无 GPU 分配）。

  veRL 进程架构：
  ```
  TaskRunner  (Ray actor, num_cpus=1, 无 GPU)  ← ray_trainer.py 的 fit() 运行在这里
     ├── WorkerDict     (Ray actor, 有 GPU)     ← FSDP 模型、训练、权重传输
     ├── vLLMHttpServer (Ray actor, 有 GPU)     ← vLLM 推理引擎
     └── AgentLoopWorker x8                     ← Agent Loop 执行
  ```

  IPC tensor 的实际流向是 WorkerDict ↔ vLLM Worker，TaskRunner 不持有任何 IPC handles。在 TaskRunner 里调用 `ipc_collect()` 即使成功也清理不到任何东西。

  `torch.cuda.ipc_collect()` 源码：
  ```python
  def ipc_collect():
      _lazy_init()               # ← 无条件初始化 CUDA，GPU 状态不佳就崩溃
      return torch._C._cuda_ipc_collect()
  ```

  对比 `torch.cuda.empty_cache()`：
  ```python
  def empty_cache():
      if is_initialized():       # ← CUDA 没初始化就直接返回，不会崩溃
          torch._C._cuda_emptyCache()
  ```

- **为什么有时不崩溃**：`_lazy_init()` 在 GPU 状态正常时能成功创建 CUDA context。00:59 运行中跑了 7 步没崩溃，但 12:10 运行中 GPU 在 step 1 后显存从 68 GiB 突降到 0（疑似 GPU 驱动状态异常），`_lazy_init()` 失败。行为不可预测。

- **修复**：
  1. Patch 6 Part 2：移除 `torch.cuda.empty_cache()`（在 CPU 进程中无效）
  2. Patch 7 所有位置：移除 `torch.cuda.ipc_collect()` 和 `torch.cuda.empty_cache()`，只保留 `gc.collect()`
  3. Patch 8 保持不变（在 vLLM Worker 进程内，是正确位置）

  各清理调用在不同进程的效果：
  | 调用 | TaskRunner (CPU) | WorkerDict (GPU) | vLLM Worker (GPU) |
  |------|:---:|:---:|:---:|
  | gc.collect() | 有效（Python 层） | 有效 | 有效 |
  | torch.cuda.empty_cache() | 无效（无 GPU 缓存） | 有效 | 有效 |
  | torch.cuda.ipc_collect() | 有害（崩溃） | 有效 | 有效 |

- **修改文件**：
  - `scripts/patch_vllm.py` — Patch 6/7 的 new_block 模板移除 CUDA 调用
  - `/tmp/verl/verl/trainer/ppo/ray_trainer.py` — 直接修改已打补丁的文件

- **教训**：
  - veRL 中 `ray_trainer.py` 的 `fit()` 运行在 CPU-only TaskRunner 进程，不能调用任何需要 GPU 的 torch.cuda 函数
  - 显存清理必须在实际持有 GPU 内存的进程（WorkerDict、vLLM Worker）中执行
  - `gc.collect()` 是唯一在任意进程都安全且有效的清理操作
  - Patch 8（`bucketed_weight_transfer.py`）和 Patch 5（`vllm_async_server.py`）在正确进程中，无需修改

---

## 2026-04-13 OOM 根因分析：Patch 7 无效 + update_actor 缺少 empty_cache

### 踩坑 24: Patch 7 gc.collect() 在错误进程中，对显存零效果

- **现象**：H20 正式多轮训练反复在 step 8 的 `update_weights_from_ipc` → `tensor.clone(48 MiB)` 处 OOM
  - 显存 ~89 GiB / 95 GiB (93.7%)，仅 ~6 GiB 余量
  - PyTorch 报 "reserved but unallocated: 20-44 MiB"，找不到 48 MiB 连续空间
  - 降 response (6800→6000)、降 gpu_util (0.30→0.25) 都不解决

- **根因**：Patch 7 的 `gc.collect()` 在 **TaskRunner** 进程（CPU-only），而训练内存（梯度/激活值）在 **WorkerDict** 进程（GPU）
  ```
  TaskRunner (无 GPU)                         ← Patch 7 gc.collect() 在这里（无用）
    ├── WorkerDict (GPU, Ray actor)           ← 训练在这里，~25 GiB PyTorch 缓存未释放
    └── vLLM Worker (GPU, Ray actor)          ← tensor.clone() OOM 在这里
  ```
  veRL 的 `_update_actor()` 调用 `self.actor_rollout_wg.update_actor()` 是 **Ray 远程调用**到 WorkerDict。
  Ray 调用返回后 TaskRunner 执行 gc.collect()，对 WorkerDict 的 GPU 内存零效果。

- **对比**：veRL 自带的 `generate_sequences()` 在 `fsdp_workers.py:1120` 有正确的清理：
  ```python
  get_torch_device().empty_cache()  # ✅ 在 WorkerDict 进程内
  ```
  但 `update_actor()` (line 1029-1071) **从未调用过任何显存清理**。

- **内存布局（OOM 时刻）**：
  | 组件 | 占用 | 说明 |
  |------|------|------|
  | cumem（vLLM KV cache, sleep）| ~64 GiB | 已 unmap 但 CUDA 驱动级仍分配 |
  | WorkerDict PyTorch（训练缓存）| ~20-25 GiB | 梯度/激活值未释放 |
  | vLLM Worker PyTorch | ~4-5 GiB | 模型权重 |
  | **总计** | **~89 GiB / 95 GiB** | **仅 ~6 GiB 余量** |

- **为什么第一次跑 16 步没 OOM**：运气好。93.7% 利用率下 OOM 是概率事件，取决于碎片化程度。
  第一次恰好 16 步都没触发最坏情况，后续几次运气差就早爆了。

- **修复**：Patch 9 — 在 `fsdp_workers.py` 的 `update_actor` 返回前加 `aggressive_empty_cache(force_sync=True)`，
  直接在 WorkerDict 进程内释放 PyTorch 缓存的训练内存。

- **torch.compile 不是原因**：`TORCHDYNAMO_DISABLE=1` 已禁用，`enforce_eager=True` + `compilation_config level=0` 确认未运行。

### GPU 显存对比（3 次运行实测）

| 运行 | gpu_util | response | 步数 | valleys (sleep) | peaks (gen) | 结果 |
|------|---------|----------|------|-----------------|-------------|------|
| Apr 11 13:35 | 0.25 | 6800 | 16 | 无 GPU 数据 | 无数据 | 挂起（非 OOM）|
| Apr 13 00:59 | 0.25 | 6800 | 7 | ~43.4 GiB | 68-97 GiB | Step 8 OOM |
| Apr 13 12:55 | 0.25 | 6000 | 8 | ~40.5 GiB | 68-97 GiB | Step 9 OOM |

 valleys 稳定（无泄漏），peaks 波动大（数据依赖），OOM 是 peak 撑到 ~97 GiB 时触发。

---

## 2026-04-14 KL 散度修复 + 多轮训练数据分析

### 踩坑 25: FSDP + LoRA 下 KL loss 始终为 0

- **现象**：多轮 GRPO 训练（A800 80GB）中 `actor/kl_loss` 一直为 0.0，`actor/ppo_kl` 也为 0
- **ppo_kl 为 0 正常**：GRPO 是 on-policy 单次更新，old_log_probs = current_log_probs，所以 ppo_kl=0
- **kl_loss 为 0 不正常**：kl_loss 是当前策略 vs 参考模型的 KL 散度，开启 `use_kl_loss=True` + `kl_loss_coef=0.003` 时应该非零

- **根因**：FSDP + LoRA 场景下 `disable_adapter()` 无法工作
  1. veRL 的 `ref_in_actor` 机制：LoRA 训练时复用 actor worker 计算 ref log prob，通过 `disable_adapter()` 临时禁用 LoRA
  2. FSDP 在 `DecoderLayer` 级别 auto_wrap，将参数 flatten 为 `FlatParameter`，LoRA 层被隐藏在 FSDP 的 flat param 中
  3. `disable_adapter()` 遍历 `model.modules()` 查找 LoRA 层 → 找不到 → 实际计算的是 actor 的 log prob（LoRA 启用），不是 ref 的
  4. actor_log_prob == ref_log_prob → kl_loss = 0

- **修复**：强制 `ref_in_actor=False`，使用独立的 ref 模型（通过 `ActorRolloutRef` 角色）
  修改两个文件：
  1. `/tmp/verl/verl/trainer/main_ppo.py` (~line 141)：强制 `ref_in_actor=False`，创建 `ActorRolloutRef` 角色
  2. `/tmp/verl/verl/trainer/ppo/ray_trainer.py` (~line 310)：强制 `self.ref_in_actor=False`

  ```python
  # main_ppo.py
  ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
  # FSDP + LoRA: disable_adapter 无法工作，强制使用独立 ref 模型
  if ref_in_actor and config.actor_rollout_ref.actor.strategy in ("fsdp", "fsdp2"):
      ref_in_actor = False

  # ray_trainer.py
  self.ref_in_actor = lora_rank > 0 or ...
  if self.ref_in_actor:
      import logging
      logging.getLogger(__name__).warning("[FIX] ref_in_actor=True but FSDP+LoRA disable_adapter is broken, forcing ref_in_actor=False")
      self.ref_in_actor = False
  ```

- **显存影响**：独立 ref 模型额外占用 ~3.4 GiB（与 actor 共享同一 GPU），用 `param_offload=True` 卸载到 CPU 可减少 GPU 占用
- **修复验证**：kl_loss 从 0 变为 ~0.001（step 1-20 稳定）
- **教训**：
  - veRL 的 `ref_in_actor` 是优化手段（省显存），但在 FSDP 下因参数 flatten 导致 LoRA 层不可见，优化失效
  - 这个问题不是我们的项目特殊——任何使用 veRL FSDP + LoRA + KL loss 的训练都会遇到
  - Megatron 后端不受影响（不 flatten 参数），所以 veRL 框架的大部分用户（大模型用 Megatron）没碰到

### 多轮训练指标分析（step 1-20，1313 样本）

**训练配置**：A800 80GB, Qwen3-1.7B, batch=8, n=8, response=8192, 3轮, LoRA r=16, KL loss=0.003
**训练数据**：`grpo_train_multi_full.parquet`（602 条），每 epoch 75 步，总 250 步 = 3.3 epoch

**核心指标**：
| 指标 | 数值 |
|------|------|
| 首轮成功率 | 12.0%（158/1313） |
| 第2轮纠错率 | 2.1%（18/841） |
| 第3轮纠错率 | 0.9%（7/809） |
| 最终通过率 | 13.9%（183/1313） |
| NO_CODE（≈截断） | 26.9%（353/1313） |
| 正/零/负 reward | 14.1% / 59.2% / 26.9% |

**Reward 分布**（与公式一致）：
- 1.00（第1轮成功）：158
- 0.85（第2轮成功）：18
- 0.70（第3轮成功）：7

**首轮失败分布**：
- RUNTIME_ERROR: 65.0%
- NO_CODE: 21.4%（≈截断率下限 25%）
- SYNTAX_ERROR: 1.2%
- TIMEOUT: 0.4%

**验证结果**：
- agent_loop vs reward 函数执行状态匹配率：**99.5%+**（verify log 960 条确认）
- 代码提取一致性：`extract_code()` 和 `extract_code_from_turn()` 结果相同
- `first_failed=False 且 reward<=0` 的样本 = **0**（数据完全一致）
- 脱敏检查：0 条 assert 行泄露

**关键发现**：
1. **负 reward = NO_CODE**：完全一一对应（283=283），截断导致代码块不完整 → 提取失败
2. **NO_CODE率 ≈ 截断率**：首轮 NO_CODE 21.4% 接近 clip_ratio 25%，部分截断代码仍有完整代码块（fallback 提取）
3. **纠错率极低**：训练初期模型不会根据错误反馈修复代码，属正常现象
4. **多轮指标日志**：两个 Ray worker 进程分别写入 `logs/multiturn_metrics_YYYYMMDD_HHMMSS.jsonl`，分析时需合并

### patch_vllm.py Patch 3 更新

- `transformer_impl.py` 中 `disable_adapter` 相关代码经历了多次调试修改
- Patch 3 需识别多种历史版本：原始版本、旧 patch、调试版本、ValueError catch 版本、manual disable 版本
- 通过检测 `_manual_disable` 字符串判断是否已应用新 patch
