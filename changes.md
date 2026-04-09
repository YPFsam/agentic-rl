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

### 多轮训练结果（成功）
- **80 步（step 21→100），score/mean: +0.046，62.5% 的步有样本通过测试**
- **response_length: 1489，clip_ratio: 0.40**
- **Entropy 0.14 → 0.06（下降 57%）**
- **显存峰值 43.4 GiB / 48 GiB (90%)**

### 训练日志
- 单轮：`logs/train_48gb.log`
- 多轮：`logs/train_multi.log`
- GPU 监控：`logs/gpu_monitor_multi.csv`
