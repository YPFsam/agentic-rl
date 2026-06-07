# 项目名称（简历用）

**基于过程奖励的多轮代码纠错强化学习训练框架**

---

## 项目描述（中文版）

### 基于过程奖励的多轮代码纠错强化学习（RL）训练框架

**技术栈：** PyTorch, veRL 0.8, vLLM, Ray, LoRA, FSDP, wandb

- 基于 veRL 框架实现多轮 AgentLoop 代码纠错训练（GRPO 算法），模型生成代码后通过异步沙盒执行获取反馈，自主迭代修正错误，最大支持 3 轮交互
- 设计并实现 turn-level 过程奖励机制：增量式状态评分 C_t - C_{t-1} 防止 reward hack（telescoping 总和有上界），格式非法轮次不更新状态、不继承未来收益，过程奖励与结果奖励分开归一化并缩放 λ=0.5，解决稀疏奖励下纠错信号弱的问题
- 扩展沙盒执行模块支持 assert 级别通过率统计（try-except 逐条包裹），为过程奖励提供部分通过信号 p_t = passed/total，实现 0.2 + 0.8×p_t 的渐进状态评分
- 实现自定义 GRPO advantage estimator（注册到 veRL 框架），从 non_tensor_batch 提取逐轮奖励数据，按 prompt group 分组归一化后构建 reward-to-go，精确映射到 token-level advantages
- 收集稀疏奖励训练中单轮和多轮的成功轨迹进行 SFT 热启动，再进行 RL 训练，最终在 HumanEval 上超过单轮 GRPO、多轮稀疏 GRPO 和基线模型
- 在 H20/A800 GPU 上完成完整训练流程，解决 vLLM + FSDP + LoRA 下的显存泄漏、KV cache 预抢占、numpy 版本兼容等多个工程问题

---

## 项目描述（英文版）

### Multi-Turn Code Correction RL Framework with Process Rewards

**Tech Stack:** PyTorch, veRL 0.8, vLLM, Ray, LoRA, FSDP, wandb

- Built a multi-turn AgentLoop for code correction training using GRPO on the veRL framework. The model generates code, receives async sandbox execution feedback, and iteratively fixes errors across up to 3 turns
- Designed a turn-level process reward mechanism: incremental state scoring C_t - C_{t-1} prevents reward hacking (telescoping sum bounded), format-invalid turns don't update state or inherit future returns, process and outcome rewards are separately normalized with λ=0.5 scaling, addressing weak correction signals under sparse rewards
- Extended the sandbox executor with per-assert pass counting (try-except wrapping each assertion), providing partial pass signals p_t = passed/total for gradual state scoring of 0.2 + 0.8×p_t
- Implemented a custom GRPO advantage estimator (registered into the veRL framework) that extracts per-turn rewards from non_tensor_batch, normalizes by prompt group, computes reward-to-go with format-invalid isolation, and maps to token-level advantages
- Collected successful trajectories from sparse-reward single-turn and multi-turn training for SFT warm-start, followed by RL fine-tuning. The final model outperforms single-turn GRPO, multi-turn sparse GRPO, and the baseline on HumanEval
- Resolved multiple production engineering issues including vLLM memory leaks with FSDP+LoRA, KV cache preemption under multi-turn amplification, and numpy version compatibility across Docker environments
