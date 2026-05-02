# Agentic-RL: 基于执行反馈的代码生成 GRPO 强化学习

对 Qwen3-1.7B 进行单轮/多轮 GRPO 训练，通过代码执行反馈和纠错循环提升代码生成能力。

## 核心结果

| 方法 | HumanEval+ | 提升 |
|------|-----------|------|
| Qwen3-1.7B baseline | 0.561 | — |
| 单轮 GRPO (st_s300) | 0.732 | +30.5% |
| 多轮 GRPO (mt_s150) | 0.732 | +30.5% |
| r8192 评估 (st_s300) | 0.787 | +40.3% |

## 技术报告

完整的实验分析、训练曲线、评估对比、工程踩坑记录见 **[REPORT.md](REPORT.md)**。

## 快速复现

```bash
# 环境搭建（云端 AutoDL）
pip install -e /tmp/verl
python3 scripts/patch_vllm.py

# 训练
bash scripts/run_cloud_multi.sh    # 多轮 GRPO
bash scripts/run_cloud_single.sh   # 单轮 GRPO

# 评估
bash scripts/eval_all_checkpoints.sh
bash scripts/eval_all_checkpoints_multiturn.sh
```

## 项目结构

```
src/
  agent_loop.py           # 多轮代码纠错 AgentLoop
  reward.py / reward_multiturn.py  # 单轮/多轮奖励函数
  evalplus_sandbox.py     # evalplus 对齐评估沙盒
  data_prepare.py         # 训练数据准备（MBPP+APPS）
scripts/
  run_cloud_multi.sh      # 多轮训练脚本
  run_cloud_single.sh     # 单轮训练脚本
  patch_vllm.py           # vLLM 补丁（8个修复）
  synth_testcases.py      # LLM 辅助构造 assert 测试用例
configs/
  agent_loop.yaml         # 自定义 AgentLoop 注册配置
```
