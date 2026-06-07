"""
注册自定义 AgentLoop 到 veRL。

veRL v0.8 提供两种注册方式：
  1. @register("name") 装饰器 — 在 import 时自动注册
  2. agent_loop_config_path — YAML 配置文件（推荐，无需 import hack）

本项目使用方式 2（configs/agent_loop.yaml），此文件保留仅供参考。

如果需要使用方式 1，需在训练启动前 import 此模块：
  import src.register_agent
"""
from src.agent_loop import CodeAgentLoop

# 尝试注册到 veRL 的全局注册表
try:
    from verl.experimental.agent_loop.agent_loop import _agent_loop_registry
    _agent_loop_registry["code_agent_loop"] = {
        "_target_": "src.agent_loop.CodeAgentLoop"
    }
except ImportError:
    pass

# 注册过程奖励 advantage estimator
from src.advantage_process_outcome import compute_grpo_process_outcome_advantage
