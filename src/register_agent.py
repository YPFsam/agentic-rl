"""
注册自定义 AgentLoop 到 veRL。

veRL 通过 agent_name 字段查找对应的 AgentLoop 类。
在训练启动脚本中 import 此模块即可完成注册。
"""
from src.agent_loop import CodeAgentLoop

# veRL 使用全局注册表管理 AgentLoop
# 注册名称必须与训练数据中 agent_name 列一致
AGENT_REGISTRY = {
    "code_agent_loop": CodeAgentLoop,
}


def get_agent_loop(name: str):
    """根据名称获取 AgentLoop 类。"""
    if name not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent_name: {name}. "
            f"Available: {list(AGENT_REGISTRY.keys())}"
        )
    return AGENT_REGISTRY[name]
