"""
门控奖励函数（Gating Reward）

veRL 自定义 reward function，通过 YAML 配置加载：
  custom_reward_function.path=src/reward.py
  custom_reward_function.name=compute_score

门控逻辑（二元稀疏奖励）：
  ┌─ 代码提取失败 → -1.0（重罚，逼迫输出代码块）
  ├─ 所有执行失败（语法/超时/运行时/断言失败）→ 一律 0.0
  └─ 测试通过     → +1.0
"""
import re
import ast
import json
import asyncio
import os
from typing import Optional
from datetime import datetime

from src.sandbox import execute_code, ExecResult, ExecStatus


# ========== 代码提取工具 ==========

def extract_code(text: str) -> Optional[str]:
    """
    从模型输出中提取 Python 代码。

    提取策略（按优先级）：
    0. 去掉 <think ...>...</think > 标签（Qwen3 思维链内部可能有草稿代码块，不应提取）
    1. ```python ... ``` 代码块
    2. 未闭合的代码块（max_tokens 截断场景）
    3. 纯文本中的 Python 代码特征检测

    Args:
        text: 模型的原始输出文本

    Returns:
        提取出的代码字符串；提取失败返回 None
    """
    # 步骤 0：先去掉 Qwen3 的 <think ...>...</think > 标签
    # 原因：think 标签内的代码块是思维草稿，不是最终答案
    cleaned = re.sub(r"<think[^>]*>.*?</think\s*>", "", text, flags=re.DOTALL).strip()

    # 策略 1a：提取所有 ```python ... ``` 代码块并拼接
    # 拼接而非取第一个/最后一个：模型可能将 helper + main 分多个代码块输出，
    # 只取任意一个都会导致函数定义缺失，拼接后才是完整解答。
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, cleaned, re.DOTALL)
    if matches:
        blocks = [m.strip() for m in matches if m.strip()]
        if blocks:
            return "\n\n".join(blocks)

    # 策略 1b：未闭合的代码块（max_tokens 截断导致 ```python 后没有 ```）
    # 例如：```python\ndef add(a,b):\n    return a   （截断，无闭合 ```）
    truncated_pattern = r"```python\s*(.*?)$"
    match = re.search(truncated_pattern, cleaned, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if code:
            return code

    # 策略 2：去掉 think 标签后的文本，检查是否像 Python 代码
    if cleaned and len(cleaned) > 20:
        # 至少包含一个 Python 关键字/语法特征，避免将普通文本误判为代码
        python_indicators = [
            r"\bdef\s+\w+\s*\(",      # def func(
            r"\bclass\s+\w+",         # class MyClass
            r"\bimport\s+\w+",        # import os
            r"\bfrom\s+\w+\s+import", # from x import
            r"\breturn\s+",           # return
            r".*\S.*=.*\S",           # 赋值语句 x = ...
            r"\bprint\s*\(",          # print(
            r"\bfor\s+\w+\s+in\b",    # for x in
            r"\bif\s+\w+",            # if condition
        ]
        if any(re.search(p, cleaned) for p in python_indicators):
            return cleaned

    return None


# ========== 异步桥接工具 ==========

def _run_async(coro):
    """
    在同步上下文中运行 async 函数。

    veRL 的 compute_score 是同步函数，但我们的沙盒是 async。
    在 Ray Worker 进程中没有运行中的事件循环，asyncio.run() 可以安全使用。
    """
    return asyncio.run(coro)


# ========== Thinking Metrics 日志 ==========

_METRICS_LOG_PATH = os.environ.get("THINKING_METRICS_LOG", "logs/thinking_metrics.jsonl")


def _log_thinking_metrics(solution_str: str, code, reward: float, extra_info: dict):
    """
    记录 thinking token 数和 reward，用于训练后分析。

    在 compute_score() 的每个 return 前调用，追加写入 JSONL。
    """
    # 去掉代码块后的文本 = thinking + 其他非代码内容
    thinking_text = re.sub(r"```python\s*.*?\s*```", "", solution_str, flags=re.DOTALL).strip()
    thinking_chars = len(thinking_text)
    # 粗略估算 token 数（中英混合约 3-4 字符 per token）
    thinking_tokens = thinking_chars // 3

    task_id = extra_info.get("task_id", "unknown") if extra_info else "unknown"

    record = {
        "task_id": task_id,
        "thinking_tokens": thinking_tokens,
        "code_length": len(code) if code else 0,
        "reward": reward,
        "has_code": code is not None,
        "timestamp": datetime.now().isoformat(),
    }

    log_dir = os.path.dirname(_METRICS_LOG_PATH)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(_METRICS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ========== 门控奖励函数（veRL 入口） ==========

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> float:
    """
    veRL 自定义奖励函数入口。

    此函数被 veRL 的 Ray Worker 调用，签名必须匹配 veRL 约定。

    Args:
        data_source: 数据来源标识（"mbpp"）
        solution_str: 模型生成的完整回答
        ground_truth: 标准答案（代码任务不用此字段）
        extra_info: 额外信息字典，包含 test_cases 和 task_id

    Returns:
        float: 奖励分数
    """
    if extra_info is None:
        extra_info = {}

    test_cases = extra_info.get("test_cases", "")

    # ===== 门控检查：代码提取 =====
    code = extract_code(solution_str)
    if code is None:
        _log_thinking_metrics(solution_str, None, -1.0, extra_info)
        return -1.0

    # ===== 语法检查 =====
    try:
        ast.parse(code)
    except SyntaxError:
        _log_thinking_metrics(solution_str, code, 0.0, extra_info)
        return 0.0

    # ===== 沙盒执行 =====
    result: ExecResult = _run_async(execute_code(code, test_cases, timeout=5.0))

    # ===== 二元稀疏奖励 =====
    if result.status == ExecStatus.SUCCESS:
        _log_thinking_metrics(solution_str, code, 1.0, extra_info)
        return 1.0
    else:
        _log_thinking_metrics(solution_str, code, 0.0, extra_info)
        return 0.0
