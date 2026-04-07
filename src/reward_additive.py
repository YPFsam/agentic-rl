"""
加法奖励函数（Additive Reward）——消融实验用

与门控奖励的对比：
  门控：代码提取失败 → 格式分归零，reward = -1.0
  加法：格式和执行独立打分，加权求和

加法奖励公式：
  reward = 0.2 × format_score + 0.8 × execution_score

潜在问题：模型可以只学格式（得 0.5~0.7 分）而不学正确代码。
这就是门控设计要解决的问题。
"""
import re
import ast
import asyncio
from typing import Optional

from src.sandbox import execute_code, ExecResult, ExecStatus
from src.reward import extract_code


def _run_async(coro):
    return asyncio.run(coro)


def _compute_format_score(text: str) -> float:
    """
    格式奖励：检查输出结构。

    评分：
      - 有 ```python ... ``` 代码块：+0.5
      - 有推理过程（文本长度 > 50）：+0.3
      - 代码块前有推理文字：+0.2
      - 以上都没有：-0.3
    """
    score = 0.0

    has_code_block = bool(re.search(r"```python\s*.*?\s*```", text, re.DOTALL))
    has_thinking = len(text.strip()) > 50

    if has_code_block:
        score += 0.5
    if has_thinking:
        score += 0.3
    if has_code_block and has_thinking:
        score += 0.2

    if score == 0.0:
        score = -0.3

    return score


def _compute_execution_score(code: str, test_cases: str) -> float:
    """
    执行奖励：沙盒执行代码。

    评分：
      - 测试通过：+1.0
      - 运行时错误：-0.2
      - 超时：-1.0
      - 语法错误：-0.3
      - 无代码：-0.5
    """
    if code is None:
        return -0.5

    try:
        ast.parse(code)
    except SyntaxError:
        return -0.3

    result: ExecResult = _run_async(execute_code(code, test_cases, timeout=5.0))

    if result.status == ExecStatus.SUCCESS:
        return 1.0
    elif result.status == ExecStatus.TIMEOUT:
        return -1.0
    elif result.status == ExecStatus.SYNTAX_ERROR:
        return -0.3
    else:
        return -0.2


def compute_score_additive(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    format_weight: float = 0.2,
    exec_weight: float = 0.8,
) -> float:
    """
    加法奖励函数（veRL 入口）。

    reward = format_weight × format_score + exec_weight × execution_score
    """
    if extra_info is None:
        extra_info = {}

    test_cases = extra_info.get("test_cases", "")

    format_score = _compute_format_score(solution_str)

    code = extract_code(solution_str)
    execution_score = _compute_execution_score(code, test_cases)

    reward = format_weight * format_score + exec_weight * execution_score
    return reward
