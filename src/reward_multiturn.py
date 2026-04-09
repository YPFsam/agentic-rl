"""
多轮门控奖励函数（纯稀疏奖励）

用于 veRL AgentLoop 训练的多轮奖励计算。

奖励公式（稀疏设计）：
  最终成功：max(0.1, 1.0 - turn_penalty × actual_turns)
    第 1 轮成功：1.0
    第 2 轮成功：0.85
    第 3 轮成功：0.70
  最终失败：按门控逻辑给 -1.0 或 0.0

轮次计数：基于 AgentLoop 写入的 [Execution Feedback] 标记，
  而非代码块数量。单轮输出多个代码块不会导致轮次误判。
"""
import re
import ast
import subprocess
import sys
import tempfile
import os
import signal
import resource
from typing import Optional

from src.sandbox import ExecResult, ExecStatus, MAX_MEMORY_BYTES, MAX_CPU_SECONDS, _preexec_fn
from src.reward import extract_code


# ========== 同步执行（避免在已有 event loop 中 asyncio.run 冲突）==========

def _execute_code_sync(code: str, test_cases: str = "", timeout: float = 5.0) -> ExecResult:
    """同步版本的沙盒执行，用于 reward 函数（veRL agent_loop 已在 uvloop 中运行）。"""
    if test_cases:
        full_code = f"{code}\n\n{test_cases}"
    else:
        full_code = code

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="sandbox_")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(full_code)

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.Popen(
            [sys.executable, tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            preexec_fn=_preexec_fn,
        )

        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait()
            return ExecResult(status=ExecStatus.TIMEOUT, stdout="", stderr="TimeoutExpired")

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        if proc.returncode == 0:
            return ExecResult(status=ExecStatus.SUCCESS, stdout=stdout, stderr=stderr)
        if "SyntaxError" in stderr:
            return ExecResult(status=ExecStatus.SYNTAX_ERROR, stdout=stdout, stderr=stderr)
        return ExecResult(status=ExecStatus.RUNTIME_ERROR, stdout=stdout, stderr=stderr)

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ========== 多轮轮次计数 ==========

# AgentLoop 写入的反馈标记，用于区分不同轮次
_TURN_SEPARATOR_PATTERN = re.compile(
    r"\[Execution Feedback(?: - Turn \d+/\d+)?\]"
)


def count_actual_turns(text: str) -> int:
    """
    统计实际交互轮数。

    基于 AgentLoop 写入的 [Execution Feedback] 标记计数，
    而非代码块数量。原因：
      - 单轮中模型可能输出多个 ```python 块（辅助函数 + 主函数）
      - 代码块数 != 交互轮数
      - Feedback 标记是 AgentLoop 明确写入的轮次分隔符

    轮数 = feedback 标记数 + 1（最后一次生成后如果没有 feedback，说明成功终止）

    Args:
        text: 完整的多轮对话轨迹

    Returns:
        实际交互轮数（≥1）
    """
    feedback_count = len(_TURN_SEPARATOR_PATTERN.findall(text))
    return feedback_count + 1


def extract_last_code(text: str) -> str | None:
    """
    从多轮轨迹中提取最后一轮的代码（用于最终评估）。

    关键设计：提取最后一轮的【所有】代码块并拼接。
    原因：模型可能在单轮中分多个代码块输出完整方案
    （如辅助函数在一个块，主函数在另一个块），拼接后才是完整解答。

    策略（按优先级）：
    1. 从最后一个 [Execution Feedback] 标记之后截取尾部文本
    2. 从尾部文本中提取所有 ```python ... ``` 块并拼接
    3. 如果没有完整代码块，尝试截断代码块提取（max_tokens 截断场景）
    4. 如果仍失败，对整段文本尝试 extract_code 兜底

    Args:
        text: 完整的多轮对话轨迹

    Returns:
        最后一轮的完整代码，提取失败返回 None
    """
    # 截取最后一个 feedback 标记之后的内容
    last_feedback_pos = 0
    for m in _TURN_SEPARATOR_PATTERN.finditer(text):
        last_feedback_pos = m.end()

    tail_text = text[last_feedback_pos:]

    # 策略 1：提取尾部文本中所有完整代码块并拼接
    blocks = _extract_all_complete_blocks(tail_text)
    if blocks:
        return "\n\n".join(blocks)

    # 策略 2：截断的未闭合代码块（max_tokens 截断）
    truncated = _extract_truncated_block(tail_text)
    if truncated:
        return truncated

    # 策略 3：对整段文本用 extract_code 兜底
    code = extract_code(text)
    if code:
        return code

    return None


def _extract_all_complete_blocks(text: str) -> list[str]:
    """从文本中提取所有完整 ```python ... ``` 代码块。"""
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [m.strip() for m in matches if m.strip()]


def _extract_truncated_block(text: str) -> str | None:
    """提取截断的未闭合代码块（```python 后无闭合 ```）。"""
    pattern = r"```python\s*(.*?)$"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if code:
            return code
    return None


# ========== 多轮门控奖励函数（veRL 入口） ==========

def compute_score_multiturn(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    turn_penalty: float = 0.15,
) -> float:
    """
    多轮门控奖励函数入口（纯稀疏奖励）。

    被 veRL Ray Worker 调用，评估多轮交互的最终奖励。

    奖励公式（稀疏设计）：
      最终成功：max(0.1, 1.0 - turn_penalty × (turns - 1))
      最终失败：按门控逻辑给 -1.0 或 0.0

    max(0.1, ...) 保底防止任何边界条件下奖励倒挂为负数。

    Args:
        data_source: 数据来源标识
        solution_str: 模型多轮交互的完整轨迹文本
        ground_truth: 标准答案（代码任务不用）
        extra_info: 包含 test_cases 和 task_id
        turn_penalty: 每多一轮的扣分（默认 0.15）

    Returns:
        float: 最终奖励分数
    """
    if extra_info is None:
        extra_info = {}

    test_cases = extra_info.get("test_cases", "")

    # ===== 提取最后一轮代码（高鲁棒性） =====
    last_code = extract_last_code(solution_str)

    if last_code is None:
        # 门控：完全没有代码 → -1.0
        return -1.0

    # ===== 统计实际轮数（基于 feedback 标记，非代码块数） =====
    num_turns = count_actual_turns(solution_str)

    # ===== 语法检查 =====
    try:
        ast.parse(last_code)
    except SyntaxError:
        return 0.0

    # ===== 沙盒执行（同步，避免 asyncio 冲突）=====
    result: ExecResult = _execute_code_sync(last_code, test_cases, timeout=5.0)

    # ===== 计算最终奖励（二元稀疏 + 保底） =====
    if result.status == ExecStatus.SUCCESS:
        penalty = turn_penalty * (num_turns - 1)
        return max(0.1, 1.0 - penalty)
    else:
        return 0.0
