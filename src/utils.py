"""
工具函数

提供项目中通用的辅助功能，如代码提取等。
主要的 extract_code 函数在 src/reward.py 中，此处提供额外工具。
"""
import re
from typing import Optional


# AssertionError 脱敏正则：不用 DOTALL，只匹配当前行，保留后续日志
_ASSERTION_PATTERN = re.compile(r"AssertionError: .+")
_SANITIZED_ASSERT_MSG = (
    "AssertionError: Test case failed — your code produced incorrect output "
    "for a hidden test case. Please review your logic and handle edge cases."
)


def sanitize_stderr(stderr: str, max_length: int = 500) -> str:
    """
    截断和清理 stderr 信息，用于多轮交互的错误反馈。

    处理顺序：先脱敏，再截断。
      1. 先替换 AssertionError 中的敏感数值（防测试用例泄露）
      2. 再执行尾部截断（保留 traceback 底部关键信息）

    Args:
        stderr: 原始标准错误输出
        max_length: 最大保留长度

    Returns:
        清理后的错误信息
    """
    # 步骤 1：先脱敏（防止截断打断 AssertionError 导致正则失效）
    stderr = _ASSERTION_PATTERN.sub(_SANITIZED_ASSERT_MSG, stderr)

    # 步骤 2：再截断（脱敏已完成，截断不会泄露敏感信息）
    if len(stderr) > max_length:
        stderr = stderr[-max_length:]

    return stderr
