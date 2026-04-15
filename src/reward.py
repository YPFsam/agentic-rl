"""
门控奖励函数（Gating Reward）— 单轮 GRPO 消融实验

veRL 自定义 reward function，通过 YAML 配置加载：
  custom_reward_function.path=src/reward.py
  custom_reward_function.name=compute_score

门控逻辑（二元稀疏奖励）：
  ┌─ 代码提取失败 → -1.0（重罚，逼迫输出代码块）
  ├─ 所有执行失败（语法/超时/运行时/断言失败）→ 一律 0.0
  └─ 测试通过     → +1.0

代码提取逻辑与 reward_multiturn.py 完全一致（严格模式，无 fallback）。
沙盒执行复用 sandbox.py 的 async execute_code + asyncio.run()。
"""
import re
import ast
import asyncio
import json
import os
import time
from typing import Optional

from src.sandbox import execute_code, ExecResult, ExecStatus


# ========== 代码提取（与 reward_multiturn.py 完全一致）==========

def _clean_think_tags(text: str) -> str:
    """移除 <think...> 标签及其内容，避免思考过程中的干扰代码。"""
    text = re.sub(r"<think[^>]*>.*?</think[^>]*>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think[^>]*>.*", "", text, flags=re.DOTALL)
    return text


def _extract_code_blocks(text: str) -> Optional[str]:
    """从文本中提取 ```python ... ``` 代码块（严格，无 fallback）。"""
    blocks = re.findall(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    blocks = [b.strip() for b in blocks if b.strip()]
    if blocks:
        return "\n\n".join(blocks)
    # 截断的未闭合代码块
    match = re.search(r"```python\s*(.*?)$", text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if code:
            return code
    return None


def extract_code(text: str) -> Optional[str]:
    """
    从模型输出中提取代码：清洗 think 标签 → 提取代码块。

    与多轮 reward_multiturn.py 的 extract_code_from_turn() 完全一致。
    """
    cleaned = _clean_think_tags(text)
    return _extract_code_blocks(cleaned)


# ========== 单轮训练指标日志（与多轮对齐）==========

_RUN_ID = time.strftime("%Y%m%d_%H%M%S")
_SINGLE_TURN_METRICS_PATH = os.environ.get(
    "SINGLE_TURN_METRICS_LOG", f"logs/single_turn_metrics_{_RUN_ID}.jsonl"
)


def _log_single_turn_metrics(
    solution_str: str,
    code: str | None,
    reward: float,
    exec_status: str,
    extra_info: dict | None,
):
    """
    记录单轮训练的完整指标到 JSONL，用于训练后分析和 debug。

    与多轮 _log_multiturn_metrics 对齐，记录：
      - task_id, reward, exec_status
      - 代码提取结果
      - thinking token 估算
    """
    if extra_info is None:
        extra_info = {}

    task_id = extra_info.get("task_id", "unknown")

    # 估算 thinking tokens
    thinking_text = re.sub(r"```python\s*.*?\s*```", "", solution_str, flags=re.DOTALL).strip()
    thinking_tokens = len(thinking_text) // 3

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "task_id": task_id,
        "reward": reward,
        "exec_status": exec_status,
        "code_extracted": code is not None,
        "code_length": len(code) if code else 0,
        "thinking_tokens": thinking_tokens,
        "response_length": len(solution_str),
    }

    try:
        log_dir = os.path.dirname(_SINGLE_TURN_METRICS_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(_SINGLE_TURN_METRICS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        import warnings
        warnings.warn(f"[single_turn_metrics] 日志写入失败: {type(e).__name__}: {e}")


# ========== 门控奖励函数（veRL 入口）==========

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> float:
    """
    veRL 自定义奖励函数入口（单轮消融）。

    此函数被 veRL 的 Ray Reward Worker 调用，签名必须匹配 veRL 约定。

    Args:
        data_source: 数据来源标识（"mbpp" / "apps"）
        solution_str: 模型生成的完整回答
        ground_truth: 标准答案（代码任务不用此字段）
        extra_info: 额外信息字典，包含 test_cases 和 task_id

    Returns:
        float: 奖励分数（-1.0 / 0.0 / +1.0）
    """
    if extra_info is None:
        extra_info = {}

    test_cases = extra_info.get("test_cases", "")

    # ===== 门控检查：代码提取 =====
    code = extract_code(solution_str)
    if code is None:
        _log_single_turn_metrics(solution_str, None, -1.0, "NO_CODE", extra_info)
        return -1.0

    # ===== 语法检查 =====
    try:
        ast.parse(code)
    except SyntaxError:
        _log_single_turn_metrics(solution_str, code, 0.0, "SYNTAX_ERROR", extra_info)
        return 0.0

    # ===== 沙盒执行（asyncio.run 安全：单轮 reward worker 无 uvloop）=====
    result: ExecResult = asyncio.run(execute_code(code, test_cases, timeout=5.0))

    # ===== 二元稀疏奖励 =====
    if result.status == ExecStatus.SUCCESS:
        _log_single_turn_metrics(solution_str, code, 1.0, "SUCCESS", extra_info)
        return 1.0
    else:
        _log_single_turn_metrics(solution_str, code, 0.0, result.status.name, extra_info)
        return 0.0
