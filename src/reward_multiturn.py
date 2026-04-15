"""
多轮门控奖励函数（纯稀疏奖励）

用于 veRL AgentLoop 训练的多轮奖励计算。

奖励公式（稀疏设计）：
  最终成功：max(0.1, 1.0 - turn_penalty × actual_turns)
    第 1 轮成功：1.0
    第 2 轮成功：0.85
    第 3 轮成功：0.70
  最终失败：按门控逻辑给 -1.0 或 0.0

数据流（v2 — 结构化）：
  agent_loop 通过 extra_info 传递：
    - last_turn_text: 最后一轮模型原始输出（reward 直接提取代码）
    - turn_details: 每轮的 exec_status 列表（reward 直接计算 corrected）
    - num_turns: 实际轮数（veRL naive.py 自动注入）
  reward 函数不再依赖正则切分 solution_str，避免模型 echo 干扰。
  保留 fallback（正则切分）兼容单元测试。
"""
import re
import ast
import subprocess
import sys
import tempfile
import os
import json
import signal
import resource
import time
from typing import Optional
from pathlib import Path

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
        hard_timeout = int(timeout) + 2
        proc = subprocess.Popen(
            ["timeout", str(hard_timeout), sys.executable, tmp_path],
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


# ========== 代码提取（从单轮模型输出） ==========

def _clean_think_tags(text: str) -> str:
    """移除 <think...> 标签及其内容，避免思考过程中的干扰代码。"""
    text = re.sub(r"<think[^>]*>.*?</think[^>]*>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think[^>]*>.*", "", text, flags=re.DOTALL)
    return text


def _extract_code_blocks(text: str) -> str | None:
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


def extract_code_from_turn(text: str) -> str | None:
    """从单轮模型输出中提取代码：清洗 think 标签 → 提取代码块。"""
    cleaned = _clean_think_tags(text)
    return _extract_code_blocks(cleaned)


# ========== 旧版正则提取（fallback，兼容单元测试） ==========

# AgentLoop 写入的反馈标记，用于区分不同轮次
_TURN_SEPARATOR_PATTERN = re.compile(
    r"\[Execution Feedback(?: - Turn \d+/\d+)?\]|\[Execution Result\]|\[Format Error\]"
)

_ERROR_FEEDBACK_PATTERN = re.compile(
    r"\[Execution Feedback(?: - Turn \d+/\d+)?\]"
)


def count_actual_turns(text: str) -> int:
    """
    统计实际交互轮数（旧版，基于正则）。

    轮数 = 错误 feedback 标记数 + 1
    """
    error_feedback_count = len(_ERROR_FEEDBACK_PATTERN.findall(text))
    return error_feedback_count + 1


def extract_last_code(text: str) -> str | None:
    """
    从多轮轨迹中提取最后一轮代码（旧版，基于正则）。

    仅在 extra_info 无结构化数据时使用（单元测试 / 兼容旧训练）。
    """
    error_matches = list(_ERROR_FEEDBACK_PATTERN.finditer(text))

    if error_matches:
        last_error = error_matches[-1]
        after_error = text[last_error.end():]

        result_pos = after_error.find('[Execution Result]')
        if result_pos >= 0:
            after_error = after_error[:result_pos]

        after_error = re.sub(r"<think[^>]*>.*?</think[^>]*>", "", after_error, flags=re.DOTALL)
        after_error = re.sub(r"<think[^>]*>.*", "", after_error, flags=re.DOTALL)

        blocks = re.findall(r"```python\s*(.*?)\s*```", after_error, re.DOTALL)
        blocks = [b.strip() for b in blocks if b.strip()]
        if blocks:
            return "\n\n".join(blocks)
        match = re.search(r"```python\s*(.*?)$", after_error, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if code:
                return code
        return None

    result_pos = text.find('[Execution Result]')
    if result_pos >= 0:
        before_result = text[:result_pos]
        before_result = re.sub(r"<think[^>]*>.*?</think[^>]*>", "", before_result, flags=re.DOTALL)
        before_result = re.sub(r"<think[^>]*>.*", "", before_result, flags=re.DOTALL)

        blocks = re.findall(r"```python\s*(.*?)\s*```", before_result, re.DOTALL)
        blocks = [b.strip() for b in blocks if b.strip()]
        if blocks:
            return "\n\n".join(blocks)
        match = re.search(r"```python\s*(.*?)$", before_result, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if code:
                return code
        return None

    code = extract_code(text)
    if code:
        return code

    return None


# ========== 多轮指标日志 ==========

_RUN_ID = time.strftime("%Y%m%d_%H%M%S")
_MULTITURN_METRICS_PATH = os.environ.get(
    "MULTITURN_METRICS_LOG", f"logs/multiturn_metrics_{_RUN_ID}.jsonl"
)


def _log_multiturn_metrics(
    solution_str: str,
    reward: float,
    num_turns: int,
    extra_info: dict,
    exec_status: str,
    turn_details: list | None = None,
):
    """
    记录多轮交互的完整指标到 JSONL，用于训练后分析。

    优先使用 turn_details（结构化数据），fallback 到正则解析。
    """
    # ===== 纠错转化判定 =====
    corrected = False
    first_turn_failed = False

    if turn_details:
        # 结构化路径：直接用 agent_loop 的 exec_status
        if len(turn_details) >= 1:
            first_status = turn_details[0].get("exec_status", "NOT_EXECUTED")
            first_turn_failed = first_status in ("ERROR", "TIMEOUT", "NO_CODE", "RUNTIME_ERROR", "SYNTAX_ERROR")
            if first_turn_failed:
                for td in turn_details[1:]:
                    if td.get("exec_status") == "SUCCESS":
                        corrected = True
                        break

        # 构建 turns 数据
        turns_data = []
        for td in turn_details:
            turns_data.append({
                "turn": td["turn"],
                "outcome": td["exec_status"],
            })

        # 估算 thinking tokens（从 generated_text 去掉代码块后的文本长度）
        total_thinking_tokens = 0
        total_code_chars = 0
        for td in turn_details:
            gen_text = td.get("generated_text", "") or ""
            # 去掉代码块后的文本 ≈ thinking
            thinking_text = re.sub(r"```python\s*.*?\s*```", "", gen_text, flags=re.DOTALL).strip()
            total_thinking_tokens += len(thinking_text) // 3
            code_blocks = re.findall(r"```python\s*(.*?)\s*```", gen_text, re.DOTALL)
            total_code_chars += sum(len(b) for b in code_blocks)

    else:
        # Fallback：正则解析 solution_str（兼容测试和旧训练数据）
        turn_outcomes = _parse_turn_outcomes(solution_str)
        if len(turn_outcomes) >= 1:
            first_outcome = turn_outcomes[0].get("outcome", "NOT_EXECUTED")
            first_turn_failed = first_outcome in ("ERROR", "TIMEOUT", "NO_CODE")
            if first_turn_failed:
                for t in turn_outcomes[1:]:
                    if t.get("outcome") == "SUCCESS":
                        corrected = True
                        break

        turns_data = []
        if turn_outcomes:
            for t in turn_outcomes:
                turns_data.append(t)
        total_thinking_tokens = 0

    task_id = extra_info.get("task_id", "unknown") if extra_info else "unknown"

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "task_id": task_id,
        "num_turns": num_turns,
        "reward": reward,
        "exec_status": exec_status,
        "first_turn_failed": first_turn_failed,
        "corrected": corrected,
        "total_thinking_tokens": total_thinking_tokens,
        "turns": turns_data,
    }

    try:
        log_dir = os.path.dirname(_MULTITURN_METRICS_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(_MULTITURN_METRICS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        import warnings
        warnings.warn(f"[multiturn_metrics] 日志写入失败: {type(e).__name__}: {e}")


def _check_sanitization(turn_details: list) -> dict:
    """
    检查脱敏效果：原始 stderr 中 assert 行是否泄露了 test case。

    对比原始 stderr vs 脱敏后 feedback（solution_str 中提取），
    确认 assert 行已被替换为 <test_case>。
    """
    import re as _re
    assert_line_pattern = _re.compile(r'^\s+assert\s+(?!<test_case>).+$', _re.MULTILINE)

    result = {
        "total_turns": len(turn_details),
        "turns_with_stderr": 0,
        "raw_leaked_count": 0,     # 原始 stderr 中包含 assert 行的轮数
        "leaked_examples": [],     # 泄露示例（最多 2 条）
    }

    for td in turn_details:
        raw_stderr = td.get("exec_stderr_raw", "")
        if not raw_stderr:
            continue
        result["turns_with_stderr"] += 1

        # 检查原始 stderr 是否有未脱敏的 assert 行
        leaked = assert_line_pattern.findall(raw_stderr)
        if leaked:
            result["raw_leaked_count"] += 1
            if len(result["leaked_examples"]) < 2:
                result["leaked_examples"].append({
                    "turn": td["turn"],
                    "leaked_line": leaked[0].strip()[:100],
                })

    return result


def _log_turn_outcomes_verify(
    solution_str: str,
    extra_info: dict,
    turn_details: list | None,
    reward_code: str | None,
    reward_exec_status: str,
):
    """
    验证日志：对比 agent_loop 的 turn_details 与 reward 的独立计算。

    用于检测数据流是否一致（extra_info 传递是否正确）。
    仅在 turn_details 存在时记录。
    """
    if not turn_details:
        return
    try:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        task_id = extra_info.get("task_id", "?") if extra_info else "?"

        # agent_loop 的最后一轮数据
        agent_last = turn_details[-1] if turn_details else {}
        agent_last_status = agent_last.get("exec_status", "?")
        agent_last_code = agent_last.get("extracted_code", None)
        agent_last_code_len = len(agent_last_code) if agent_last_code else 0

        # reward 提取的代码与 agent_loop 提取的代码是否一致
        code_match = (reward_code == agent_last_code) if (reward_code and agent_last_code) else None

        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "task_id": task_id,
            "num_turns": extra_info.get("num_turns", "?"),
            "agent_turns": len(turn_details),
            # agent_loop 侧
            "agent_last_exec_status": agent_last_status,
            "agent_last_code_extracted": agent_last.get("code_extracted", "?"),
            "agent_last_code_len": agent_last_code_len,
            # reward 侧
            "reward_code_extracted": reward_code is not None,
            "reward_code_len": len(reward_code) if reward_code else 0,
            "reward_exec_status": reward_exec_status,
            # 对比
            "code_match": code_match,
            "exec_status_match": agent_last_status == reward_exec_status,
            # 脱敏验证：原始 stderr 中是否有 assert 行泄露 test case
            "sanitize_check": _check_sanitization(turn_details),
        }

        log_file = log_dir / f"verify_turn_outcomes_{_RUN_ID}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        import warnings
        warnings.warn(f"[verify] 日志写入失败: {type(e).__name__}: {e}")


def _parse_turn_outcomes(solution_str: str) -> list[dict]:
    """
    从多轮轨迹文本中解析每轮的执行结果（旧版 fallback）。

    基于 AgentLoop 写入的反馈标记判断每轮结果。
    """
    parts = _TURN_SEPARATOR_PATTERN.split(solution_str)

    outcomes = []
    for i, part in enumerate(parts):
        if i == 0:
            outcomes.append({"turn": 1, "outcome": "NOT_EXECUTED"})
            continue

        feedback_text = part[:500]

        if "All test cases passed!" in feedback_text:
            if outcomes:
                outcomes[-1]["outcome"] = "SUCCESS"
            outcomes.append({"turn": i + 1, "outcome": "NOT_EXECUTED"})
        elif "No Python code block found" in feedback_text:
            if outcomes:
                outcomes[-1]["outcome"] = "NO_CODE"
            outcomes.append({"turn": i + 1, "outcome": "NOT_EXECUTED"})
        elif "TimeoutError" in feedback_text:
            if outcomes:
                outcomes[-1]["outcome"] = "TIMEOUT"
            outcomes.append({"turn": i + 1, "outcome": "NOT_EXECUTED"})
        else:
            if outcomes:
                outcomes[-1]["outcome"] = "ERROR"
            outcomes.append({"turn": i + 1, "outcome": "NOT_EXECUTED"})

    return outcomes


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

    数据流（v2）：
      优先使用 extra_info 中的结构化数据：
        - last_turn_text: 最后一轮模型原始输出 → 直接提取代码
        - num_turns: 实际轮数 → 直接用于惩罚计算
        - turn_details: 每轮 exec_status → 直接计算 corrected
      fallback 到正则切分 solution_str（兼容单元测试）。
    """
    if extra_info is None:
        extra_info = {}

    test_cases = extra_info.get("test_cases", "")

    # ===== 获取结构化数据（agent_loop 注入） =====
    last_turn_text = extra_info.get("last_turn_text")
    turn_details = extra_info.get("turn_details")
    num_turns_structured = extra_info.get("num_turns")

    # ===== 提取最后一轮代码 =====
    if last_turn_text is not None:
        # v2 路径：直接从最后一轮模型输出提取代码（无正则切分）
        last_code = extract_code_from_turn(last_turn_text)
        used_structured = True
    else:
        # fallback 路径：正则切分 solution_str（兼容测试）
        last_code = extract_last_code(solution_str)
        used_structured = False

    if last_code is None:
        _log_multiturn_metrics(
            solution_str, -1.0,
            num_turns_structured or count_actual_turns(solution_str),
            extra_info, "NO_CODE", turn_details=turn_details,
        )
        # 验证日志
        if turn_details:
            _log_turn_outcomes_verify(
                solution_str, extra_info, turn_details,
                None, "NO_CODE",
            )
        return -1.0

    # ===== 轮次计数 =====
    if num_turns_structured is not None:
        num_turns = num_turns_structured
    else:
        num_turns = count_actual_turns(solution_str)

    # ===== 语法检查 =====
    try:
        ast.parse(last_code)
    except SyntaxError:
        _log_multiturn_metrics(
            solution_str, 0.0, num_turns,
            extra_info, "SYNTAX_ERROR", turn_details=turn_details,
        )
        if turn_details:
            _log_turn_outcomes_verify(
                solution_str, extra_info, turn_details,
                last_code, "SYNTAX_ERROR",
            )
        return 0.0

    # ===== 沙盒执行 =====
    result: ExecResult = _execute_code_sync(last_code, test_cases, timeout=5.0)

    # ===== 验证日志：对比 agent_loop vs reward 的执行结果 =====
    if turn_details:
        _log_turn_outcomes_verify(
            solution_str, extra_info, turn_details,
            last_code, result.status.name,
        )

    # ===== 计算最终奖励 =====
    if result.status == ExecStatus.SUCCESS:
        penalty = turn_penalty * (num_turns - 1)
        reward = max(0.1, 1.0 - penalty)
        _log_multiturn_metrics(
            solution_str, reward, num_turns,
            extra_info, "SUCCESS", turn_details=turn_details,
        )
        return reward
    else:
        _log_multiturn_metrics(
            solution_str, 0.0, num_turns,
            extra_info, result.status.name, turn_details=turn_details,
        )
        return 0.0
