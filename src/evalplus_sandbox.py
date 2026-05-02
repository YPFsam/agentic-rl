"""
评估用沙盒 — 直接调用 evalplus 内部函数，保证判卷 100% 一致

核心设计：
  1. pass/fail 判断：直接调用 evalplus.eval.untrusted_check()
     - 与 `python -m evalplus.evaluate` 走完全相同的代码路径
     - 在独立子进程中执行（multiprocessing 隔离）
     - 使用 evalplus 的 safety utils（swallow_io, time_limit, reliability_guard）
  2. 错误详情捕获：在 untrusted_check 判定 fail 时，
     在子进程中用 evalplus 的 exec 模型捕获异常信息（用于多轮反馈）
  3. 脱敏：与 agent_loop._format_error 一致，隐藏 test case 输入/输出
  4. Debug 日志：每次检查打印 task_id、status、n_passed/n_total、耗时

与训练沙盒 (src/sandbox.py) 的区别：
  - 训练沙盒：python code+test_cases.py（脚本模式，assert 语句）
  - 评估沙盒：exec(prompt+completion) → fn(*input)（evalplus 函数调用模式）
  - 判卷结果与 evalplus 完全一致，不会出现沙盒/evalplus 判断不一致的问题
"""
import multiprocessing
import re
import time
import traceback
from dataclasses import dataclass

import numpy as np

# 直接导入 evalplus 内部函数（不是重新实现，是直接调用 evalplus 的代码）
from evalplus.eval import PASS, FAIL, TIMEOUT, untrusted_check, is_floats
from evalplus.eval.utils import swallow_io, time_limit
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.evaluate import get_groundtruth


# ========== 脱敏（与 agent_loop._format_error 一致）==========

# traceback 中的 assert 行包含完整测试输入和期望输出，替换为占位符
_ASSERT_LINE_PATTERN = re.compile(r'^(\s+)assert\s+.+$', re.MULTILINE)
_ASSERT_LINE_REPLACEMENT = r'\1assert <test_case>'

# evalplus 模式下 output mismatch 信息中的 expected 值也需脱敏
_EXPECTED_LINE_PATTERN = re.compile(r'^(\s*)expected:\s+.+$', re.MULTILINE)
_EXPECTED_LINE_REPLACEMENT = r'\1expected: <test_case>'

# 错误信息最大长度（与 agent_loop.max_error_length 一致）
MAX_ERROR_LENGTH = 500


@dataclass
class EvalPlusResult:
    """evalplus 执行结果"""
    passed: bool           # 是否通过所有 base tests（与 evalplus 判卷一致）
    status: str            # "pass" / "fail" / "timeout"（evalplus 原始状态）
    error_msg: str         # 错误信息（已脱敏，用于多轮反馈）
    error_msg_raw: str     # 原始错误信息（debug 日志用）
    details: list          # 每个 test case 的通过/失败详情
    n_passed: int          # 通过的 test case 数
    n_total: int           # 总 test case 数
    check_time_s: float    # evalplus_check 总耗时（含 untrusted_check + 错误捕获）

    @property
    def is_timeout(self) -> bool:
        return self.status == "timeout"


# 模块级缓存（首次调用时初始化，避免重复加载数据）
_problems = None
_expected_output = None
_dataset_hash = None


def _init_evalplus_data():
    """初始化 evalplus 数据（problem 定义 + groundtruth expected outputs）。

    等价于 evalplus.evaluate 中 evaluate() 函数的数据加载逻辑：
      - get_human_eval_plus() → 题目定义（prompt, base_input, plus_input, atol, entry_point）
      - get_groundtruth() → 用 trusted_exec 计算每个 test case 的 expected output
    """
    global _problems, _expected_output, _dataset_hash

    if _problems is not None:
        return

    print("[evalplus_sandbox] 初始化 evalplus 数据...", flush=True)
    t0 = time.time()
    _problems = get_human_eval_plus()
    _dataset_hash = get_human_eval_plus_hash()
    _expected_output = get_groundtruth(_problems, _dataset_hash, [])
    elapsed = time.time() - t0
    print(f"[evalplus_sandbox] 初始化完成: {len(_problems)} 个题目, "
          f"hash={_dataset_hash[:8]}, 耗时 {elapsed:.1f}s", flush=True)


def _sanitize_error_msg(raw: str) -> str:
    """脱敏错误信息（与 agent_loop._format_error 逻辑对齐）。

    处理步骤：
      1. 替换 traceback 中的 assert 行（agent_loop 同款，隐藏测试输入/输出）
      2. 替换 output mismatch 信息中的 expected 值（evalplus 模式特有）
      3. 尾部截断到 MAX_ERROR_LENGTH（保留 traceback 底部关键信息）

    脱敏效果：
      - AssertionError + assert 行 → assert <test_case>
      - output mismatch 的 expected: [...] → expected: <test_case>
      - RuntimeError / SyntaxError → 保留错误类型和模型出错代码行
    """
    # 步骤 1：脱敏 assert 行（必须在截断前，防止截断切掉 assert 行）
    sanitized = _ASSERT_LINE_PATTERN.sub(_ASSERT_LINE_REPLACEMENT, raw)
    # 步骤 2：脱敏 expected 值（evalplus 模式下 output mismatch 信息）
    sanitized = _EXPECTED_LINE_PATTERN.sub(_EXPECTED_LINE_REPLACEMENT, sanitized)
    # 步骤 3：尾部截断
    if len(sanitized) > MAX_ERROR_LENGTH:
        sanitized = sanitized[-MAX_ERROR_LENGTH:]
    return sanitized


def evalplus_check(task_id: str, completion: str) -> EvalPlusResult:
    """
    使用 evalplus 的 untrusted_check() 检查代码是否通过所有 base tests。

    执行流程与 evalplus.evaluate.check_correctness() 完全一致：
      1. solution = problems[task_id]["prompt"] + completion
         （与 evaluate.py L207-211 一致）
      2. 调用 untrusted_check()：
         - 在 multiprocessing.Process 中执行
         - exec(solution) → fn = globals[entry_point]
         - 对每个 base_input 调用 fn(*input)，比较输出与 expected
         - 使用 swallow_io + time_limit + reliability_guard 隔离
      3. 返回 (status, details) — status 为 "pass"/"fail"/"timeout"

    Args:
        task_id: HumanEval 题目 ID，如 "HumanEval/0"
        completion: 模型生成的代码补全

    Returns:
        EvalPlusResult: 包含通过状态、错误信息和测试详情
    """
    t_start = time.time()
    _init_evalplus_data()

    if task_id not in _problems:
        return EvalPlusResult(
            passed=False, status="error",
            error_msg=f"Unknown task_id: {task_id}",
            error_msg_raw=f"Unknown task_id: {task_id}",
            details=[], n_passed=0, n_total=0,
            check_time_s=time.time() - t_start,
        )

    problem = _problems[task_id]

    # 与 evalplus evaluate.py L207-211 完全一致
    solution = problem["prompt"] + completion

    # 调用 evalplus 的 untrusted_check（与 evaluate.py L97-108 一致）
    t_check_start = time.time()
    status, details = untrusted_check(
        dataset="humaneval",
        code=solution,
        inputs=problem["base_input"],
        entry_point=problem["entry_point"],
        expected=_expected_output[task_id]["base"],
        atol=problem["atol"],
        ref_time=_expected_output[task_id]["base_time"],
        fast_check=False,
    )
    check_time = time.time() - t_check_start

    n_total = len(problem["base_input"])
    details_list = details.tolist() if hasattr(details, 'tolist') else list(details)
    n_passed = sum(details_list) if details_list else 0

    # 捕获错误信息（用于多轮反馈）
    error_msg_raw = ""
    error_msg = ""
    if status != PASS:
        t_capture_start = time.time()
        error_msg_raw = _capture_error_message(
            solution=solution,
            entry_point=problem["entry_point"],
            inputs=problem["base_input"],
            expected=_expected_output[task_id]["base"],
            atol=problem["atol"],
            details=details_list,
            task_id=task_id,
        )
        capture_time = time.time() - t_capture_start
        error_msg = _sanitize_error_msg(error_msg_raw)

        # Debug 日志：失败时打印详细信息
        print(f"  [evalplus_sandbox] FAIL {task_id} | "
              f"tests={n_passed}/{n_total} | "
              f"untrusted_check={check_time:.1f}s capture={capture_time:.1f}s | "
              f"error_raw={error_msg_raw[:120]}", flush=True)
    else:
        # Debug 日志：通过时简要打印
        print(f"  [evalplus_sandbox] PASS {task_id} | tests={n_passed}/{n_total} | "
              f"check={check_time:.1f}s", flush=True)

    total_time = time.time() - t_start
    return EvalPlusResult(
        passed=(status == PASS),
        status=status,
        error_msg=error_msg,
        error_msg_raw=error_msg_raw,
        details=details_list,
        n_passed=n_passed,
        n_total=n_total,
        check_time_s=total_time,
    )


def _capture_error_message(
    solution: str,
    entry_point: str,
    inputs: list,
    expected: list,
    atol: float,
    details: list,
    task_id: str = "",
) -> str:
    """
    在子进程中运行 solution，捕获第一个失败的 test case 的错误信息。

    执行模型与 evalplus unsafe_execute() 完全一致：
      exec(solution) → fn = globals[entry_point] → fn(*input)

    使用 multiprocessing 隔离（与 evalplus 一致），超时保护 10s。

    返回原始错误信息（未经脱敏），由调用方通过 _sanitize_error_msg 脱敏。
    """
    result_queue = multiprocessing.Queue()

    def _filter_traceback(tb_str: str) -> str:
        """过滤 traceback 中的 sandbox 内部帧，只保留用户代码（<string>）部分。

        避免向模型暴露 evalplus_sandbox.py 内部路径，让反馈只包含
        用户代码的错误类型、行号和消息（与旧 sandbox 的 stderr 格式对齐）。
        """
        lines = tb_str.split('\n')
        filtered = []
        in_user_frame = False
        for line in lines:
            # 保留用户代码帧（<string> 开头）
            if 'File "<string>"' in line:
                in_user_frame = True
                filtered.append(line)
            elif in_user_frame:
                # 用户帧后面的内容（错误行代码、错误消息）
                if line.strip().startswith('File "') and '"<string>"' not in line:
                    # 遇到新的非用户帧，停止
                    in_user_frame = False
                else:
                    filtered.append(line)
            # 始终保留错误类型行（最后一行，如 RecursionError、SyntaxError 等）
            if line.strip() and not line.startswith(' ') and not line.startswith('File') \
               and not line.startswith('Traceback') and not line.startswith('During'):
                if line not in filtered:
                    filtered.append(line)

        result = '\n'.join(filtered).strip()
        # 如果过滤后为空，保留原始错误类型和消息（最后一行）
        if not result:
            last_line = tb_str.strip().split('\n')[-1] if tb_str.strip() else "Unknown error"
            result = last_line
        return result

    def _worker():
        try:
            exec_globals = {}
            # 与 evalplus unsafe_execute 一致：先 swallow_io 中 exec
            with swallow_io():
                exec(solution, exec_globals)

            fn = exec_globals.get(entry_point)
            if fn is None:
                result_queue.put(f"NameError: function '{entry_point}' not defined")
                return

            # 逐 test case 执行，捕获第一个错误
            for i, inp in enumerate(inputs):
                try:
                    with time_limit(5.0):
                        with swallow_io():
                            out = fn(*inp)

                    exp = expected[i]

                    # 与 evalplus unsafe_execute 的比较逻辑一致
                    exact_match = (out == exp)

                    # 浮点数 atol 处理（与 unsafe_execute 一致）
                    if atol == 0 and is_floats(exp):
                        atol_local = 1e-6
                    else:
                        atol_local = atol

                    if not exact_match and atol_local != 0:
                        try:
                            assert type(out) == type(exp)
                            if isinstance(exp, (list, tuple)):
                                assert len(out) == len(exp)
                            assert np.allclose(out, exp, rtol=1e-7, atol=atol_local)
                        except AssertionError:
                            exp_repr = repr(exp)
                            out_repr = repr(out)
                            if len(exp_repr) > 200:
                                exp_repr = exp_repr[:200] + "..."
                            if len(out_repr) > 200:
                                out_repr = out_repr[:200] + "..."
                            result_queue.put(
                                f"AssertionError: test case {i} output mismatch\n"
                                f"  expected: {exp_repr}\n"
                                f"  got:      {out_repr}"
                            )
                            return
                    elif not exact_match:
                        exp_repr = repr(exp)
                        out_repr = repr(out)
                        if len(exp_repr) > 200:
                            exp_repr = exp_repr[:200] + "..."
                        if len(out_repr) > 200:
                            out_repr = out_repr[:200] + "..."
                        result_queue.put(
                            f"AssertionError: test case {i} output mismatch\n"
                            f"  expected: {exp_repr}\n"
                            f"  got:      {out_repr}"
                        )
                        return

                except Exception as e:
                    tb = traceback.format_exception(type(e), e, e.__traceback__)
                    tb_str = _filter_traceback("".join(tb))
                    if len(tb_str) > 500:
                        tb_str = tb_str[-500:]
                    result_queue.put(tb_str)
                    return

            # 所有 test case 通过（不应发生，因为 untrusted_check 说 fail）
            result_queue.put("")

        except SyntaxError as e:
            tb = traceback.format_exception(type(e), e, e.__traceback__)
            tb_str = _filter_traceback("".join(tb))
            if len(tb_str) > 500:
                tb_str = tb_str[-500:]
            result_queue.put(f"SyntaxError:\n{tb_str}")
        except Exception as e:
            tb = traceback.format_exception(type(e), e, e.__traceback__)
            tb_str = _filter_traceback("".join(tb))
            if len(tb_str) > 500:
                tb_str = tb_str[-500:]
            result_queue.put(f"ExecError:\n{tb_str}")

    p = multiprocessing.Process(target=_worker)
    p.start()
    p.join(timeout=10)

    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
        if p.is_alive():
            p.kill()
            time.sleep(0.1)
        return f"TimeoutError: code execution exceeded 10s"

    if not result_queue.empty():
        msg = result_queue.get()
        if msg == "":
            # 本地全部通过但 untrusted_check 失败 → 可能是精度/竞态差异
            failed_cases = [i for i, d in enumerate(details) if not d]
            n_failed = len(failed_cases)
            if failed_cases:
                return (f"Failed {n_failed}/{len(inputs)} test cases "
                        f"(first failure at test case {failed_cases[0]})")
            return "Unknown failure (untrusted_check failed but all local tests passed)"
        return msg

    return "Unknown error (subprocess exited without result)"
