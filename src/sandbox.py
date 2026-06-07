"""
async 代码沙盒执行模块

为 veRL AgentLoop 提供 async 代码执行能力。
在隔离子进程中运行 Python 代码，支持超时控制和批量执行。

注意：必须用 async 实现，因为 veRL AgentLoop.run() 是 async 方法，
内部需要并发执行多个沙盒调用（num_generations 个并行 rollout）。
"""
import asyncio
import sys
import tempfile
import os
import re
import signal
import resource
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class ExecStatus(Enum):
    """代码执行结果状态"""
    SUCCESS = "success"               # 代码执行成功，所有 assert 通过
    SYNTAX_ERROR = "syntax_error"     # 语法错误（AST 解析失败或 Python 报 SyntaxError）
    RUNTIME_ERROR = "runtime_error"   # 运行时错误（含 AssertionError）
    TIMEOUT = "timeout"               # 超时（死循环）


# ========== 沙盒子进程资源限制 ==========

# 单进程最大虚拟内存（字节）。防止模型生成恶意代码吞噬宿主机内存。
MAX_MEMORY_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB

# 单进程最大 CPU 时间（秒）。
MAX_CPU_SECONDS = 3


def _preexec_fn():
    """preexec_fn：在子进程 fork 后、exec 前设置资源限制和进程组。

    防御层设计（由外到内）：
      0. OS timeout 命令（wall-clock 硬超时）——最外层，不依赖 asyncio/进程组
      1. os.setsid() — 创建新进程组，超时时可 killpg 杀掉整个进程组（含孙子进程）
      2. Docker 容器隔离（文件系统 + 网络）
      3. RLIMIT_AS 2GB（内存）——防 while True: a+=[1]*10**9 吞内存
      4. RLIMIT_CPU 3s（CPU 时间）——防 while True: pass 霸占 CPU，内核级强制
      5. asyncio.wait_for 5s（wall-clock 超时）——第一层超时，处理 sleep/IO 阻塞
    """
    os.setsid()  # 创建新进程组，使 killpg 能杀掉孙子进程
    resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY_BYTES, MAX_MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (MAX_CPU_SECONDS, MAX_CPU_SECONDS))


@dataclass
class ExecResult:
    """代码执行结果"""
    status: ExecStatus
    stdout: str
    stderr: str
    n_passed: int = 0   # 通过的 assert 数量（仅 count_passes=True 时填充）
    n_total: int = 0    # 总 assert 数量（仅 count_passes=True 时填充）

    @property
    def passed(self) -> bool:
        """是否通过所有测试"""
        return self.status == ExecStatus.SUCCESS


async def execute_code(
    code: str,
    test_cases: str = "",
    timeout: float = 5.0,
    count_passes: bool = False,
) -> ExecResult:
    """
    在隔离子进程中异步执行 Python 代码。

    将代码和测试用例拼接后写入临时文件，用 asyncio 子进程执行。

    Args:
        code: 模型生成的 Python 代码
        test_cases: 测试用例（assert 语句，每行一个）
        timeout: 超时秒数，默认 5 秒
        count_passes: 是否统计每个 assert 的通过数（用于过程奖励 C_t 计算）

    Returns:
        ExecResult 包含执行状态、标准输出和标准错误
    """
    # 拼接代码和测试用例
    if count_passes and test_cases:
        full_code = _build_counting_wrapper(code, test_cases)
    elif test_cases:
        full_code = f"{code}\n\n{test_cases}"
    else:
        full_code = code

    # 写入临时文件
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="sandbox_")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(full_code)

        # 用 asyncio 子进程执行
        # 外层用系统 timeout 命令作为 OS 级硬超时保障：
        #   - 不依赖 asyncio 事件循环（防止事件循环积压导致 wait_for 回调延迟）
        #   - 不依赖进程组（防 os.fork()+setpgrp 逃逸）
        #   - timeout 追踪的是 PID，比 killpg 更可靠
        # 内层 asyncio.wait_for 仍然保留，作为第一层超时
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        hard_timeout = int(timeout) + 2  # 比 asyncio 超时多 2s，作为最后防线
        proc = await asyncio.create_subprocess_exec(
            "timeout", str(hard_timeout),
            sys.executable, tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            preexec_fn=_preexec_fn,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            # 杀掉整个进程组（含孙子进程），防 os.fork()/multiprocessing 逃逸
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            await proc.wait()
            return ExecResult(
                status=ExecStatus.TIMEOUT,
                stdout="",
                stderr="TimeoutExpired",
            )

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        # 从 stdout 解析 assert 通过计数
        n_passed, n_total = _parse_pass_count(stdout)

        if proc.returncode == 0:
            return ExecResult(
                status=ExecStatus.SUCCESS,
                stdout=stdout,
                stderr=stderr,
                n_passed=n_passed,
                n_total=n_total,
            )

        # 根据错误类型判断状态
        if "SyntaxError" in stderr:
            return ExecResult(
                status=ExecStatus.SYNTAX_ERROR,
                stdout=stdout,
                stderr=stderr,
                n_passed=n_passed,
                n_total=n_total,
            )
        else:
            return ExecResult(
                status=ExecStatus.RUNTIME_ERROR,
                stdout=stdout,
                stderr=stderr,
                n_passed=n_passed,
                n_total=n_total,
            )

    finally:
        # 确保清理临时文件
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


_PASS_COUNT_RE = re.compile(r'__PASS_COUNT__:(\d+)/(\d+)')


def _build_counting_wrapper(code: str, test_cases: str) -> str:
    """
    构建 assert 计数包装代码。

    将 test_cases 中的每个 assert 逐个用 try-except 包裹，
    统计通过数，最后通过 stdout 输出 __PASS_COUNT__:X/Y。

    注意：函数定义和 check() 调用不参与 assert 计数，
    只有独立的 assert 语句被计数。
    """
    lines = test_cases.strip().split('\n')
    assert_lines = []
    other_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('assert ') or stripped.startswith('assert\t'):
            assert_lines.append(stripped)
        else:
            other_lines.append(line)

    n_total = len(assert_lines)
    if n_total == 0:
        # 没有 assert 语句，直接执行原始代码
        return f"{code}\n\n{test_cases}"

    # 构建计数包装
    wrapper_lines = [
        code,
        "",
        "# === assert 计数包装 ===",
        f"__n_passed__ = 0",
        f"__n_total__ = {n_total}",
    ]
    # 先执行非 assert 的代码（函数定义等）
    if other_lines:
        wrapper_lines.append("")
        wrapper_lines.extend(other_lines)
    # 逐个执行 assert 并计数
    for assert_line in assert_lines:
        wrapper_lines.append("try:")
        wrapper_lines.append(f"    {assert_line}")
        wrapper_lines.append("    __n_passed__ += 1")
        wrapper_lines.append("except Exception:")
        wrapper_lines.append("    pass")
    wrapper_lines.append(f'print(f"__PASS_COUNT__:{{__n_passed__}}/{{__n_total__}}")')

    return "\n".join(wrapper_lines)


def _parse_pass_count(stdout: str) -> tuple[int, int]:
    """从 stdout 中解析 __PASS_COUNT__:X/Y"""
    match = _PASS_COUNT_RE.search(stdout)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0


async def execute_batch(
    codes: list[str],
    test_cases_list: list[str],
    timeout: float = 5.0,
    max_concurrent: int = 12,
) -> list[ExecResult]:
    """
    并发批量执行代码。

    使用 asyncio.gather + Semaphore 并发执行多个代码片段。
    Semaphore 限制同时运行的沙盒子进程数量，防止 CPU 饥饿。

    Args:
        codes: 代码列表
        test_cases_list: 对应的测试用例列表（与 codes 等长）
        timeout: 每个代码的超时秒数
        max_concurrent: 最大并发沙盒数（默认 12）

    Returns:
        ExecResult 列表，与 codes 一一对应
    """
    assert len(codes) == len(test_cases_list), \
        f"codes 和 test_cases_list 长度不一致: {len(codes)} vs {len(test_cases_list)}"

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _limited_exec(code: str, tc: str) -> ExecResult:
        async with semaphore:
            return await execute_code(code, tc, timeout)

    tasks = [
        _limited_exec(code, tc)
        for code, tc in zip(codes, test_cases_list)
    ]
    return await asyncio.gather(*tasks)
