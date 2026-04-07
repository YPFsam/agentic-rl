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
      0. os.setsid() — 创建新进程组，超时时可 killpg 杀掉整个进程组（含孙子进程）
      1. Docker 容器隔离（文件系统 + 网络）——最外层
      2. RLIMIT_AS 2GB（内存）——防 while True: a+=[1]*10**9 吞内存
      3. RLIMIT_CPU 3s（CPU 时间）——防 while True: pass 霸占 CPU，内核级强制
      4. asyncio.wait_for 5s（wall-clock 超时）——兜底，处理 sleep/IO 阻塞
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

    @property
    def passed(self) -> bool:
        """是否通过所有测试"""
        return self.status == ExecStatus.SUCCESS


async def execute_code(
    code: str,
    test_cases: str = "",
    timeout: float = 5.0,
) -> ExecResult:
    """
    在隔离子进程中异步执行 Python 代码。

    将代码和测试用例拼接后写入临时文件，用 asyncio 子进程执行。

    Args:
        code: 模型生成的 Python 代码
        test_cases: 测试用例（assert 语句，每行一个）
        timeout: 超时秒数，默认 5 秒

    Returns:
        ExecResult 包含执行状态、标准输出和标准错误
    """
    # 拼接代码和测试用例
    if test_cases:
        full_code = f"{code}\n\n{test_cases}"
    else:
        full_code = code

    # 写入临时文件
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="sandbox_")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(full_code)

        # 用 asyncio 子进程执行
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        proc = await asyncio.create_subprocess_exec(
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

        if proc.returncode == 0:
            return ExecResult(
                status=ExecStatus.SUCCESS,
                stdout=stdout,
                stderr=stderr,
            )

        # 根据错误类型判断状态
        if "SyntaxError" in stderr:
            return ExecResult(
                status=ExecStatus.SYNTAX_ERROR,
                stdout=stdout,
                stderr=stderr,
            )
        else:
            return ExecResult(
                status=ExecStatus.RUNTIME_ERROR,
                stdout=stdout,
                stderr=stderr,
            )

    finally:
        # 确保清理临时文件
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


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
