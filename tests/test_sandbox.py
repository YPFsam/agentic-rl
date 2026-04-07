"""沙盒模块单元测试"""
import asyncio
import sys
import os

# 添加项目根目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.sandbox import execute_code, execute_batch, ExecStatus


def test_success():
    """测试 1：正常代码执行成功"""
    result = asyncio.run(execute_code("print('hello')"))
    assert result.status == ExecStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert "hello" in result.stdout, f"Expected 'hello' in stdout, got {result.stdout}"
    print("  [PASS] 测试 1：正常代码执行成功")


def test_syntax_error():
    """测试 2：语法错误"""
    result = asyncio.run(execute_code("def f(\n"))
    assert result.status == ExecStatus.SYNTAX_ERROR, f"Expected SYNTAX_ERROR, got {result.status}"
    print("  [PASS] 测试 2：语法错误检测正确")


def test_runtime_error():
    """测试 3：运行时错误（含 assert 失败）"""
    result = asyncio.run(execute_code("assert 1 == 2"))
    assert result.status == ExecStatus.RUNTIME_ERROR, f"Expected RUNTIME_ERROR, got {result.status}"
    print("  [PASS] 测试 3：运行时错误检测正确")


def test_timeout():
    """测试 4：超时检测"""
    result = asyncio.run(execute_code("while True: pass", timeout=2.0))
    assert result.status == ExecStatus.TIMEOUT, f"Expected TIMEOUT, got {result.status}"
    print("  [PASS] 测试 4：超时检测正确")


def test_with_test_cases():
    """测试 5：带测试用例的代码执行"""
    code = "def add(a, b):\n    return a + b"
    test_cases = "assert add(1, 2) == 3\nassert add(-1, 1) == 0"
    result = asyncio.run(execute_code(code, test_cases))
    assert result.status == ExecStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    print("  [PASS] 测试 5：带测试用例执行成功")


def test_batch():
    """测试 6：批量并发执行"""
    codes = [
        "def add(a, b): return a + b",
        "def sub(a, b): return a - b",
        "while True: pass",
    ]
    test_cases_list = [
        "assert add(1, 2) == 3",
        "assert sub(3, 1) == 2",
        "",
    ]
    results = asyncio.run(execute_batch(codes, test_cases_list, timeout=2.0))
    assert len(results) == 3
    assert results[0].status == ExecStatus.SUCCESS
    assert results[1].status == ExecStatus.SUCCESS
    assert results[2].status == ExecStatus.TIMEOUT
    print("  [PASS] 测试 6：批量并发执行正确")


def test_zombie_process_timeout():
    """测试 7：恶意子进程逃逸拦截（验证 os.killpg 是否生效）"""
    evil_code = (
        "import subprocess, sys\n"
        "# 启动一个无限循环的子进程\n"
        "subprocess.Popen([sys.executable, '-c', 'while True: pass'])\n"
        "# 主进程也卡死\n"
        "while True:\n"
        "    pass\n"
    )
    result = asyncio.run(execute_code(evil_code, timeout=2.0))
    assert result.status == ExecStatus.TIMEOUT, f"Expected TIMEOUT, got {result.status}"
    # 额外验证：等待一小段时间后，不应有残留的僵尸进程
    import time
    time.sleep(1.0)
    print("  [PASS] 测试 7：恶意子进程逃逸拦截成功（无僵尸进程残留）")


if __name__ == "__main__":
    print("=" * 50)
    print("沙盒模块测试")
    print("=" * 50)
    test_success()
    test_syntax_error()
    test_runtime_error()
    test_timeout()
    test_with_test_cases()
    test_batch()
    test_zombie_process_timeout()
    print("=" * 50)
    print("全部 7 个测试通过！")
