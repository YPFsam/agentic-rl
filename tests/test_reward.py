"""门控奖励函数单元测试"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.reward import compute_score, extract_code


def test_extract_code():
    """测试代码提取功能"""
    # 从 ```python 代码块提取
    text1 = "Some thinking\n```python\ndef add(a, b):\n    return a + b\n```"
    code1 = extract_code(text1)
    assert code1 is not None
    assert "def add" in code1
    print("  [PASS] 代码块提取正确")

    # 无代码块 → None
    text2 = "Just plain text without any code"
    code2 = extract_code(text2)
    assert code2 is None
    print("  [PASS] 无代码时返回 None")

    # 截断的未闭合代码块（max_tokens 截断场景）
    text3 = "Some thinking\n```python\ndef add(a, b):\n    return a"
    code3 = extract_code(text3)
    assert code3 is not None
    assert "def add" in code3
    print("  [PASS] 截断的未闭合代码块提取正确")

    # 多代码块：拼接所有代码块（helper + add 一起执行才不会 NameError）
    text4 = (
        "Here is a helper:\n```python\ndef helper(): return 1\n```\n"
        "And the main function:\n```python\ndef add(a, b): return a + b\n```"
    )
    code4 = extract_code(text4)
    assert code4 is not None
    assert "helper" in code4, "多代码块应拼接，helper 不能丢"
    assert "add" in code4, "多代码块应拼接，add 不能丢"
    print("  [PASS] 多代码块拼接正确（helper + add 都在）")

    # think 标签内的代码块不应被提取
    text5 = (
        "<think reasoning>\nLet me draft: ```python\ndef draft(): pass\n```\n</think >\n"
        "```python\ndef add(a, b): return a + b\n```"
    )
    code5 = extract_code(text5)
    assert code5 is not None
    assert "add" in code5, "应提取正文代码块，不是 think 内的草稿"
    assert "draft" not in code5, "think 内的草稿代码不应被提取"
    print("  [PASS] think 标签内的代码块被正确忽略")


def test_perfect_output():
    """测试 1：完美输出（代码正确 + 测试通过）"""
    solution = (
        "Let me think about this step by step.\n"
        "```python\ndef add(a, b):\n    return a + b\n```"
    )
    extra_info = {"test_cases": "assert add(1, 2) == 3\nassert add(0, 0) == 0"}
    score = compute_score("mbpp", solution, "", extra_info)
    assert score == 1.0, f"Expected 1.0, got {score}"
    print("  [PASS] 测试 1：完美输出 → reward = 1.0")


def test_no_code():
    """测试 2：无代码输出（门控触发）"""
    solution = "The answer is 42."
    extra_info = {"test_cases": "assert True"}
    score = compute_score("mbpp", solution, "", extra_info)
    assert score == -1.0, f"Expected -1.0, got {score}"
    print("  [PASS] 测试 2：无代码 → reward = -1.0（门控触发）")


def test_wrong_code():
    """测试 3：代码错误（运行时错误 / assert 失败）"""
    solution = (
        "```python\ndef add(a, b):\n    return a - b\n```"
    )
    extra_info = {"test_cases": "assert add(1, 2) == 3"}
    score = compute_score("mbpp", solution, "", extra_info)
    assert score == 0.0, f"Expected 0.0, got {score}"
    print("  [PASS] 测试 3：代码错误 → reward = 0.0")


def test_gibberish():
    """测试 4：完全乱输出"""
    solution = "I don't know the answer."
    extra_info = {"test_cases": "assert True"}
    score = compute_score("mbpp", solution, "", extra_info)
    assert score == -1.0, f"Expected -1.0, got {score}"
    print("  [PASS] 测试 4：乱输出 → reward = -1.0（门控触发）")


def test_syntax_error_code():
    """测试 5：语法错误代码"""
    solution = "```python\ndef f(\n```"
    extra_info = {"test_cases": ""}
    score = compute_score("mbpp", solution, "", extra_info)
    assert score == 0.0, f"Expected 0.0, got {score}"
    print("  [PASS] 测试 5：语法错误 → reward = 0.0")


def test_timeout_code():
    """测试 6：死循环代码"""
    solution = "```python\nwhile True:\n    pass\n```"
    extra_info = {"test_cases": ""}
    score = compute_score("mbpp", solution, "", extra_info)
    assert score == 0.0, f"Expected 0.0, got {score}"
    print("  [PASS] 测试 6：超时代码 → reward = 0.0")


if __name__ == "__main__":
    print("=" * 50)
    print("门控奖励函数测试")
    print("=" * 50)
    test_extract_code()
    test_perfect_output()
    test_no_code()
    test_wrong_code()
    test_gibberish()
    test_syntax_error_code()
    test_timeout_code()
    print("=" * 50)
    print("全部 10 个测试通过！")
