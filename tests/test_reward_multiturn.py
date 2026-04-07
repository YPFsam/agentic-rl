"""多轮奖励函数单元测试"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.reward_multiturn import (
    compute_score_multiturn,
    count_actual_turns,
    extract_last_code,
)


def test_count_actual_turns():
    """测试轮次计数：基于 feedback 标记，不是代码块数"""
    # 单轮无 feedback
    assert count_actual_turns("just some text") == 1
    print("  [PASS] 单轮（无 feedback）→ 1 轮")

    # 1 个 feedback = 2 轮
    text_2turns = (
        "```python\ndef f(): pass\n```\n"
        "[Execution Feedback - Turn 1/3]\nError\n"
        "```python\ndef f(): return 1\n```"
    )
    assert count_actual_turns(text_2turns) == 2
    print("  [PASS] 1 个 feedback → 2 轮")

    # 2 个 feedback = 3 轮
    text_3turns = (
        "```python\ndef f(): pass\n```\n"
        "[Execution Feedback - Turn 1/3]\nError\n"
        "```python\ndef f(): return 1\n```\n"
        "[Execution Feedback - Turn 2/3]\nError\n"
        "```python\ndef f(): return 2\n```"
    )
    assert count_actual_turns(text_3turns) == 3
    print("  [PASS] 2 个 feedback → 3 轮")

    # 关键：单轮但有多个代码块 → 仍算 1 轮（非代码块数！）
    text_multi_blocks = (
        "```python\ndef helper(): return 1\n```\n"
        "```python\ndef main(): return helper()\n```"
    )
    assert count_actual_turns(text_multi_blocks) == 1
    print("  [PASS] 单轮多代码块 → 1 轮（不是 2！）")


def test_extract_last_code():
    """测试最后一轮代码提取"""
    # 成功终止：最后一个 feedback 之后有代码
    text_success = (
        "```python\ndef f(): pass\n```\n"
        "[Execution Feedback]\nError\n"
        "```python\ndef add(a,b): return a+b\n```"
    )
    code = extract_last_code(text_success)
    assert code is not None
    assert "def add" in code
    print("  [PASS] 成功场景：提取最后一轮代码")


def test_perfect_turn1():
    """测试 1：第 1 轮就成功 → reward = 1.0"""
    solution = "```python\ndef add(a, b):\n    return a + b\n```"
    extra_info = {"test_cases": "assert add(1, 2) == 3"}
    score = compute_score_multiturn("mbpp", solution, "", extra_info)
    assert score == 1.0, f"Expected 1.0, got {score}"
    print("  [PASS] 第 1 轮成功 → reward = 1.0")


def test_perfect_turn2():
    """测试 2：第 2 轮成功 → reward = 0.85"""
    solution = (
        "```python\ndef add(a, b):\n    return a - b\n```\n"
        "[Execution Feedback - Turn 1/3]\nRuntimeError\n"
        "```python\ndef add(a, b):\n    return a + b\n```"
    )
    extra_info = {"test_cases": "assert add(1, 2) == 3"}
    score = compute_score_multiturn("mbpp", solution, "", extra_info)
    assert score == 0.85, f"Expected 0.85, got {score}"
    print("  [PASS] 第 2 轮成功 → reward = 0.85")


def test_perfect_turn3():
    """测试 3：第 3 轮成功 → reward = 0.70"""
    solution = (
        "```python\ndef f(): pass\n```\n"
        "[Execution Feedback - Turn 1/3]\nError1\n"
        "```python\ndef f(): x\n```\n"
        "[Execution Feedback - Turn 2/3]\nError2\n"
        "```python\ndef add(a, b):\n    return a + b\n```"
    )
    extra_info = {"test_cases": "assert add(1, 2) == 3"}
    score = compute_score_multiturn("mbpp", solution, "", extra_info)
    assert score == 0.70, f"Expected 0.70, got {score}"
    print("  [PASS] 第 3 轮成功 → reward = 0.70")


def test_multicode_no_overpenalty():
    """测试 4：单轮但输出多个代码块 → 不应被多扣轮次惩罚"""
    # 关键测试：模型一次输出 3 个代码块（辅助函数+主函数+测试），但只有 1 轮
    solution = (
        "```python\ndef helper(x):\n    return x * 2\n```\n"
        "Here's the main function:\n"
        "```python\ndef add(a, b):\n    return a + b\n```\n"
        "```python\nprint(add(1,2))\n```"
    )
    extra_info = {"test_cases": "assert add(1, 2) == 3"}
    score = compute_score_multiturn("mbpp", solution, "", extra_info)
    # 应该是 1.0（1 轮，无惩罚），不是 1.0 - 0.15*2 = 0.7
    assert score == 1.0, f"Expected 1.0 (1 turn, no penalty), got {score}"
    print("  [PASS] 单轮多代码块 → reward = 1.0（无轮次惩罚）")


def test_no_code():
    """测试 5：无代码 → gate -1.0"""
    solution = "I don't know the answer."
    extra_info = {"test_cases": "assert True"}
    score = compute_score_multiturn("mbpp", solution, "", extra_info)
    assert score == -1.0, f"Expected -1.0, got {score}"
    print("  [PASS] 无代码 → reward = -1.0（门控触发）")


def test_final_failure():
    """测试 6：3 轮全部失败 → 0.0"""
    solution = (
        "```python\ndef add(a): return a\n```\n"
        "[Execution Feedback - Turn 1/3]\nError1\n"
        "```python\ndef add(a): return a\n```\n"
        "[Execution Feedback - Turn 2/3]\nError2\n"
        "```python\ndef add(a): return a\n```"
    )
    extra_info = {"test_cases": "assert add(1, 2) == 3"}
    score = compute_score_multiturn("mbpp", solution, "", extra_info)
    assert score == 0.0, f"Expected 0.0, got {score}"
    print("  [PASS] 3 轮全失败 → reward = 0.0")


def test_reward_floor():
    """测试 7：奖励下限 0.1 保底，即使极端场景也不倒挂"""
    # 构造极端场景：10 个 feedback（11 轮），但最终成功
    # 如果没有保底：1.0 - 0.15 * 10 = -0.5（倒挂！）
    parts = []
    for i in range(10):
        parts.append(f"```python\ndef f(): pass\n```")
        parts.append(f"[Execution Feedback - Turn {i+1}/11]\nError")
    parts.append("```python\ndef add(a, b):\n    return a + b\n```")
    solution = "\n".join(parts)
    extra_info = {"test_cases": "assert add(1, 2) == 3"}
    score = compute_score_multiturn("mbpp", solution, "", extra_info)
    assert score >= 0.0, f"Reward should never be negative, got {score}"
    assert score == 0.1, f"Expected floor 0.1, got {score}"
    print("  [PASS] 极端 11 轮成功 → reward = 0.1（保底生效）")


if __name__ == "__main__":
    print("=" * 50)
    print("多轮奖励函数测试")
    print("=" * 50)
    test_count_actual_turns()
    test_extract_last_code()
    test_perfect_turn1()
    test_perfect_turn2()
    test_perfect_turn3()
    test_multicode_no_overpenalty()
    test_no_code()
    test_final_failure()
    test_reward_floor()
    print("=" * 50)
    print("全部 9 个测试通过！")
