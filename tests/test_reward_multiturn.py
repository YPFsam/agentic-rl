"""多轮奖励函数单元测试

测试覆盖两条路径：
  1. v2 结构化路径：extra_info 包含 last_turn_text + turn_details
  2. fallback 正则路径：无结构化数据，从 solution_str 正则切分
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.reward_multiturn import (
    compute_score_multiturn,
    count_actual_turns,
    extract_last_code,
    extract_code_from_turn,
)


# ========== 旧版正则路径测试（fallback） ==========

def test_count_actual_turns():
    """测试轮次计数：基于 feedback 标记，不是代码块数"""
    assert count_actual_turns("just some text") == 1
    print("  [PASS] 单轮（无 feedback）→ 1 轮")

    text_2turns = (
        "```python\ndef f(): pass\n```\n"
        "[Execution Feedback - Turn 1/3]\nError\n"
        "```python\ndef f(): return 1\n```"
    )
    assert count_actual_turns(text_2turns) == 2
    print("  [PASS] 1 个 error feedback → 2 轮")

    text_3turns = (
        "```python\ndef f(): pass\n```\n"
        "[Execution Feedback - Turn 1/3]\nError\n"
        "```python\ndef f(): return 1\n```\n"
        "[Execution Feedback - Turn 2/3]\nError\n"
        "```python\ndef f(): return 2\n```"
    )
    assert count_actual_turns(text_3turns) == 3
    print("  [PASS] 2 个 error feedback → 3 轮")

    text_multi_blocks = (
        "```python\ndef helper(): return 1\n```\n"
        "```python\ndef main(): return helper()\n```"
    )
    assert count_actual_turns(text_multi_blocks) == 1
    print("  [PASS] 单轮多代码块 → 1 轮（不是 2！）")

    text_turn1_success = (
        "```python\ndef add(a, b): return a + b\n```\n"
        "[Execution Result]\nAll test cases passed!"
    )
    assert count_actual_turns(text_turn1_success) == 1
    print("  [PASS] 第 1 轮成功（含 SUCCESS 标记）→ 1 轮（不是 2！）")

    text_turn2_success = (
        "```python\ndef f(): pass\n```\n"
        "[Execution Feedback - Turn 1/3]\nError\n"
        "```python\ndef add(a, b): return a + b\n```\n"
        "[Execution Result]\nAll test cases passed!"
    )
    assert count_actual_turns(text_turn2_success) == 2
    print("  [PASS] 第 2 轮成功（1 error + 1 success）→ 2 轮")


def test_extract_last_code():
    """测试最后一轮代码提取（旧版 fallback）"""
    text_success = (
        "```python\ndef f(): pass\n```\n"
        "[Execution Feedback]\nError\n"
        "```python\ndef add(a,b): return a+b\n```"
    )
    code = extract_last_code(text_success)
    assert code is not None
    assert "def add" in code
    print("  [PASS] 成功场景：提取最后一轮代码")


def test_extract_last_code_with_success_marker():
    """测试：[Execution Result] 标记后，只提取最后一轮代码（不拼接历史轮次）"""
    text = (
        "```python\ndef f(): pass\n```\n"
        "[Execution Feedback - Turn 1/3]\n"
        "Your code produced the following error:\n"
        "```\nRuntimeError\n```\n"
        "Please analyze the error and fix your code.\n"
        "Output your revised solution in a python code block.\n"
        "```python\ndef add(a, b):\n    return a + b\n```\n"
        "[Execution Result]\n"
        "All test cases passed!"
    )
    code = extract_last_code(text)
    assert code is not None, "应提取到代码"
    assert "def add" in code, f"应包含第 2 轮代码 def add，实际: {code}"
    assert "def f" not in code, f"不应包含第 1 轮代码 def f，实际: {code}"
    print("  [PASS] 成功标记后正确提取最后一轮代码（不拼接历史轮次）")


def test_extract_last_code_ignores_think_tags():
    """测试：<thinking> 标签内的代码片段不会被提取"""
    text = (
        "```python\ndef f(): pass\n```\n"
        "[Execution Feedback - Turn 1/3]\nRuntimeError\n"
        "<thinking>\nLet me try:\n```python\ndef wrong(): pass\n```\n</thinking>\n"
        "```python\ndef add(a, b):\n    return a + b\n```\n"
        "[Execution Result]\n"
        "All test cases passed!"
    )
    code = extract_last_code(text)
    assert code is not None, "应提取到代码"
    assert "def add" in code, f"应包含正确代码 def add，实际: {code}"
    assert "def wrong" not in code, f"不应包含思考过程中的代码，实际: {code}"
    assert "def f" not in code, f"不应包含第 1 轮代码，实际: {code}"
    print("  [PASS] 正确忽略 <thinking> 标签中的干扰代码")


# ========== 旧版 fallback compute_score_multiturn 测试 ==========

def test_perfect_turn1():
    """第 1 轮就成功 → reward = 1.0"""
    solution = "```python\ndef add(a, b):\n    return a + b\n```"
    extra_info = {"test_cases": "assert add(1, 2) == 3"}
    score = compute_score_multiturn("mbpp", solution, "", extra_info)
    assert score == 1.0, f"Expected 1.0, got {score}"
    print("  [PASS] 第 1 轮成功 → reward = 1.0")


def test_perfect_turn2():
    """第 2 轮成功 → reward = 0.85"""
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
    """第 3 轮成功 → reward = 0.70"""
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
    """单轮但输出多个代码块 → 不应被多扣轮次惩罚"""
    solution = (
        "```python\ndef helper(x):\n    return x * 2\n```\n"
        "Here's the main function:\n"
        "```python\ndef add(a, b):\n    return a + b\n```\n"
        "```python\nprint(add(1,2))\n```"
    )
    extra_info = {"test_cases": "assert add(1, 2) == 3"}
    score = compute_score_multiturn("mbpp", solution, "", extra_info)
    assert score == 1.0, f"Expected 1.0 (1 turn, no penalty), got {score}"
    print("  [PASS] 单轮多代码块 → reward = 1.0（无轮次惩罚）")


def test_no_code():
    """无代码 → gate -1.0"""
    solution = "I don't know the answer."
    extra_info = {"test_cases": "assert True"}
    score = compute_score_multiturn("mbpp", solution, "", extra_info)
    assert score == -1.0, f"Expected -1.0, got {score}"
    print("  [PASS] 无代码 → reward = -1.0（门控触发）")


def test_final_failure():
    """3 轮全部失败 → 0.0"""
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
    """奖励下限 0.1 保底"""
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


def test_turn2_success_with_marker():
    """第 2 轮成功（带 [Execution Result] 标记）→ reward = 0.85"""
    solution = (
        "```python\ndef add(a, b):\n    return a - b\n```\n"
        "[Execution Feedback - Turn 1/3]\nRuntimeError\n"
        "```python\ndef add(a, b):\n    return a + b\n```\n"
        "[Execution Result]\n"
        "All test cases passed!"
    )
    extra_info = {"test_cases": "assert add(1, 2) == 3"}
    score = compute_score_multiturn("mbpp", solution, "", extra_info)
    assert score == 0.85, f"Expected 0.85, got {score}"
    print("  [PASS] 第 2 轮成功（带成功标记）→ reward = 0.85")


def test_turn1_success_with_marker():
    """第 1 轮成功（带 [Execution Result] 标记）→ reward = 1.0"""
    solution = (
        "```python\ndef add(a, b):\n    return a + b\n```\n"
        "[Execution Result]\n"
        "All test cases passed!"
    )
    extra_info = {"test_cases": "assert add(1, 2) == 3"}
    score = compute_score_multiturn("mbpp", solution, "", extra_info)
    assert score == 1.0, f"Expected 1.0, got {score}"
    print("  [PASS] 第 1 轮成功（带成功标记）→ reward = 1.0")


def test_turn2_success_with_think_tags():
    """第 2 轮成功（含思考过程）→ reward = 0.85"""
    solution = (
        "```python\ndef add(a, b):\n    return a - b\n```\n"
        "[Execution Feedback - Turn 1/3]\nRuntimeError\n"
        "<thinking>I need to fix the operator\n```python\ndef tmp(): pass\n```\n</thinking>\n"
        "```python\ndef add(a, b):\n    return a + b\n```\n"
        "[Execution Result]\n"
        "All test cases passed!"
    )
    extra_info = {"test_cases": "assert add(1, 2) == 3"}
    score = compute_score_multiturn("mbpp", solution, "", extra_info)
    assert score == 0.85, f"Expected 0.85, got {score}"
    print("  [PASS] 第 2 轮成功（含思考过程）→ reward = 0.85")


# ========== v2 结构化数据路径测试 ==========

def test_structured_turn1_success():
    """v2: extra_info 有 last_turn_text → 第 1 轮成功 → reward = 1.0"""
    solution = "```python\ndef add(a, b):\n    return a + b\n```"  # 不使用
    extra_info = {
        "test_cases": "assert add(1, 2) == 3",
        "last_turn_text": "```python\ndef add(a, b):\n    return a + b\n```",
        "num_turns": 1,
        "turn_details": [
            {"turn": 1, "exec_status": "SUCCESS", "code_extracted": True, "generated_text_len": 50},
        ],
    }
    score = compute_score_multiturn("mbpp", solution, "", extra_info)
    assert score == 1.0, f"Expected 1.0, got {score}"
    print("  [PASS] v2: 第 1 轮成功（结构化）→ reward = 1.0")


def test_structured_turn2_success():
    """v2: 第 2 轮成功 → reward = 0.85（轮次惩罚正确）"""
    solution = "ignored"  # 不使用
    extra_info = {
        "test_cases": "assert add(1, 2) == 3",
        "last_turn_text": "```python\ndef add(a, b):\n    return a + b\n```",
        "num_turns": 2,
        "turn_details": [
            {"turn": 1, "exec_status": "RUNTIME_ERROR", "code_extracted": True, "generated_text_len": 50},
            {"turn": 2, "exec_status": "SUCCESS", "code_extracted": True, "generated_text_len": 50},
        ],
    }
    score = compute_score_multiturn("mbpp", solution, "", extra_info)
    assert score == 0.85, f"Expected 0.85, got {score}"
    print("  [PASS] v2: 第 2 轮成功（结构化）→ reward = 0.85")


def test_structured_turn3_success():
    """v2: 第 3 轮成功 → reward = 0.70"""
    extra_info = {
        "test_cases": "assert add(1, 2) == 3",
        "last_turn_text": "```python\ndef add(a, b):\n    return a + b\n```",
        "num_turns": 3,
        "turn_details": [
            {"turn": 1, "exec_status": "RUNTIME_ERROR", "code_extracted": True, "generated_text_len": 50},
            {"turn": 2, "exec_status": "RUNTIME_ERROR", "code_extracted": True, "generated_text_len": 50},
            {"turn": 3, "exec_status": "RUNTIME_ERROR", "code_extracted": True, "generated_text_len": 50},
        ],
    }
    score = compute_score_multiturn("mbpp", "ignored", "", extra_info)
    assert score == 0.70, f"Expected 0.70, got {score}"
    print("  [PASS] v2: 第 3 轮成功（结构化）→ reward = 0.70")


def test_structured_no_code():
    """v2: 最后一轮没有代码 → reward = -1.0"""
    extra_info = {
        "test_cases": "assert True",
        "last_turn_text": "I don't know the answer.",
        "num_turns": 3,
        "turn_details": [
            {"turn": 1, "exec_status": "RUNTIME_ERROR", "code_extracted": True, "generated_text_len": 50},
            {"turn": 2, "exec_status": "NO_CODE", "code_extracted": False, "generated_text_len": 30},
            {"turn": 3, "exec_status": "NO_CODE", "code_extracted": False, "generated_text_len": 30},
        ],
    }
    score = compute_score_multiturn("mbpp", "ignored", "", extra_info)
    assert score == -1.0, f"Expected -1.0, got {score}"
    print("  [PASS] v2: 最后一轮无代码（结构化）→ reward = -1.0")


def test_structured_think_tags_cleaned():
    """v2: last_turn_text 含 <thinkng> 标签 → 正确清洗并提取代码"""
    extra_info = {
        "test_cases": "assert add(1, 2) == 3",
        "last_turn_text": (
            "<thinking>Let me try ```python\ndef wrong(): pass\n```\n</thinking>\n"
            "```python\ndef add(a, b):\n    return a + b\n```"
        ),
        "num_turns": 2,
        "turn_details": [
            {"turn": 1, "exec_status": "RUNTIME_ERROR", "code_extracted": True, "generated_text_len": 50},
            {"turn": 2, "exec_status": "RUNTIME_ERROR", "code_extracted": True, "generated_text_len": 100},
        ],
    }
    score = compute_score_multiturn("mbpp", "ignored", "", extra_info)
    assert score == 0.85, f"Expected 0.85, got {score}"
    print("  [PASS] v2: think 标签清洗（结构化）→ reward = 0.85")


def test_structured_echo_immune():
    """v2: 模型 echo [Execution Feedback] 在 last_turn_text 中 → 不影响提取"""
    extra_info = {
        "test_cases": "assert add(1, 2) == 3",
        "last_turn_text": (
            "Let me fix it based on [Execution Feedback - Turn 2/3]\n"
            "The error was about...\n"
            "```python\ndef add(a, b):\n    return a + b\n```"
        ),
        "num_turns": 3,
        "turn_details": [
            {"turn": 1, "exec_status": "RUNTIME_ERROR", "code_extracted": True, "generated_text_len": 50},
            {"turn": 2, "exec_status": "RUNTIME_ERROR", "code_extracted": True, "generated_text_len": 50},
            {"turn": 3, "exec_status": "RUNTIME_ERROR", "code_extracted": True, "generated_text_len": 100},
        ],
    }
    score = compute_score_multiturn("mbpp", "ignored", "", extra_info)
    assert score == 0.70, f"Expected 0.70 (3 turns), got {score}"
    print("  [PASS] v2: echo 免疫（结构化）→ reward = 0.70")


def test_structured_final_failure():
    """v2: 3 轮全失败 → 0.0"""
    extra_info = {
        "test_cases": "assert add(1, 2) == 3",
        "last_turn_text": "```python\ndef add(a): return a\n```",
        "num_turns": 3,
        "turn_details": [
            {"turn": 1, "exec_status": "RUNTIME_ERROR", "code_extracted": True, "generated_text_len": 50},
            {"turn": 2, "exec_status": "RUNTIME_ERROR", "code_extracted": True, "generated_text_len": 50},
            {"turn": 3, "exec_status": "RUNTIME_ERROR", "code_extracted": True, "generated_text_len": 50},
        ],
    }
    score = compute_score_multiturn("mbpp", "ignored", "", extra_info)
    assert score == 0.0, f"Expected 0.0, got {score}"
    print("  [PASS] v2: 3 轮全失败（结构化）→ reward = 0.0")


def test_extract_code_from_turn_basic():
    """extract_code_from_turn 单元测试"""
    # 正常代码块
    text = "Some text\n```python\ndef f(): return 1\n```\nMore text"
    code = extract_code_from_turn(text)
    assert code is not None and "def f" in code

    # 含 think 标签
    text = "<thinking>```python\ndef wrong(): pass\n```\n</thinking>\n```python\ndef right(): pass\n```"
    code = extract_code_from_turn(text)
    assert code is not None and "def right" in code and "def wrong" not in code

    # 无代码
    code = extract_code_from_turn("Just text, no code")
    assert code is None

    print("  [PASS] extract_code_from_turn 基础功能")


if __name__ == "__main__":
    print("=" * 50)
    print("多轮奖励函数测试")
    print("=" * 50)

    print("\n--- 旧版 fallback 测试 ---")
    test_count_actual_turns()
    test_extract_last_code()
    test_extract_last_code_with_success_marker()
    test_extract_last_code_ignores_think_tags()
    test_perfect_turn1()
    test_perfect_turn2()
    test_perfect_turn3()
    test_multicode_no_overpenalty()
    test_no_code()
    test_final_failure()
    test_reward_floor()
    test_turn2_success_with_marker()
    test_turn1_success_with_marker()
    test_turn2_success_with_think_tags()

    print("\n--- v2 结构化数据路径测试 ---")
    test_structured_turn1_success()
    test_structured_turn2_success()
    test_structured_turn3_success()
    test_structured_no_code()
    test_structured_think_tags_cleaned()
    test_structured_echo_immune()
    test_structured_final_failure()
    test_extract_code_from_turn_basic()

    print("\n" + "=" * 50)
    print("全部测试通过！")
