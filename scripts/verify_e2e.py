"""
端到端验证脚本（阶段零点五）

手动跑一遍 prompt → model.generate() → extract_code → sandbox → reward，
确认全流程正确后再上 veRL 训练。

用法：
  python scripts/verify_e2e.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.reward import extract_code, compute_score
from src.sandbox import execute_code, ExecStatus


SYSTEM_PROMPT = (
    "Solve the programming problem below. "
    "First think step by step, then provide your solution "
    "in a ```python code block. "
    "Your code will be tested with assert statements."
)


def test_model_generate():
    """测试 1：模型能生成包含 ```python 代码块的输出"""
    print("测试 1：模型生成 + 代码提取")
    model_name = "Qwen/Qwen3-1.7B"

    print(f"  加载模型 {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    model.eval()

    prompt = "Write a function to add two numbers."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    import torch
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.0, do_sample=False)

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  模型输出前 200 字符: {generated[:200]}")

    code = extract_code(generated)
    assert code is not None, "代码提取失败！模型输出中没有 ```python 代码块"
    print(f"  [PASS] 代码提取成功，长度: {len(code)} 字符")
    return model, tokenizer


def test_sandbox_with_model_code(model, tokenizer):
    """测试 2：模型生成的代码能在沙盒中执行"""
    print("\n测试 2：沙盒执行模型生成的代码")

    prompt = (
        "def add(a, b):\n"
        "    '''Return the sum of a and b.'''\n"
        "Write the complete function."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    import torch
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.0, do_sample=False)

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    code = extract_code(generated)
    if code:
        result = asyncio.run(execute_code(code, "assert add(1, 2) == 3", timeout=5.0))
        print(f"  沙盒执行状态: {result.status}")
        if result.status == ExecStatus.SUCCESS:
            print(f"  [PASS] 沙盒执行成功")
        else:
            print(f"  [INFO] 沙盒执行失败（正常，基座模型可能写不对），stderr: {result.stderr[:100]}")
    else:
        print("  [SKIP] 未提取到代码，跳过沙盒测试")


def test_reward_pipeline():
    """测试 3：完整的 reward 计算管道"""
    print("\n测试 3：完整 reward 计算管道")

    # 正确代码
    solution_correct = "```python\ndef add(a, b):\n    return a + b\n```"
    reward = compute_score("mbpp", solution_correct, "", {"test_cases": "assert add(1, 2) == 3"})
    assert reward == 1.0, f"正确代码 reward 应为 1.0，实际: {reward}"
    print(f"  [PASS] 正确代码: reward={reward}")

    # 错误代码
    solution_wrong = "```python\ndef add(a, b):\n    return a - b\n```"
    reward = compute_score("mbpp", solution_wrong, "", {"test_cases": "assert add(1, 2) == 3"})
    assert reward == 0.0, f"错误代码 reward 应为 0.0，实际: {reward}"
    print(f"  [PASS] 错误代码: reward={reward}")

    # 无代码
    solution_no_code = "The answer is 42."
    reward = compute_score("mbpp", solution_no_code, "", {"test_cases": "assert True"})
    assert reward == -1.0, f"无代码 reward 应为 -1.0，实际: {reward}"
    print(f"  [PASS] 无代码输出: reward={reward}")


if __name__ == "__main__":
    print("=" * 60)
    print("阶段零点五：端到端验证")
    print("=" * 60)

    try:
        model, tokenizer = test_model_generate()
        test_sandbox_with_model_code(model, tokenizer)
        test_reward_pipeline()

        print("\n" + "=" * 60)
        print("全部端到端验证通过！可以安全进入阶段一。")
        print("=" * 60)
    except Exception as e:
        print(f"\n验证失败: {e}")
        print("请先修复上述问题，再进入阶段一。")
        sys.exit(1)
