"""
APPS 数据清洗脚本：用 LLM API 将 input/output 对转为 assert 测试用例。

用法：
  python scripts/synth_testcases.py \\
    --api_key YOUR_KEY \\
    --output data/apps_cleaned.jsonl \\
    --max_concurrent 8
"""
import os
import sys
import json
import re
import asyncio
import argparse
import random
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sandbox import execute_code, ExecStatus


# ========== LLM Prompt 模板 ==========

SYNTH_PROMPT = """\
You are a Python testing expert. Given a programming problem and its input/output test cases,
generate Python assert statements that verify a solution function against these test cases.

Rules:
1. Each assert must be a single line, callable as `assert func(args) == expected`
2. Use the function name from the problem description
3. For complex outputs (lists, sets, dicts), use sorted() or set() comparison if needed
4. Do NOT include any explanation, only the assert statements
5. If output is multi-line or complex, wrap in appropriate Python literal

Problem:
{prompt}

Test cases (input -> output):
{test_cases}

Generate ONLY the assert statements, one per line:"""


# ========== 单条数据处理 ==========

async def synth_asserts_for_task(
    client,
    task: dict,
    semaphore: asyncio.Semaphore,
    max_retries: int = 2,
) -> dict | None:
    """用 LLM 为单个 APPS 任务生成 assert 测试用例。"""
    from openai import AsyncOpenAI

    async with semaphore:
        prompt_text = task["prompt"]
        io_pairs = task.get("test_cases", [])

        if not io_pairs:
            return None

        io_text = ""
        for i, pair in enumerate(io_pairs[:5]):
            inp = pair.get("input", "").strip()
            out = pair.get("output", "").strip()
            io_text += f"Input {i+1}: {inp}\nExpected Output {i+1}: {out}\n\n"

        user_msg = SYNTH_PROMPT.format(prompt=prompt_text[:1000], test_cases=io_text)

        for attempt in range(max_retries + 1):
            try:
                response = await client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": user_msg}],
                    temperature=0.0,
                    max_tokens=1024,
                    timeout=30,
                )
                llm_output = response.choices[0].message.content.strip()
                break
            except Exception as e:
                if attempt == max_retries:
                    print(f"  [SKIP] task {task.get('task_id', '?')} API 失败: {e}")
                    return None
                await asyncio.sleep(2 ** attempt)

        # 提取 assert 语句
        assert_lines = []
        for line in llm_output.split("\n"):
            line = line.strip()
            if line.startswith("assert "):
                line = line.rstrip("`")
                assert_lines.append(line)

        if not assert_lines:
            return None

        test_code = "\n".join(assert_lines)
        task_id = task.get("task_id", f"apps_{id(task)}")

        return {
            "task_id": str(task_id),
            "prompt": prompt_text,
            "test_cases": test_code,
            "data_source": "apps",
        }


# ========== 沙盒验证 ==========

async def validate_with_sandbox(
    sample: dict,
    reference_solution: str | None = None,
    timeout: float = 5.0,
) -> bool:
    """用沙盒验证 LLM 生成的 assert 是否正确。"""
    test_code = sample["test_cases"]

    if reference_solution:
        full_code = f"{reference_solution}\n\n{test_code}"
        result = await execute_code(full_code, timeout=timeout)
        return result.status == ExecStatus.SUCCESS
    else:
        import ast
        try:
            ast.parse(test_code)
            return True
        except SyntaxError:
            return False


# ========== 主流程 ==========

async def main_async(args):
    from openai import AsyncOpenAI
    from datasets import load_dataset

    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )

    print("加载 APPS 数据集...")
    ds = load_dataset("codeparrot/apps", trust_remote_code=True)

    all_tasks = []
    for item in ds["train"]:
        difficulty = item.get("difficulty", "")
        if difficulty not in ("introductory",):
            continue

        test_cases_raw = item.get("test_cases", "[]")
        try:
            io_pairs = json.loads(test_cases_raw) if isinstance(test_cases_raw, str) else test_cases_raw
        except json.JSONDecodeError:
            continue

        if not io_pairs:
            continue

        solutions_raw = item.get("solutions", "[]")
        try:
            solutions = json.loads(solutions_raw) if isinstance(solutions_raw, str) else solutions_raw
            ref_solution = solutions[0] if solutions else None
        except (json.JSONDecodeError, IndexError):
            ref_solution = None

        all_tasks.append({
            "task_id": f"apps_{item.get('problem_id', len(all_tasks))}",
            "prompt": item.get("question", ""),
            "test_cases": io_pairs,
            "reference_solution": ref_solution,
        })

    random.shuffle(all_tasks)
    if args.max_samples > 0:
        all_tasks = all_tasks[:args.max_samples]

    print(f"待处理: {len(all_tasks)} 个 APPS 入门题")

    semaphore = asyncio.Semaphore(args.max_concurrent)
    tasks = [synth_asserts_for_task(client, t, semaphore) for t in all_tasks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_samples = []
    skip_count = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception) or result is None:
            skip_count += 1
            continue

        ref_sol = all_tasks[i].get("reference_solution")
        is_valid = await validate_with_sandbox(result, ref_sol)
        if is_valid:
            valid_samples.append(result)
        else:
            skip_count += 1

    print(f"\n生成完成: {len(valid_samples)} 条有效 / {skip_count} 条跳过")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for s in valid_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"已保存: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="APPS 数据清洗：LLM 生成 Assert + 沙盒验证")
    parser.add_argument("--api_key", type=str, required=True, help="DeepSeek API Key")
    parser.add_argument("--base_url", type=str, default="https://api.deepseek.com", help="API base URL")
    parser.add_argument("--output", type=str, default="data/apps_cleaned.jsonl", help="输出文件路径")
    parser.add_argument("--max_concurrent", type=int, default=8, help="最大并发请求数")
    parser.add_argument("--max_samples", type=int, default=3000, help="最大处理样本数")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
