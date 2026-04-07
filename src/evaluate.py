"""
统一评估脚本

支持三个 benchmark：
  1. HumanEval+（164 题，evalplus 增强测试）
  2. MBPP+（500 题，evalplus 增强测试）
  3. LiveCodeBench（定期更新的竞赛题）

用法：
  python src/evaluate.py \\
    --model_path Qwen/Qwen3-1.7B \\
    --tag baseline \\
    --benchmark humaneval
"""
import os
import re
import gc
import json
import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 项目内部导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.reward import extract_code


# ========== System Prompt（与训练一致） ==========

SYSTEM_PROMPT = (
    "Solve the programming problem below. "
    "First think step by step, then provide your solution "
    "in a ```python code block. "
    "Your code will be tested with assert statements."
)


# ========== Benchmark 数据加载 ==========

def load_humaneval():
    """加载 HumanEval 数据集。"""
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    return ds, "task_id", "prompt"


def load_mbpp():
    """加载 MBPP test 数据集。"""
    from datasets import load_dataset
    ds = load_dataset("mbpp", "sanitized", split="test")
    return ds, "task_id", "prompt"


def load_livecodebench():
    """加载 LiveCodeBench 数据集。"""
    from datasets import load_dataset
    ds = load_dataset("livecodebench/code_generation", split="test")
    return ds, "task_id", "question_content"


BENCHMARK_LOADERS = {
    "humaneval": load_humaneval,
    "mbpp": load_mbpp,
    "livecodebench": load_livecodebench,
}


# ========== 生成函数 ==========

def generate_for_benchmark(
    model,
    tokenizer,
    benchmark: str,
    output_file: str,
    max_new_tokens: int = 2048,
    num_samples: int = 1,
    temperature: float = 0.0,
):
    """
    为指定 benchmark 生成答案。

    核心流程：
    1. 加载 benchmark 数据
    2. 对每道题：构造 prompt → 模型生成 → extract_code() → 写入 jsonl
    3. 支持断点续传

    重要：必须用 extract_code() 提取纯净代码，不能直接写入原始输出。
    evalplus 要求 completion 字段只包含可执行的 Python 代码。
    """
    loader = BENCHMARK_LOADERS.get(benchmark)
    if loader is None:
        raise ValueError(f"未知 benchmark: {benchmark}，可选: {list(BENCHMARK_LOADERS.keys())}")

    dataset, id_key, prompt_key = loader()

    # 断点续传：读取已完成的 task_id
    completed = set()
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        completed.add(json.loads(line)["task_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        if completed:
            print(f"  断点续传：跳过 {len(completed)} 个已完成题目")

    total = len(dataset)
    model.eval()

    do_sample = temperature > 0.0

    with open(output_file, "a", encoding="utf-8") as f:
        for i, item in enumerate(dataset):
            task_id = item[id_key] if id_key in item else f"{benchmark}/{i}"

            # 转换为 EvalPlus 兼容的 task_id 格式
            # EvalPlus 对 MBPP 严格要求 "Mbpp/X"，HumanEval 已是 "HumanEval/X"
            if benchmark == "mbpp" and not str(task_id).startswith("Mbpp/"):
                task_id = f"Mbpp/{task_id}"

            if task_id in completed:
                continue

            prompt_text = item[prompt_key]

            # 构造 chat messages
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ]

            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            for sample_idx in range(num_samples):
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                # 解码模型输出（只取生成的部分）
                generated = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )

                # 关键：用 extract_code() 提取纯净代码
                completion = extract_code(generated)
                if completion is None:
                    # 没有提取到代码，使用原始输出
                    completion = generated.strip()

                result = {"task_id": task_id, "completion": completion}
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            f.flush()

            completed.add(task_id)
            done = len(completed)
            if done % 20 == 0 or done == total:
                print(f"  进度: {done}/{total}")

    print(f"  生成完成: {output_file}")
    return output_file


# ========== EvalPlus 判卷 ==========

def run_evalplus(samples_file: str, benchmark: str, k_values: list = None) -> dict:
    """
    调用 evalplus 判卷，返回 pass@k 结果。

    优先从 evalplus 生成的 *_eval_results.json 读取结果（最可靠），
    仅在 JSON 文件不存在时回退到正则解析 stdout。

    Args:
        samples_file: 生成的 jsonl 文件路径
        benchmark: "humaneval" 或 "mbpp"
        k_values: 要计算的 k 值列表

    Returns:
        dict: {"pass@1": float, "pass@10": float, ...}
    """
    if k_values is None:
        k_values = [1, 10]

    print(f"\nEvalPlus 判卷: {benchmark}")
    print(f"  输入文件: {samples_file}")

    dataset_flag = "humaneval" if benchmark == "humaneval" else "mbpp"

    cmd = [sys.executable, "-m", "evalplus.evaluate",
           "--dataset", dataset_flag,
           "--samples", samples_file]

    result = subprocess.run(
        cmd,
        capture_output=True, text=True, timeout=600,
    )

    print(result.stdout)
    if result.stderr and "warning" not in result.stderr.lower():
        print(result.stderr)

    # 优先读取 evalplus 生成的 JSON 结果文件
    results = {}
    json_result_file = samples_file + "_eval_results.json"
    if os.path.exists(json_result_file):
        try:
            with open(json_result_file, "r", encoding="utf-8") as f:
                eval_data = json.load(f)

            # evalplus JSON 结构：{"pass@1": 0.xxx, "pass@10": 0.xxx, ...}
            # 也可能是嵌套结构 {"humaneval": {"pass@1": ...}} 或包含 "base" "plus" 子键
            for k in k_values:
                key = f"pass@{k}"
                # 直接在顶层找
                if key in eval_data:
                    results[key] = eval_data[key]
                # 在 "base" 或 "plus" 子键中找（取 plus 版本优先）
                elif isinstance(eval_data, dict):
                    for sub_key in ("plus", "base"):
                        sub = eval_data.get(sub_key, {})
                        if isinstance(sub, dict) and key in sub:
                            results[key] = sub[key]
                            break

            if results:
                print(f"  从 JSON 结果文件读取: {results}")
                return results
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  警告：读取 JSON 结果文件失败 ({e})，回退到正则解析")

    # 回退：从 stdout 正则解析
    for line in result.stdout.split("\n"):
        for k in k_values:
            pattern = f"pass@{k}"
            if pattern in line.lower():
                nums = re.findall(r"(\d+\.\d+)%?", line)
                if nums:
                    results[pattern] = float(nums[0])

    return results


# ========== LiveCodeBench 判卷 ==========

def run_livecodebench_eval(samples_file: str) -> dict:
    """
    LiveCodeBench 判卷（简化版）。
    完整评估建议使用 LiveCodeBench 官方工具。
    """
    import ast

    total = 0
    syntax_pass = 0

    with open(samples_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            code = item.get("completion", "")
            total += 1
            try:
                ast.parse(code)
                syntax_pass += 1
            except SyntaxError:
                pass

    rate = syntax_pass / total if total > 0 else 0
    print(f"\nLiveCodeBench 语法通过率: {rate:.2%} ({syntax_pass}/{total})")
    print("  注意：完整评估请使用 LiveCodeBench 官方工具")

    return {"syntax_pass_rate": rate, "total": total}


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(description="统一评估脚本")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--tag", type=str, required=True, help="实验标签")
    parser.add_argument("--benchmark", type=str, nargs="+", default=["humaneval"],
                        choices=["humaneval", "mbpp", "livecodebench"])
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    args = parser.parse_args()

    if args.num_samples > 1 and args.temperature == 0.0:
        print("警告：num_samples > 1 但 temperature=0.0，建议设置 --temperature 0.6")

    # ===== 加载模型 =====
    print(f"加载模型: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"模型加载完成，设备: {model.device}")

    # ===== 逐 benchmark 评估 =====
    all_results = {}

    for benchmark in args.benchmark:
        print(f"\n{'=' * 60}")
        print(f"评估 {benchmark} (tag={args.tag})")
        print(f"{'=' * 60}")

        output_file = os.path.join(args.output_dir, f"{benchmark}_{args.tag}.jsonl")
        generate_for_benchmark(
            model, tokenizer, benchmark, output_file,
            max_new_tokens=args.max_new_tokens,
            num_samples=args.num_samples,
            temperature=args.temperature,
        )

        if benchmark in ("humaneval", "mbpp"):
            results = run_evalplus(output_file, benchmark)
        else:
            results = run_livecodebench_eval(output_file)

        all_results[benchmark] = results

    # ===== 清理 =====
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # ===== 汇总输出 =====
    print(f"\n{'=' * 60}")
    print(f"评估汇总 (tag={args.tag})")
    print(f"{'=' * 60}")
    for benchmark, results in all_results.items():
        print(f"  {benchmark}: {results}")

    summary_path = os.path.join(args.output_dir, f"summary_{args.tag}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
