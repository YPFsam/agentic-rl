"""
单轮评估脚本（串行版本）

与多轮评估脚本 (scripts/evaluate_multiturn.py) 保持一致的推理方式：
  - 串行逐条生成（无 batch padding，确保温度=0 下确定性输出）
  - max_new_tokens=4800（与多轮 max_response_length 一致）
  - extract_code() 提取代码，失败时写空字符串（不写 thinking 全文）
  - 每样本 ETA + 全局 extract_ok/fail 统计

支持三个 benchmark：
  1. HumanEval（164 题）
  2. MBPP（500 题）
  3. LiveCodeBench

用法：
  python src/evaluate.py \
    --model_path Qwen/Qwen3-1.7B \
    --tag baseline \
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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.reward import extract_code


# ========== System Prompt（与训练/多轮评估一致）==========

SYSTEM_PROMPT = (
    "Solve the programming problem below. "
    "First think step by step, then provide your solution "
    "in a ```python code block. "
    "Your code will be tested with assert statements. "
    "Provide only ONE complete Python code block — do not include alternative solutions or extra code blocks."
)


# ========== Benchmark 数据加载 ==========

def load_humaneval():
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    return ds, "task_id", "prompt"


def load_mbpp():
    from datasets import load_dataset
    ds = load_dataset("mbpp", "sanitized", split="test")
    return ds, "task_id", "prompt"


def load_livecodebench():
    from datasets import load_dataset
    ds = load_dataset("livecodebench/code_generation", split="test")
    return ds, "task_id", "question_content"


BENCHMARK_LOADERS = {
    "humaneval": load_humaneval,
    "mbpp": load_mbpp,
    "livecodebench": load_livecodebench,
}


# ========== 串行生成函数 ==========

def generate_for_benchmark(
    model,
    tokenizer,
    benchmark: str,
    output_file: str,
    max_new_tokens: int = 4800,
    temperature: float = 0.0,
):
    """
    串行单轮生成（与多轮评估脚本的推理方式完全一致，无 batch padding）。

    extract_code() 失败时写空字符串（不写 thinking 全文），确保 evalplus 不误判。
    """
    loader = BENCHMARK_LOADERS.get(benchmark)
    if loader is None:
        raise ValueError(f"未知 benchmark: {benchmark}，可选: {list(BENCHMARK_LOADERS.keys())}")

    dataset, id_key, prompt_key = loader()

    # 断点续传
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
            print(f"  [断点续传] 跳过 {len(completed)} 个已完成题目")

    total = len(dataset)
    model.eval()
    do_sample = temperature > 0.0

    print(f"  [DEBUG] 总题数: {total}, max_new_tokens: {max_new_tokens}")
    print(f"  [DEBUG] temperature: {temperature}")

    eval_start = time.monotonic()
    done_count = len(completed)
    extract_ok = 0
    extract_fail = 0

    with open(output_file, "a", encoding="utf-8") as f:
        for item in dataset:
            task_id = item[id_key] if id_key in item else f"{benchmark}/{i}"

            if benchmark == "mbpp" and not str(task_id).startswith("Mbpp/"):
                task_id = f"Mbpp/{task_id}"

            if task_id in completed:
                continue

            prompt_text = item[prompt_key]

            # 构造 chat messages（与多轮评估完全一致）
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ]

            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            input_token_count = inputs["input_ids"].shape[1]

            t_gen_start = time.monotonic()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated = tokenizer.decode(
                outputs[0][input_token_count:],
                skip_special_tokens=True,
            )
            gen_time = time.monotonic() - t_gen_start
            gen_token_count = len(outputs[0]) - input_token_count

            # 提取代码（与多轮评估一致）
            completion = extract_code(generated)
            if completion is None:
                # 没有提取到代码，写空字符串（不写 thinking 全文）
                completion = ""
                extract_fail += 1
                status = "NO_CODE"
            else:
                extract_ok += 1
                status = "OK"

            f.write(json.dumps({"task_id": task_id, "completion": completion}, ensure_ascii=False) + "\n")
            f.flush()

            done_count += 1
            elapsed = time.monotonic() - eval_start
            speed = done_count / elapsed if elapsed > 0 else 0
            eta = (total - done_count) / speed if speed > 0 else 0
            print(f"  [{done_count}/{total}] {task_id} {status} | "
                  f"gen={gen_time:.1f}s tok={gen_token_count} | ETA {eta:.0f}s")

    eval_time = time.monotonic() - eval_start
    print(f"\n  生成完成: {output_file} (耗时 {eval_time:.1f}s)")
    print(f"  统计: extract_ok={extract_ok} fail={extract_fail}")
    return output_file


# ========== EvalPlus 判卷 ==========

def run_evalplus(samples_file: str, benchmark: str, k_values: list = None) -> dict:
    """
    调用 evalplus 判卷，返回 pass@k 结果。

    优先从 evalplus 生成的 *_eval_results.json 读取结果（最可靠），
    仅在 JSON 文件不存在时回退到正则解析 stdout。
    """
    if k_values is None:
        k_values = [1, 10]

    print(f"\nEvalPlus 判卷: {benchmark}")
    print(f"  输入文件: {samples_file}")

    dataset_flag = "humaneval" if benchmark == "humaneval" else "mbpp"

    cmd = [sys.executable, "-m", "evalplus.evaluate",
           "--dataset", dataset_flag,
           "--samples", samples_file]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("  evalplus 判卷超时，跳过")
        return {}

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

            for k in k_values:
                key = f"pass@{k}"
                if key in eval_data:
                    results[key] = eval_data[key]
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
    parser = argparse.ArgumentParser(description="单轮评估脚本（串行版本）")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--tag", type=str, required=True, help="实验标签")
    parser.add_argument("--benchmark", type=str, nargs="+", default=["humaneval"],
                        choices=["humaneval", "mbpp", "livecodebench"])
    parser.add_argument("--max_new_tokens", type=int, default=4800,
                        help="最大生成 token 数（与多轮 max_response_length 一致）")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    args = parser.parse_args()

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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
