#!/usr/bin/env python3
"""
多轮推理评估脚本（串行版本）

Token 预算与训练 agent_loop 完全一致：
  - response_length = 4800（多轮共享总预算）
  - remaining = response_length - budget_used
  - 每轮 max_new_tokens = remaining（与 agent_loop 一致）
  - 反馈文本也计入预算

用法：
  python scripts/evaluate_multiturn.py \
    --model_path Qwen/Qwen3-1.7B \
    --tag baseline \
    --max_turns 3

日志：
  logs/multiturn_eval_{tag}_{timestamp}.jsonl  — 每样本逐轮详情
"""
import os
import re
import gc
import json
import argparse
import subprocess
import sys
import time
import asyncio
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.sandbox import execute_code, ExecResult, ExecStatus
from src.reward import extract_code


# ========== 常量（与训练/agent_loop 一致）==========

SYSTEM_PROMPT = (
    "Solve the programming problem below. "
    "First think step by step, then provide your solution "
    "in a ```python code block. "
    "Your code will be tested with assert statements. "
    "Provide only ONE complete Python code block — do not include alternative solutions or extra code blocks."
)

ERROR_FEEDBACK_TEMPLATE = (
    "\n\n[Execution Feedback - Turn {turn}/{max_turns}]\n"
    "Your code produced the following error:\n"
    "```\n{error}\n```\n"
    "Please analyze the error and fix your code.\n"
    "Output your revised solution in a python code block.\n"
)

_ASSERT_LINE_PATTERN = re.compile(r'^(\s+)assert\s+.+$', re.MULTILINE)
_ASSERT_LINE_REPLACEMENT = r'\1assert <test_case>'


def format_error(result: ExecResult, turn: int, max_turns: int, timeout: float = 5.0,
                 max_error_length: int = 500) -> str:
    """格式化错误信息，与 agent_loop._format_error 一致。"""
    if result.status == ExecStatus.TIMEOUT:
        return f"TimeoutError: Code execution exceeded {timeout}s limit (possible infinite loop)."
    stderr = result.stderr
    stderr = _ASSERT_LINE_PATTERN.sub(_ASSERT_LINE_REPLACEMENT, stderr)
    if len(stderr) > max_error_length:
        stderr = stderr[-max_error_length:]
    if result.status == ExecStatus.SYNTAX_ERROR:
        return f"SyntaxError:\n{stderr}"
    else:
        return f"RuntimeError:\n{stderr}"


# ========== 日志 ==========

class EvalLogger:
    """每样本一条 JSONL，记录逐轮详情。"""

    def __init__(self, log_path: str):
        self._path = log_path
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def log_sample(self, task_id: str, passed_at_turn: int,
                   turn_details: list[dict], final_code: str | None,
                   prompt: str, test_cases: str, entry_point: str):
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "task_id": task_id,
            "passed_at_turn": passed_at_turn,
            "total_turns": len(turn_details),
            "final_pass": passed_at_turn > 0,
            "final_code_length": len(final_code) if final_code else 0,
            "entry_point": entry_point,
            "test_cases": test_cases,
            "prompt": prompt[:500],  # 截断防止日志过大
            "turns": turn_details,
        }
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            warnings.warn(f"[multiturn_eval] 日志写入失败: {type(e).__name__}: {e}")


# ========== 数据加载 ==========

def load_humaneval():
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    return ds


# ========== 串行多轮评估 ==========

def multiturn_evaluate(
    model,
    tokenizer,
    output_file: str,
    log_path: str,
    max_turns: int = 3,
    max_response_length: int = 4800,
    temperature: float = 0.0,
    timeout: float = 5.0,
):
    """
    串行多轮推理评估。

    Token 预算与训练 agent_loop 完全一致：
      - response_length = max_response_length（4800）
      - remaining = response_length - budget_used
      - 每轮 max_new_tokens = remaining（agent_loop: min(remaining, response_length)）
      - 反馈文本 token 也计入 budget_used
    """
    dataset = load_humaneval()
    logger = EvalLogger(log_path)
    total = len(dataset)
    model.eval()
    do_sample = temperature > 0.0

    # 断点续传
    completed_ids = set()
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        completed_ids.add(json.loads(line)["task_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        if completed_ids:
            print(f"  [断点续传] 跳过 {len(completed_ids)} 个已完成题目")

    # 统计（动态初始化，适配任意 max_turns）
    stats = {
        "total": total,
        **{f"turn{t}_pass": 0 for t in range(1, max_turns + 1)},
        "no_code": 0,
        "final_pass": 0,
    }

    print(f"  [DEBUG] 总题数: {total}, max_turns: {max_turns}, max_response_length: {max_response_length}")
    print(f"  [DEBUG] temperature: {temperature}, timeout: {timeout}s")

    eval_start = time.monotonic()
    done_count = len(completed_ids)

    with open(output_file, "a", encoding="utf-8") as f:
        for item in dataset:
            task_id = item["task_id"]

            if task_id in completed_ids:
                continue

            # 构造测试用例
            entry_point = item.get("entry_point", "")
            test_raw = item.get("test", "")
            test_cases = (test_raw + f"\ncheck({entry_point})\n") if test_raw and entry_point else ""

            # 对话历史
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["prompt"]},
            ]

            # ===== 以下与 agent_loop.run() 逐行对齐 =====
            response_length = max_response_length  # 对应 agent_loop 的 self.rollout_config.response_length
            budget_used = 0                         # 对应 agent_loop 的 len(response_ids)
            final_completion = None
            passed_at_turn = 0
            turn_details = []
            code = None       # 初始化，防止循环未执行时 NameError
            generated = ""    # 初始化，同上

            for turn in range(max_turns):
                # 预算检查（agent_loop: remaining = response_length - len(response_ids)）
                remaining = response_length - budget_used
                if remaining <= 0:
                    break

                # 动态 max_new_tokens（agent_loop: min(remaining, original_max)，original_max 默认 = response_length）
                max_new_tokens = remaining

                # 生成
                t_gen_start = time.monotonic()
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
                input_token_count = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                generated = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                gen_time = time.monotonic() - t_gen_start

                # 计入预算（agent_loop: response_ids.extend(generated_ids)）
                gen_token_count = len(outputs[0]) - inputs["input_ids"].shape[1]
                budget_used += gen_token_count

                # 预算检查（agent_loop: if len(response_ids) > response_length: truncate; break）
                if budget_used > response_length:
                    budget_used = response_length
                    break

                # 提取代码
                code = extract_code(generated)

                if code is None:
                    turn_detail = {
                        "turn": turn + 1,
                        "gen_time_s": round(gen_time, 2),
                        "gen_token_count": gen_token_count,
                        "input_token_count": input_token_count,
                        "max_new_tokens": max_new_tokens,
                        "generated_length": len(generated),
                        "generated_text": generated,  # debug: 模型原始输出
                        "code_extracted": False,
                        "extracted_code": None,
                        "exec_status": "NO_CODE",
                        "exec_stderr": None,
                        "exec_time_s": 0,
                        "budget_used": budget_used,
                        "budget_remaining": response_length - budget_used,
                    }
                    if turn < max_turns - 1:
                        # 与 agent_loop L171-175 对齐：有 \n\n 前缀
                        feedback_text = (
                            "\n\n[Format Error]\n"
                            "Error: No Python code block found in your response. "
                            "Please use a python code block.\n"
                        )
                        turn_detail["feedback_sent"] = feedback_text
                        messages.append({"role": "assistant", "content": generated})
                        messages.append({"role": "user", "content": feedback_text})
                        fb_tokens = len(tokenizer.encode(feedback_text, add_special_tokens=False))
                        budget_used += fb_tokens
                    turn_details.append(turn_detail)
                    continue

                # 沙盒执行
                t_exec_start = time.monotonic()
                result: ExecResult = asyncio.run(execute_code(code, test_cases, timeout=timeout))
                exec_time = time.monotonic() - t_exec_start

                turn_detail = {
                    "turn": turn + 1,
                    "gen_time_s": round(gen_time, 2),
                    "gen_token_count": gen_token_count,
                    "input_token_count": input_token_count,
                    "max_new_tokens": max_new_tokens,
                    "generated_length": len(generated),
                    "generated_text": generated,  # debug: 模型原始输出
                    "code_extracted": True,
                    "extracted_code": code,       # debug: 提取的代码
                    "exec_status": result.status.name,
                    "exec_stderr": result.stderr,  # debug: 完整 stderr
                    "exec_time_s": round(exec_time, 3),
                    "budget_used": budget_used,
                    "budget_remaining": response_length - budget_used,
                }

                if result.status == ExecStatus.SUCCESS:
                    final_completion = code
                    passed_at_turn = turn + 1
                    stats[f"turn{turn + 1}_pass"] += 1
                    stats["final_pass"] += 1
                    turn_details.append(turn_detail)
                    break
                else:
                    turn_details.append(turn_detail)
                    if turn < max_turns - 1:
                        error_msg = format_error(result, turn, max_turns, timeout)
                        feedback = ERROR_FEEDBACK_TEMPLATE.format(
                            turn=turn + 1, max_turns=max_turns, error=error_msg,
                        )
                        turn_detail["feedback_sanitized"] = error_msg
                        turn_detail["feedback_sent"] = feedback
                        messages.append({"role": "assistant", "content": generated})
                        messages.append({"role": "user", "content": feedback})
                        fb_tokens = len(tokenizer.encode(feedback, add_special_tokens=False))
                        budget_used += fb_tokens

            # 记录结果
            if final_completion is None:
                if code is None:
                    stats["no_code"] += 1
                    final_completion = generated.strip() if generated else ""
                else:
                    final_completion = code

            f.write(json.dumps({"task_id": task_id, "completion": final_completion}, ensure_ascii=False) + "\n")
            f.flush()
            logger.log_sample(task_id, passed_at_turn, turn_details, final_completion,
                              prompt=item["prompt"], test_cases=test_cases, entry_point=entry_point)

            done_count += 1
            # 每样本控制台 debug 输出
            turn_summary = " → ".join(
                f"T{t['turn']}:{t['exec_status']}" for t in turn_details
            ) if turn_details else "NO_TURNS"
            result_mark = "PASS" if passed_at_turn > 0 else "FAIL"
            last = turn_details[-1] if turn_details else {}
            print(f"  [{done_count}/{total}] {task_id} {result_mark} at T{passed_at_turn} | "
                  f"{turn_summary} | budget={last.get('budget_used','?')}/{response_length} "
                  f"gen={last.get('gen_time_s','?')}s")

    eval_time = time.monotonic() - eval_start
    print(f"\n  生成完成: {output_file} (耗时 {eval_time:.1f}s)")
    print(f"  详情日志: {log_path}")
    return stats


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(description="多轮推理评估（串行版本）")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--max_turns", type=int, default=3, help="最大推理轮数（与训练一致）")
    parser.add_argument("--max_response_length", type=int, default=4800,
                        help="多轮共享 token 预算（与训练 response_length 一致）")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    args = parser.parse_args()

    print(f"加载模型: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    print(f"模型加载完成，设备: {model.device}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n{'=' * 60}")
    print(f"多轮推理评估 (tag={args.tag}, max_turns={args.max_turns})")
    print(f"{'=' * 60}")

    output_file = os.path.join(args.output_dir, f"humaneval_{args.tag}_multiturn.jsonl")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/multiturn_eval_{args.tag}_{timestamp}.jsonl"

    stats = multiturn_evaluate(
        model, tokenizer, output_file,
        log_path=log_path,
        max_turns=args.max_turns,
        max_response_length=args.max_response_length,
        temperature=args.temperature,
    )

    # evalplus 判卷
    print(f"\nEvalPlus 判卷...")
    cmd = [sys.executable, "-m", "evalplus.evaluate",
           "--dataset", "humaneval", "--samples", output_file]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(result.stdout)

    # 汇总
    total = stats["total"]
    print(f"\n{'=' * 60}")
    print(f"多轮推理评估汇总 (tag={args.tag})")
    print(f"{'=' * 60}")
    print(f"  总题数:       {total}")
    for t in range(1, args.max_turns + 1):
        key = f"turn{t}_pass"
        if stats.get(key, 0) > 0:
            print(f"  T{t} 通过:    {stats[key]} ({stats[key]/total*100:.1f}%)")
    print(f"  总通过:       {stats['final_pass']} ({stats['final_pass']/total*100:.1f}%)")
    print(f"  无代码:       {stats['no_code']}")

    summary = {
        "tag": args.tag,
        "model_path": args.model_path,
        "max_turns": args.max_turns,
        "max_response_length": args.max_response_length,
        "stats": stats,
        "evalplus_output": result.stdout,
        "log_path": log_path,
    }
    summary_path = os.path.join(args.output_dir, f"summary_{args.tag}_multiturn.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n汇总已保存: {summary_path}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
