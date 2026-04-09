"""
数据准备脚本

从 MBPP 数据集提取训练数据（本地 train+val ~464条，云端 MBPP + APPS ~3000条），
生成 veRL GRPO 训练所需的 parquet 格式文件。

**重要：不用 MBPP test split**
  MBPP test split 的题目与 MBPP+ 评估 benchmark 重叠，
  如果用 test split 训练再在 MBPP+ 上评估，属于数据泄露。
  因此只使用 train + validation splits（~464条）。

veRL 数据格式要求（parquet）：
  - data_source: str — 数据集来源
  - prompt: list[dict] — chat messages（system + user）
  - ability: str — 任务类型
  - reward_model: dict — 包含 ground_truth
  - extra_info: dict — 包含 test_cases, task_id 等
"""
import os
import re
import json
import random
import argparse

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
import datasets


# ========== System Prompt ==========

SYSTEM_PROMPT = (
    "Solve the programming problem below. "
    "First think step by step, then provide your solution "
    "in a ```python code block. "
    "Your code will be tested with assert statements."
)


# ========== MBPP 数据加载 ==========

def load_mbpp_data() -> list[dict]:
    """
    加载 MBPP 数据集（train + validation），返回标准格式的样本列表。

    Returns:
        样本列表，每个样本包含 task_id, prompt, test_cases, data_source
    """
    ds = load_dataset("mbpp", "full")
    samples = []

    for split_name in ["train", "validation"]:
        for item in ds[split_name]:
            task_id = item.get("task_id", f"mbpp_{len(samples)}")
            prompt_text = item["text"]
            test_list = item["test_list"]

            test_code = "\n".join(test_list)

            samples.append({
                "task_id": str(task_id),
                "prompt": prompt_text,
                "test_cases": test_code,
                "data_source": "mbpp",
            })

    print(f"MBPP 加载完成: {len(samples)} 条")
    return samples


# ========== 数据质量检查 ==========

def validate_mbpp_data(samples: list[dict]) -> list[dict]:
    """
    验证 MBPP 数据质量。检查 test_cases 语法是否正确。

    Args:
        samples: MBPP 样本列表

    Returns:
        过滤后的样本列表
    """
    import ast
    valid = []
    skipped = 0
    for s in samples:
        tc = s.get("test_cases", "")
        try:
            ast.parse(tc)
            valid.append(s)
        except SyntaxError:
            skipped += 1
    if skipped > 0:
        print(f"MBPP 数据质量检查: 过滤 {skipped} 条 test_cases 语法错误的样本，剩余 {len(valid)} 条")
    return valid


# ========== 跨数据源去重 ==========

def deduplicate_cross_source(samples: list[dict], threshold: float = 0.7) -> list[dict]:
    """
    跨数据源去重：基于 prompt 词集合的 Jaccard 相似度检测重复。
    按顺序保留，先出现的优先（即 MBPP 优先于 APPS）。

    Args:
        samples: 混合后的样本列表（MBPP 在前）
        threshold: Jaccard 相似度阈值，超过则视为重复

    Returns:
        去重后的样本列表
    """
    def normalize(text):
        return re.sub(r'\s+', ' ', text.lower().strip())

    kept_norms = []
    kept = []
    dup_count = 0

    for s in samples:
        norm = normalize(s["prompt"])
        words_new = set(norm.split())
        is_dup = False
        for existing_norm in kept_norms:
            words_old = set(existing_norm.split())
            if not words_new or not words_old:
                continue
            jaccard = len(words_new & words_old) / len(words_new | words_old)
            if jaccard > threshold:
                is_dup = True
                break

        if is_dup:
            dup_count += 1
        else:
            kept_norms.append(norm)
            kept.append(s)

    if dup_count > 0:
        print(f"跨数据源去重: 移除 {dup_count} 条重复样本，剩余 {len(kept)} 条")
    else:
        print("跨数据源去重: 未发现重复样本")
    return kept


# ========== APPS 清洗数据加载 ==========

def load_apps_cleaned(max_samples: int = 3000) -> list[dict]:
    """
    加载 LLM 清洗后的 APPS 数据（带 assert 测试用例）。

    Args:
        max_samples: 最多加载的样本数

    Returns:
        样本列表
    """
    cleaned_path = os.path.join(os.path.dirname(__file__) or ".", "..", "data", "apps_cleaned.jsonl")
    if not os.path.exists(cleaned_path):
        print(f"警告：APPS 清洗数据不存在 ({cleaned_path})，请先运行 scripts/synth_testcases.py")
        return []

    samples = []
    with open(cleaned_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                samples.append({
                    "task_id": item["task_id"],
                    "prompt": item["prompt"],
                    "test_cases": item["test_cases"],
                    "data_source": item["data_source"],
                })
                if len(samples) >= max_samples:
                    break

    print(f"APPS-easy-cleaned 加载完成: {len(samples)} 条")
    return samples


# ========== 生成 veRL parquet ==========

def create_verl_parquet(
    samples: list[dict],
    output_path: str,
) -> None:
    """
    将样本列表转换为 veRL 格式的 parquet 文件。

    Args:
        samples: 样本列表
        output_path: 输出 parquet 文件路径
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    records = []
    for s in samples:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": s["prompt"]},
        ]
        records.append({
            "data_source": s["data_source"],
            "prompt": messages,
            "ability": "code_generation",
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {
                "test_cases": s["test_cases"],
                "task_id": s["task_id"],
            },
        })

    # 用 datasets 库保存 parquet，自动处理 dict/list 类型序列化
    ds = datasets.Dataset.from_list(records)
    ds.to_parquet(output_path)
    print(f"veRL parquet 已保存: {output_path} ({len(samples)} 条)")


# ========== 生成多轮训练 parquet ==========

def create_multiturn_parquet(
    samples: list[dict],
    output_path: str,
    agent_name: str = "code_agent_loop",
) -> None:
    """
    生成多轮训练用的 parquet 文件。

    与 create_verl_parquet 的区别：新增 agent_name 列。

    Args:
        samples: 样本列表
        output_path: 输出路径
        agent_name: AgentLoop 名称
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    records = []
    for s in samples:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": s["prompt"]},
        ]
        records.append({
            "data_source": s["data_source"],
            "prompt": messages,
            "ability": "code_generation",
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {
                "test_cases": s["test_cases"],
                "task_id": s["task_id"],
            },
            "agent_name": agent_name,
        })

    ds = datasets.Dataset.from_list(records)
    ds.to_parquet(output_path)
    print(f"多轮训练 parquet 已保存: {output_path} ({len(samples)} 条)")


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(description="准备 veRL 训练数据")
    parser.add_argument(
        "--output", type=str,
        default="data/grpo_train.parquet",
        help="输出 parquet 文件路径",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="使用扩充数据（MBPP train+val + APPS-easy-cleaned，~3000条，云端用）",
    )
    parser.add_argument(
        "--multiturn", action="store_true",
        help="生成多轮训练 parquet（含 agent_name 列）",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # 加载并验证 MBPP 数据
    samples = load_mbpp_data()
    samples = validate_mbpp_data(samples)

    if args.full:
        apps_samples = load_apps_cleaned()
        samples.extend(apps_samples)
        # 跨数据源去重（MBPP 在前，优先保留）
        samples = deduplicate_cross_source(samples)

    # 打乱顺序
    random.shuffle(samples)

    # 生成 parquet
    if args.multiturn:
        create_multiturn_parquet(samples, args.output)
    else:
        create_verl_parquet(samples, args.output)

    print(f"\n数据准备完成！共 {len(samples)} 条训练样本")
    print(f"运行以下命令开始训练：")
    if args.multiturn:
        print(f"  bash scripts/run_local_multi.sh")
    else:
        print(f"  bash scripts/run_local_single.sh")


if __name__ == "__main__":
    main()
