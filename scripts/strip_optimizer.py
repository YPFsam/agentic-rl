#!/usr/bin/env python3
"""
Checkpoint 优化器剥离脚本

用法:
    python scripts/strip_optimizer.py                    # 处理所有实验目录
    python scripts/strip_optimizer.py --exp qwen3-4b-grpo-single  # 只处理指定实验
    python scripts/strip_optimizer.py --dry-run          # 仅预览，不实际删除

功能:
    - 保留每个实验最新的 checkpoint 完整不动（含优化器，方便断点续训）
    - 删除其余 checkpoint 的 optim_*.pt 文件（节省 ~60% 空间）
    - 保留 model_*.pt 和其他文件（可用于评估）

4B 模型估算:
    - 完整 checkpoint: ~49 GiB (model ~18G + optim ~31G)
    - 剥离后: ~18 GiB (仅 model)
"""
import argparse
import glob
import os
import re


def find_checkpoints(ckpt_dir):
    """扫描所有实验目录，返回 {exp_name: {step: path}} 的结构"""
    experiments = {}
    if not os.path.isdir(ckpt_dir):
        return experiments

    for exp_dir in glob.glob(os.path.join(ckpt_dir, "*")):
        if not os.path.isdir(exp_dir):
            continue
        exp_name = os.path.basename(exp_dir)
        steps = {}
        for step_dir in glob.glob(os.path.join(exp_dir, "global_step_*")):
            match = re.search(r"global_step_(\d+)", step_dir)
            if match:
                step = int(match.group(1))
                steps[step] = step_dir
        if steps:
            experiments[exp_name] = dict(sorted(steps.items()))

    return experiments


def get_optimizer_files(step_dir):
    """找到 step_dir 下所有 optim_*.pt 文件"""
    return glob.glob(os.path.join(step_dir, "**", "optim_*.pt"), recursive=True)


def get_step_size(step_dir):
    """计算 step_dir 的总大小 (bytes)"""
    total = 0
    for root, dirs, files in os.walk(step_dir):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def format_size(size_bytes):
    """格式化文件大小"""
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:.1f} GiB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / 1024**2:.1f} MiB"
    else:
        return f"{size_bytes / 1024:.1f} KiB"


def strip_experiment(exp_name, steps, dry_run=False):
    """剥离一个实验的旧 checkpoint 优化器状态"""
    if len(steps) <= 1:
        print(f"  [{exp_name}] 只有 {len(steps)} 个 checkpoint，无需剥离")
        return 0

    sorted_steps = sorted(steps.keys())
    latest_step = sorted_steps[-1]
    old_steps = sorted_steps[:-1]

    print(f"  [{exp_name}] {len(steps)} 个 checkpoint, 保留最新 step {latest_step}")

    total_freed = 0
    for step in old_steps:
        step_dir = steps[step]
        optim_files = get_optimizer_files(step_dir)

        if not optim_files:
            print(f"    step {step}: 已无优化器文件，跳过")
            continue

        step_freed = 0
        for f in optim_files:
            size = os.path.getsize(f)
            step_freed += size
            if dry_run:
                print(f"    step {step}: [预览] 删除 {os.path.relpath(f, step_dir)} ({format_size(size)})")
            else:
                os.remove(f)
                print(f"    step {step}: 已删除 {os.path.relpath(f, step_dir)} ({format_size(size)})")

        total_freed += step_freed
        if step_freed > 0:
            if dry_run:
                after_size = get_step_size(step_dir) - step_freed
            else:
                after_size = get_step_size(step_dir)
            print(f"    step {step}: {format_size(step_freed)} → 剩余 {format_size(after_size)}")

    return total_freed


def main():
    parser = argparse.ArgumentParser(description="Checkpoint 优化器剥离")
    parser.add_argument("--ckpt_dir", default="/root/autodl-tmp/checkpoints",
                        help="Checkpoint 根目录")
    parser.add_argument("--exp", default=None,
                        help="只处理指定实验目录")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅预览，不实际删除文件")
    args = parser.parse_args()

    experiments = find_checkpoints(args.ckpt_dir)

    if not experiments:
        print(f"未找到 checkpoint，路径: {args.ckpt_dir}")
        return

    if args.exp:
        if args.exp in experiments:
            experiments = {args.exp: experiments[args.exp]}
        else:
            print(f"未找到实验: {args.exp}")
            print(f"可用实验: {list(experiments.keys())}")
            return

    print(f"{'[预览模式] ' if args.dry_run else ''}扫描 {args.ckpt_dir}")
    print()

    total_freed = 0
    for exp_name, steps in experiments.items():
        freed = strip_experiment(exp_name, steps, dry_run=args.dry_run)
        total_freed += freed

    print()
    if total_freed > 0:
        action = "可释放" if args.dry_run else "已释放"
        print(f"总计 {action}: {format_size(total_freed)}")
    else:
        print("无需剥离")


if __name__ == "__main__":
    main()
