"""
Thinking Length 与 Reward 关联分析

从训练日志中提取 thinking_tokens 和 reward，
绘制散点图 + 分箱统计 + 训练趋势图。

用法：
  python analysis/plot_thinking_analysis.py --log logs/thinking_metrics.jsonl --output_dir plots/
"""
import os
import json
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.size"] = 12


def load_metrics(log_path: str) -> list[dict]:
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def plot_thinking_vs_reward(records: list[dict], output_path: str):
    thinking = [r["thinking_tokens"] for r in records if r.get("has_code")]
    rewards = [r["reward"] for r in records if r.get("has_code")]

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(thinking, rewards, alpha=0.15, s=10, c=rewards, cmap="RdYlGn")

    if len(thinking) > 50:
        bins = np.percentile(thinking, np.arange(0, 101, 10))
        bin_means = []
        bin_centers = []
        for i in range(len(bins) - 1):
            mask = [(bins[i] <= t < bins[i + 1]) for t in thinking]
            if any(mask):
                bin_r = [r for r, m in zip(rewards, mask) if m]
                bin_means.append(np.mean(bin_r))
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
        ax.plot(bin_centers, bin_means, "r-o", linewidth=2, markersize=6, label="bin mean")

    ax.set_xlabel("Thinking Tokens (estimated)")
    ax.set_ylabel("Reward")
    ax.set_title("Thinking Length vs Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.colorbar(scatter, ax=ax, label="Reward")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Thinking-Reward 散点图已保存: {output_path}")


def plot_thinking_over_time(records: list[dict], output_path: str, window: int = 100):
    thinking = [r["thinking_tokens"] for r in records]
    rewards = [r["reward"] for r in records]

    def smooth(data, w):
        if len(data) < w:
            return data
        return [np.mean(data[max(0, i - w // 2):i + w // 2 + 1]) for i in range(len(data))]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color1 = "tab:blue"
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Thinking Tokens (smoothed)", color=color1)
    ax1.plot(smooth(thinking, window), color=color1, alpha=0.8, linewidth=1.5)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:green"
    ax2.set_ylabel("Reward (smoothed)", color=color2)
    ax2.plot(smooth(rewards, window), color=color2, alpha=0.6, linewidth=1.2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title("Training Dynamics: Thinking Length & Reward Over Time")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Thinking 趋势图已保存: {output_path}")


def print_statistics(records: list[dict]):
    thinking = [r["thinking_tokens"] for r in records]
    rewards = [r["reward"] for r in records]

    high_reward = [r["thinking_tokens"] for r in records if r["reward"] > 0.5]
    low_reward = [r["thinking_tokens"] for r in records if r["reward"] <= 0.0]

    print(f"\n{'=' * 50}")
    print(f"Thinking Length 分析统计")
    print(f"{'=' * 50}")
    print(f"  总记录数: {len(records)}")
    print(f"  Thinking tokens 均值: {np.mean(thinking):.0f}")
    print(f"  Thinking tokens 中位数: {np.median(thinking):.0f}")
    print(f"  Reward > 0.5 时 thinking 均值: {np.mean(high_reward):.0f} ({len(high_reward)} 条)")
    print(f"  Reward <= 0 时 thinking 均值: {np.mean(low_reward):.0f} ({len(low_reward)} 条)")

    if high_reward and low_reward:
        ratio = np.mean(high_reward) / np.mean(low_reward) if np.mean(low_reward) > 0 else float('inf')
        print(f"  高/低 reward thinking 比值: {ratio:.2f}x")
        if ratio > 1.2:
            print(f"  → 高 reward 样本的 thinking 显著更长")
        elif ratio < 0.8:
            print(f"  → 高 reward 样本的 thinking 更短（内化了能力）")
        else:
            print(f"  → thinking 长度与 reward 无明显线性关系")


def main():
    parser = argparse.ArgumentParser(description="Thinking Length 分析")
    parser.add_argument("--log", type=str, default="logs/thinking_metrics.jsonl")
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument("--smooth_window", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    records = load_metrics(args.log)
    if not records:
        print("未找到记录，请确认训练已产生 thinking_metrics.jsonl")
        return

    print(f"加载 {len(records)} 条记录")
    print_statistics(records)
    plot_thinking_vs_reward(records, os.path.join(args.output_dir, "thinking_vs_reward.png"))
    plot_thinking_over_time(records, os.path.join(args.output_dir, "thinking_trend.png"),
                            window=args.smooth_window)


if __name__ == "__main__":
    main()
