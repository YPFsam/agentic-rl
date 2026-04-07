"""
消融实验对比图

从评估结果 JSON 中读取各模型的 pass@1，
生成柱状图用于面试/论文展示。

用法：
  python analysis/plot_comparison.py --results_dir eval_results/ --output_dir plots/
"""
import os
import json
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.size"] = 12


def plot_ablation_bar(results: dict, output_path: str):
    """绘制消融实验柱状对比图。"""
    models = list(results.keys())
    benchmarks = list(results[models[0]].keys())

    x = np.arange(len(benchmarks))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#888888", "#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336"]

    for i, model in enumerate(models):
        values = [results[model].get(b, 0) for b in benchmarks]
        bars = ax.bar(x + i * width, values, width, label=model, color=colors[i % len(colors)])
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel("Benchmark")
    ax.set_ylabel("pass@1 (%)")
    ax.set_title("Ablation Study: Reward Design & Multi-turn Correction")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(benchmarks)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"消融对比图已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="eval_results")
    parser.add_argument("--output_dir", type=str, default="plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    for filename in os.listdir(args.results_dir):
        if filename.startswith("summary_") and filename.endswith(".json"):
            tag = filename.replace("summary_", "").replace(".json", "")
            with open(os.path.join(args.results_dir, filename)) as f:
                data = json.load(f)
            results[tag] = {}
            for benchmark, metrics in data.items():
                if "pass@1" in metrics:
                    results[tag][benchmark] = metrics["pass@1"]

    if not results:
        print("未找到评估结果，请先运行 src/evaluate.py")
        return

    plot_ablation_bar(results, os.path.join(args.output_dir, "ablation_comparison.png"))


if __name__ == "__main__":
    main()
