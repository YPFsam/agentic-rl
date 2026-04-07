"""
Checkpoint step vs pass@1 曲线绘制

从 eval_results/checkpoints/ 下读取各 checkpoint 的评估结果，
绘制 step-pass@1 曲线，用于判断最优 checkpoint 和过拟合。
"""
import os
import json
import re
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 12


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="eval_results/checkpoints")
    parser.add_argument("--output_dir", default="plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = {}

    for filename in os.listdir(args.results_dir):
        if not filename.startswith("summary_") or not filename.endswith(".json"):
            continue
        match = re.search(r"_s(\d+)", filename)
        if not match:
            continue
        step = int(match.group(1))

        with open(os.path.join(args.results_dir, filename)) as f:
            results = json.load(f)

        for benchmark, metrics in results.items():
            if "pass@1" in metrics:
                if benchmark not in data:
                    data[benchmark] = []
                data[benchmark].append((step, metrics["pass@1"]))

    if not data:
        print("未找到评估结果")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for benchmark, points in data.items():
        points.sort(key=lambda x: x[0])
        steps, pass_rates = zip(*points)
        ax.plot(steps, pass_rates, marker="o", linewidth=2, label=benchmark)

        best_step, best_rate = max(points, key=lambda x: x[1])
        ax.annotate(f"best: step {best_step} ({best_rate:.1f}%)",
                     xy=(best_step, best_rate),
                     xytext=(10, 10), textcoords="offset points",
                     fontsize=9, color="red")

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("pass@1 (%)")
    ax.set_title("Checkpoint Evaluation: Step vs pass@1")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = os.path.join(args.output_dir, "checkpoint_eval_curve.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Checkpoint 评估曲线已保存: {output_path}")

    for benchmark, points in data.items():
        best_step, best_rate = max(points, key=lambda x: x[1])
        print(f"  {benchmark}: 最优 checkpoint = step {best_step}, pass@1 = {best_rate:.1f}%")


if __name__ == "__main__":
    main()
