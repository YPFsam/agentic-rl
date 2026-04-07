"""
训练曲线可视化脚本

从 TensorBoard event 文件提取训练指标，生成 PNG 图表。

用法：
  python analysis/plot_training.py --logdir outputs/grpo_single --output_dir plots/
"""
import os
import argparse
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 12
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 150


def read_tensorboard_logs(logdir: str) -> dict:
    """从 TensorBoard event 文件读取标量数据。"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    metrics = {}
    event_files = glob.glob(os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True)

    if not event_files:
        print(f"警告：在 {logdir} 下未找到 TensorBoard event 文件")
        return metrics

    for event_file in event_files:
        ea = EventAccumulator(event_file)
        ea.Reload()

        for tag in ea.Tags().get("scalars", []):
            if tag not in metrics:
                metrics[tag] = []
            for event in ea.Scalars(tag):
                metrics[tag].append((event.step, event.value))

    for tag in metrics:
        metrics[tag].sort(key=lambda x: x[0])

    return metrics


def plot_single_metric(steps, values, title, ylabel, output_path, smooth_window=5):
    """绘制单条指标曲线（含滑动平均）。"""
    fig, ax = plt.subplots()
    ax.plot(steps, values, alpha=0.3, label="raw", linewidth=0.8)

    if len(values) > smooth_window:
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - smooth_window // 2)
            end = min(len(values), i + smooth_window // 2 + 1)
            smoothed.append(sum(values[start:end]) / (end - start))
        ax.plot(steps, smoothed, label=f"smooth (w={smooth_window})", linewidth=2)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  已保存: {output_path}")


def plot_reward_and_kl(metrics: dict, output_dir: str):
    """绘制 reward + KL 双轴图（核心展示图）。"""
    reward_key = None
    kl_key = None
    for k in metrics:
        if "reward" in k.lower() and "mean" in k.lower():
            reward_key = k
        if k.lower() == "kl" or ("kl" in k.lower() and "coef" not in k.lower()):
            kl_key = k

    if reward_key is None:
        print("  跳过 reward+KL 图：未找到 reward 指标")
        return

    reward_steps, reward_values = zip(*metrics[reward_key])

    fig, ax1 = plt.subplots()

    color1 = "tab:blue"
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Reward (mean)", color=color1)
    ax1.plot(reward_steps, reward_values, color=color1, alpha=0.8, linewidth=1.5)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    if kl_key and kl_key in metrics:
        ax2 = ax1.twinx()
        color2 = "tab:red"
        kl_steps, kl_values = zip(*metrics[kl_key])
        ax2.set_ylabel("KL Divergence", color=color2)
        ax2.plot(kl_steps, kl_values, color=color2, alpha=0.6, linewidth=1.2)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.axhline(y=5.0, color=color2, linestyle="--", alpha=0.3, label="warning=5.0")
        ax2.axhline(y=10.0, color=color2, linestyle=":", alpha=0.3, label="danger=10.0")

    fig.suptitle("Training Dynamics: Reward & KL Divergence")
    fig.tight_layout()
    output_path = os.path.join(output_dir, "reward_and_kl.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  已保存: {output_path}")


def plot_all(logdir: str, output_dir: str):
    """生成所有训练曲线图。"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"读取日志: {logdir}")
    metrics = read_tensorboard_logs(logdir)

    if not metrics:
        print("未读取到任何指标，退出")
        return

    print(f"发现 {len(metrics)} 个指标: {list(metrics.keys())}")

    plot_reward_and_kl(metrics, output_dir)

    metric_configs = {
        "reward": ("Reward", "Reward Value"),
        "loss": ("Training Loss", "Loss"),
        "response_length": ("Response Length", "Tokens"),
        "kl": ("KL Divergence", "KL"),
        "entropy": ("Policy Entropy", "Entropy"),
    }

    for keyword, (title, ylabel) in metric_configs.items():
        for tag, data in metrics.items():
            if keyword in tag.lower():
                steps, values = zip(*data)
                safe_name = tag.replace("/", "_")
                plot_single_metric(
                    steps, values,
                    title=title,
                    ylabel=ylabel,
                    output_path=os.path.join(output_dir, f"{safe_name}.png"),
                )

    print(f"\n所有图表已保存到: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="plots")
    args = parser.parse_args()
    plot_all(args.logdir, args.output_dir)
