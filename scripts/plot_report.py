"""
生成 REPORT.md 所需的训练曲线和评估对比图。
输出到 report_figures/ 目录。
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

OUT = Path("report_figures")
OUT.mkdir(exist_ok=True)

# ============================================================
# 1. 加载 wandb 数据并合并
# ============================================================
def load_wandb():
    files = {
        "multi_1_225": "report_data/multi_turn_1_225_history.json",
        "multi_200_300": "report_data/multi_turn_200_300_history.json",
        "single_1_300": "report_data/single_turn_1_300_history.json",
        "single_300_350": "report_data/single_turn_300_350_history.json",
    }
    data = {}
    for name, path in files.items():
        with open(path) as f:
            data[name] = json.load(f)

    # 多轮：step 1-225（取1-200）+ step 201-300（去重叠）
    multi = {}
    for key in ['critic/rewards/mean', 'actor/kl_loss', 'response_length/clip_ratio',
                'response_length/mean', 'actor/entropy', 'actor/ppo_kl', 'num_turns/mean']:
        vals_1_225 = {d['step']: d[key] for d in data['multi_1_225'] if d.get(key) is not None}
        vals_200_300 = {d['step']: d[key] for d in data['multi_200_300'] if d.get(key) is not None}
        merged = {}
        merged.update(vals_1_225)
        merged.update(vals_200_300)  # 后者覆盖重叠部分
        if merged:
            steps = sorted(merged.keys())
            multi[key] = (steps, [merged[s] for s in steps])

    # 单轮：step 1-300 + step 301-350（去重叠）
    single = {}
    for key in ['critic/rewards/mean', 'actor/kl_loss', 'response_length/clip_ratio',
                'response_length/mean', 'actor/entropy', 'actor/ppo_kl', 'num_turns/mean']:
        vals_1_300 = {d['step']: d[key] for d in data['single_1_300'] if d.get(key) is not None}
        vals_300_350 = {d['step']: d[key] for d in data['single_300_350'] if d.get(key) is not None}
        merged = {}
        merged.update(vals_1_300)
        merged.update(vals_300_350)
        if merged:
            steps = sorted(merged.keys())
            single[key] = (steps, [merged[s] for s in steps])

    return multi, single


def ma(values, window=20):
    """滑动平均"""
    arr = np.array(values, dtype=float)
    result = np.full_like(arr, np.nan)
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        result[i] = (cumsum[i+1] - cumsum[start]) / (i - start + 1)
    return result


# ============================================================
# 图 1: reward/mean 对比（含 MA20）
# ============================================================
def plot_reward(multi, single):
    fig, ax = plt.subplots(figsize=(10, 5))

    m_steps, m_vals = multi['critic/rewards/mean']
    s_steps, s_vals = single['critic/rewards/mean']

    ax.plot(m_steps, m_vals, alpha=0.25, color='#2563eb', linewidth=0.8)
    ax.plot(s_steps, s_vals, alpha=0.25, color='#dc2626', linewidth=0.8)

    ax.plot(m_steps, ma(m_vals, 20), color='#2563eb', linewidth=2, label='Multi-turn GRPO (MA20)')
    ax.plot(s_steps, ma(s_vals, 20), color='#dc2626', linewidth=2, label='Single-turn GRPO (MA20)')

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward (mean)')
    ax.set_title('Training Reward: Multi-turn vs Single-turn GRPO')
    ax.legend()
    ax.set_xlim(1, 350)

    fig.savefig(OUT / 'reward_mean.png')
    plt.close(fig)
    print(f"  -> reward_mean.png")


# ============================================================
# 图 2: KL 散度对比
# ============================================================
def plot_kl(multi, single):
    fig, ax = plt.subplots(figsize=(10, 5))

    m_steps, m_vals = multi['actor/kl_loss']
    s_steps, s_vals = single['actor/kl_loss']

    ax.plot(m_steps, ma(m_vals, 20), color='#2563eb', linewidth=2, label='Multi-turn GRPO (MA20)')
    ax.plot(s_steps, ma(s_vals, 20), color='#dc2626', linewidth=2, label='Single-turn GRPO (MA20)')
    ax.plot(m_steps, m_vals, alpha=0.2, color='#2563eb', linewidth=0.8)
    ax.plot(s_steps, s_vals, alpha=0.2, color='#dc2626', linewidth=0.8)

    ax.set_xlabel('Step')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence: Multi-turn vs Single-turn GRPO')
    ax.legend()
    ax.set_xlim(1, 350)

    fig.savefig(OUT / 'kl_loss.png')
    plt.close(fig)
    print(f"  -> kl_loss.png")


# ============================================================
# 图 3: 截断率对比
# ============================================================
def plot_clip_ratio(multi, single):
    fig, ax = plt.subplots(figsize=(10, 5))

    m_steps, m_vals = multi['response_length/clip_ratio']
    s_steps, s_vals = single['response_length/clip_ratio']

    ax.plot(m_steps, ma(m_vals, 10), color='#2563eb', linewidth=2, label='Multi-turn GRPO (MA10)')
    ax.plot(s_steps, ma(s_vals, 10), color='#dc2626', linewidth=2, label='Single-turn GRPO (MA10)')
    ax.plot(m_steps, m_vals, alpha=0.2, color='#2563eb', linewidth=0.8)
    ax.plot(s_steps, s_vals, alpha=0.2, color='#dc2626', linewidth=0.8)

    ax.set_xlabel('Step')
    ax.set_ylabel('Clip Ratio')
    ax.set_title('Response Truncation Rate (clip_ratio): Multi-turn vs Single-turn')
    ax.legend()
    ax.set_xlim(1, 350)

    fig.savefig(OUT / 'clip_ratio.png')
    plt.close(fig)
    print(f"  -> clip_ratio.png")


# ============================================================
# 图 4: 多轮训练 T1/T2/T3 纠错率趋势（从日志计算）
# ============================================================
def plot_correction_rates():
    # 加载两个日志
    with open("logs/multiturn_metrics_20260415_002828.jsonl") as f:
        log1 = [json.loads(l) for l in f]
    with open("logs/multiturn_metrics_20260415_140459.jsonl") as f:
        log2 = [json.loads(l) for l in f]

    samples_per_step = 64

    def compute_rates(data, step_offset=0):
        """计算每25步窗口的 T1/T2/T3 纠错率"""
        windows = defaultdict(lambda: {"total": 0, "t1_ok": 0, "t2_att": 0, "t2_ok": 0,
                                        "t3_att": 0, "t3_ok": 0, "nocode": 0})
        for i, d in enumerate(data):
            s = step_offset + i // samples_per_step + 1
            w_start = ((s - 1) // 25) * 25 + 1
            w = f"{w_start}-{w_start + 24}"
            w_data = windows[w]
            w_data["total"] += 1
            turns = d.get("turns", [])
            if turns[0]["outcome"] == "SUCCESS":
                w_data["t1_ok"] += 1
            if len(turns) >= 2:
                w_data["t2_att"] += 1
                if turns[1]["outcome"] == "SUCCESS":
                    w_data["t2_ok"] += 1
            if len(turns) >= 3:
                w_data["t3_att"] += 1
                if turns[2]["outcome"] == "SUCCESS":
                    w_data["t3_ok"] += 1
            if d.get("exec_status") == "NO_CODE":
                w_data["nocode"] += 1
        return windows

    # 用续训日志（step 200-300，无重叠）
    rates_200_300 = compute_rates(log2, step_offset=200)

    # 用早期日志（step 1-225，有重叠但取 1-200）
    rates_1_200 = compute_rates(log1[:200 * samples_per_step], step_offset=0)

    # 合并
    all_rates = {}
    for w, d in rates_1_200.items():
        w_start = int(w.split('-')[0])
        if w_start <= 200:
            all_rates[w] = d
    for w, d in rates_200_300.items():
        all_rates[w] = d

    # 排序
    sorted_windows = sorted(all_rates.keys(), key=lambda x: int(x.split('-')[0]))

    steps_mid = []
    t1_rates, t2_rates, t3_rates, nocode_rates = [], [], [], []

    for w in sorted_windows:
        d = all_rates[w]
        if d["total"] == 0:
            continue
        start = int(w.split('-')[0])
        steps_mid.append(start + 12)
        t1_rates.append(d["t1_ok"] / d["total"] * 100)
        t2_rates.append(d["t2_ok"] / max(d["t2_att"], 1) * 100)
        t3_rates.append(d["t3_ok"] / max(d["t3_att"], 1) * 100)
        nocode_rates.append(d["nocode"] / d["total"] * 100)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(steps_mid, t1_rates, 'o-', color='#16a34a', label='T1 Success Rate', linewidth=2, markersize=4)
    ax1.plot(steps_mid, t2_rates, 's-', color='#2563eb', label='T2 Correction Rate', linewidth=2, markersize=4)
    ax1.plot(steps_mid, t3_rates, '^-', color='#9333ea', label='T3 Correction Rate', linewidth=2, markersize=4)
    ax1.set_ylabel('Rate (%)')
    ax1.set_title('Multi-turn Training: Success/Correction Rates by Step Window (25-step)')
    ax1.legend()
    ax1.set_ylim(0, 50)
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps_mid, nocode_rates, 'D-', color='#ea580c', label='NO_CODE Rate', linewidth=2, markersize=4)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Rate (%)')
    ax2.set_title('NO_CODE Rate by Step Window')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.savefig(OUT / 'correction_rates.png')
    plt.close(fig)
    print(f"  -> correction_rates.png")


# ============================================================
# 图 5: 评估结果 2x2 折线图
# 上=Base, 下=HEval+, 左=单轮评估, 右=多轮评估
# 每图中两条线: 多轮训练 checkpoints vs 单轮训练 checkpoints
# ============================================================
def plot_eval_bars():
    steps = [0, 100, 150, 200, 250, 300]

    # 多轮训练 checkpoints (mt_s0=baseline, mt_s100, ..., mt_s300)
    mt_base_single = [0.610, 0.726, 0.787, 0.750, 0.780, 0.720]
    mt_base_multi  = [0.622, 0.738, 0.805, 0.768, 0.799, 0.750]
    mt_heval_single = [0.561, 0.677, 0.732, 0.695, 0.720, 0.677]
    mt_heval_multi  = [0.573, 0.683, 0.750, 0.707, 0.732, 0.701]

    # 单轮训练 checkpoints
    st_base_single = [0.610, 0.726, 0.738, 0.744, 0.744, 0.768]
    st_base_multi  = [0.622, 0.750, 0.762, 0.756, 0.756, 0.805]
    st_heval_single = [0.561, 0.665, 0.665, 0.683, 0.689, 0.732]
    st_heval_multi  = [0.573, 0.689, 0.683, 0.695, 0.695, 0.762]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)

    marker_kw = dict(markersize=7, linewidth=2)
    mt_style = dict(color='#2563eb', marker='o', label='Multi-turn Trained')
    st_style = dict(color='#dc2626', marker='s', label='Single-turn Trained')

    # Top-left: Base, Single-turn Eval
    ax = axes[0][0]
    ax.plot(steps, mt_base_single, **mt_style, **marker_kw)
    ax.plot(steps, st_base_single, **st_style, **marker_kw)
    ax.set_ylabel('Score')
    ax.set_title('HumanEval Base (Single-turn Eval)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.55, 0.85)

    # Top-right: Base, Multi-turn Eval
    ax = axes[0][1]
    ax.plot(steps, mt_base_multi, **mt_style, **marker_kw)
    ax.plot(steps, st_base_multi, **st_style, **marker_kw)
    ax.set_title('HumanEval Base (Multi-turn Eval)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.55, 0.85)

    # Bottom-left: HEval+, Single-turn Eval
    ax = axes[1][0]
    ax.plot(steps, mt_heval_single, **mt_style, **marker_kw)
    ax.plot(steps, st_heval_single, **st_style, **marker_kw)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Score')
    ax.set_title('HumanEval+ (Single-turn Eval)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.50, 0.80)

    # Bottom-right: HEval+, Multi-turn Eval
    ax = axes[1][1]
    ax.plot(steps, mt_heval_multi, **mt_style, **marker_kw)
    ax.plot(steps, st_heval_multi, **st_style, **marker_kw)
    ax.set_xlabel('Training Step')
    ax.set_title('HumanEval+ (Multi-turn Eval)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.50, 0.80)

    fig.suptitle('Evaluation Scores Across Checkpoints', fontsize=14, fontweight='bold')
    fig.savefig(OUT / 'eval_heval_plus.png')
    plt.close(fig)
    print(f"  -> eval_heval_plus.png")


# ============================================================
# 图 6: r4800 vs r8192 对比
# ============================================================
def plot_budget_comparison():
    checkpoints = ['baseline', 'mt_s150', 'st_s300']
    heval_4800 = [0.573, 0.750, 0.762]
    heval_8192 = [0.671, 0.780, 0.787]
    base_4800 =  [0.622, 0.805, 0.805]
    base_8192 =  [0.646, 0.854, 0.835]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    x = np.arange(len(checkpoints))
    width = 0.3

    # HumanEval+
    ax1.bar(x - width/2, heval_4800, width, label='r4800', color='#93c5fd', edgecolor='#2563eb')
    ax1.bar(x + width/2, heval_8192, width, label='r8192', color='#2563eb', edgecolor='#1d4ed8')
    ax1.set_ylabel('HumanEval+ Score')
    ax1.set_title('HumanEval+')
    ax1.set_xticks(x)
    ax1.set_xticklabels(checkpoints)
    ax1.legend(loc='upper left')
    ax1.set_ylim(0.5, 0.85)
    ax1.grid(True, axis='y', alpha=0.3)
    for i, (v1, v2) in enumerate(zip(heval_4800, heval_8192)):
        ax1.text(i - width/2, v1 + 0.006, f'{v1:.3f}', ha='center', fontsize=8)
        ax1.text(i + width/2, v2 + 0.006, f'{v2:.3f}', ha='center', fontsize=8)

    # HumanEval/Base
    ax2.bar(x - width/2, base_4800, width, label='r4800', color='#fca5a5', edgecolor='#dc2626')
    ax2.bar(x + width/2, base_8192, width, label='r8192', color='#dc2626', edgecolor='#991b1b')
    ax2.set_ylabel('HumanEval/Base Score')
    ax2.set_title('HumanEval (Base)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(checkpoints)
    ax2.legend(loc='upper left')
    ax2.set_ylim(0.5, 0.9)
    ax2.grid(True, axis='y', alpha=0.3)
    for i, (v1, v2) in enumerate(zip(base_4800, base_8192)):
        ax2.text(i - width/2, v1 + 0.006, f'{v1:.3f}', ha='center', fontsize=8)
        ax2.text(i + width/2, v2 + 0.006, f'{v2:.3f}', ha='center', fontsize=8)

    fig.suptitle('Response Budget Comparison: r4800 vs r8192', fontsize=13)
    fig.savefig(OUT / 'budget_comparison.png')
    plt.close(fig)
    print(f"  -> budget_comparison.png")


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    print("Loading wandb data...")
    multi, single = load_wandb()

    print("Generating plots...")
    plot_reward(multi, single)
    plot_kl(multi, single)
    plot_clip_ratio(multi, single)
    plot_correction_rates()
    plot_eval_bars()
    plot_budget_comparison()

    print(f"\nDone! All figures saved to {OUT}/")
