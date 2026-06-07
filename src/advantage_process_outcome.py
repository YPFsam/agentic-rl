"""
GRPO 过程+结果分离归一化 advantage estimator。

注册为 "grpo_process_outcome"，配合 compute_score_process_multiturn 使用。

核心算法：
  1. 从 non_tensor_batch 提取 per-turn 的 process rewards 和 outcome reward
  2. 按 uid group 分开归一化 process 和 outcome
  3. 过程奖励缩放 0.5
  4. Reward-to-go：格式非法轮次只拿自己的 z_t，不继承未来收益
  5. 映射到 token-level advantages
"""
import torch
import numpy as np
from collections import defaultdict
from typing import Optional

from verl.trainer.ppo.core_algos import register_adv_est, AdvantageEstimator


@register_adv_est("grpo_process_outcome")
def compute_grpo_process_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[object] = None,
    non_tensor_batch: Optional[dict] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    过程+结果分离归一化的 GRPO advantage estimator。

    Args:
        token_level_rewards: (batch_size, response_length) — 未使用，保留接口兼容
        response_mask: (batch_size, response_length) — assistant=1, env_feedback=0
        index: (batch_size,) — uid 用于 group 归一化
        epsilon: 防除零
        non_tensor_batch: 包含 turn_process_rewards, outcome_reward,
                          turn_token_boundaries, format_invalid_turns

    Returns:
        advantages: (batch_size, response_length)
        returns: (batch_size, response_length) — 与 advantages 相同
    """
    if non_tensor_batch is None:
        # fallback 到标准 GRPO
        from verl.trainer.ppo import core_algos
        return core_algos.compute_grpo_outcome_advantage(
            token_level_rewards, response_mask, index, epsilon, norm_adv_by_std_in_grpo,
        )

    device = response_mask.device
    batch_size, response_length = response_mask.size()

    # 提取 per-sample 数据
    all_process_rewards = non_tensor_batch.get("turn_process_rewards")
    all_outcome_rewards = non_tensor_batch.get("outcome_reward")
    all_boundaries = non_tensor_batch.get("turn_token_boundaries")
    all_format_invalid = non_tensor_batch.get("format_invalid_turns")

    # 数据不完整时 fallback
    if any(x is None for x in [all_process_rewards, all_outcome_rewards,
                                 all_boundaries, all_format_invalid]):
        from verl.trainer.ppo import core_algos
        return core_algos.compute_grpo_outcome_advantage(
            token_level_rewards, response_mask, index, epsilon, norm_adv_by_std_in_grpo,
        )

    # Step 1: 按 uid group 收集所有 process / outcome rewards
    proc_by_group = defaultdict(list)
    outcome_by_group = defaultdict(list)

    for i in range(batch_size):
        uid = index[i]
        # process rewards
        for r in all_process_rewards[i]:
            proc_by_group[uid].append(r)
        # outcome reward
        outcome_by_group[uid].append(all_outcome_rewards[i])

    # Step 2: 计算 group 统计量
    proc_stats = {}
    for g, vals in proc_by_group.items():
        arr = torch.tensor(vals, dtype=torch.float32)
        if len(vals) == 1:
            proc_stats[g] = (torch.tensor(0.0), torch.tensor(1.0))
        else:
            proc_stats[g] = (arr.mean(), arr.std())

    outcome_stats = {}
    for g, vals in outcome_by_group.items():
        arr = torch.tensor(vals, dtype=torch.float32)
        if len(vals) == 1:
            outcome_stats[g] = (torch.tensor(0.0), torch.tensor(1.0))
        else:
            outcome_stats[g] = (arr.mean(), arr.std())

    # Step 3: 构建 token-level advantages
    advantages = torch.zeros(batch_size, response_length, device=device, dtype=torch.float32)

    for i in range(batch_size):
        uid = index[i]
        proc_mean, proc_std = proc_stats[uid]
        out_mean, out_std = outcome_stats[uid]

        process_rewards = all_process_rewards[i]
        outcome_reward = all_outcome_rewards[i]
        boundaries = all_boundaries[i]
        format_invalid = all_format_invalid[i]

        T = len(process_rewards)
        if T == 0:
            continue

        # 归一化 z 值
        z_values = []
        for t in range(T):
            r_proc = process_rewards[t]
            z_proc = 0.5 * (r_proc - proc_mean.item()) / (proc_std.item() + epsilon)
            z_values.append(z_proc)

        # 最后一轮加上归一化的 outcome
        r_O = outcome_reward
        z_O = (r_O - out_mean.item()) / (out_std.item() + epsilon)
        z_values[T - 1] += z_O

        # Reward-to-go
        R_turn = []
        for t in range(T):
            if format_invalid[t]:
                # 格式非法：只拿自己的惩罚
                R_turn.append(z_values[t])
            else:
                # 正常：继承未来收益
                R_turn.append(sum(z_values[t:]))

        # 映射到 token positions
        for t in range(min(T, len(boundaries))):
            start = boundaries[t].get("assistant_start", 0)
            end = boundaries[t].get("assistant_end", 0)
            # clamp
            start = max(0, min(start, response_length))
            end = max(0, min(end, response_length))
            if start < end:
                advantages[i, start:end] = R_turn[t]

    # 应用 response_mask（环境反馈 token = 0）
    advantages = advantages * response_mask

    return advantages, advantages
