"""
过程奖励单元测试。

覆盖：
  1. _compute_state_score 各种执行状态
  2. _compute_process_and_outcome_rewards 各种 turn 组合
  3. format_invalid 不继承未来收益
  4. advantage estimator mock batch 验证
"""
import pytest
import torch
import numpy as np


class TestStateScore:
    """测试 _compute_state_score 函数。"""

    def test_no_code_returns_none(self):
        from src.reward_multiturn import _compute_state_score
        assert _compute_state_score("NO_CODE") is None

    def test_success_returns_1(self):
        from src.reward_multiturn import _compute_state_score
        assert _compute_state_score("SUCCESS") == 1.0

    def test_timeout_returns_0(self):
        from src.reward_multiturn import _compute_state_score
        assert _compute_state_score("TIMEOUT") == 0.0

    def test_syntax_error_returns_0(self):
        from src.reward_multiturn import _compute_state_score
        assert _compute_state_score("SYNTAX_ERROR") == 0.0

    def test_runtime_error_no_tests(self):
        from src.reward_multiturn import _compute_state_score
        assert _compute_state_score("RUNTIME_ERROR", 0, 0) == 0.0

    def test_runtime_error_partial_pass(self):
        from src.reward_multiturn import _compute_state_score
        # 3/5 pass -> 0.2 + 0.8 * 0.6 = 0.68
        assert abs(_compute_state_score("RUNTIME_ERROR", 3, 5) - 0.68) < 1e-6

    def test_runtime_error_half_pass(self):
        from src.reward_multiturn import _compute_state_score
        # 5/10 pass -> 0.2 + 0.8 * 0.5 = 0.6
        assert abs(_compute_state_score("RUNTIME_ERROR", 5, 10) - 0.6) < 1e-6


class TestProcessRewards:
    """测试 _compute_process_and_outcome_rewards。"""

    def test_single_turn_success(self):
        from src.reward_multiturn import _compute_process_and_outcome_rewards
        turns = [{"exec_status": "SUCCESS", "n_passed": 0, "n_total": 0}]
        proc, r_O, fmt_invalid = _compute_process_and_outcome_rewards(turns)
        assert proc == [1.0]          # C_1 - C_0 = 1.0 - 0.0
        assert r_O == 1.0             # 第 1 轮成功
        assert fmt_invalid == [False]

    def test_two_turn_correction(self):
        from src.reward_multiturn import _compute_process_and_outcome_rewards
        turns = [
            {"exec_status": "RUNTIME_ERROR", "n_passed": 0, "n_total": 0},
            {"exec_status": "SUCCESS", "n_passed": 0, "n_total": 0},
        ]
        proc, r_O, fmt_invalid = _compute_process_and_outcome_rewards(turns)
        assert proc == [0.0, 1.0]     # T1: 0-0=0, T2: 1.0-0=1.0
        assert r_O == 0.85            # 第 2 轮成功: 1.0 - 0.15*1
        assert fmt_invalid == [False, False]

    def test_no_code_then_success(self):
        """NO_CODE 不更新 C_t，后续恢复只跟上一个有效状态比较。"""
        from src.reward_multiturn import _compute_process_and_outcome_rewards
        turns = [
            {"exec_status": "NO_CODE", "n_passed": 0, "n_total": 0},
            {"exec_status": "SUCCESS", "n_passed": 0, "n_total": 0},
        ]
        proc, r_O, fmt_invalid = _compute_process_and_outcome_rewards(turns)
        assert proc == [-1.0, 1.0]    # T1: NO_CODE=-1, T2: C=1.0 - C_prev=0 = 1.0
        assert r_O == 0.85
        assert fmt_invalid == [True, False]

    def test_no_code_no_update_state(self):
        """NO_CODE 不更新状态，下一轮的 C_t - C_{t-1} 使用前一个有效状态。"""
        from src.reward_multiturn import _compute_process_and_outcome_rewards
        turns = [
            {"exec_status": "RUNTIME_ERROR", "n_passed": 3, "n_total": 5},
            {"exec_status": "NO_CODE", "n_passed": 0, "n_total": 0},
            {"exec_status": "RUNTIME_ERROR", "n_passed": 4, "n_total": 5},
        ]
        proc, r_O, fmt_invalid = _compute_process_and_outcome_rewards(turns)
        # T1: C=0.68, proc=0.68-0=0.68
        # T2: NO_CODE, proc=-1, C stays 0.68
        # T3: C=0.2+0.8*0.8=0.84, proc=0.84-0.68=0.16
        assert abs(proc[0] - 0.68) < 1e-6
        assert proc[1] == -1.0
        assert abs(proc[2] - 0.16) < 1e-6
        assert r_O == 0.0   # 最终没成功
        assert fmt_invalid == [False, True, False]

    def test_all_no_code(self):
        from src.reward_multiturn import _compute_process_and_outcome_rewards
        turns = [
            {"exec_status": "NO_CODE", "n_passed": 0, "n_total": 0},
            {"exec_status": "NO_CODE", "n_passed": 0, "n_total": 0},
        ]
        proc, r_O, fmt_invalid = _compute_process_and_outcome_rewards(turns)
        assert proc == [-1.0, -1.0]
        assert r_O == -1.0
        assert fmt_invalid == [True, True]

    def test_three_turn_success(self):
        from src.reward_multiturn import _compute_process_and_outcome_rewards
        turns = [
            {"exec_status": "RUNTIME_ERROR", "n_passed": 0, "n_total": 0},
            {"exec_status": "RUNTIME_ERROR", "n_passed": 0, "n_total": 0},
            {"exec_status": "SUCCESS", "n_passed": 0, "n_total": 0},
        ]
        proc, r_O, fmt_invalid = _compute_process_and_outcome_rewards(turns)
        assert proc == [0.0, 0.0, 1.0]
        assert r_O == 0.70   # 1.0 - 0.15*2
        assert fmt_invalid == [False, False, False]

    def test_empty_turns(self):
        from src.reward_multiturn import _compute_process_and_outcome_rewards
        proc, r_O, fmt_invalid = _compute_process_and_outcome_rewards([])
        assert proc == []
        assert r_O == -1.0
        assert fmt_invalid == []


class TestProcessRewardDict:
    """测试 compute_score_process_multiturn 返回值结构。"""

    def test_returns_dict(self):
        from src.reward_multiturn import compute_score_process_multiturn
        extra_info = {
            "turn_details": [
                {"exec_status": "SUCCESS", "n_passed": 0, "n_total": 0},
            ],
            "turn_token_boundaries": [
                {"assistant_start": 0, "assistant_end": 100, "exec_status": "SUCCESS"},
            ],
        }
        result = compute_score_process_multiturn(
            "test", "solution", "gt", extra_info=extra_info,
        )
        assert isinstance(result, dict)
        assert "score" in result
        assert "turn_process_rewards" in result
        assert "outcome_reward" in result
        assert "turn_token_boundaries" in result
        assert "format_invalid_turns" in result
        assert result["n_turns"] == 1


class TestAdvantageEstimator:
    """测试 grpo_process_outcome advantage estimator。"""

    def test_registration(self):
        from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY
        assert "grpo_process_outcome" in ADV_ESTIMATOR_REGISTRY

    def test_basic_two_samples(self):
        from src.advantage_process_outcome import compute_grpo_process_outcome_advantage
        response_length = 200

        # 样本 0: T1 success, R0 = 1.0 + 1.0 = 2.0
        # 样本 1: T1 fail, T2 success, R0 = 0.0 + 0.85 = 0.85
        token_level_rewards = torch.zeros(2, response_length)
        response_mask = torch.zeros(2, response_length)
        # 样本 0: turn 1 tokens [0:50] = assistant
        response_mask[0, 0:50] = 1
        # 样本 1: turn 1 tokens [0:30] = assistant, [30:40] = env, turn 2 [40:80] = assistant
        response_mask[1, 0:30] = 1
        response_mask[1, 30:40] = 0
        response_mask[1, 40:80] = 1

        index = np.array(["p0", "p0"])  # 同一个 prompt group

        non_tensor_batch = {
            "turn_process_rewards": [
                [1.0],           # 样本 0: T1 proc = 1.0
                [0.0, 1.0],      # 样本 1: T1 proc = 0.0, T2 proc = 1.0
            ],
            "outcome_reward": [1.0, 0.85],
            "turn_token_boundaries": [
                [{"assistant_start": 0, "assistant_end": 50}],
                [{"assistant_start": 0, "assistant_end": 30},
                 {"assistant_start": 40, "assistant_end": 80}],
            ],
            "format_invalid_turns": [
                [False],
                [False, False],
            ],
        }

        adv, ret = compute_grpo_process_outcome_advantage(
            token_level_rewards, response_mask, index,
            non_tensor_batch=non_tensor_batch,
        )

        assert adv.shape == (2, response_length)
        # 环境反馈 token 应该为 0
        assert adv[1, 30:40].abs().max() == 0
        # assistant token 不全为 0（有信号）
        assert adv[0, 0:50].abs().sum() > 0
        assert adv[1, 0:30].abs().sum() > 0

    def test_format_invalid_isolation(self):
        """格式非法轮次不继承未来收益。"""
        from src.advantage_process_outcome import compute_grpo_process_outcome_advantage
        response_length = 200

        # 样本: T1 NO_CODE, T2 success
        token_level_rewards = torch.zeros(1, response_length)
        response_mask = torch.zeros(1, response_length)
        response_mask[0, 0:30] = 1   # T1 assistant
        response_mask[0, 30:50] = 0  # env feedback
        response_mask[0, 50:100] = 1 # T2 assistant

        index = np.array(["p0"])

        non_tensor_batch = {
            "turn_process_rewards": [[-1.0, 1.0]],
            "outcome_reward": [0.85],
            "turn_token_boundaries": [[
                {"assistant_start": 0, "assistant_end": 30},
                {"assistant_start": 50, "assistant_end": 100},
            ]],
            "format_invalid_turns": [[True, False]],
        }

        adv, ret = compute_grpo_process_outcome_advantage(
            token_level_rewards, response_mask, index,
            non_tensor_batch=non_tensor_batch,
        )

        # T1 (format invalid) 只有自己的 z_t，不包含 T2 的值
        t1_adv = adv[0, 0:30]
        t2_adv = adv[0, 50:100]
        # T1 和 T2 的 advantage 不应该相同（T2 包含未来收益）
        assert not torch.allclose(t1_adv.mean().expand_as(t2_adv), t2_adv)


class TestSandboxCountPasses:
    """测试 sandbox count_passes 功能。"""

    def test_count_passes_basic(self):
        """执行代码并统计 assert 通过数。"""
        import asyncio
        from src.sandbox import execute_code

        code = "x = 5"
        test_cases = "assert x == 5\nassert x > 3\nassert x == 10"  # 2/3 pass

        result = asyncio.run(execute_code(code, test_cases, timeout=5.0, count_passes=True))
        assert result.n_total == 3
        assert result.n_passed == 2

    def test_count_passes_all_pass(self):
        import asyncio
        from src.sandbox import execute_code

        code = "def add(a, b): return a + b"
        test_cases = "assert add(1, 2) == 3\nassert add(0, 0) == 0"

        result = asyncio.run(execute_code(code, test_cases, timeout=5.0, count_passes=True))
        assert result.n_total == 2
        assert result.n_passed == 2
        assert result.status.value == "success"

    def test_count_passes_no_count_by_default(self):
        """默认不计数。"""
        import asyncio
        from src.sandbox import execute_code

        code = "x = 5"
        test_cases = "assert x == 5"
        result = asyncio.run(execute_code(code, test_cases, timeout=5.0))
        assert result.n_passed == 0
        assert result.n_total == 0
