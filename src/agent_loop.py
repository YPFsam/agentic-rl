"""
多轮代码纠错 AgentLoop

继承 veRL AgentLoopBase，实现：
  LLM 生成代码 → async 沙盒执行 → 错误反馈 → LLM 修正 → 再执行 → ... → 最终评分

关键设计：
  - response_mask: LLM 生成的 token 标记为 1（参与梯度），环境反馈标记为 0
  - 错误反馈模板：包含错误类型 + 截断的 stderr
  - max_turns 控制最大交互轮数
  - 纯稀疏奖励：只看最终执行结果 + turn_penalty

veRL 架构链路：
  GRPOTrainer → AgentLoopManager.generate_sequences
    → AgentLoopWorker → CodeAgentLoop.run()

注意：veRL v0.8 中 AgentLoop 位于 experimental 目录：
  verl.experimental.agent_loop.agent_loop.AgentLoopBase
"""
import re
import ast
from typing import Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

from src.sandbox import execute_code, ExecStatus
from src.reward import extract_code


# ========== 错误反馈模板 ==========

ERROR_FEEDBACK_TEMPLATE = (
    "\n\n[Execution Feedback - Turn {turn}/{max_turns}]\n"
    "Your code produced the following error:\n"
    "```\n{error}\n```\n"
    "Please analyze the error and fix your code.\n"
    "Output your revised solution in a ```python code block.\n"
)

SUCCESS_FEEDBACK = (
    "\n\n[Execution Feedback]\n"
    "All test cases passed! Your code is correct.\n"
)


class CodeAgentLoop(AgentLoopBase):
    """
    多轮代码纠错 AgentLoop。

    每轮交互流程：
    1. LLM 生成代码（response_mask=1）
    2. 从 LLM 输出中提取 Python 代码
    3. 在 async 沙盒中执行代码 + 测试用例
    4. 如果成功 → 拼接成功反馈（response_mask=0），结束
    5. 如果失败 → 拼接错误反馈（response_mask=0），继续下一轮

    参数（通过 veRL 配置传入）：
        max_turns: int — 最大交互轮数（含第一次生成）
        timeout: float — 代码执行超时秒数
        max_error_length: int — 错误信息截断长度
    """

    def __init__(self, *args, max_turns: int = 3, timeout: float = 5.0,
                 max_error_length: int = 500, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_turns = max_turns
        self.timeout = timeout
        self.max_error_length = max_error_length

    async def run(self, sampling_params: dict, **kwargs) -> AgentLoopOutput:
        """
        执行多轮代码纠错循环。

        Args:
            sampling_params: veRL 传入的采样参数（temperature, top_p, max_tokens 等）
            **kwargs: 数据集字段（raw_prompt, test_cases 等来自 non_tensor_batch）

        Returns:
            AgentLoopOutput 包含 prompt_ids, response_ids, response_mask
        """
        # 从 kwargs 获取原始 prompt（消息列表）和测试用例
        raw_prompt = kwargs.get("raw_prompt", [])
        test_cases = kwargs.get("test_cases", "")

        # 1. tokenize prompt → prompt_ids
        prompt_ids = await self.apply_chat_template(raw_prompt)

        # 2. 初始化输出容器
        response_ids = []
        response_mask = []

        # current_ids 追踪当前对话的完整 token 序列（prompt 为起点）
        current_ids = list(prompt_ids)

        # 唯一请求 ID，用于 server_manager 的 sticky session
        request_id = uuid4().hex

        for turn in range(self.max_turns):
            # ===== 1. LLM 生成 =====
            llm_output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=current_ids,
                sampling_params=sampling_params,
            )

            # LLM 生成的 token → mask=1（参与策略梯度）
            generated_ids = llm_output.ids
            response_ids.extend(generated_ids)
            response_mask.extend([1] * len(generated_ids))
            current_ids.extend(generated_ids)

            # ===== 2. 提取代码 =====
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            code = extract_code(generated_text)

            if code is None:
                # 代码提取失败，给格式提示后继续
                if turn < self.max_turns - 1:
                    feedback = (
                        "\n\n[Execution Feedback]\n"
                        "Error: No Python code block found in your response. "
                        "Please use ```python ... ``` format.\n"
                    )
                    self._append_feedback(feedback, response_ids, response_mask, current_ids)
                continue

            # ===== 3. 沙盒执行 =====
            result = await execute_code(code, test_cases, timeout=self.timeout)

            if result.status == ExecStatus.SUCCESS:
                # 成功 → 拼接成功反馈，结束循环
                self._append_feedback(
                    SUCCESS_FEEDBACK, response_ids, response_mask, current_ids
                )
                break
            else:
                # 失败 → 构造错误反馈，继续
                error_msg = self._format_error(result, turn)
                feedback = ERROR_FEEDBACK_TEMPLATE.format(
                    turn=turn + 1,
                    max_turns=self.max_turns,
                    error=error_msg,
                )
                self._append_feedback(feedback, response_ids, response_mask, current_ids)

        return AgentLoopOutput(
            prompt_ids=list(prompt_ids),
            response_ids=response_ids,
            response_mask=response_mask,
        )

    def _append_feedback(self, feedback: str, response_ids: list,
                         response_mask: list, current_ids: list):
        """
        将环境反馈拼入 response，mask 设为 0。

        环境反馈（错误信息、成功信息）不参与策略梯度更新，
        因为它们不是模型生成的，模型不应该"学习"去生成错误信息。
        """
        feedback_ids = self.tokenizer.encode(feedback, add_special_tokens=False)
        response_ids.extend(feedback_ids)
        response_mask.extend([0] * len(feedback_ids))
        current_ids.extend(feedback_ids)

    # ---------- AssertionError 脱敏 ----------
    # 不用 re.DOTALL：只匹配当前行，保留 AssertionError 之后的其他日志信息
    _ASSERTION_PATTERN = re.compile(r"AssertionError: .+")
    _SANITIZED_ASSERT_MSG = (
        "AssertionError: Test case failed — your code produced incorrect output "
        "for a hidden test case. Please review your logic and handle edge cases."
    )

    def _format_error(self, result, turn: int) -> str:
        """
        格式化错误信息。

        处理顺序：先脱敏，再截断。
          1. 先替换 AssertionError 中的敏感数值（防测试用例泄露）
          2. 再执行尾部截断（保留 traceback 底部关键信息）

        如果先截断再脱敏，截断可能从 "AssertionError" 中间切断，
        导致正则匹配失败，敏感信息泄露。
        """
        if result.status == ExecStatus.TIMEOUT:
            return f"TimeoutError: Code execution exceeded {self.timeout}s limit (possible infinite loop)."

        stderr = result.stderr

        # 步骤 1：先脱敏 AssertionError（防止截断打断关键词导致正则失效）
        stderr = self._ASSERTION_PATTERN.sub(self._SANITIZED_ASSERT_MSG, stderr)

        # 步骤 2：再执行尾部截断（此时脱敏已完成，截断不会泄露敏感信息）
        if len(stderr) > self.max_error_length:
            stderr = stderr[-self.max_error_length:]

        if result.status == ExecStatus.SYNTAX_ERROR:
            return f"SyntaxError:\n{stderr}"
        else:
            return f"RuntimeError:\n{stderr}"
