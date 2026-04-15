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
import json
import re
import ast
import time
from typing import Optional
from uuid import uuid4
from pathlib import Path

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.utils.profiler import simple_timer

from src.sandbox import execute_code, ExecStatus
from src.reward import extract_code


# ========== 错误反馈模板 ==========

ERROR_FEEDBACK_TEMPLATE = (
    "\n\n[Execution Feedback - Turn {turn}/{max_turns}]\n"
    "Your code produced the following error:\n"
    "```\n{error}\n```\n"
    "Please analyze the error and fix your code.\n"
    "Output your revised solution in a python code block.\n"
)

SUCCESS_FEEDBACK = (
    "\n\n[Execution Result]\n"
    "All test cases passed! Your code is correct.\n"
)

# ========== 异常检测阈值 ==========
_ANOMALY_GEN_THRESHOLD = 200.0   # 单轮 LLM 生成超过 200s 视为异常
_ANOMALY_TOOL_THRESHOLD = 10.0   # 单轮沙盒执行超过 10s 视为异常


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
        # test_cases 在 extra_info 嵌套 dict 里，不在 non_tensor_batch 顶层
        test_cases = kwargs.get("extra_info", {}).get("test_cases", "")

        # 1. tokenize prompt → prompt_ids
        prompt_ids = await self.apply_chat_template(raw_prompt)

        # 2. 初始化输出容器
        response_ids = []
        response_mask = []

        # current_ids 追踪当前对话的完整 token 序列（prompt 为起点）
        current_ids = list(prompt_ids)

        # 唯一请求 ID，用于 server_manager 的 sticky session
        request_id = uuid4().hex

        # veRL 要求的 metrics 字段
        metrics = {}
        actual_turns = 0

        response_length = self.rollout_config.response_length

        # 异常样本追踪：每轮收集详情
        turn_details_raw = []

        for turn in range(self.max_turns):
            # 预算检查：剩余空间不足则提前终止
            remaining = response_length - len(response_ids)
            if remaining <= 0:
                break

            # ===== 1. LLM 生成 =====
            # 动态调整 max_new_tokens：不超过剩余 response 预算
            # 否则 vLLM 会尝试分配超出预算的 KV cache，导致：
            #   (a) 多轮累积序列超过 max_model_len → preemption 死锁
            #   (b) 第 2 轮生成的代码被事后截断，修正毫无意义
            adjusted_params = dict(sampling_params)
            original_max = adjusted_params.get("max_new_tokens", response_length)
            adjusted_params["max_new_tokens"] = min(remaining, original_max)

            t_gen_start = time.monotonic()
            with simple_timer("generate_sequences", metrics):
                llm_output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=current_ids,
                    sampling_params=adjusted_params,
                )
            gen_time = time.monotonic() - t_gen_start
            actual_turns += 1

            # LLM 生成的 token → mask=1（参与策略梯度）
            generated_ids = llm_output.token_ids
            response_ids.extend(generated_ids)
            response_mask.extend([1] * len(generated_ids))
            current_ids.extend(generated_ids)

            # 预算检查：生成后超限则截断并终止
            if len(response_ids) > response_length:
                response_ids = response_ids[:response_length]
                response_mask = response_mask[:response_length]
                break

            # ===== 2. 提取代码 =====
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            code = extract_code(generated_text)

            if code is None:
                # 代码提取失败，记录轮次详情后继续
                turn_details_raw.append({
                    "turn": turn + 1,
                    "generate_time": gen_time,
                    "generated_text": generated_text,
                    "code_extracted": False,
                    "extracted_code": None,
                    "exec_status": "NO_CODE",
                    "exec_stderr": None,
                    "tool_time": 0.0,
                })
                if turn < self.max_turns - 1:
                    feedback = (
                        "\n\n[Format Error]\n"
                        "Error: No Python code block found in your response. "
                        "Please use a python code block.\n"
                    )
                    self._append_feedback(feedback, response_ids, response_mask, current_ids)
                continue

            # ===== 3. 沙盒执行 =====
            t_tool_start = time.monotonic()
            with simple_timer("tool_calls", metrics):
                result = await execute_code(code, test_cases, timeout=self.timeout)
            tool_time = time.monotonic() - t_tool_start

            if result.status == ExecStatus.SUCCESS:
                # 成功 → 记录详情 + 拼接成功反馈，结束循环
                turn_details_raw.append({
                    "turn": turn + 1,
                    "generate_time": gen_time,
                    "generated_text": generated_text,
                    "code_extracted": True,
                    "extracted_code": code,
                    "exec_status": "SUCCESS",
                    "exec_stderr": None,
                    "tool_time": tool_time,
                })
                self._append_feedback(
                    SUCCESS_FEEDBACK, response_ids, response_mask, current_ids
                )
                break
            else:
                # 失败 → 记录详情 + 构造错误反馈，继续
                turn_details_raw.append({
                    "turn": turn + 1,
                    "generate_time": gen_time,
                    "generated_text": generated_text,
                    "code_extracted": True,
                    "extracted_code": code,
                    "exec_status": result.status.name,
                    "exec_stderr_raw": (result.stderr or "")[:500],
                    "tool_time": tool_time,
                })
                # 最后一轮不追加错误反馈：避免 (a) 轮次计数虚高 (b) extract_last_code 提取失败
                if turn < self.max_turns - 1:
                    error_msg = self._format_error(result, turn)
                    feedback = ERROR_FEEDBACK_TEMPLATE.format(
                        turn=turn + 1,
                        max_turns=self.max_turns,
                        error=error_msg,
                    )
                    self._append_feedback(feedback, response_ids, response_mask, current_ids)

        # 最终保障：截断到 response_length（多轮累计可能超出预算）
        if len(response_ids) > response_length:
            response_ids = response_ids[:response_length]
            response_mask = response_mask[:response_length]

        # ===== 传递结构化轮次数据给 reward 函数 =====
        # reward 函数直接使用这些字段，不再依赖正则切分 solution_str
        # 避免模型 echo 反馈标记导致轮次计数和代码提取出错
        extra_info = kwargs.get("extra_info")
        if extra_info is not None:
            # 最后一轮的模型原始输出（不含环境反馈）
            # reward 从此文本清洗 think 标签并提取代码
            if turn_details_raw:
                extra_info["last_turn_text"] = turn_details_raw[-1]["generated_text"]
            else:
                extra_info["last_turn_text"] = ""

            # 每轮结构化数据：exec_status 由 agent_loop 沙盒执行确定，不受 echo 影响
            # 传完整原始数据方便 debug，每步 64 样本约 0.3 MB，可忽略
            extra_info["turn_details"] = turn_details_raw

        # 异常样本检测与记录
        self._log_if_anomalous(
            turn_details_raw=turn_details_raw,
            raw_prompt=raw_prompt,
            response_length=response_length,
            actual_turns=actual_turns,
            response_ids=response_ids,
        )

        return AgentLoopOutput(
            prompt_ids=list(prompt_ids),
            response_ids=response_ids,
            response_mask=response_mask,
            num_turns=actual_turns,
            metrics=metrics,
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

    def _log_if_anomalous(self, turn_details_raw, raw_prompt, response_length,
                          actual_turns, response_ids):
        """
        检测异常样本并记录完整详情到 logs/anomalous_samples.jsonl。

        异常条件（满足任一即记录）：
          - 单轮 LLM 生成耗时 > 200s（正常 ~100-135s）
          - 单轮沙盒执行耗时 > 10s（正常 <1s，超时阈值 5s）
        """
        is_anomalous = False
        for detail in turn_details_raw:
            if detail["generate_time"] > _ANOMALY_GEN_THRESHOLD:
                is_anomalous = True
                break
            if detail["tool_time"] > _ANOMALY_TOOL_THRESHOLD:
                is_anomalous = True
                break

        if not is_anomalous:
            return

        try:
            log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)

            record = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "prompt": raw_prompt,
                "max_turns": self.max_turns,
                "timeout": self.timeout,
                "response_length": response_length,
                "actual_turns": actual_turns,
                "final_response_length": len(response_ids),
                "response_clipped": len(response_ids) >= response_length,
                "turn_details": turn_details_raw,
            }

            # 每个 Python 进程用独立文件，避免重启训练数据混杂
            if not hasattr(self, "_run_id"):
                self._run_id = time.strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"anomalous_samples_{self._run_id}.jsonl"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            # 日志失败不应影响训练
            pass

    # ---------- test case 脱敏 ----------
    # traceback 中的 assert 行包含完整测试输入和期望输出，替换为占位符
    # 只替换 assert 行，保留错误类型、模型出错代码行、错误消息等纠错信号
    _ASSERT_LINE_PATTERN = re.compile(r'^(\s+)assert\s+.+$', re.MULTILINE)
    _ASSERT_LINE_REPLACEMENT = r'\1assert <test_case>'

    def _format_error(self, result, turn: int) -> str:
        """
        格式化错误信息：脱敏 assert 行 + 截断。

        处理顺序：
          1. 替换 traceback 中 assert 行的测试内容（保留 assert 关键字）
          2. 尾部截断（保留 traceback 底部关键信息）

        脱敏效果：
          - AssertionError: assert <test_case>（去掉输入输出）
          - TypeError/IndexError: assert 行脱敏 + 模型出错代码行完整保留
          - SyntaxError: 不受影响（无 assert 行）
        """
        if result.status == ExecStatus.TIMEOUT:
            return f"TimeoutError: Code execution exceeded {self.timeout}s limit (possible infinite loop)."

        stderr = result.stderr

        # 步骤 1：脱敏 assert 行（必须在截断前，防止截断切掉 assert 行）
        stderr = self._ASSERT_LINE_PATTERN.sub(self._ASSERT_LINE_REPLACEMENT, stderr)

        # 步骤 2：尾部截断
        if len(stderr) > self.max_error_length:
            stderr = stderr[-self.max_error_length:]

        if result.status == ExecStatus.SYNTAX_ERROR:
            return f"SyntaxError:\n{stderr}"
        else:
            return f"RuntimeError:\n{stderr}"
