"""
测试 vLLM 在 WSL2 容器内是否能正常运行。
"""
import os

# 在 import vllm 之前设置环境变量
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import torch
from vllm import LLM, SamplingParams


def main():
    MODEL = "Qwen/Qwen3-1.7B"

    print(f"=== GPU: {torch.cuda.get_device_name(0)} ===")
    print(f"=== VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB ===")
    print(f"=== CUDA: {torch.version.cuda} ===")

    print("\n[1/3] 加载模型（gpu_memory_utilization=0.2, max_model_len=1024）...")
    try:
        llm = LLM(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=1024,
            enforce_eager=True,
            dtype="auto",
        )
        print("  ✅ 模型加载成功")
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        return False

    print("\n[2/3] 生成测试...")
    try:
        sampling = SamplingParams(max_tokens=64, temperature=0.0)
        outputs = llm.generate(["def hello_world():"], sampling)
        text = outputs[0].outputs[0].text
        print(f"  ✅ 生成成功: {text[:100]}")
    except Exception as e:
        print(f"  ❌ 生成失败: {e}")
        return False

    print("\n[3/3] 释放模型...")
    del llm
    torch.cuda.empty_cache()
    print("  ✅ 释放成功")

    print("\n=== 全部测试通过！vLLM 在 WSL2 下可正常工作 ===")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
