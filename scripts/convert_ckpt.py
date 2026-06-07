#!/usr/bin/env python3
"""将 veRL FSDP checkpoint 转换为 HuggingFace 格式（单步版本，供并行调用）"""
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

step_dir = sys.argv[1]
output_dir = sys.argv[2]

model_pt = os.path.join(step_dir, "actor", "model_world_size_1_rank_0.pt")
hf_dir = os.path.join(step_dir, "actor", "huggingface")

if os.path.exists(os.path.join(output_dir, "model.safetensors.index.json")) or \
   os.path.exists(os.path.join(output_dir, "model.safetensors")):
    print(f"[跳过] {output_dir} 已存在")
    sys.exit(0)

print(f"[转换] {step_dir} -> {output_dir}")
state_dict = torch.load(model_pt, map_location="cpu", weights_only=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype=torch.bfloat16)
model.load_state_dict(state_dict, strict=False)
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer = AutoTokenizer.from_pretrained(hf_dir)
tokenizer.save_pretrained(output_dir)
del model, state_dict
print(f"[完成] {output_dir}")
