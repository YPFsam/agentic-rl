#!/usr/bin/env python3
"""
vLLM 兼容性补丁集合

Patch 1: Qwen2 LoRA base_layer 权重名兼容
  问题：veRL 开启 LoRA 后，参数名变为 qkv_proj.base_layer.weight，
  但发送的权重名是 qkv_proj.weight，导致 KeyError。

Patch 2: numpy.int64 tensor 索引修复
  问题：PyTorch 2.8 不支持 numpy.int64 直接做 tensor 索引，
  vLLM _dummy_run 中 np.cumsum 产生 int64 数组导致崩溃。

用法：python scripts/patch_vllm.py
"""

import sys
import glob

def find_vllm_file(pattern):
    """查找 vLLM 包内文件路径"""
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    return candidates[0]
    if not candidates:
        print("ERROR: 找不到 vllm qwen2.py")
        sys.exit(1)
    return candidates[0]


def patch_qwen2():
    path = find_vllm_file("/root/miniconda3/lib/python*/site-packages/vllm/model_executor/models/qwen2.py")
    if not path:
        print("SKIP: 找不到 vllm qwen2.py")
        return
    print(f"[Patch 1] 目标文件: {path}")

    with open(path, "r") as f:
        src = f.read()

    # 检查是否已打过 patch
    if "_resolve_param" in src:
        print("已打过 patch，跳过")
        return

    # 在 load_weights 方法的 params_dict 行之后插入 _resolve_param 函数
    old = "        params_dict = dict(self.named_parameters(remove_duplicate=False))\n        loaded_params"
    new = """        params_dict = dict(self.named_parameters(remove_duplicate=False))

        # LoRA wrapper renames params: qkv_proj.weight -> qkv_proj.base_layer.weight
        def _resolve_param(name, pd):
            if name in pd:
                return name
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                lora_name = f"{parts[0]}.base_layer.{parts[1]}"
                if lora_name in pd:
                    return lora_name
            return name

        loaded_params"""

    if old not in src:
        print("ERROR: 找不到目标代码，可能 vLLM 版本不兼容")
        sys.exit(1)

    src = src.replace(old, new, 1)

    # 在 stacked_params_mapping 循环内添加 _resolve_param 调用
    # 找 stacked_params_mapping 循环里的 param = params_dict[name]
    old2 = """                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:"""

    new2 = """                name = _resolve_param(name, params_dict)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:"""

    if old2 not in src:
        print("ERROR: 找不到 stacked_params 代码块")
        sys.exit(1)

    src = src.replace(old2, new2, 1)

    # 在 else 分支也添加 _resolve_param
    old3 = """                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)"""

    new3 = """                if is_pp_missing_parameter(name, self):
                    continue
                name = _resolve_param(name, params_dict)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)"""

    if old3 not in src:
        print("ERROR: 找不到 else 分支代码块")
        sys.exit(1)

    src = src.replace(old3, new3, 1)

    with open(path, "w") as f:
        f.write(src)

    print("[Patch 1] 成功！")


def patch_numpy_indexing():
    """修复 numpy.int64 tensor 索引问题"""
    path = find_vllm_file("/root/miniconda3/lib/python*/site-packages/vllm/v1/worker/gpu_model_runner.py")
    if not path:
        print("SKIP: 找不到 gpu_model_runner.py")
        return

    print(f"[Patch 2] 目标文件: {path}")

    with open(path, "r") as f:
        src = f.read()

    old = "return hidden_states, hidden_states[logit_indices]"
    new = "return hidden_states, hidden_states[logit_indices.tolist()]"

    if new in src:
        print("[Patch 2] 已打过 patch，跳过")
        return

    if old not in src:
        print("[Patch 2] ERROR: 找不到目标代码")
        sys.exit(1)

    src = src.replace(old, new)

    with open(path, "w") as f:
        f.write(src)

    print("[Patch 2] 成功！")


if __name__ == "__main__":
    patch_qwen2()
    patch_numpy_indexing()
