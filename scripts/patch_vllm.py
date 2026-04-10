#!/usr/bin/env python3
"""
vLLM 兼容性补丁集合

Patch 1: Qwen2 LoRA base_layer 权重名兼容
  问题：veRL 开启 LoRA 后，参数名变为 qkv_proj.base_layer.weight，
  但发送的权重名是 qkv_proj.weight，导致 KeyError。

Patch 2: numpy.int64 tensor 索引修复
  问题：PyTorch 2.8 不支持 numpy.int64 直接做 tensor 索引，
  vLLM _dummy_run 中 np.cumsum 产生 int64 数组导致崩溃。

Patch 3: veRL disable_adapter 方法名不兼容
  问题：veRL 调用 disable_adapter()（单数），但 Qwen3 等模型的
  LoRA 接口是 disable_adapters()（复数），导致 AttributeError。

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


def patch_verl_disable_adapter():
    """修复 veRL disable_adapter 方法名不兼容（单数 vs 复数）+ adapter 未加载时 ValueError"""
    verl_path = "/tmp/verl/verl/workers/engine/fsdp/transformer_impl.py"
    if not os.path.exists(verl_path):
        print("SKIP: 找不到 veRL transformer_impl.py")
        return

    print(f"[Patch 3] 目标文件: {verl_path}")

    with open(verl_path, "r") as f:
        src = f.read()

    # 已打过新版 patch（含 ValueError catch）
    if "except ValueError" in src:
        print("[Patch 3] 已打过新版 patch，跳过")
        return

    # 原始 veRL 代码
    old_vanilla = """    def disable_adapter(self) -> ContextManager:
        return self.module.disable_adapter()"""

    # 旧版 patch（缺 ValueError catch）
    old_patched = """    def disable_adapter(self) -> ContextManager:
        # 兼容 disable_adapter (PEFT) 和 disable_adapters (Qwen3 等模型)
        if hasattr(self.module, 'disable_adapter'):
            return self.module.disable_adapter()
        elif hasattr(self.module, 'disable_adapters'):
            return self.module.disable_adapters()
        else:
            from contextlib import nullcontext
            return nullcontext()"""

    new = """    def disable_adapter(self) -> ContextManager:
        # 兼容 disable_adapter (PEFT) 和 disable_adapters (Qwen3 等模型)
        # adapter 未加载时调用会抛 ValueError，需 catch
        from contextlib import nullcontext
        try:
            if hasattr(self.module, 'disable_adapter'):
                return self.module.disable_adapter()
            elif hasattr(self.module, 'disable_adapters'):
                return self.module.disable_adapters()
        except ValueError:
            pass
        return nullcontext()"""

    if old_patched in src:
        src = src.replace(old_patched, new, 1)
    elif old_vanilla in src:
        src = src.replace(old_vanilla, new, 1)
    else:
        print("[Patch 3] ERROR: 找不到目标代码")
        sys.exit(1)

    with open(verl_path, "w") as f:
        f.write(src)

    print("[Patch 3] 成功！")


def patch_numpy_core():
    """修复 Docker 镜像中 numpy 1.26.4 _core/ 残留的 2.x 文件

    Docker 镜像构建时先装 numpy 2.x，再降级到 1.26.4，但 _core/ 目录
    残留了 2.x 的 _type_aliases.py、arrayprint.py、numerictypes.py 等文件。
    vLLM spawn 子进程 import numpy._core.arrayprint 时触发 unsignedinteger
    等属性的 AttributeError，导致 EngineDeadError。
    每次克隆实例后必须运行。
    """
    import os
    import glob

    site_pkgs = glob.glob("/root/miniconda3/lib/python*/site-packages/numpy/_core")
    if not site_pkgs:
        print("[Patch 4] SKIP: 找不到 numpy/_core/ 目录")
        return

    core_dir = site_pkgs[0]
    print(f"[Patch 4] 目标目录: {core_dir}")

    # numpy 1.26.4 的 _core/ 只应有这 8 个 .py 文件（兼容性 shim）
    allowed_py = {
        "__init__.py", "__init__.pyi",
        "_dtype.py", "_dtype_ctypes.py", "_internal.py",
        "_multiarray_umath.py", "multiarray.py", "umath.py",
    }

    actual_files = set(f for f in os.listdir(core_dir)
                       if f.endswith((".py", ".pyi")) and f != "__pycache__")

    extra_files = actual_files - allowed_py

    if not extra_files:
        print("[Patch 4] _core/ 目录干净，无需修复")
        return

    print(f"[Patch 4] 发现 {len(extra_files)} 个残留文件: {sorted(extra_files)}")

    # 删除残留文件
    for f in extra_files:
        path = os.path.join(core_dir, f)
        os.remove(path)
        print(f"  删除: {f}")

    # 清理 __pycache__
    pycache = os.path.join(core_dir, "__pycache__")
    if os.path.isdir(pycache):
        import shutil
        shutil.rmtree(pycache)
        print("  清理 __pycache__/")

    # 验证：用子进程测试（当前进程可能已缓存旧 numpy）
    import subprocess
    result = subprocess.run(
        ["python3", "-c", "import numpy; print(numpy.__version__); "
         "from numpy.core import _type_aliases; print('_type_aliases OK')"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode == 0:
        print(f"[Patch 4] 成功！({result.stdout.strip().split(chr(10))[-1]})")
    else:
        print(f"[Patch 4] 验证失败，需要彻底重装 numpy:")
        print(f"  {result.stderr.strip().split(chr(10))[-1]}")
        print("  运行: cd /root/miniconda3/lib/python3.12/site-packages/ && rm -rf numpy* && pip install numpy==1.26.4")


if __name__ == "__main__":
    import os
    patch_qwen2()
    patch_numpy_indexing()
    patch_verl_disable_adapter()
    patch_numpy_core()
