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
    """修复 veRL disable_adapter：FSDP 包装后 peft 方法不可用，手动禁用 LoRA 层"""
    verl_path = "/tmp/verl/verl/workers/engine/fsdp/transformer_impl.py"
    if not os.path.exists(verl_path):
        print("SKIP: 找不到 veRL transformer_impl.py")
        return

    print(f"[Patch 3] 目标文件: {verl_path}")

    with open(verl_path, "r") as f:
        src = f.read()

    # 已打过新版 patch（含手动禁用 fallback）
    if "_manual_disable" in src:
        print("[Patch 3] 已打过新版 patch（手动禁用 LoRA），跳过")
        return

    # 原始 veRL 代码
    old_vanilla = """    def disable_adapter(self) -> ContextManager:
        return self.module.disable_adapter()"""

    # 旧版 patch（含 ValueError catch 但 fallback 到 nullcontext）
    old_versions = [
        # debug 版本
        """    def disable_adapter(self) -> ContextManager:
        # 兼容 disable_adapter (PEFT) 和 disable_adapters (Qwen3 等模型)
        # adapter 未加载时调用会抛 ValueError，需 catch
        from contextlib import nullcontext
        import logging
        logger = logging.getLogger(__name__)
        try:
            if hasattr(self.module, 'disable_adapter'):
                logger.warning("[ADAPTER_DEBUG] using module.disable_adapter()")
                return self.module.disable_adapter()
            elif hasattr(self.module, 'disable_adapters'):
                logger.warning("[ADAPTER_DEBUG] using module.disable_adapters()")
                return self.module.disable_adapters()
            else:
                logger.warning("[ADAPTER_DEBUG] no disable_adapter method found on module type=%s, falling back to nullcontext", type(self.module).__name__)
        except (ValueError, Exception) as e:
            logger.warning("[ADAPTER_DEBUG] disable_adapter raised %s: %s, falling back to nullcontext", type(e).__name__, e)
        return nullcontext()""",
        # ValueError catch 版本
        """    def disable_adapter(self) -> ContextManager:
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
        return nullcontext()""",
        # 旧版 patch（缺 ValueError catch）
        """    def disable_adapter(self) -> ContextManager:
        # 兼容 disable_adapter (PEFT) 和 disable_adapters (Qwen3 等模型)
        if hasattr(self.module, 'disable_adapter'):
            return self.module.disable_adapter()
        elif hasattr(self.module, 'disable_adapters'):
            return self.module.disable_adapters()
        else:
            from contextlib import nullcontext
            return nullcontext()""",
    ]

    new = """    def disable_adapter(self) -> ContextManager:
        # 兼容 disable_adapter (PEFT) 和 disable_adapters (Qwen3 等模型)
        # FSDP 包装后 peft 方法可能不可用或抛 ValueError，需手动禁用 LoRA 层
        from contextlib import contextmanager, nullcontext

        # 优先尝试标准 peft/Qwen3 方法
        try:
            if hasattr(self.module, 'disable_adapter'):
                return self.module.disable_adapter()
            elif hasattr(self.module, 'disable_adapters'):
                return self.module.disable_adapters()
        except (ValueError, Exception):
            pass

        # Fallback: 手动遍历模块设置 _disable_adapters 标志
        # peft LoRA forward 检查此标志来跳过 LoRA 分支，只走 base_layer
        from peft.tuners.tuners_utils import BaseTunerLayer

        @contextmanager
        def _manual_disable():
            disabled = []
            for m in self.module.modules():
                if isinstance(m, BaseTunerLayer):
                    if not getattr(m, '_disable_adapters', False):
                        m._disable_adapters = True
                        disabled.append(m)
            try:
                yield
            finally:
                for m in disabled:
                    m._disable_adapters = False

        return _manual_disable()"""

    patched = False
    for old in old_versions:
        if old in src:
            src = src.replace(old, new, 1)
            patched = True
            break

    if not patched and old_vanilla in src:
        src = src.replace(old_vanilla, new, 1)
        patched = True

    if not patched:
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


def patch_vllm_sleep_prefix_cache():
    """在 vLLM sleep(level=1) 后额外清理 prefix cache，防止多轮交互后 KV cache 残留"""
    import os
    target = "/tmp/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py"
    if not os.path.exists(target):
        print("SKIP: 找不到 vllm_async_server.py")
        return

    print(f"[Patch 5] 目标文件: {target}")

    with open(target, "r") as f:
        src = f.read()

    old = """        elif self.rollout_mode == RolloutMode.COLOCATED:
            await self.engine.sleep(level=1)
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")"""

    new = """        elif self.rollout_mode == RolloutMode.COLOCATED:
            await self.engine.sleep(level=1)
            # [Patch 5] 额外清理 prefix cache，防止多轮交互后 KV cache 残留
            await self.engine.reset_prefix_cache()
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")"""

    if new in src:
        print("[Patch 5] 已打过 patch，跳过")
        return

    if old not in src:
        print("[Patch 5] ERROR: 找不到目标代码（可能已被修改或版本不兼容）")
        sys.exit(1)

    src = src.replace(old, new, 1)

    with open(target, "w") as f:
        f.write(src)

    print("[Patch 5] 成功！")


def patch_ray_trainer_memory_cleanup():
    """在训练循环每步末尾加显存回收 + GPU 显存指标日志"""
    import os
    target = "/tmp/verl/verl/trainer/ppo/ray_trainer.py"
    if not os.path.exists(target):
        print("SKIP: 找不到 ray_trainer.py")
        return

    print(f"[Patch 6] 目标文件: {target}")

    with open(target, "r") as f:
        src = f.read()

    # Part 1: 添加 import gc
    if "import gc\n" in src:
        print("[Patch 6] Part 1: import gc 已存在，跳过")
    else:
        old_import = "import torch\n"
        new_import = "import gc\nimport torch\n"
        if old_import not in src:
            print("[Patch 6] ERROR: 找不到 'import torch' 行")
            sys.exit(1)
        src = src.replace(old_import, new_import, 1)
        print("[Patch 6] Part 1: 添加 import gc 成功")

    # Part 2: 在 logger.log 之后、progress_bar.update 之前插入显存回收 + 日志
    old_block = """                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1"""

    new_block = """                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                # [Patch 6] 显式释放当前 step 的 batch 和中间变量，防止引用残留
                del batch, gen_batch_output
                gc.collect()

                progress_bar.update(1)
                self.global_steps += 1"""

    if "del batch, gen_batch_output" in src:
        print("[Patch 6] Part 2: 显存回收代码已存在，跳过")
    elif old_block not in src:
        print("[Patch 6] ERROR: 找不到目标代码块（logger.log / progress_bar）")
        sys.exit(1)
    else:
        src = src.replace(old_block, new_block, 1)
        print("[Patch 6] Part 2: 添加显存回收代码成功")

    # Part 3: 在 metrics 收集阶段添加 GPU 显存指标（在 compute_throughout_metrics 之后）
    old_metrics = "                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))"
    new_metrics = """                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # [Patch 6] 记录 GPU 显存指标，用于 wandb 排查显存泄漏
                if torch.cuda.is_available():
                    metrics["gpu/allocated_gib"] = round(torch.cuda.memory_allocated() / (1024**3), 2)
                    metrics["gpu/reserved_gib"] = round(torch.cuda.memory_reserved() / (1024**3), 2)
                    metrics["gpu/max_allocated_gib"] = round(torch.cuda.max_memory_allocated() / (1024**3), 2)"""

    if "gpu/allocated_gib" in src:
        print("[Patch 6] Part 3: GPU 显存指标已存在，跳过")
    elif old_metrics not in src:
        print("[Patch 6] Part 3: WARN: 找不到 compute_throughout_metrics，跳过 GPU 指标")
    else:
        src = src.replace(old_metrics, new_metrics, 1)
        print("[Patch 6] Part 3: 添加 GPU 显存指标成功")

    with open(target, "w") as f:
        f.write(src)

    print("[Patch 6] 成功！")


def patch_ray_trainer_aggressive_cleanup():
    """在 _update_actor 和 update_weights 后加 gc.collect()，释放 Python 层引用残留

    注意：不能在 ray_trainer.py 中调用 torch.cuda.ipc_collect() 或 torch.cuda.empty_cache()，
    因为 fit() 运行在 TaskRunner 进程（ray.remote(num_cpus=1)，无 GPU 访问权限）。
    IPC handles 在 WorkerDict 和 vLLM Worker 进程之间共享，TaskRunner 不持有任何 IPC handles。
    """
    import os
    target = "/tmp/verl/verl/trainer/ppo/ray_trainer.py"
    if not os.path.exists(target):
        print("SKIP: 找不到 ray_trainer.py")
        return

    print(f"[Patch 7] 目标文件: {target}")

    with open(target, "r") as f:
        src = f.read()

    if "[Patch 7]" in src:
        print("[Patch 7] 已打过 patch，跳过")
        return

    changed = False

    # 1. _update_actor 后加 gc.collect()（Python 层清理）
    old_actor = """                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration."""
    new_actor = """                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        # [Patch 7] gc.collect() 释放 actor 训练后 Python 层残留引用
                        gc.collect()

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration."""

    if old_actor in src:
        src = src.replace(old_actor, new_actor, 1)
        changed = True
        print("[Patch 7] Part 1: _update_actor 后清理 成功")
    else:
        print("[Patch 7] Part 1: WARN: 找不到 _update_actor 代码块")

    # 2. update_weights 后加 gc.collect()（正常路径）
    old_uw = """                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])"""
    new_uw = """                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)
                        # [Patch 7] gc.collect() 释放权重传输协调中的 Python 层引用
                        gc.collect()

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])"""

    if old_uw in src:
        src = src.replace(old_uw, new_uw, 1)
        changed = True
        print("[Patch 7] Part 2: update_weights 后清理 成功")
    else:
        print("[Patch 7] Part 2: WARN: 找不到 update_weights 代码块")

    # 3. update_weights 后加 gc.collect()（critic warmup 路径）
    old_warmup = """                        # Still in critic warmup, only update weights to wake up rollout replicas.
                        self.checkpoint_manager.update_weights(self.global_steps)"""
    new_warmup = """                        # Still in critic warmup, only update weights to wake up rollout replicas.
                        self.checkpoint_manager.update_weights(self.global_steps)
                        # [Patch 7] gc.collect()（warmup 路径）
                        gc.collect()"""

    if old_warmup in src:
        src = src.replace(old_warmup, new_warmup, 1)
        changed = True
        print("[Patch 7] Part 3: warmup update_weights 后清理 成功")
    else:
        print("[Patch 7] Part 3: WARN: 找不到 warmup 代码块")

    if not changed:
        print("[Patch 7] ERROR: 没有成功应用任何修改")
        return

    with open(target, "w") as f:
        f.write(src)

    print("[Patch 7] 成功！")


def patch_weight_transfer_cleanup():
    """在 bucketed_weight_transfer.py 每处理完一个 bucket 后清理 clone 副本"""
    import os
    target = "/tmp/verl/verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py"
    if not os.path.exists(target):
        print("SKIP: 找不到 bucketed_weight_transfer.py")
        return

    print(f"[Patch 8] 目标文件: {target}")

    with open(target, "r") as f:
        src = f.read()

    if "[Patch 8]" in src:
        print("[Patch 8] 已打过 patch，跳过")
        return

    old = """                on_bucket_received(weights)
                del weights, tensor
                if metadata["is_last"]:"""
    new = """                on_bucket_received(weights)
                del weights, tensor
                # [Patch 8] 每处理完一个 bucket 后清理 clone 副本残留，防止累积
                get_torch_device().empty_cache()
                if metadata["is_last"]:"""

    if old not in src:
        print("[Patch 8] ERROR: 找不到目标代码块")
        return

    src = src.replace(old, new, 1)

    with open(target, "w") as f:
        f.write(src)

    print("[Patch 8] 成功！")


def patch_update_actor_empty_cache():
    """在 WorkerDict 的 update_actor 结束后加 aggressive_empty_cache()

    核心问题：
      veRL 的 update_actor() 在 WorkerDict 进程（有 GPU）中执行训练，
      训练完成后 PyTorch 缓存了梯度/激活值不释放。
      而后续的 update_weights 需要在 vLLM Worker 中 tensor.clone()，
      此时 GPU 已被 cumem(~64G) + PyTorch 缓存(~25G) 占满 → OOM。

      ray_trainer.py 的 Patch 7 gc.collect() 运行在 TaskRunner（无 GPU），
      对 WorkerDict 的显存完全无效！必须在 WorkerDict 进程内清理。

    对照：generate_sequences() 在第 1120 行已经有 empty_cache()，
      但 update_actor() 从未调用过任何清理。
    """
    import os
    target = "/tmp/verl/verl/workers/fsdp_workers.py"
    if not os.path.exists(target):
        print("SKIP: 找不到 fsdp_workers.py")
        return

    print(f"[Patch 9] 目标文件: {target}")

    with open(target, "r") as f:
        src = f.read()

    if "[Patch 9]" in src:
        print("[Patch 9] 已打过 patch，跳过")
        return

    # 在 update_actor 的 return output 之前插入 empty_cache
    old = """        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return output"""

    new = """        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        # [Patch 9] 训练后释放 PyTorch 缓存的梯度/激活值，
        # 为后续 update_weights 的 tensor.clone() 腾出空间。
        # 必须在 WorkerDict 进程内调用（ray_trainer.py 的 gc.collect 在 TaskRunner 无效）
        aggressive_empty_cache(force_sync=True)

        return output"""

    if old not in src:
        print("[Patch 9] ERROR: 找不到目标代码块")
        return

    src = src.replace(old, new, 1)

    with open(target, "w") as f:
        f.write(src)

    print("[Patch 9] 成功！")


if __name__ == "__main__":
    import os
    patch_qwen2()
    patch_numpy_indexing()
    patch_verl_disable_adapter()
    patch_numpy_core()
    patch_vllm_sleep_prefix_cache()
    patch_ray_trainer_memory_cleanup()
    patch_ray_trainer_aggressive_cleanup()
    patch_weight_transfer_cleanup()
    patch_update_actor_empty_cache()
