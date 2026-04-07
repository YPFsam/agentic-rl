"""
环境验证脚本 — 一键验证所有组件

在 Docker 容器内运行：
  python scripts/verify_env.py
"""
import sys

checks = []

# 1. Python 版本
py_ver = sys.version_info
checks.append(("Python >= 3.10", py_ver >= (3, 10)))

# 2. PyTorch + CUDA
try:
    import torch
    checks.append(("PyTorch installed", True))
    checks.append(("CUDA available", torch.cuda.is_available()))
    checks.append(("GPU detected", torch.cuda.device_count() > 0))
    if torch.cuda.is_available():
        checks.append(("VRAM >= 10GB",
                        torch.cuda.get_device_properties(0).total_memory >= 10 * 1024**3))
except ImportError:
    checks.append(("PyTorch installed", False))

# 3. veRL
try:
    import verl
    checks.append(("veRL installed", True))
except ImportError:
    checks.append(("veRL installed", False))

# 4. datasets + pyarrow
try:
    import datasets, pyarrow
    checks.append(("datasets + pyarrow", True))
except ImportError:
    checks.append(("datasets + pyarrow", False))

# 5. evalplus
try:
    import evalplus
    checks.append(("evalplus installed", True))
except ImportError:
    checks.append(("evalplus installed", False))

# 输出结果
print("=" * 50)
print("环境验证结果")
print("=" * 50)
all_pass = True
for name, passed in checks:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    if not passed:
        all_pass = False
print("=" * 50)
if all_pass:
    print("全部通过！可以开始阶段零。")
else:
    print("有项目未通过，请检查安装。")
