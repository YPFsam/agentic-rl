#!/bin/bash
# ============================================================
# 本地环境安装脚本（WSL2 + Docker Desktop）
# ============================================================
set -e

echo "=========================================="
echo "Agentic-RL 本地环境安装"
echo "=========================================="

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# 步骤 1：检查 Docker Desktop WSL 集成
echo ""
echo "[步骤 1/4] 检查 Docker Desktop WSL 集成..."
if ! docker info &>/dev/null; then
    echo "错误：Docker 未就绪。"
    echo "请在 Windows 端 Docker Desktop → Settings → Resources → WSL integration 中"
    echo "启用当前 WSL2 发行版（通常是 Ubuntu）。"
    echo "启用后重新运行此脚本。"
    exit 1
fi
echo "  Docker 已就绪 ✓"

# 步骤 2：检查 NVIDIA Container Toolkit
echo ""
echo "[步骤 2/4] 检查 NVIDIA Container Toolkit..."
if ! command -v nvidia-container-toolkit &>/dev/null && \
   ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo "安装 NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
      sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo "  NVIDIA Container Toolkit 安装完成 ✓"
else
    echo "  NVIDIA Container Toolkit 已安装 ✓"
fi

# 步骤 3：构建 Docker 镜像
echo ""
echo "[步骤 3/4] 构建 Docker 镜像（首次构建约需 20-30 分钟）..."
docker build -t agentic-rl:latest .
echo "  Docker 镜像构建完成 ✓"

# 步骤 4：验证 GPU 直通
echo ""
echo "[步骤 4/4] 验证 GPU 直通..."
docker run --rm --runtime=nvidia --gpus all agentic-rl:latest \
  python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f'VRAM: {vram:.1f} GB')
"

echo ""
echo "=========================================="
echo "本地环境安装完成！"
echo ""
echo "后续步骤："
echo "  1. 创建容器并挂载项目目录："
echo "     docker create --runtime=nvidia --gpus all \\"
echo "       --net=host --shm-size=\"10g\" \\"
echo "       -v $PROJECT_DIR:/workspace/agentic-rl \\"
echo "       --name agentic-rl agentic-rl:latest sleep infinity"
echo ""
echo "  2. 启动并进入容器："
echo "     docker start agentic-rl"
echo "     docker exec -it agentic-rl bash"
echo ""
echo "  3. 在容器内运行环境验证："
echo "     cd /workspace/agentic-rl"
echo "     python scripts/verify_env.py"
echo "=========================================="
