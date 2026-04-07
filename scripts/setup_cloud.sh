#!/bin/bash
# ============================================================
# 云端环境安装脚本（AutoDL 80GB GPU）
# ============================================================
set -e

echo "=========================================="
echo "Agentic-RL 云端环境安装（AutoDL）"
echo "=========================================="

# ===== 缓存重定向（系统盘只有 30GB，不做必爆）=====

export HF_HOME=/root/autodl-tmp/hf_cache
mkdir -p /root/autodl-tmp/hf_cache

export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache
mkdir -p /root/autodl-tmp/modelscope_cache

export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
mkdir -p /root/autodl-tmp/pip_cache

export TORCH_HOME=/root/autodl-tmp/torch_cache
mkdir -p /root/autodl-tmp/torch_cache

mkdir -p /root/autodl-tmp/outputs

# 写入 .bashrc 持久化
cat >> /root/.bashrc << 'EOF'
export HF_HOME=/root/autodl-tmp/hf_cache
export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache
export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
export TORCH_HOME=/root/autodl-tmp/torch_cache
EOF
source /root/.bashrc

echo "缓存重定向完成 ✓"

# ===== 安装依赖 =====

echo ""
echo "安装 veRL 和项目依赖..."

# 换源加速
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 安装 veRL（从源码安装最新版）
if ! python3 -c "import verl" &>/dev/null; then
    git clone https://github.com/volcengine/verl.git /tmp/verl
    cd /tmp/verl
    pip install -e .
    cd ~
    rm -rf /tmp/verl
    echo "veRL 安装完成 ✓"
else
    echo "veRL 已安装 ✓"
fi

# 安装项目依赖
pip install datasets>=3.0.0 pyarrow>=15.0.0 \
    evalplus>=0.3.0 modelscope>=1.14.0 \
    matplotlib tensorboard wandb pyyaml

echo "项目依赖安装完成 ✓"

# ===== 下载模型 =====

echo ""
echo "下载 Qwen3-4B 模型（ModelScope 内网加速）..."

python3 -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen3-4B', cache_dir='/root/autodl-tmp/models')
print('Qwen3-4B 下载完成')
"

# ===== 环境验证 =====

echo ""
echo "环境验证..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f'VRAM: {vram:.1f} GB')

import verl
print('veRL: OK')

import os
model_path = '/root/autodl-tmp/models/Qwen/Qwen3-4B'
if os.path.exists(model_path):
    print(f'模型已就绪: {model_path}')
else:
    print('警告：模型未找到，请检查下载路径')
"

echo ""
echo "=========================================="
echo "云端环境安装完成！"
echo ""
echo "后续步骤："
echo "  1. 同步代码：git clone <仓库> ~/agentic-rl"
echo "  2. 准备数据：cd ~/agentic-rl && python src/data_prepare.py --output data/grpo_train_full.parquet --full"
echo "  3. 启动训练：tmux new -s train && bash scripts/run_cloud_single.sh"
echo "=========================================="
