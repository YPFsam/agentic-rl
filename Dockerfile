FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# 避免交互式安装卡住
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv \
    git curl wget tmux \
    && rm -rf /var/lib/apt/lists/*

# 设置 python3 为默认
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# PyTorch + CUDA 12.4
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# veRL 核心（从源码安装，确保最新 API）
RUN git clone https://github.com/volcengine/verl.git /tmp/verl && \
    cd /tmp/verl && \
    pip install --no-cache-dir -e . && \
    rm -rf /tmp/verl

# 项目依赖
RUN pip install --no-cache-dir \
    datasets>=3.0.0 \
    pyarrow>=15.0.0 \
    evalplus>=0.3.0 \
    modelscope>=1.14.0 \
    matplotlib>=3.8.0 \
    tensorboard>=2.15.0 \
    wandb>=0.17.0 \
    pyyaml>=6.0

# 设置工作目录
WORKDIR /workspace/agentic-rl

# 默认命令
CMD ["/bin/bash"]
