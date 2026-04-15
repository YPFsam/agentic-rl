#!/bin/bash
# 通过 tmux 运行 H20 冒烟测试，训练日志 + GPU 显存日志分离记录
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="$LOG_DIR/smoke_h20_train_${TIMESTAMP}.log"
GPU_LOG="$LOG_DIR/smoke_h20_gpu_${TIMESTAMP}.log"

echo "=== H20 冒烟测试启动 ==="
echo "训练日志: $TRAIN_LOG"
echo "显存日志: $GPU_LOG"
echo "查看方式:"
echo "  tmux attach -t h20smoke     # 训练进程"
echo "  tmux attach -t h20gpu       # GPU 显存监控"
echo "  tail -f $TRAIN_LOG          # 训练日志"
echo "  tail -f $GPU_LOG            # 显存日志"
echo ""

# 会话 1：GPU 显存监控（每 5 秒采样）
tmux new-session -d -s h20gpu -x 200 -y 50
tmux send-keys -t h20gpu "echo '=== H20 GPU 显存监控 开始: $(date) ===' | tee $GPU_LOG" Enter
tmux send-keys -t h20gpu "echo '时间,显存已用(MiB),显存总计(MiB),GPU利用率(%)' | tee -a $GPU_LOG" Enter
tmux send-keys -t h20gpu "nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -l 5 2>&1 | tee -a $GPU_LOG" Enter

# 会话 2：训练脚本
tmux new-session -d -s h20smoke -x 200 -y 50
tmux send-keys -t h20smoke "echo '=== H20 冒烟测试 开始: $(date) ===' | tee $TRAIN_LOG" Enter
tmux send-keys -t h20smoke "bash $PROJECT_DIR/scripts/run_smoke_test_h20.sh 2>&1 | tee -a $TRAIN_LOG" Enter

echo "已启动两个 tmux 会话，按 Ctrl+B 然后 D 可分离窗口"
