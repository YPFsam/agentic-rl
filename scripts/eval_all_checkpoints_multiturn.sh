#!/bin/bash
# 顺序多轮评估所有 checkpoint
#
# Checkpoint 来源：
#   多轮训练（global 开头）: /root/autodl-tmp/checkpoints/agentic-rl-cloud/h20-formal-2turn-r4800/global_step_{100,150,200,250,300}_hf
#   单轮训练:                /root/autodl-tmp/hf_models/step{100,150,200,250,300}
#
# Tag 命名：
#   mt_step{N}  = 多轮训练的 checkpoint（global 开头）
#   st_step{N}  = 单轮训练的 checkpoint
#
# 基模 baseline 已评估，不重复跑（tag=baseline_evalplus, 102/164）
#
# 用法：
#   bash scripts/eval_all_checkpoints_multiturn.sh

set -e

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

MT_CKPT_DIR="/root/autodl-tmp/checkpoints/agentic-rl-cloud/h20-formal-2turn-r4800"
ST_CKPT_DIR="/root/autodl-tmp/hf_models"
OUTPUT_DIR="eval_results_v2"
STEPS=(100 150 200 250 300)

mkdir -p "$OUTPUT_DIR"

echo "===== 顺序多轮评估 10 个 checkpoint ====="
echo "多轮训练 (mt): global_step_*_hf in $MT_CKPT_DIR"
echo "单轮训练 (st): step* in $ST_CKPT_DIR"
echo "评估方式: 多轮推理 + evalplus sandbox 判卷"
echo "max_turns=3, max_response_length=4800"
echo ""

# ===== 1. 多轮训练 checkpoint（global 开头）=====
for step in "${STEPS[@]}"; do
    MODEL_PATH="$MT_CKPT_DIR/global_step_${step}_hf"
    TAG="mt_step${step}"

    if [ ! -f "$MODEL_PATH/config.json" ]; then
        echo "SKIP: $MODEL_PATH 不存在"
        continue
    fi

    echo ">>>>> 多轮训练 step${step} | tag=$TAG | path=$MODEL_PATH >>>>>"
    python scripts/evaluate_multiturn.py \
        --model_path "$MODEL_PATH" \
        --tag "$TAG" \
        --max_turns 3 \
        --max_response_length 4800 \
        --output_dir "$OUTPUT_DIR"
    echo "<<<<< 多轮训练 step${step} 完成 <<<<<"
    echo ""
done

# ===== 2. 单轮训练 checkpoint =====
for step in "${STEPS[@]}"; do
    MODEL_PATH="$ST_CKPT_DIR/step${step}"
    TAG="st_step${step}"

    if [ ! -f "$MODEL_PATH/config.json" ]; then
        echo "SKIP: $MODEL_PATH 不存在"
        continue
    fi

    echo ">>>>> 单轮训练 step${step} | tag=$TAG | path=$MODEL_PATH >>>>>"
    python scripts/evaluate_multiturn.py \
        --model_path "$MODEL_PATH" \
        --tag "$TAG" \
        --max_turns 3 \
        --max_response_length 4800 \
        --output_dir "$OUTPUT_DIR"
    echo "<<<<< 单轮训练 step${step} 完成 <<<<<"
    echo ""
done

echo "===== 全部 10 个 checkpoint 多轮评估完成 ====="
