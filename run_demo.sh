#!/bin/bash
# 系统演示运行脚本

set -e

# ==================== GPU配置 ====================
export CUDA_VISIBLE_DEVICES=3

# ==================== HuggingFace镜像源 ====================
export HF_ENDPOINT=https://hf-mirror.com

# ==================== 配置 ====================
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
LORA_PATH=""  # 可选：微调模型路径
MODE="cli"    # cli 或 web
PORT=7860

# ==================== 解析参数 ====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --lora)
            LORA_PATH="$2"
            shift 2
            ;;
        --web)
            MODE="web"
            shift
            ;;
        --cli)
            MODE="cli"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --share)
            SHARE="--share"
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# ==================== 开始演示 ====================
echo "=========================================="
echo "数学推理问答系统"
echo "=========================================="
echo "模型: $MODEL_NAME"
if [ -n "$LORA_PATH" ]; then
    echo "LoRA: $LORA_PATH"
fi
echo "模式: $MODE"
if [ "$MODE" == "web" ]; then
    echo "端口: $PORT"
fi
echo "=========================================="
echo ""

CMD="python src/demo.py \
    --model $MODEL_NAME \
    --mode $MODE"

if [ -n "$LORA_PATH" ]; then
    CMD="$CMD --lora_path $LORA_PATH"
fi

if [ "$MODE" == "web" ]; then
    CMD="$CMD --port $PORT $SHARE"
fi

eval $CMD
