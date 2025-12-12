#!/bin/bash
# 小学数学推理微调训练脚本

set -e

# ==================== GPU配置 ====================
export CUDA_VISIBLE_DEVICES=3

# ==================== HuggingFace镜像源 ====================
export HF_ENDPOINT=https://hf-mirror.com

# ==================== 配置 ====================
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR="./outputs/qwen-math-lora"
EPOCHS=3
BATCH_SIZE=4
LEARNING_RATE=2e-4
LORA_R=16
MAX_LENGTH=512

# ==================== 环境检查 ====================
echo "=========================================="
echo "环境检查"
echo "=========================================="

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "GPU信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo "警告: 未检测到GPU"
fi

echo ""
echo "Python版本: $(python --version)"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# ==================== 开始训练 ====================
echo "=========================================="
echo "开始训练"
echo "=========================================="
echo "模型: $MODEL_NAME"
echo "输出目录: $OUTPUT_DIR"
echo "训练轮数: $EPOCHS"
echo "批次大小: $BATCH_SIZE"
echo "学习率: $LEARNING_RATE"
echo "LoRA秩: $LORA_R"
echo "=========================================="
echo ""

python src/train.py \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --lora_r $LORA_R \
    --max_length $MAX_LENGTH

echo ""
echo "=========================================="
echo "训练完成！"
echo "模型已保存至: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "启动TensorBoard查看训练日志："
echo "  tensorboard --logdir=$OUTPUT_DIR/logs"
echo ""