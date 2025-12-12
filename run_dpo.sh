#!/bin/bash
# DPO对齐训练脚本

set -e

# ==================== GPU配置 ====================
export CUDA_VISIBLE_DEVICES=5

# ==================== HuggingFace镜像源 ====================
export HF_ENDPOINT=https://hf-mirror.com

# ==================== 配置 ====================
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
SFT_MODEL="./outputs/qwen-math-lora"  # SFT微调后的模型路径
OUTPUT_DIR="./outputs/qwen-math-dpo"
DATA_PATH="./data/preference_data.json"
EPOCHS=3
BATCH_SIZE=2
LEARNING_RATE=5e-5
BETA=0.1

# ==================== 开始训练 ====================
echo "=========================================="
echo "DPO 对齐训练"
echo "=========================================="
echo "基础模型: $MODEL_NAME"
echo "SFT模型: $SFT_MODEL"
echo "输出目录: $OUTPUT_DIR"
echo "Beta (KL惩罚): $BETA"
echo "=========================================="
echo ""

python src/train_dpo.py \
    --base_model "$MODEL_NAME" \
    --sft_model "$SFT_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --data_path "$DATA_PATH" \
    --beta $BETA \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE

echo ""
echo "=========================================="
echo "DPO训练完成！"
echo "模型已保存至: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "启动TensorBoard查看训练日志："
echo "  tensorboard --logdir=$OUTPUT_DIR/logs"
echo ""