#!/bin/bash
# 提示工程（CoT、ToT等）评估脚本

set -e

# ==================== GPU配置 ====================
export CUDA_VISIBLE_DEVICES=3

# ==================== HuggingFace镜像源 ====================
export HF_ENDPOINT=https://hf-mirror.com

# ==================== 配置 ====================
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
LORA_PATH=""
OUTPUT_DIR="./outputs/dpo"
NUM_SAMPLES=100

# 要评估的策略列表
STRATEGIES="zero_shot few_shot cot zero_shot_cot structured_cot"

# ==================== 解析参数 ====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --lora)
            LORA_PATH="$2"
            shift 2
            ;;
        --samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# ==================== 开始评估 ====================
echo "=========================================="
echo "提示工程评估"
echo "=========================================="
echo "模型: $MODEL_NAME"
if [ -n "$LORA_PATH" ]; then
    echo "LoRA: $LORA_PATH"
fi
echo "评估样本数: $NUM_SAMPLES"
echo "策略: $STRATEGIES"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="
echo ""

CMD="python src/evaluate_prompts.py \
    --model $MODEL_NAME \
    --num_samples $NUM_SAMPLES \
    --output_dir $OUTPUT_DIR \
    --strategies $STRATEGIES"

if [ -n "$LORA_PATH" ]; then
    CMD="$CMD --lora_path $LORA_PATH"
fi

eval $CMD

echo ""
echo "=========================================="
echo "评估完成！"
echo "结果已保存至: $OUTPUT_DIR"
echo "=========================================="
