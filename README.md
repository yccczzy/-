# 小学数学推理微调项目
大模型课程大作业，本项目实现了大语言模型在数学推理任务上的完整训练流程，包括：

1. **监督微调 (SFT)** - 使用LoRA在GSM8K数据集上微调
2. **DPO对齐** - 使用偏好数据进行直接偏好优化
3. **提示工程评估** - 对比CoT、ToT等多种提示策略
4. **系统演示** - 交互式数学问答界面

## 问题记录
问题记录1：国内连接不了HF下载模型

问题记录2：国内连接不了HF下载数据集
## 解决办法
解决办法1：借鉴https://blog.csdn.net/m0_61474277/article/details/140348032?spm=1001.2014.3001.5506
具体来说就是在虚拟环境的 “../miniconda3/envs/xxx(具体环境名字)/lib/python3.1/site-packages/huggingface_hub/constants.py”中找到：constants.py文件
将原来的默认网址：（在第65行）
_HF_DEFAULT_ENDPOINT = "https://huggingface.co"
修改为镜像网址：
_HF_DEFAULT_ENDPOINT = "https://hf-mirror.com"

解决办法2：在运行代码前加export HF_ENDPOINT=https://hf-mirror.com

## 项目结构

```
llm-math-finetune/
├── src/
│   ├── train.py              # SFT训练脚本
│   ├── train_dpo.py          # DPO对齐训练脚本
│   ├── evaluate.py           # 模型评估脚本
│   ├── evaluate_prompts.py   # 提示工程评估脚本
│   └── demo.py               # 交互式演示系统
├── configs/
│   └── config.yaml           # 配置文件
├── data/
│   ├── sample_chinese_math.json   # 示例中文数学数据
│   └── preference_data.json       # DPO偏好数据
├── run_train.sh             # SFT训练
├── run_dpo.sh               # DPO训练
├── run_eval.sh              # 模型评估
├── run_prompt_eval.sh       # 提示工程评估
├── run_demo.sh              # 系统演示
└── requirements.txt
```

## 环境配置

### 1. 创建虚拟环境

```bash
conda create -n llm python=3.10
conda activate llm
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 运行训练

```bash
# 使用默认配置
bash run_train.sh

# 或者直接运行Python脚本
python src/train.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --output_dir "./outputs/qwen-math-lora" \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4
```

### 2. DPO对齐训练

```bash
# 在SFT之后运行
bash run_dpo.sh
```

### 3. 提示工程评估

```bash
# 评估基础模型
bash run_prompt_eval.sh

# 评估微调后模型
bash run_prompt_eval.sh --lora ./outputs/qwen-math-lora --samples 200
```

### 4. 系统演示

```bash
# 命令行交互
bash run_demo.sh --cli

# Web界面
bash run_demo.sh --web --port 7860

# 使用微调模型
bash run_demo.sh --web --lora ./outputs/qwen-math-lora
```

## 配置说明

主要配置参数（在`configs/config.yaml`中）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| model.name | Qwen/Qwen2.5-7B-Instruct | 基础模型 |
| lora.r | 16 | LoRA秩 |
| lora.alpha | 32 | LoRA缩放因子 |
| training.num_epochs | 3 | 训练轮数 |
| training.batch_size | 4 | 批次大小 |
| training.learning_rate | 2e-4 | 学习率 |

## 数据集

默认使用[GSM8K](https://huggingface.co/datasets/openai/gsm8k)数据集：
- 训练集：7,473个样本
- 测试集：1,319个样本

如需使用自定义数据集，请准备JSON格式文件：

```json
[
  {
    "question": "问题文本",
    "answer": "包含解题步骤的答案"
  }
]
```

### 提示策略对比

| 策略 | 说明 |
|------|------|
| Zero-shot | 直接问答 |
| Few-shot | 给出示例 |
| CoT | 链式思维推理 |
| Zero-shot CoT | "让我们一步步思考" |
| Structured CoT | 结构化推理格式 |
| ToT | 树状思维（多角度） |

## 硬件要求

- GPU：建议16GB以上显存
- 本实验用RTX 5880 Ada 48GB显存可使用更大batch size


## 预期结果

在GSM8K数据集上：
- 基础模型：~60-65%准确率
- 微调后：~70-80%准确率（视训练轮数和数据而定）    
