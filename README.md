# 小学数学推理微调项目
大模型课程大作业，本项目使用LoRA方法对Qwen2.5-7B-Instruct模型进行小学数学推理能力的微调。
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
│   ├── train.py          # 训练脚本
│   └── evaluate.py       # 评估脚本
├── configs/
│   └── config.yaml       # 配置文件
├── data/                 # 数据目录（可选）
├── outputs/              # 输出目录
├── requirements.txt      # Python依赖
├── run_train.sh         # 训练运行脚本
├── run_eval.sh          # 评估运行脚本
└── README.md
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

### 2. 运行评估

```bash
# 评估微调模型
bash run_eval.sh

# 对比基础模型和微调模型
bash run_eval.sh --compare

# 指定评估样本数
bash run_eval.sh --samples 500 --compare
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

## 硬件要求

- GPU：建议16GB以上显存（使用LoRA）

## 预期结果

在GSM8K数据集上：
- 基础模型：~60-65%准确率
- 微调后：~70-80%准确率（视训练轮数和数据而定）    
