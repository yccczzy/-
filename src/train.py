#!/usr/bin/env python3
"""
小学数学推理微调训练脚本
使用LoRA对Qwen2.5进行高效微调
"""

import os
import json
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


# ==================== 配置 ====================
@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_length: int = 512
    use_4bit: bool = False 


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class TrainConfig:
    output_dir: str = "./outputs/qwen-math-lora"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    fp16: bool = True
    seed: int = 42


# ==================== 数据处理 ====================
SYSTEM_PROMPT = """你是一个专业的小学数学老师。请仔细阅读题目，一步一步地思考和解答问题。在解答过程中：
1. 首先理解题目要求
2. 列出解题步骤
3. 进行计算
4. 给出最终答案"""

def format_example(example: dict) -> str:
    """将数据格式化为对话格式"""
    question = example.get("question", example.get("problem", ""))
    answer = example.get("answer", example.get("solution", ""))
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    return messages


def preprocess_function(examples, tokenizer, max_length):
    """预处理数据集"""
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    
    for i in range(len(examples["question"])):
        example = {k: examples[k][i] for k in examples.keys()}
        messages = format_example(example)
        
        # 使用tokenizer的chat template
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # Tokenize
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        
        # 创建labels（只计算assistant回复的loss）
        labels = tokenized["input_ids"].copy()
        
        # 找到assistant回复的起始位置，之前的token设为-100
        assistant_start = text.find("assistant")
        if assistant_start != -1:
            # 找到assistant标记后的实际内容开始位置
            prefix = text[:assistant_start]
            prefix_tokens = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            for j in range(min(len(prefix_tokens), len(labels))):
                labels[j] = -100
        
        model_inputs["input_ids"].append(tokenized["input_ids"])
        model_inputs["attention_mask"].append(tokenized["attention_mask"])
        model_inputs["labels"].append(labels)
    
    return model_inputs


def load_math_dataset(data_path: Optional[str] = None):
    """加载数学数据集"""
    if data_path and os.path.exists(data_path):
        # 从本地加载
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Dataset.from_list(data)
    
    # 使用GSM8K数据集（经典小学数学数据集）
    print("从HuggingFace加载GSM8K数据集...")
    dataset = load_dataset("openai/gsm8k", "main")
    
    # 转换格式
    def convert_gsm8k(example):
        return {
            "question": example["question"],
            "answer": example["answer"]
        }
    
    train_dataset = dataset["train"].map(convert_gsm8k)
    test_dataset = dataset["test"].map(convert_gsm8k)
    
    return train_dataset, test_dataset


def create_sample_chinese_math_data():
    """创建示例中文数学数据"""
    samples = [
        {
            "question": "小明有5个苹果，小红给了他3个苹果，请问小明现在有多少个苹果？",
            "answer": "让我们一步一步来解决这个问题：\n\n1. 小明原来有5个苹果\n2. 小红又给了他3个苹果\n3. 所以总共是：5 + 3 = 8个苹果\n\n答：小明现在有8个苹果。"
        },
        {
            "question": "一个长方形的长是8厘米，宽是5厘米，求这个长方形的周长。",
            "answer": "让我们一步一步来解决这个问题：\n\n1. 长方形的周长公式是：周长 = (长 + 宽) × 2\n2. 已知长 = 8厘米，宽 = 5厘米\n3. 代入公式：周长 = (8 + 5) × 2 = 13 × 2 = 26厘米\n\n答：这个长方形的周长是26厘米。"
        },
        {
            "question": "学校图书馆有故事书240本，科技书比故事书少60本，科技书有多少本？",
            "answer": "让我们一步一步来解决这个问题：\n\n1. 故事书有240本\n2. 科技书比故事书少60本\n3. 所以科技书的数量是：240 - 60 = 180本\n\n答：科技书有180本。"
        },
        {
            "question": "一辆汽车每小时行驶60千米，从甲地到乙地用了3小时，甲地到乙地有多远？",
            "answer": "让我们一步一步来解决这个问题：\n\n1. 汽车的速度是每小时60千米\n2. 行驶时间是3小时\n3. 根据路程 = 速度 × 时间\n4. 甲地到乙地的距离 = 60 × 3 = 180千米\n\n答：甲地到乙地有180千米。"
        },
        {
            "question": "小华买了3本笔记本，每本8元，付给售货员50元，应找回多少钱？",
            "answer": "让我们一步一步来解决这个问题：\n\n1. 每本笔记本8元，买了3本\n2. 总共花费：8 × 3 = 24元\n3. 付给售货员50元\n4. 应找回：50 - 24 = 26元\n\n答：应找回26元。"
        },
    ]
    return samples


# ==================== 模型加载 ====================
def load_model_and_tokenizer(model_config: ModelConfig):
    """加载模型和tokenizer"""
    print(f"正在加载模型: {model_config.model_name}")
    
    # 量化配置（如果需要）
    bnb_config = None
    if model_config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    if model_config.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    model.config.use_cache = False
    
    return model, tokenizer


def setup_lora(model, lora_config: LoRAConfig):
    """配置LoRA"""
    config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    return model


# ==================== 训练 ====================
def train(
    model_config: ModelConfig,
    lora_config: LoRAConfig,
    train_config: TrainConfig,
    data_path: Optional[str] = None,
):
    """主训练函数"""
    # 设置随机种子
    torch.manual_seed(train_config.seed)
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(model_config)
    
    # 配置LoRA
    model = setup_lora(model, lora_config)
    
    # 加载数据
    train_dataset, eval_dataset = load_math_dataset(data_path)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(eval_dataset)}")
    
    # 预处理数据
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, model_config.max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="处理训练数据",
    )
    
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer, model_config.max_length),
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="处理评估数据",
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        num_train_epochs=train_config.num_train_epochs,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        warmup_ratio=train_config.warmup_ratio,
        logging_steps=train_config.logging_steps,
        save_steps=train_config.save_steps,
        eval_steps=train_config.eval_steps,
        eval_strategy="steps",
        save_total_limit=train_config.save_total_limit,
        fp16=train_config.fp16,
        seed=train_config.seed,
        report_to="tensorboard",  # 使用TensorBoard记录训练日志
        logging_dir=os.path.join(train_config.output_dir, "logs"),  # TensorBoard日志目录
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    # 数据收集器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("=" * 50)
    print("开始训练...")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    trainer.train()
    
    # 保存模型
    print("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained(train_config.output_dir)
    
    print("=" * 50)
    print("训练完成！")
    print(f"模型保存至: {train_config.output_dir}")
    print("=" * 50)
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description="小学数学推理微调")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen-math-lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    
    args = parser.parse_args()
    
    model_config = ModelConfig(
        model_name=args.model_name,
        max_length=args.max_length,
    )
    
    lora_config = LoRAConfig(r=args.lora_r)
    
    train_config = TrainConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    
    train(model_config, lora_config, train_config, args.data_path)


if __name__ == "__main__":
    main()