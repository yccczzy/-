#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) 对齐训练脚本
兼容新版trl库
"""

import os
import json
import argparse
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig


# ==================== 配置 ====================
@dataclass
class DPOTrainConfig:
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    sft_model_path: Optional[str] = None
    output_dir: str = "./outputs/qwen-math-dpo"
    
    # DPO参数
    beta: float = 0.1
    max_length: int = 512
    max_prompt_length: int = 256
    
    # 训练参数
    num_train_epochs: int = 3  # 增加到3个epoch
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    
    # LoRA参数
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


# ==================== 偏好数据 ====================
def create_math_preference_data():
    """创建数学推理偏好数据"""
    preference_data = [
        {
            "prompt": "请解答以下数学问题：小明有12个苹果，他给了小红5个，又买了3个，现在小明有多少个苹果？",
            "chosen": "让我一步一步解决这个问题：\n\n1. 小明原来有12个苹果\n2. 给了小红5个后：12 - 5 = 7个\n3. 又买了3个后：7 + 3 = 10个\n\n所以小明现在有10个苹果。\n\n答案：10",
            "rejected": "12 - 5 + 3 = 10，答案是10。"
        },
        {
            "prompt": "请解答以下数学问题：一个长方形的长是15厘米，宽是8厘米，求它的面积和周长。",
            "chosen": "让我分别计算面积和周长：\n\n计算面积：\n面积 = 长 × 宽\n面积 = 15 × 8 = 120平方厘米\n\n计算周长：\n周长 = (长 + 宽) × 2\n周长 = (15 + 8) × 2 = 23 × 2 = 46厘米\n\n答案：面积是120平方厘米，周长是46厘米。",
            "rejected": "面积=15*8=120，周长=(15+8)*2=46"
        },
        {
            "prompt": "请解答以下数学问题：商店有苹果45千克，卖出了一些后还剩18千克，卖出了多少千克？",
            "chosen": "让我来分析这个问题：\n\n已知条件：\n- 原来有苹果：45千克\n- 现在剩余：18千克\n\n求：卖出了多少千克？\n\n解题步骤：\n卖出的重量 = 原有重量 - 剩余重量\n卖出的重量 = 45 - 18 = 27千克\n\n答案：商店卖出了27千克苹果。",
            "rejected": "45-18=27千克"
        },
        {
            "prompt": "请解答以下数学问题：一辆汽车2小时行驶了160千米，照这样的速度，5小时能行驶多少千米？",
            "chosen": "让我用比例方法解决这个问题：\n\n第一步：计算汽车的速度\n速度 = 路程 ÷ 时间\n速度 = 160 ÷ 2 = 80千米/小时\n\n第二步：计算5小时行驶的路程\n路程 = 速度 × 时间\n路程 = 80 × 5 = 400千米\n\n答案：照这样的速度，5小时能行驶400千米。",
            "rejected": "160/2=80千米/小时，80*5=400千米"
        },
        {
            "prompt": "请解答以下数学问题：学校买来25盒粉笔，每盒40支，用去了560支，还剩多少支？",
            "chosen": "让我一步步解决这个问题：\n\n第一步：计算总共有多少支粉笔\n总数 = 盒数 × 每盒支数\n总数 = 25 × 40 = 1000支\n\n第二步：计算剩余多少支\n剩余 = 总数 - 用去的\n剩余 = 1000 - 560 = 440支\n\n答案：还剩440支粉笔。",
            "rejected": "25×40-560=1000-560=440支"
        },
        {
            "prompt": "请解答以下数学问题：果园里有桃树和梨树共420棵，桃树的棵数是梨树的2倍，桃树和梨树各有多少棵？",
            "chosen": "这是一个和倍问题，让我用设未知数的方法解决：\n\n设梨树有x棵，则桃树有2x棵\n\n根据题意：\nx + 2x = 420\n3x = 420\nx = 140\n\n所以：\n- 梨树：140棵\n- 桃树：2 × 140 = 280棵\n\n验证：140 + 280 = 420 ✓\n\n答案：桃树有280棵，梨树有140棵。",
            "rejected": "设梨树x棵，x+2x=420，x=140，梨树140棵，桃树280棵"
        },
        {
            "prompt": "请解答以下数学问题：小明骑自行车从家到学校，每分钟行200米，需要15分钟，如果每分钟行250米，需要几分钟？",
            "chosen": "这是一个路程一定的问题，让我来解决：\n\n第一步：计算家到学校的距离\n距离 = 速度 × 时间\n距离 = 200 × 15 = 3000米\n\n第二步：计算新速度下需要的时间\n时间 = 距离 ÷ 速度\n时间 = 3000 ÷ 250 = 12分钟\n\n答案：如果每分钟行250米，需要12分钟。",
            "rejected": "距离=200×15=3000米，时间=3000÷250=12分钟"
        },
        {
            "prompt": "请解答以下数学问题：一个三角形的面积是24平方厘米，底是8厘米，高是多少厘米？",
            "chosen": "让我用三角形面积公式来解决：\n\n三角形面积公式：\n面积 = 底 × 高 ÷ 2\n\n已知：\n- 面积 = 24平方厘米\n- 底 = 8厘米\n\n求高：\n根据公式变形：高 = 面积 × 2 ÷ 底\n高 = 24 × 2 ÷ 8\n高 = 48 ÷ 8\n高 = 6厘米\n\n验证：面积 = 8 × 6 ÷ 2 = 24平方厘米 ✓\n\n答案：这个三角形的高是6厘米。",
            "rejected": "24×2÷8=6厘米"
        },
    ]
    return preference_data


def load_preference_dataset(data_path: Optional[str] = None):
    """加载偏好数据集"""
    # 默认使用项目中的偏好数据文件
    if data_path is None:
        data_path = "./data/preference_data.json"
    
    if os.path.exists(data_path):
        print(f"从本地加载偏好数据: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        print("使用内置数学偏好数据...")
        data = create_math_preference_data()
    
    # 确保数据格式正确
    formatted_data = []
    for item in data:
        formatted_data.append({
            "prompt": str(item["prompt"]),
            "chosen": str(item["chosen"]),
            "rejected": str(item["rejected"]),
        })
    
    return Dataset.from_list(formatted_data)


# ==================== DPO训练 ====================
def train_dpo(config: DPOTrainConfig):
    """DPO训练主函数"""
    print("=" * 60)
    print("DPO 对齐训练")
    print("=" * 60)
    
    # 加载tokenizer
    print(f"加载Tokenizer: {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print(f"加载模型: {config.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,  # 使用bfloat16更稳定
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 如果有SFT模型，先加载SFT的LoRA权重并合并
    if config.sft_model_path and os.path.exists(config.sft_model_path):
        print(f"加载SFT LoRA权重: {config.sft_model_path}")
        model = PeftModel.from_pretrained(model, config.sft_model_path)
        model = model.merge_and_unload()
        print("SFT权重已合并")
    
    # 配置新的LoRA用于DPO
    print("配置LoRA...")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 加载参考模型
    print("加载参考模型...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 加载偏好数据
    print("加载偏好数据集...")
    train_dataset = load_preference_dataset()
    print(f"训练样本数: {len(train_dataset)}")
    
    # DPO配置
    training_args = DPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        beta=config.beta,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        bf16=True,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        report_to="tensorboard",
        logging_dir=os.path.join(config.output_dir, "logs"),
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )
    
    # 创建DPO Trainer
    print("创建DPO Trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    # 开始训练
    print("=" * 60)
    print("开始DPO训练...")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    trainer.train()
    
    # 保存模型
    print("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    print("=" * 60)
    print("DPO训练完成！")
    print(f"模型保存至: {config.output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="DPO对齐训练")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--sft_model", type=str, default=None, help="SFT微调后的LoRA路径")
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen-math-dpo")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    
    args = parser.parse_args()
    
    config = DPOTrainConfig(
        base_model=args.base_model,
        sft_model_path=args.sft_model,
        output_dir=args.output_dir,
        beta=args.beta,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    
    train_dpo(config)


if __name__ == "__main__":
    main()