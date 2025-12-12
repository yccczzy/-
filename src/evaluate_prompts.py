#!/usr/bin/env python3
"""
提示工程评估脚本
对比不同提示策略的效果：
- Zero-shot: 直接问答
- Few-shot: 给出示例
- CoT (Chain-of-Thought): 链式思维
- ToT (Tree-of-Thought): 树状思维
"""

import os
import re
import json
import argparse
from typing import Optional, List, Dict, Callable
from datetime import datetime
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm


# ==================== 提示模板 ====================

# 1. Zero-shot 直接问答
ZERO_SHOT_TEMPLATE = """请解答以下数学问题：

问题：{question}

答案："""


# 2. Few-shot 少样本学习
FEW_SHOT_TEMPLATE = """请解答以下数学问题。

示例1：
问题：小明有8个苹果，吃了3个，还剩几个？
答案：8 - 3 = 5个

示例2：
问题：一本书有120页，小红看了45页，还剩多少页？
答案：120 - 45 = 75页

现在请解答：
问题：{question}

答案："""


# 3. Chain-of-Thought (CoT) 链式思维
COT_TEMPLATE = """请一步一步地思考并解答以下数学问题。

问题：{question}

让我们逐步思考："""


# 4. Zero-shot CoT
ZERO_SHOT_COT_TEMPLATE = """问题：{question}

让我们一步一步地思考这个问题。"""


# 5. Tree-of-Thought (ToT) 树状思维
TOT_TEMPLATE = """请解答以下数学问题。使用以下方法：
1. 首先，考虑3种不同的解题思路
2. 评估每种思路的可行性
3. 选择最佳思路并执行
4. 验证答案

问题：{question}

思路1："""


# 6. Self-Consistency (多次采样取众数)
SELF_CONSISTENCY_TEMPLATE = COT_TEMPLATE  # 使用CoT模板，多次采样


# 7. 结构化CoT
STRUCTURED_COT_TEMPLATE = """请按照以下格式解答数学问题：

【问题分析】首先理解题目要求
【已知条件】列出题目给出的信息
【解题思路】说明解题方法
【计算过程】展示详细计算
【最终答案】给出答案

问题：{question}

【问题分析】"""


@dataclass
class PromptStrategy:
    name: str
    template: str
    description: str
    num_samples: int = 1  # 用于self-consistency
    temperature: float = 0.1


STRATEGIES = {
    "zero_shot": PromptStrategy(
        name="Zero-shot",
        template=ZERO_SHOT_TEMPLATE,
        description="直接问答，不给任何提示"
    ),
    "few_shot": PromptStrategy(
        name="Few-shot",
        template=FEW_SHOT_TEMPLATE,
        description="给出2个示例后再问答"
    ),
    "cot": PromptStrategy(
        name="Chain-of-Thought",
        template=COT_TEMPLATE,
        description="链式思维，一步步推理"
    ),
    "zero_shot_cot": PromptStrategy(
        name="Zero-shot CoT",
        template=ZERO_SHOT_COT_TEMPLATE,
        description="零样本链式思维"
    ),
    "tot": PromptStrategy(
        name="Tree-of-Thought",
        template=TOT_TEMPLATE,
        description="树状思维，多角度分析"
    ),
    "structured_cot": PromptStrategy(
        name="Structured CoT",
        template=STRUCTURED_COT_TEMPLATE,
        description="结构化链式思维"
    ),
    "self_consistency": PromptStrategy(
        name="Self-Consistency",
        template=SELF_CONSISTENCY_TEMPLATE,
        description="多次采样取众数",
        num_samples=5,
        temperature=0.7
    ),
}


# ==================== 答案提取 ====================
def extract_answer(text: str) -> Optional[str]:
    """从回答中提取最终数字答案"""
    patterns = [
        r"####\s*(\-?[\d,]+\.?\d*)",
        r"答[：:]\s*(\-?[\d,]+\.?\d*)",
        r"答案[是为：:]\s*(\-?[\d,]+\.?\d*)",
        r"最终答案[是为：:]\s*(\-?[\d,]+\.?\d*)",
        r"=\s*(\-?[\d,]+\.?\d*)\s*[。\n个千克米]",
        r"还剩[：:]?\s*(\-?[\d,]+\.?\d*)",
        r"共[有收]?\s*(\-?[\d,]+\.?\d*)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).replace(",", "")
    
    numbers = re.findall(r"\-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    
    return None


def normalize_answer(answer: str) -> str:
    """标准化答案"""
    if answer is None:
        return ""
    answer = answer.replace(",", "").strip()
    try:
        num = float(answer)
        if num == int(num):
            return str(int(num))
        return str(num)
    except:
        return answer


def majority_vote(answers: List[str]) -> str:
    """多数投票"""
    from collections import Counter
    valid_answers = [a for a in answers if a]
    if not valid_answers:
        return ""
    counter = Counter(valid_answers)
    return counter.most_common(1)[0][0]


# ==================== 评估器 ====================
class PromptEvaluator:
    """提示工程评估器"""
    
    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.device = device
        
        print(f"加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        if lora_path and os.path.exists(lora_path):
            print(f"加载LoRA权重: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        do_sample: bool = False,
    ) -> str:
        """生成回答"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        return response
    
    def evaluate_strategy(
        self,
        strategy: PromptStrategy,
        dataset,
        num_samples: Optional[int] = None,
    ) -> Dict:
        """评估单个策略"""
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        results = []
        correct = 0
        total = 0
        
        for item in tqdm(dataset, desc=f"评估 {strategy.name}"):
            question = item.get("question", item.get("problem", ""))
            gold_answer = item.get("answer", item.get("solution", ""))
            gold_extracted = normalize_answer(extract_answer(gold_answer))
            
            # 构建prompt
            prompt = strategy.template.format(question=question)
            
            if strategy.num_samples > 1:
                # Self-consistency: 多次采样
                pred_answers = []
                for _ in range(strategy.num_samples):
                    response = self.generate(
                        prompt, 
                        temperature=strategy.temperature,
                        do_sample=True
                    )
                    pred = normalize_answer(extract_answer(response))
                    pred_answers.append(pred)
                pred_extracted = majority_vote(pred_answers)
                response = f"[多次采样结果: {pred_answers}] -> 多数投票: {pred_extracted}"
            else:
                response = self.generate(prompt, temperature=strategy.temperature)
                pred_extracted = normalize_answer(extract_answer(response))
            
            is_correct = gold_extracted == pred_extracted
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                "question": question,
                "gold": gold_extracted,
                "prediction": pred_extracted,
                "response": response[:500],  # 截断保存
                "correct": is_correct,
            })
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            "strategy": strategy.name,
            "description": strategy.description,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "details": results,
        }
    
    def evaluate_all_strategies(
        self,
        dataset,
        strategies: List[str] = None,
        num_samples: int = 100,
        output_dir: str = "./outputs/prompt_eval",
    ) -> Dict:
        """评估所有策略"""
        if strategies is None:
            strategies = list(STRATEGIES.keys())
        
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = {}
        summary = []
        
        for strategy_name in strategies:
            if strategy_name not in STRATEGIES:
                print(f"未知策略: {strategy_name}, 跳过")
                continue
            
            strategy = STRATEGIES[strategy_name]
            print(f"\n{'='*60}")
            print(f"评估策略: {strategy.name}")
            print(f"描述: {strategy.description}")
            print("=" * 60)
            
            result = self.evaluate_strategy(strategy, dataset, num_samples)
            all_results[strategy_name] = result
            
            summary.append({
                "strategy": strategy.name,
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "total": result["total"],
            })
            
            # 保存单个策略结果
            with open(os.path.join(output_dir, f"{strategy_name}_results.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"准确率: {result['accuracy']:.2%}")
        
        # 打印汇总
        print("\n" + "=" * 60)
        print("策略对比汇总")
        print("=" * 60)
        print(f"{'策略':<25} {'准确率':<15} {'正确/总数':<15}")
        print("-" * 60)
        for s in sorted(summary, key=lambda x: x['accuracy'], reverse=True):
            print(f"{s['strategy']:<25} {s['accuracy']:.2%}          {s['correct']}/{s['total']}")
        print("=" * 60)
        
        # 保存汇总
        final_results = {
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
            "num_samples": num_samples,
            "details": {k: {**v, "details": v["details"][:10]} for k, v in all_results.items()}
        }
        
        with open(os.path.join(output_dir, "comparison_results.json"), "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存至: {output_dir}")
        
        return final_results


def load_test_dataset():
    """加载测试数据集"""
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="提示工程评估")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./outputs/prompt_eval")
    parser.add_argument("--strategies", type=str, nargs="+", 
                        default=["zero_shot", "few_shot", "cot", "zero_shot_cot", "structured_cot"],
                        help="要评估的策略列表")
    
    args = parser.parse_args()
    
    evaluator = PromptEvaluator(args.model, args.lora_path)
    dataset = load_test_dataset()
    
    evaluator.evaluate_all_strategies(
        dataset,
        strategies=args.strategies,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
