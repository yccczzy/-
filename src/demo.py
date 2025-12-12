import os
import argparse
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel

# 尝试导入gradio用于Web界面
try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False
    print("提示: 安装 gradio 可启用Web界面 (pip install gradio)")


# ==================== 推理模式 ====================
REASONING_MODES = {
    "standard": {
        "name": "标准模式",
        "system": "你是一个专业的数学老师，请直接解答问题。",
        "prefix": "",
    },
    "cot": {
        "name": "链式思维 (CoT)",
        "system": "你是一个专业的数学老师。请一步一步地思考，展示完整的推理过程。",
        "prefix": "让我们一步一步来思考这个问题：\n\n",
    },
    "detailed": {
        "name": "详细解析",
        "system": "你是一个专业的数学老师。请按照以下格式解答：\n1. 【题目分析】\n2. 【解题思路】\n3. 【计算过程】\n4. 【答案验证】",
        "prefix": "",
    },
    "simple": {
        "name": "简洁模式",
        "system": "你是数学老师，请简洁地解答问题，给出关键步骤和答案。",
        "prefix": "",
    },
}


class MathAssistant:
    """数学问答助手"""
    
    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.device = device
        
        print(f"正在加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
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
        print("模型加载完成！")
    
    def solve(
        self,
        question: str,
        mode: str = "cot",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        stream: bool = False,
    ) -> str:
        """解答数学问题"""
        if mode not in REASONING_MODES:
            mode = "cot"
        
        config = REASONING_MODES[mode]
        
        messages = [
            {"role": "system", "content": config["system"]},
            {"role": "user", "content": question},
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        if config["prefix"]:
            text += config["prefix"]
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        if stream:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        else:
            streamer = None
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                streamer=streamer,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        if config["prefix"] and not response.startswith(config["prefix"]):
            response = config["prefix"] + response
        
        return response


def run_cli(assistant: MathAssistant):
    """命令行交互界面"""
    print("\n" + "=" * 60)
    print("小学数学推理问答系统")
    print("=" * 60)
    print("\n可用模式:")
    for key, value in REASONING_MODES.items():
        print(f"  {key}: {value['name']}")
    print("\n命令:")
    print("  /mode <模式名>  - 切换推理模式")
    print("  /quit          - 退出")
    print("=" * 60)
    
    current_mode = "cot"
    print(f"\n当前模式: {REASONING_MODES[current_mode]['name']}")
    
    while True:
        try:
            user_input = input("\n📝 请输入数学问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        
        if not user_input:
            continue
        
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()
            
            if cmd == "/quit":
                print("再见！")
                break
            elif cmd == "/mode" and len(parts) > 1:
                new_mode = parts[1]
                if new_mode in REASONING_MODES:
                    current_mode = new_mode
                    print(f"✅ 已切换到: {REASONING_MODES[current_mode]['name']}")
                else:
                    print(f"❌ 未知模式: {new_mode}")
            else:
                print("❌ 未知命令")
            continue
        
        print(f"\n🤔 思考中... (模式: {REASONING_MODES[current_mode]['name']})\n")
        print("-" * 40)
        
        response = assistant.solve(user_input, mode=current_mode, stream=True)
        
        print("-" * 40)


def create_gradio_interface(assistant: MathAssistant):
    """创建Gradio Web界面"""
    
    def solve_problem(question, mode, temperature):
        if not question.strip():
            return "请输入数学问题"
        response = assistant.solve(
            question, 
            mode=mode, 
            temperature=temperature
        )
        return response
    
    # 模式选择列表
    mode_choices = [k for k in REASONING_MODES.keys()]
    
    # 使用 gr.Interface 更简洁、兼容性更好
    demo = gr.Interface(
        fn=solve_problem,
        inputs=[
            gr.Textbox(
                label="输入数学问题",
                placeholder="例如：小明有5个苹果，吃了2个，还剩几个？",
                lines=3,
            ),
            gr.Dropdown(
                choices=mode_choices,
                value="cot",
                label="推理模式 (cot=链式思维, detailed=详细解析, standard=标准, simple=简洁)",
            ),
            gr.Slider(
                minimum=0,
                maximum=1,
                value=0.1,
                step=0.1,
                label="温度 (创造性)",
            ),
        ],
        outputs=gr.Textbox(label="解答结果", lines=15),
        title="🧮 小学数学推理问答系统",
        description="基于大语言模型的数学问题解答助手，支持多种推理模式。",
        examples=[
            ["小明有12个苹果，给了小红5个，又买了8个，现在有多少个？", "cot", 0.1],
            ["一个长方形的长是15米，宽是8米，求周长和面积。", "detailed", 0.1],
            ["甲乙两人共有钱240元，甲的钱数是乙的2倍，两人各有多少钱？", "cot", 0.1],
        ],
        allow_flagging="never",
    )
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="数学推理问答系统")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["cli", "web"], default="cli",
                        help="运行模式: cli=命令行, web=网页界面")
    parser.add_argument("--port", type=int, default=7860, help="Web界面端口")
    parser.add_argument("--share", action="store_true", help="创建公共链接")
    
    args = parser.parse_args()
    
    assistant = MathAssistant(args.model, args.lora_path)
    
    if args.mode == "web":
        if not HAS_GRADIO:
            print("错误: 需要安装 gradio (pip install gradio)")
            return
        demo = create_gradio_interface(assistant)
        demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
    else:
        run_cli(assistant)


if __name__ == "__main__":
    main()