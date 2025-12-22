"""
DeepSeek V3.2 演示脚本
用于快速测试和演示 DeepSeek V3.2 模型
"""

import torch
import argparse
from nanochat.deepseek_model import DeepSeekModel, create_deepseek_model
from nanochat.deepseek_config import get_deepseek_config
from nanochat.tokenizer import get_tokenizer
from nanochat.engine import Engine
from nanochat.common import print0, autodetect_device_type


def interactive_chat(model: DeepSeekModel, tokenizer, device: torch.device, 
                    max_tokens: int = 100, temperature: float = 0.7):
    """交互式聊天演示"""
    
    engine = Engine(model, tokenizer)
    
    print0("DeepSeek V3.2 交互式聊天模式")
    print0("输入 'quit' 退出，输入 'clear' 清空对话历史")
    print0("=" * 60)
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("\n你: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'clear':
                conversation_history = []
                print0("对话历史已清空")
                continue
            elif not user_input:
                continue
            
            # 构建对话上下文
            if conversation_history:
                context = "\n".join(conversation_history[-6:]) + "\n" + user_input
            else:
                context = user_input
            
            # 添加系统提示
            system_prompt = "你是一个有用的AI助手。请用中文回答用户的问题。"
            full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{context}\n<|assistant|>\n"
            
            # 生成回复
            tokens = tokenizer(full_prompt, prepend="<|bos|>")
            
            with torch.no_grad():
                generated, _ = engine.generate_batch(
                    tokens, 
                    num_samples=1, 
                    max_tokens=max_tokens, 
                    temperature=temperature
                )
            
            response = tokenizer.decode(generated[0])
            
            # 提取助手的回复（去掉提示部分）
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1].strip()
            
            print0(f"DeepSeek: {response}")
            
            # 更新对话历史
            conversation_history.append(f"用户: {user_input}")
            conversation_history.append(f"助手: {response}")
            
        except KeyboardInterrupt:
            print0("\n\n再见!")
            break
        except Exception as e:
            print0(f"错误: {e}")


def batch_generation(model: DeepSeekModel, tokenizer, device: torch.device, 
                    prompts: list, max_tokens: int = 50, temperature: float = 0.7):
    """批量生成演示"""
    
    engine = Engine(model, tokenizer)
    
    print0("DeepSeek V3.2 批量生成演示")
    print0("=" * 60)
    
    for i, prompt in enumerate(prompts):
        print0(f"\n提示 {i+1}: {prompt}")
        
        tokens = tokenizer(prompt, prepend="<|bos|>")
        
        with torch.no_grad():
            generated, _ = engine.generate_batch(
                tokens, 
                num_samples=1, 
                max_tokens=max_tokens, 
                temperature=temperature
            )
        
        response = tokenizer.decode(generated[0])
        print0(f"生成: {response}")
        print0("-" * 50)


def test_model_capabilities(model: DeepSeekModel, tokenizer, device: torch.device):
    """测试模型能力"""
    
    print0("DeepSeek V3.2 能力测试")
    print0("=" * 60)
    
    test_cases = [
        {
            "category": "知识问答",
            "prompts": [
                "中国的首都是哪里？",
                "Python是一种什么类型的编程语言？",
                "太阳系有多少颗行星？"
            ]
        },
        {
            "category": "代码生成", 
            "prompts": [
                "写一个Python函数计算斐波那契数列",
                "实现一个快速排序算法",
                "用JavaScript写一个简单的TODO应用"
            ]
        },
        {
            "category": "创意写作",
            "prompts": [
                "写一首关于人工智能的诗",
                "创作一个关于时间旅行的短故事开头",
                "描述未来城市的景象"
            ]
        },
        {
            "category": "逻辑推理",
            "prompts": [
                "如果所有猫都会爬树，而汤姆是一只猫，那么汤姆会爬树吗？",
                "有三个人，A说B在说谎，B说C在说谎，C说A和B都在说谎。谁在说真话？",
                "一个篮子里有苹果和橙子，苹果比橙子多3个，总共有15个水果。苹果和橙子各有多少个？"
            ]
        }
    ]
    
    engine = Engine(model, tokenizer)
    
    for test_case in test_cases:
        print0(f"\n{test_case['category']}:")
        print0("-" * 40)
        
        for prompt in test_case["prompts"]:
            print0(f"问题: {prompt}")
            
            tokens = tokenizer(prompt, prepend="<|bos|>")
            
            with torch.no_grad():
                generated, _ = engine.generate_batch(
                    tokens, 
                    num_samples=1, 
                    max_tokens=100, 
                    temperature=0.3  # 低温度以获得更确定的回答
                )
            
            response = tokenizer.decode(generated[0])
            print0(f"回答: {response}")
            print0()


def load_model(model_path: str, model_size: str = "medium", 
              max_seq_len: int = 4096, device_type: str = "") -> tuple:
    """加载模型和分词器"""
    
    # 设备检测
    device_type = autodetect_device_type() if device_type == "" else device_type
    device = torch.device(device_type)
    
    # 加载分词器
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    
    # 创建模型配置
    config = get_deepseek_config(model_size)
    config.vocab_size = vocab_size
    config.max_seq_len = max_seq_len
    
    # 根据设备调整配置
    if device_type != "cuda":
        config.dsa_enabled = False
        config.moe_enabled = False
    
    # 创建模型
    model = create_deepseek_model(config)
    model.to(device)
    
    # 加载模型权重
    try:
        if model_path:
            checkpoint = torch.load(model_path, map_location=device)
            
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            # 使用随机初始化的权重
            model.init_weights()
            print0("使用随机初始化的模型权重")
    except Exception as e:
        print0(f"无法加载模型权重: {e}")
        print0("使用随机初始化的模型权重")
        model.init_weights()
    
    model.eval()
    
    print0(f"模型已加载到 {device_type}")
    print0(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print0(f"最大序列长度: {max_seq_len}")
    print0(f"DSA启用: {config.dsa_enabled}")
    print0(f"MoE启用: {config.moe_enabled}")
    
    return model, tokenizer, device


def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description="DeepSeek V3.2 演示脚本")
    parser.add_argument("--model-path", type=str, default="",
                       help="模型检查点路径（可选，使用随机权重如果未提供）")
    parser.add_argument("--model-size", type=str, default="small",
                       choices=["small", "medium", "large", "xlarge", "full"],
                       help="模型规模")
    parser.add_argument("--max-seq-len", type=int, default=4096,
                       help="最大序列长度")
    parser.add_argument("--device", type=str, default="",
                       help="设备类型 (cuda|cpu|mps)")
    parser.add_argument("--mode", type=str, default="interactive",
                       choices=["interactive", "batch", "test"],
                       help="演示模式")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="生成的最大token数")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="生成温度")
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer, device = load_model(
        model_path=args.model_path,
        model_size=args.model_size,
        max_seq_len=args.max_seq_len,
        device_type=args.device
    )
    
    # 根据模式执行演示
    if args.mode == "interactive":
        interactive_chat(model, tokenizer, device, 
                        max_tokens=args.max_tokens, 
                        temperature=args.temperature)
    
    elif args.mode == "batch":
        prompts = [
            "深度学习的基本原理是",
            "人工智能的未来发展趋势",
            "如何学习机器学习",
            "自然语言处理的应用场景",
            "编程语言的选择标准"
        ]
        batch_generation(model, tokenizer, device, prompts,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature)
    
    elif args.mode == "test":
        test_model_capabilities(model, tokenizer, device)


if __name__ == "__main__":
    main()