"""
DeepSeek V3.2 评估脚本
支持长序列推理和特殊评估指标
"""

import os
import json
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

from nanochat.deepseek_model import DeepSeekModel, create_deepseek_model
from nanochat.deepseek_config import get_deepseek_config
from nanochat.tokenizer import get_tokenizer
from nanochat.common import print0, autodetect_device_type
from nanochat.engine import Engine


def evaluate_deepseek_model(model_path: str, 
                           model_size: str = "medium",
                           max_seq_len: int = 16384,
                           eval_tokens: int = 1000000,
                           device_type: str = "") -> Dict[str, Any]:
    """
    评估 DeepSeek V3.2 模型
    
    Args:
        model_path: 模型检查点路径
        model_size: 模型规模
        max_seq_len: 最大序列长度
        eval_tokens: 评估token数
        device_type: 设备类型
        
    Returns:
        评估结果
    """
    
    print0("开始 DeepSeek V3.2 模型评估...")
    
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
    if os.path.isdir(model_path):
        # 从检查点目录加载最新模型
        checkpoint_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
        if not checkpoint_files:
            raise FileNotFoundError(f"在 {model_path} 中没有找到检查点文件")
        
        # 按步数排序，选择最新的
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_checkpoint = os.path.join(model_path, checkpoint_files[-1])
        
        print0(f"加载最新检查点: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
    else:
        # 直接加载模型文件
        checkpoint = torch.load(model_path, map_location=device)
    
    # 加载模型权重
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print0(f"模型已加载，参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 评估结果字典
    results = {}
    
    # 1. 计算困惑度 (Perplexity)
    print0("计算困惑度...")
    perplexity = evaluate_perplexity(model, tokenizer, device, eval_tokens, max_seq_len)
    results["perplexity"] = perplexity
    print0(f"困惑度: {perplexity:.4f}")
    
    # 2. 计算 bits per byte (BPB)
    print0("计算 bits per byte...")
    bpb = evaluate_bpb(model, tokenizer, device, eval_tokens, max_seq_len)
    results["bpb"] = bpb
    print0(f"Bits per byte: {bpb:.4f}")
    
    # 3. 长序列推理测试
    print0("测试长序列推理...")
    long_seq_results = evaluate_long_sequence(model, tokenizer, device, max_seq_len)
    results["long_sequence"] = long_seq_results
    
    # 4. 文本生成质量评估
    print0("评估文本生成质量...")
    generation_results = evaluate_generation(model, tokenizer, device)
    results["generation"] = generation_results
    
    # 5. 内存使用评估
    print0("评估内存使用...")
    memory_results = evaluate_memory_usage(model, device, max_seq_len)
    results["memory"] = memory_results
    
    # 6. 推理速度评估
    print0("评估推理速度...")
    speed_results = evaluate_inference_speed(model, device, max_seq_len)
    results["speed"] = speed_results
    
    print0("DeepSeek V3.2 模型评估完成!")
    
    return results


def evaluate_perplexity(model: DeepSeekModel, tokenizer, device: torch.device, 
                       eval_tokens: int, max_seq_len: int) -> float:
    """计算困惑度"""
    
    # 使用简单文本进行测试
    test_texts = [
        "DeepSeek V3.2 是一个先进的大型语言模型，",
        "人工智能技术正在快速发展，",
        "机器学习模型需要大量的数据进行训练，",
        "自然语言处理是人工智能的重要分支，",
        "深度学习技术已经在多个领域取得突破性进展，"
    ]
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in test_texts:
            # 分词
            tokens = tokenizer(text, prepend="<|bos|>")
            
            if len(tokens) >= max_seq_len:
                tokens = tokens[:max_seq_len-1]
            
            # 准备输入和目标
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            targets = torch.tensor([tokens[1:] + [tokenizer.eos_token_id]], dtype=torch.long, device=device)
            
            # 前向传播
            loss = model(input_ids, targets)
            
            total_loss += loss.item() * len(tokens)
            total_tokens += len(tokens)
            
            if total_tokens >= eval_tokens:
                break
    
    # 计算平均损失和困惑度
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity


def evaluate_bpb(model: DeepSeekModel, tokenizer, device: torch.device, 
                eval_tokens: int, max_seq_len: int) -> float:
    """计算 bits per byte"""
    
    # BPB = (交叉熵损失) / log(2)
    test_texts = [
        "The quick brown fox jumps over the lazy dog. " * 10,
        "Machine learning is a subset of artificial intelligence. " * 10,
        "Deep learning models require large amounts of data. " * 10,
    ]
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in test_texts:
            tokens = tokenizer(text, prepend="<|bos|>")
            
            if len(tokens) >= max_seq_len:
                tokens = tokens[:max_seq_len-1]
            
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            targets = torch.tensor([tokens[1:] + [tokenizer.eos_token_id]], dtype=torch.long, device=device)
            
            loss = model(input_ids, targets)
            
            total_loss += loss.item() * len(tokens)
            total_tokens += len(tokens)
            
            if total_tokens >= eval_tokens:
                break
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    bpb = avg_loss / math.log(2)  # 转换为 bits per byte
    
    return bpb


def evaluate_long_sequence(model: DeepSeekModel, tokenizer, device: torch.device, 
                          max_seq_len: int) -> Dict[str, Any]:
    """评估长序列处理能力"""
    
    results = {}
    
    # 测试不同序列长度
    seq_lengths = [1024, 4096, 8192, 16384, 32768]
    seq_lengths = [l for l in seq_lengths if l <= max_seq_len]
    
    memory_usage = []
    inference_time = []
    
    for seq_len in seq_lengths:
        print0(f"测试序列长度: {seq_len}")
        
        # 生成测试序列
        test_tokens = torch.randint(0, tokenizer.get_vocab_size() - 100, (1, seq_len), device=device)
        
        # 测量内存使用
        torch.cuda.empty_cache() if device.type == "cuda" else None
        
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        
        start_memory = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
        
        # 测量推理时间
        start_time = time.time()
        
        with torch.no_grad():
            logits = model(test_tokens)
        
        end_time = time.time()
        
        if device.type == "cuda":
            end_memory = torch.cuda.max_memory_allocated(device)
            memory_used = (end_memory - start_memory) / 1024 / 1024  # MB
        else:
            memory_used = 0
        
        inference_duration = end_time - start_time
        tokens_per_second = seq_len / inference_duration
        
        memory_usage.append({"seq_len": seq_len, "memory_mb": memory_used})
        inference_time.append({"seq_len": seq_len, "tokens_per_second": tokens_per_second})
        
        print0(f"  内存使用: {memory_used:.2f} MB")
        print0(f"  推理速度: {tokens_per_second:.2f} tokens/秒")
    
    results["memory_usage"] = memory_usage
    results["inference_speed"] = inference_time
    
    return results


def evaluate_generation(model: DeepSeekModel, tokenizer, device: torch.device) -> Dict[str, Any]:
    """评估文本生成质量"""
    
    results = {}
    generations = []
    
    prompts = [
        "DeepSeek V3.2 的主要特点包括",
        "人工智能的未来发展方向是",
        "机器学习的核心算法有",
        "自然语言处理技术的应用包括",
        "深度学习模型的训练需要"
    ]
    
    engine = Engine(model, tokenizer)
    
    for prompt in prompts:
        tokens = tokenizer(prompt, prepend="<|bos|>")
        
        with torch.no_grad():
            generated, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=50, temperature=0.7)
        
        generated_text = tokenizer.decode(generated[0])
        generations.append({"prompt": prompt, "generation": generated_text})
        
        print0(f"提示: {prompt}")
        print0(f"生成: {generated_text}")
        print0("-" * 50)
    
    results["generations"] = generations
    
    # 简单的连贯性评分（基于生成长度和重复性）
    coherence_scores = []
    for gen in generations:
        text = gen["generation"]
        
        # 简单的评分逻辑
        length_score = min(len(text) / 100, 1.0)  # 长度分数
        
        # 检查重复性（简单的重复检测）
        words = text.split()
        unique_words = set(words)
        repetition_score = len(unique_words) / len(words) if words else 1.0
        
        coherence_score = (length_score + repetition_score) / 2
        coherence_scores.append(coherence_score)
    
    results["avg_coherence_score"] = sum(coherence_scores) / len(coherence_scores)
    
    return results


def evaluate_memory_usage(model: DeepSeekModel, device: torch.device, max_seq_len: int) -> Dict[str, Any]:
    """评估内存使用情况"""
    
    results = {}
    
    if device.type == "cuda":
        # 测量模型参数内存
        param_memory = sum(p.element_size() * p.nelement() for p in model.parameters()) / 1024 / 1024
        
        # 测量激活内存（近似）
        batch_size = 1
        hidden_size = model.config.n_embd
        
        # 近似激活内存计算
        activation_memory = (batch_size * max_seq_len * hidden_size * 2 * 4) / 1024 / 1024  # 近似值
        
        total_memory = param_memory + activation_memory
        
        results["parameter_memory_mb"] = param_memory
        results["activation_memory_mb"] = activation_memory
        results["total_memory_mb"] = total_memory
        
        print0(f"参数内存: {param_memory:.2f} MB")
        print0(f"激活内存: {activation_memory:.2f} MB")
        print0(f"总内存估计: {total_memory:.2f} MB")
    
    return results


def evaluate_inference_speed(model: DeepSeekModel, device: torch.device, max_seq_len: int) -> Dict[str, Any]:
    """评估推理速度"""
    
    results = {}
    
    # 预热
    warmup_tokens = torch.randint(0, 1000, (1, 128), device=device)
    with torch.no_grad():
        _ = model(warmup_tokens)
    
    # 测试不同批大小的速度
    batch_sizes = [1, 2, 4, 8]
    seq_len = min(1024, max_seq_len)  # 使用较短的序列进行速度测试
    
    speeds = []
    
    for batch_size in batch_sizes:
        if batch_size * seq_len > max_seq_len:
            continue
        
        test_tokens = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        
        # 测量推理时间
        start_time = time.time()
        
        with torch.no_grad():
            _ = model(test_tokens)
        
        end_time = time.time()
        
        duration = end_time - start_time
        tokens_per_second = (batch_size * seq_len) / duration
        
        speeds.append({"batch_size": batch_size, "tokens_per_second": tokens_per_second})
        
        print0(f"批大小 {batch_size}: {tokens_per_second:.2f} tokens/秒")
    
    results["inference_speeds"] = speeds
    
    return results


def save_evaluation_results(results: Dict[str, Any], output_path: str):
    """保存评估结果"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print0(f"评估结果已保存: {output_path}")


def print_evaluation_summary(results: Dict[str, Any]):
    """打印评估总结"""
    
    print0("\n" + "="*60)
    print0("DeepSeek V3.2 模型评估总结")
    print0("="*60)
    
    print0(f"困惑度 (Perplexity): {results.get('perplexity', 'N/A'):.4f}")
    print0(f"Bits per Byte (BPB): {results.get('bpb', 'N/A'):.4f}")
    print0(f"平均连贯性评分: {results.get('generation', {}).get('avg_coherence_score', 'N/A'):.4f}")
    
    if "memory" in results:
        mem = results["memory"]
        print0(f"参数内存: {mem.get('parameter_memory_mb', 'N/A'):.2f} MB")
        print0(f"总内存估计: {mem.get('total_memory_mb', 'N/A'):.2f} MB")
    
    if "speed" in results:
        speeds = results["speed"].get("inference_speeds", [])
        if speeds:
            fastest = max(speeds, key=lambda x: x["tokens_per_second"])
            print0(f"最快推理速度: {fastest['tokens_per_second']:.2f} tokens/秒 (批大小: {fastest['batch_size']})")
    
    print0("="*60)


if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="DeepSeek V3.2 模型评估")
    parser.add_argument("--model-path", type=str, required=True,
                       help="模型检查点路径")
    parser.add_argument("--model-size", type=str, default="medium",
                       choices=["small", "medium", "large", "xlarge", "full"],
                       help="模型规模")
    parser.add_argument("--max-seq-len", type=int, default=16384,
                       help="最大序列长度")
    parser.add_argument("--eval-tokens", type=int, default=1000000,
                       help="评估token数")
    parser.add_argument("--device", type=str, default="",
                       help="设备类型 (cuda|cpu|mps)")
    parser.add_argument("--output", type=str, default="deepseek_eval_results.json",
                       help="输出结果文件路径")
    
    args = parser.parse_args()
    
    # 执行评估
    start_time = time.time()
    
    try:
        results = evaluate_deepseek_model(
            model_path=args.model_path,
            model_size=args.model_size,
            max_seq_len=args.max_seq_len,
            eval_tokens=args.eval_tokens,
            device_type=args.device
        )
        
        # 保存结果
        save_evaluation_results(results, args.output)
        
        # 打印总结
        print_evaluation_summary(results)
        
        elapsed_time = time.time() - start_time
        print0(f"评估总耗时: {elapsed_time:.2f} 秒")
        
    except Exception as e:
        print0(f"评估失败: {e}")
        import traceback
        traceback.print_exc()