"""
DeepSeek V3.2 数据预处理脚本
支持长序列和特殊数据格式的处理
"""

import os
import json
import torch
from datasets import load_dataset, Dataset
from typing import List, Dict, Any

from nanochat.tokenizer import get_tokenizer


def prepare_deepseek_dataset(dataset_name: str = "wikitext", 
                           dataset_config: str = "wikitext-103-raw-v1",
                           max_seq_len: int = 16384,
                           output_dir: str = "deepseek_data") -> str:
    """
    准备 DeepSeek V3.2 训练数据集
    
    Args:
        dataset_name: 数据集名称
        dataset_config: 数据集配置
        max_seq_len: 最大序列长度
        output_dir: 输出目录
        
    Returns:
        处理后的数据目录路径
    """
    
    print(f"正在准备 DeepSeek V3.2 数据集: {dataset_name}/{dataset_config}")
    print(f"最大序列长度: {max_seq_len}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载分词器
    tokenizer = get_tokenizer()
    
    # 加载数据集
    try:
        dataset = load_dataset(dataset_name, dataset_config)
    except Exception as e:
        print(f"无法加载数据集 {dataset_name}: {e}")
        print("尝试使用默认的 wikitext 数据集...")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    
    def tokenize_function(examples):
        """分词函数"""
        # 连接文本并分词
        text = " ".join(examples["text"])
        tokens = tokenizer(text, prepend="<|bos|>")
        return {"tokens": tokens}
    
    def chunk_tokens(examples, chunk_size: int = max_seq_len):
        """将token分块"""
        tokens = examples["tokens"]
        chunks = []
        
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i + chunk_size]
            if len(chunk) == chunk_size:  # 只保留完整块
                chunks.append(chunk)
        
        return {"chunks": chunks}
    
    # 处理训练集
    print("处理训练集...")
    train_dataset = dataset["train"]
    
    # 分词
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=train_dataset.column_names
    )
    
    # 分块
    chunked_train = tokenized_train.map(
        lambda x: chunk_tokens(x, max_seq_len),
        batched=True,
        batch_size=100,
        remove_columns=tokenized_train.column_names
    )
    
    # 保存训练集
    train_output_path = os.path.join(output_dir, "train_tokens.pt")
    train_tokens = []
    for chunk in chunked_train["chunks"]:
        train_tokens.extend(chunk)
    
    train_tokens_tensor = torch.tensor(train_tokens, dtype=torch.long)
    torch.save(train_tokens_tensor, train_output_path)
    print(f"训练集已保存: {train_output_path} (形状: {train_tokens_tensor.shape})")
    
    # 处理验证集
    print("处理验证集...")
    if "validation" in dataset:
        val_dataset = dataset["validation"]
    elif "test" in dataset:
        val_dataset = dataset["test"]
    else:
        # 如果没有验证集，从训练集分割
        train_size = int(0.9 * len(train_dataset))
        val_dataset = train_dataset.select(range(train_size, len(train_dataset)))
    
    # 分词
    tokenized_val = val_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=val_dataset.column_names
    )
    
    # 分块
    chunked_val = tokenized_val.map(
        lambda x: chunk_tokens(x, max_seq_len),
        batched=True,
        batch_size=100,
        remove_columns=tokenized_val.column_names
    )
    
    # 保存验证集
    val_output_path = os.path.join(output_dir, "val_tokens.pt")
    val_tokens = []
    for chunk in chunked_val["chunks"]:
        val_tokens.extend(chunk)
    
    val_tokens_tensor = torch.tensor(val_tokens, dtype=torch.long)
    torch.save(val_tokens_tensor, val_output_path)
    print(f"验证集已保存: {val_output_path} (形状: {val_tokens_tensor.shape})")
    
    # 保存数据集信息
    dataset_info = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "max_seq_len": max_seq_len,
        "train_samples": len(train_tokens),
        "val_samples": len(val_tokens),
        "vocab_size": tokenizer.get_vocab_size(),
        "total_tokens": len(train_tokens) * max_seq_len + len(val_tokens) * max_seq_len
    }
    
    info_path = os.path.join(output_dir, "dataset_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"数据集信息已保存: {info_path}")
    print(f"训练样本数: {len(train_tokens):,}")
    print(f"验证样本数: {len(val_tokens):,}")
    print(f"总token数: {dataset_info['total_tokens']:,}")
    
    return output_dir


def create_synthetic_data(num_samples: int = 1000, 
                         max_seq_len: int = 16384,
                         output_dir: str = "synthetic_data") -> str:
    """
    创建合成数据用于测试和调试
    
    Args:
        num_samples: 样本数量
        max_seq_len: 序列长度
        output_dir: 输出目录
        
    Returns:
        合成数据目录路径
    """
    
    print(f"正在创建合成数据: {num_samples} 个样本，序列长度 {max_seq_len}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载分词器
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    
    # 创建合成数据
    synthetic_tokens = []
    
    for i in range(num_samples):
        # 生成随机token序列
        tokens = torch.randint(0, vocab_size - 100, (max_seq_len,))  # 留出一些空间给特殊token
        synthetic_tokens.append(tokens)
    
    # 分割训练/验证集
    train_size = int(0.9 * num_samples)
    train_tokens = synthetic_tokens[:train_size]
    val_tokens = synthetic_tokens[train_size:]
    
    # 保存数据
    train_output_path = os.path.join(output_dir, "train_tokens.pt")
    val_output_path = os.path.join(output_dir, "val_tokens.pt")
    
    torch.save(torch.stack(train_tokens), train_output_path)
    torch.save(torch.stack(val_tokens), val_output_path)
    
    # 保存数据集信息
    dataset_info = {
        "dataset_type": "synthetic",
        "max_seq_len": max_seq_len,
        "train_samples": len(train_tokens),
        "val_samples": len(val_tokens),
        "vocab_size": vocab_size,
        "total_tokens": len(train_tokens) * max_seq_len + len(val_tokens) * max_seq_len
    }
    
    info_path = os.path.join(output_dir, "dataset_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"合成数据已保存到: {output_dir}")
    print(f"训练样本数: {len(train_tokens):,}")
    print(f"验证样本数: {len(val_tokens):,}")
    
    return output_dir


def load_deepseek_data(data_dir: str, split: str = "train") -> torch.Tensor:
    """
    加载处理好的 DeepSeek 数据
    
    Args:
        data_dir: 数据目录
        split: 数据集分割 ("train" 或 "val")
        
    Returns:
        token张量
    """
    
    file_path = os.path.join(data_dir, f"{split}_tokens.pt")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    tokens = torch.load(file_path)
    print(f"加载 {split} 数据: {tokens.shape}")
    
    return tokens


def validate_dataset(data_dir: str) -> Dict[str, Any]:
    """
    验证数据集完整性
    
    Args:
        data_dir: 数据目录
        
    Returns:
        验证结果
    """
    
    info_path = os.path.join(data_dir, "dataset_info.json")
    
    if not os.path.exists(info_path):
        return {"valid": False, "error": "数据集信息文件不存在"}
    
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    except Exception as e:
        return {"valid": False, "error": f"无法读取数据集信息: {e}"}
    
    # 检查数据文件
    train_path = os.path.join(data_dir, "train_tokens.pt")
    val_path = os.path.join(data_dir, "val_tokens.pt")
    
    if not os.path.exists(train_path):
        return {"valid": False, "error": "训练数据文件不存在"}
    
    if not os.path.exists(val_path):
        return {"valid": False, "error": "验证数据文件不存在"}
    
    # 加载数据验证
    try:
        train_tokens = torch.load(train_path)
        val_tokens = torch.load(val_path)
        
        # 验证形状
        max_seq_len = dataset_info["max_seq_len"]
        
        if len(train_tokens.shape) != 2 or train_tokens.shape[1] != max_seq_len:
            return {"valid": False, "error": f"训练数据形状错误: {train_tokens.shape}"}
        
        if len(val_tokens.shape) != 2 or val_tokens.shape[1] != max_seq_len:
            return {"valid": False, "error": f"验证数据形状错误: {val_tokens.shape}"}
        
        # 验证token范围
        vocab_size = dataset_info["vocab_size"]
        if train_tokens.max() >= vocab_size or train_tokens.min() < 0:
            return {"valid": False, "error": "训练数据包含无效token"}
        
        if val_tokens.max() >= vocab_size or val_tokens.min() < 0:
            return {"valid": False, "error": "验证数据包含无效token"}
        
    except Exception as e:
        return {"valid": False, "error": f"数据验证失败: {e}"}
    
    return {
        "valid": True,
        "dataset_info": dataset_info,
        "train_shape": train_tokens.shape,
        "val_shape": val_tokens.shape
    }


if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepSeek V3.2 数据预处理")
    parser.add_argument("--dataset", type=str, default="wikitext", 
                       help="数据集名称")
    parser.add_argument("--config", type=str, default="wikitext-103-raw-v1", 
                       help="数据集配置")
    parser.add_argument("--max-seq-len", type=int, default=16384, 
                       help="最大序列长度")
    parser.add_argument("--output-dir", type=str, default="deepseek_data", 
                       help="输出目录")
    parser.add_argument("--synthetic", action="store_true", 
                       help="创建合成数据")
    parser.add_argument("--num-samples", type=int, default=1000, 
                       help="合成数据样本数")
    
    args = parser.parse_args()
    
    if args.synthetic:
        # 创建合成数据
        data_dir = create_synthetic_data(
            num_samples=args.num_samples,
            max_seq_len=args.max_seq_len,
            output_dir=args.output_dir
        )
    else:
        # 准备真实数据集
        data_dir = prepare_deepseek_dataset(
            dataset_name=args.dataset,
            dataset_config=args.config,
            max_seq_len=args.max_seq_len,
            output_dir=args.output_dir
        )
    
    # 验证数据集
    validation_result = validate_dataset(data_dir)
    
    if validation_result["valid"]:
        print("✓ 数据集验证通过")
        print(f"训练数据形状: {validation_result['train_shape']}")
        print(f"验证数据形状: {validation_result['val_shape']}")
    else:
        print("✗ 数据集验证失败:")
        print(f"错误: {validation_result['error']}")