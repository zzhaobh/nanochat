"""
nanoDeepSeek演示脚本

快速演示nanoDeepSeek模型的功能，包括：
- 模型创建
- 简单训练循环
- 文本生成
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from nanochat.nanodeepseek import NanoDeepSeek, NanoDeepSeekConfig, create_nano_deepseek
from nanochat.common import compute_init, autodetect_device_type

print0 = print  # 简化演示

print0("=" * 80)
print0("nanoDeepSeek模型演示")
print0("=" * 80)

# -----------------------------------------------------------------------------
# 1. 计算初始化
device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
print0(f"设备类型: {device_type}")
print0(f"设备: {device}")

# -----------------------------------------------------------------------------
# 2. 创建nanoDeepSeek模型
print0("\n" + "=" * 80)
print0("2. 创建nanoDeepSeek模型")
print0("=" * 80)

# 创建小规模模型用于快速演示
model = create_nano_deepseek(scale="small", sequence_len=512, vocab_size=50257)
model_config = model.config

print0(f"模型配置:")
print0(f"  - 层数: {model_config.n_layer}")
print0(f"  - 隐藏层维度: {model_config.n_embd}")
print0(f"  - 注意力头数: {model_config.n_head}")
print0(f"  - 键值头数: {model_config.n_kv_head}")
print0(f"  - 序列长度: {model_config.sequence_len}")
print0(f"  - 词汇表大小: {model_config.vocab_size}")

# 初始化权重
model.to(device)
model.init_weights()

num_params = sum(p.numel() for p in model.parameters())
print0(f"参数数量: {num_params:,}")

# -----------------------------------------------------------------------------
# 3. 创建简单的合成数据集
print0("\n" + "=" * 80)
print0("3. 创建简单的合成数据集")
print0("=" * 80)

class SimpleTextDataset(Dataset):
    """简单的文本数据集，用于演示"""
    
    def __init__(self, vocab_size, seq_len, num_samples):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        # 生成随机token序列作为演示数据
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 输入：前seq_len-1个token
        # 目标：后seq_len-1个token
        x = self.data[idx, :-1]
        y = self.data[idx, 1:]
        return x, y

# 创建数据集和加载器
batch_size = 4
dataset = SimpleTextDataset(
    vocab_size=model_config.vocab_size,
    seq_len=model_config.sequence_len,
    num_samples=100
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print0(f"数据集大小: {len(dataset)}")
print0(f"批次大小: {batch_size}")
print0(f"每个epoch批次数: {len(dataloader)}")

# -----------------------------------------------------------------------------
# 4. 简单训练循环
print0("\n" + "=" * 80)
print0("4. 简单训练循环")
print0("=" * 80)

# 设置优化器
optimizers = model.setup_optimizers()
adamw_optimizer, muon_optimizer = optimizers

# 训练参数
num_epochs = 3
print0(f"训练配置:")
print0(f"  - Epoch数: {num_epochs}")
print0(f"  - AdamW学习率: {adamw_optimizer.param_groups[0]['lr']:.4f}")
print0(f"  - Muon学习率: {muon_optimizer.param_groups[0]['lr']:.4f}")

model.train()
total_steps = 0

for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        # 前向传播
        loss = model(x, y)
        
        # 反向传播
        for opt in optimizers:
            opt.zero_grad()
        loss.backward()
        for opt in optimizers:
            opt.step()
        
        epoch_loss += loss.item()
        num_batches += 1
        total_steps += 1
        
        if batch_idx % 5 == 0:
            print0(f"  Epoch {epoch+1}/{num_epochs} | "
                  f"Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f}")
    
    avg_epoch_loss = epoch_loss / num_batches
    print0(f"Epoch {epoch+1}/{num_epochs} 平均损失: {avg_epoch_loss:.4f}")

print0(f"\n训练完成！总步数: {total_steps}")

# -----------------------------------------------------------------------------
# 5. 文本生成演示
print0("\n" + "=" * 80)
print0("5. 文本生成演示")
print0("=" * 80)

model.eval()

# 创建一些随机的起始token序列
print0("\n从随机序列生成文本:")
for i in range(3):
    # 生成随机起始序列
    start_tokens = torch.randint(0, min(100, model_config.vocab_size), (8,)).tolist()
    print0(f"\n示例 {i+1}:")
    print0(f"起始tokens: {start_tokens}")
    
    # 生成新token
    generated_tokens = list(start_tokens)
    gen_stream = model.generate(
        tokens=start_tokens,
        max_tokens=10,
        temperature=0.8,
        top_k=20
    )
    
    for token_id in gen_stream:
        generated_tokens.append(token_id)
        print0(f"  生成token: {token_id}")
    
    print0(f"完整序列: {generated_tokens[:20]}...")

# -----------------------------------------------------------------------------
# 6. 模型架构细节
print0("\n" + "=" * 80)
print0("6. 模型架构细节")
print0("=" * 80)

print0("\n层结构:")
for i, block in enumerate(model.transformer.h):
    print0(f"  层 {i}:")
    print0(f"    - 注意力层: {block.attn}")
    print0(f"    - 前馈网络: {block.mlp}")

print0(f"\nFLOPs估计:")
flops_per_token = model.estimate_flops()
print0(f"  每token FLOPs: {flops_per_token:e}")
print0(f"  每序列FLOPs: {flops_per_token * model_config.sequence_len:e}")

print0("\n" + "=" * 80)
print0("演示完成！")
print0("=" * 80)
print0("\n总结:")
print0(f"  - 模型参数: {num_params:,}")
print0(f"  - 训练步数: {total_steps}")
print0(f"  - 每token估计FLOPs: {flops_per_token:e}")
