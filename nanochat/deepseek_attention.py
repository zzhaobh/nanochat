"""
DeepSeek V3.2 MLA (Multi-head Latent Attention) 实现
基于 DeepSeek V3.2 官方架构的多头潜在注意力机制
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from nanochat.common import print0


def apply_rotary_emb(x: torch.Tensor, cos_sin: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """应用旋转位置编码到张量"""
    cos, sin = cos_sin
    
    # 确保 x 是 4D 张量 (batch, seq_len, num_heads, head_dim)
    assert x.ndim == 4
    
    # 分割为实部和虚部
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    
    # 应用旋转
    y1 = x1 * cos + x2 * sin
    y2 = -x1 * sin + x2 * cos
    
    return torch.cat([y1, y2], dim=-1)


class DeepSeekMLA(nn.Module):
    """DeepSeek V3.2 多头潜在注意力机制 (MLA)"""
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        
        # 基础注意力参数
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        
        # MLA 核心参数
        self.kv_lora_rank = getattr(config, 'kv_lora_rank', 512)  # KV 压缩维度
        self.q_lora_rank = getattr(config, 'q_lora_rank', 1536)   # Query 压缩维度
        self.rope_dim = getattr(config, 'rope_dim', 64)           # RoPE 维度
        self.v_head_dim = getattr(config, 'v_head_dim', 128)      # Value head 维度
        
        # 计算维度分配
        self.nope_head_dim = (self.n_embd // self.n_head) - self.rope_dim  # 非 RoPE 部分维度
        
        # MLA 投影层
        # Query 投影: 输入 -> q_lora_rank -> 多头查询
        self.wq_down = nn.Linear(self.n_embd, self.q_lora_rank, bias=False)
        self.wq_up = nn.Linear(self.q_lora_rank, self.n_head * (self.nope_head_dim + self.rope_dim), bias=False)
        
        # KV 投影: 输入 -> kv_lora_rank -> 多头键值 (MLA 核心)
        self.wkv_down = nn.Linear(self.n_embd, self.kv_lora_rank, bias=False)
        self.wkv_up = nn.Linear(self.kv_lora_rank, 
                               self.n_head * (self.nope_head_dim + self.rope_dim + self.v_head_dim), bias=False)
        
        # 输出投影
        self.wo = nn.Linear(self.n_head * self.v_head_dim, self.n_embd, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化 - 使用 DeepSeek 推荐的初始化策略"""
        for module in [self.wq_down, self.wq_up, self.wkv_down, self.wkv_up, self.wo]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, cos_sin: Tuple[torch.Tensor, torch.Tensor], 
                kv_cache: Optional[dict] = None) -> torch.Tensor:
        """前向传播"""
        B, T, C = x.size()
        
        # 1. 生成 Query (MLA 查询路径)
        q_latent = self.wq_down(x)  # (B, T, q_lora_rank)
        q = self.wq_up(q_latent)    # (B, T, n_head * (nope_dim + rope_dim))
        
        # 分割 Query 为 NOPE 和 ROPE 部分
        q = q.view(B, T, self.n_head, self.nope_head_dim + self.rope_dim)
        q_nope, q_rope = torch.split(q, [self.nope_head_dim, self.rope_dim], dim=-1)
        
        # 2. 生成 KV Latent (MLA 核心 - KV 压缩路径)
        kv_latent = self.wkv_down(x)  # (B, T, kv_lora_rank) - 这就是需要缓存的部分
        
        # 处理 KV Cache
        if kv_cache is not None:
            # 将 kv_latent 插入缓存 (MLA 只缓存压缩后的 latent)
            kv_latent = kv_cache.insert_kv(self.layer_idx, kv_latent)
            Tk = kv_latent.size(1)  # 总序列长度
        else:
            Tk = T
        
        # 3. 从 KV Latent 恢复完整的 KV Heads
        kv = self.wkv_up(kv_latent)  # (B, Tk, n_head * (nope_dim + rope_dim + v_dim))
        
        # 分割 KV 为 K_nope, K_rope, V
        kv = kv.view(B, Tk, self.n_head, self.nope_head_dim + self.rope_dim + self.v_head_dim)
        k_nope, k_rope, v = torch.split(kv, [self.nope_head_dim, self.rope_dim, self.v_head_dim], dim=-1)
        
        # 4. 应用旋转位置编码 (只对 ROPE 部分)
        q_rope = apply_rotary_emb(q_rope, cos_sin)
        k_rope = apply_rotary_emb(k_rope, cos_sin)
        
        # 5. 拼接完整的 Key 和 Query
        k = torch.cat([k_nope, k_rope], dim=-1)  # (B, Tk, n_head, head_dim)
        q = torch.cat([q_nope, q_rope], dim=-1)  # (B, T, n_head, head_dim)
        
        # 6. QK Norm (DeepSeek 使用的 RMSNorm)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        
        # 7. 转置维度用于注意力计算 (B, T, H, D) -> (B, H, T, D)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        # 8. 注意力计算 (使用 PyTorch 的高效实现)
        enable_gqa = self.n_head != self.n_kv_head
        
        if kv_cache is None or T == Tk:
            # 训练模式或全序列推理
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif T == 1:
            # 单token推理
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # 推理时的多token处理
            attn_mask = torch.zeros((T, Tk), dtype=torch.bool, device=x.device)
            prefix_len = Tk - T
            attn_mask[:, :prefix_len] = True  # 关注前缀
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((T, T), dtype=torch.bool, device=x.device))
            
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)
        
        # 9. 重组输出并投影
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.wo(y)
        
        return y


def test_mla_attention():
    """测试 MLA 注意力机制"""
    from nanochat.deepseek_config import DeepSeekConfig
    
    config = DeepSeekConfig(
        n_embd=256,
        n_head=4,
        n_kv_head=2,
        max_seq_len=2048,
        kv_lora_rank=128,   # MLA 参数
        q_lora_rank=384,    # MLA 参数
        rope_dim=32,        # MLA 参数
        v_head_dim=64       # MLA 参数
    )
    
    model = DeepSeekMLA(config, layer_idx=0)
    
    # 测试输入
    batch_size = 2
    seq_len = 1024
    x = torch.randn(batch_size, seq_len, config.n_embd)
    
    # 模拟旋转编码
    cos = torch.randn(1, seq_len, 1, config.rope_dim // 2)
    sin = torch.randn(1, seq_len, 1, config.rope_dim // 2)
    
    # 前向传播
    output = model(x, (cos, sin))
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"KV Latent 维度: {config.kv_lora_rank}")
    print(f"压缩比: {config.n_embd / config.kv_lora_rank:.2f}x")
    print("MLA Attention test passed!")
    
    return output


if __name__ == "__main__":
    test_mla_attention()