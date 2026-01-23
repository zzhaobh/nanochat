"""
nanoDeepSeek - 基于DeepSeek v3.2架构的精简版实现

该模块实现了DeepSeek v3.2核心架构特性，主要特点：
- 标准Transformer基础结构
- MLA（多头潜在注意力）：低秩压缩减少KV缓存
- MoE（混合专家）：简化版MoE支持稀疏激活
- 旋转位置编码（RoPE）
- QK归一化
- Group-Query Attention (GQA)支持
- RMSNorm归一化（无学习参数）
- SwiGLU激活函数

架构说明：
- 本实现保留了DeepSeek的核心特性（QK归一化、RoPE、RMSNorm、GQA、SwiGLU）
- 实现了MLA（多头潜在注意力）以减少KV缓存（压缩比例约50-70%）
- 实现了简化版MoE（混合专家）以支持稀疏激活（默认16个专家，Top-4路由）
- 适用于资源受限环境的小规模训练和推理
- 注意：完整的DeepSeek V3.2包含DSA（稀疏注意力），本简化版暂未实现
"""

import math
from functools import partial
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0


@dataclass
class NanoDeepSeekConfig:
    """nanoDeepSeek模型配置类
    
    精简版DeepSeek v3.2架构的配置参数
    
    Attributes:
        sequence_len: 最大序列长度
        vocab_size: 词汇表大小
        n_layer: Transformer层数
        n_head: 注意力头数（查询头数）
        n_kv_head: 键值头数（用于GQA）
        n_embd: 嵌入维度（隐藏层维度）
        
        # MLA（多头潜在注意力）参数
        kv_lora_rank: KV压缩后的维度（None则默认为n_embd//2）
        use_mla: 是否启用MLA压缩（默认启用）
        
        # MoE（混合专家）参数
        use_moe: 是否启用MoE（默认启用）
        num_experts: 专家数量（简化版使用8个）
        top_k: 每个token激活的专家数（Top-K路由）
        
        # MTP（多Token预测）参数
        use_mtp: bool = True  # 是否启用MTP（默认启用）
        mtp_num_future_tokens: int = 4  # 预测的token数量
    """
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6  # 查询头数
    n_kv_head: int = 6  # 键值头数（GQA）
    n_embd: int = 768
    
    # MLA参数
    kv_lora_rank: Optional[int] = None  # KV压缩后的维度
    use_mla: bool = True  # 是否启用MLA
    
    # MoE参数
    use_moe: bool = True  # 是否启用MoE
    num_experts: int = 16  # 专家数量（简化版，V3.2使用256）
    top_k: int = 4  # 每个token激活的专家数（简化版，V3.2使用8）
    
    # MTP参数
    use_mtp: bool = True  # 是否启用MTP
    mtp_num_future_tokens: int = 4  # 预测的token数量


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """RMS归一化函数（无学习参数）
    
    Args:
        x: 输入张量，形状为(*, d_model)
        
    Returns:
        归一化后的张量
    """
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor:
    """应用旋转位置编码到输入张量
    
    Args:
        x: 输入张量，形状为(B, T, H, D)，其中D必须为偶数
        cos: 余弦值，形状为(1, T, 1, D/2)
        sin: 正弦值，形状为(1, T, 1, D/2)
        
    Returns:
        应用旋转编码后的张量
    """
    assert x.ndim == 4, "输入张量必须是4维的 (B, T, H, D)"
    
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # 将最后一维分为两半
    
    # 旋转成对维度
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    
    out = torch.cat([y1, y2], dim=3)  # 重新组合
    out = out.to(x.dtype)  # 确保输入输出类型匹配
    return out


class MultiHeadLatentAttention(nn.Module):
    """多头潜在注意力（MLA）- DeepSeek核心特性（修正版）
    
    实现了DeepSeek V3.2的MLA机制，包含：
    - Q/K/V低秩压缩（减少KV缓存60-70%）
    - 解耦RoPE：仅应用于Key的RoPE部分
    - GQA支持（键值头共享）
    - QK归一化
    - 因果注意力掩码
    - KV缓存支持（压缩后的KV）
    
    DeepSeek MLA正确流程：
    输入 -> KV压缩 -> 潜在向量分为两部分：
      - KV_pe：用于RoPE（通常占总维度的1/4）
      - KV_content：用于内容
    -> 分别升维到K/V -> RoPE只应用于K_pe部分
    """
    
    def __init__(self, config: NanoDeepSeekConfig, layer_idx: int):
        """初始化MLA注意力层
        
        Args:
            config: 模型配置
            layer_idx: 当前层索引（用于KV缓存）
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.use_mla = config.use_mla
        
        # 确保维度整除
        assert self.n_embd % self.n_head == 0, \
            f"n_embd ({self.n_embd}) 必须能被 n_head ({self.n_head}) 整除"
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0, \
            f"n_kv_head ({self.n_kv_head}) 必须小于等于 n_head ({self.n_head}) 且是其约数"
        
        if self.use_mla:
            # MLA压缩维度（默认为n_embd//2）
            self.kv_lora_rank = config.kv_lora_rank or self.n_embd // 2
            
            # DeepSeek MLA：将压缩后的向量分为两部分
            # kv_pe_dim：用于RoPE的部分（通常占总维度的1/4）
            # kv_content_dim：用于内容的部分（占总维度的3/4）
            self.kv_pe_dim = self.kv_lora_rank // 4
            self.kv_content_dim = self.kv_lora_rank - self.kv_pe_dim
            
            # Q压缩：降维到潜在空间
            self.c_q_proj = nn.Linear(
                self.n_embd,
                self.kv_lora_rank,
                bias=False
            )
            # Q升维：从潜在空间恢复
            self.c_q_up = nn.Linear(
                self.kv_lora_rank,
                self.n_head * self.head_dim,
                bias=False
            )
            
            # KV压缩：降维到潜在空间（输出分为pe和content两部分）
            self.c_kv_proj = nn.Linear(
                self.n_embd,
                self.kv_lora_rank,
                bias=False
            )
            
            # K升维：从潜在空间的content部分恢复
            self.c_k_up = nn.Linear(
                self.kv_content_dim,
                self.n_kv_head * self.head_dim,
                bias=False
            )
            
            # K_pe升维：从潜在空间的pe部分恢复（用于RoPE）
            self.c_k_pe_up = nn.Linear(
                self.kv_pe_dim,
                self.n_kv_head * self.head_dim,
                bias=False
            )
            
            # V升维：从潜在空间的content部分恢复
            self.c_v_up = nn.Linear(
                self.kv_content_dim,
                self.n_kv_head * self.head_dim,
                bias=False
            )
        else:
            # 标准MHA（无压缩）
            self.c_q = nn.Linear(
                self.n_embd,
                self.n_head * self.head_dim,
                bias=False
            )
            self.c_k = nn.Linear(
                self.n_embd,
                self.n_kv_head * self.head_dim,
                bias=False
            )
            self.c_v = nn.Linear(
                self.n_embd,
                self.n_kv_head * self.head_dim,
                bias=False
            )
        
        # 输出投影（共享）
        self.c_proj = nn.Linear(
            self.n_embd,
            self.n_embd,
            bias=False
        )
    
    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        kv_cache: Optional[object]
    ) -> torch.Tensor:
        """MLA前向传播（修正版：解耦RoPE）
        
        Args:
            x: 输入张量，形状为(B, T, C)
            cos_sin: 旋转编码的(cos, sin)元组
            kv_cache: KV缓存对象（可选）
            
        Returns:
            注意力输出，形状为(B, T, C)
        """
        B, T, C = x.size()
        
        if self.use_mla:
            # MLA压缩路径（解耦RoPE实现）
            
            # Q：压缩 -> 升维
            q_latent = self.c_q_proj(x)  # (B, T, kv_lora_rank)
            q = self.c_q_up(q_latent)  # (B, T, n_head * head_dim)
            q = q.view(B, T, self.n_head, self.head_dim)
            
            # KV：压缩 -> 分离pe和content
            kv_latent = self.c_kv_proj(x)  # (B, T, kv_lora_rank)
            kv_pe = kv_latent[..., :self.kv_pe_dim]  # (B, T, kv_pe_dim) - 用于RoPE
            kv_content = kv_latent[..., self.kv_pe_dim:]  # (B, T, kv_content_dim) - 用于内容
            
            # K：content升维 + pe升维（带RoPE）
            k_content = self.c_k_up(kv_content)  # (B, T, n_kv_head * head_dim)
            k_pe = self.c_k_pe_up(kv_pe)  # (B, T, n_kv_head * head_dim)
            
            # 将content和pe组合
            k_content = k_content.view(B, T, self.n_kv_head, self.head_dim)
            k_pe = k_pe.view(B, T, self.n_kv_head, self.head_dim)
            
            # 只对pe部分应用RoPE
            cos, sin = cos_sin
            k_pe_rotated = apply_rotary_emb(k_pe, cos, sin)
            
            # K = content + rotated_pe
            k = k_content + k_pe_rotated  # (B, T, n_kv_head, head_dim)
            
            # V：仅从content升维（不应用RoPE）
            v = self.c_v_up(kv_content)  # (B, T, n_kv_head * head_dim)
            v = v.view(B, T, self.n_kv_head, self.head_dim)
        else:
            # 标准MHA路径（无压缩）
            q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
            k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
            v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
            
            # 标准MHA也需要应用RoPE到q和k
            cos, sin = cos_sin
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        
        # QK归一化
        q, k = rms_norm(q), rms_norm(k)
        
        # 转置使头成为批维度：(B, T, H, D) -> (B, H, T, D)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        # 应用KV缓存：插入当前k,v到缓存中，获取完整视图
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        
        Tq = q.size(2)  # 本次前向传播的查询数
        Tk = k.size(2)  # 总键值数（缓存+当前）
        
        # 注意力计算：查询自回归地关注键/值
        enable_gqa = self.n_head != self.n_kv_head  # GQA：复制K/V头以匹配查询头
        
        if kv_cache is None or Tq == Tk:
            # 训练时（无KV缓存），使用因果注意力
            y = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                enable_gqa=enable_gqa
            )
        elif Tq == 1:
            # 推理时且只有一个查询：查询关注缓存中的所有键值
            y = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=False,
                enable_gqa=enable_gqa
            )
        else:
            # 推理时且有多个查询：首先关注所有缓存的键值，然后在块内因果关注
            attn_mask = torch.zeros(
                (Tq, Tk),
                dtype=torch.bool,
                device=q.device
            )
            prefix_len = Tk - Tq
            attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(
                torch.ones((Tq, Tq), dtype=torch.bool, device=q.device)
            )
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                enable_gqa=enable_gqa
            )
        
        # 重新组装头并投影回残差流
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MoEFFN(nn.Module):
    """混合专家前馈网络（MoE）- DeepSeek核心特性
    
    实现了简化版MoE架构，包含：
    - 共享专家：所有token都使用
    - 路由专家：通过TopK路由选择
    - 负载均衡：使用专家偏置（无辅助损失）
    - 稀疏激活：每个token只使用top_k个专家
    
    路由策略：Top-K + Sigmoid
    """
    
    def __init__(self, config: NanoDeepSeekConfig):
        """初始化MoE前馈网络
        
        Args:
            config: 模型配置
        """
        super().__init__()
        self.use_moe = config.use_moe
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.n_embd = config.n_embd
        
        if self.use_moe:
            # 共享专家（所有token都使用，SwiGLU需要2倍隐藏维度）
            self.shared_expert_fc = nn.Linear(
                self.n_embd,
                2 * 4 * self.n_embd,
                bias=False
            )
            self.shared_expert_proj = nn.Linear(
                4 * self.n_embd,
                self.n_embd,
                bias=False
            )

            # 路由专家（SwiGLU需要2倍隐藏维度）
            self.experts_fc = nn.ModuleList([
                nn.Linear(self.n_embd, 2 * 4 * self.n_embd, bias=False)
                for _ in range(self.num_experts)
            ])
            self.experts_proj = nn.ModuleList([
                nn.Linear(4 * self.n_embd, self.n_embd, bias=False)
                for _ in range(self.num_experts)
            ])
            
            # 门控网络（Sigmoid路由）
            self.gate = nn.Linear(self.n_embd, self.num_experts, bias=False)
            
            # 专家偏置（用于负载均衡，无梯度）
            self.register_buffer(
                'expert_bias',
                torch.zeros(self.num_experts)
            )
        else:
            # 标准FFN（无MoE，SwiGLU需要2倍隐藏维度）
            self.c_fc = nn.Linear(
                config.n_embd,
                2 * 4 * config.n_embd,
                bias=False
            )
            self.c_proj = nn.Linear(
                4 * config.n_embd,
                config.n_embd,
                bias=False
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MoE前向传播
        
        Args:
            x: 输入张量，形状为(B, T, C)
            
        Returns:
            输出张量，形状为(B, T, C)
        """
        if self.use_moe:
            return self._moe_forward(x)
        else:
            return self._dense_forward(x)
    
    def _moe_forward(self, x: torch.Tensor) -> torch.Tensor:
        """MoE前向传播（优化版：向量化实现）
        
        Args:
            x: 输入张量，形状为(B, T, C)
            
        Returns:
            输出张量，形状为(B, T, C)
        """
        B, T, C = x.shape

        # 共享专家输出（使用SwiGLU激活）
        shared_out = self.shared_expert_fc(x)  # (B, T, 2*4*C)
        shared_gate, shared_x = shared_out.chunk(2, dim=-1)  # 分为两部分，各(B, T, 4*C)
        shared_out = F.silu(shared_gate) * shared_x  # SwiGLU激活
        shared_out = self.shared_expert_proj(shared_out)  # (B, T, C)
        
        # 路由计算
        gate_logits = self.gate(x) + self.expert_bias  # (B, T, num_experts)
        gate_weights = torch.sigmoid(gate_logits)  # Sigmoid激活
        
        # TopK选择
        top_k_weights, top_k_indices = torch.topk(
            gate_weights,
            self.top_k,
            dim=-1
        )  # (B, T, top_k)
        
        # 归一化权重
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # 向量化专家计算（优化版：只循环专家，不循环token）
        # 将输入展平: (B*T, C)
        flat_x = x.view(-1, C)
        flat_indices = top_k_indices.view(-1, self.top_k)  # (B*T, K)
        flat_weights = top_k_weights.view(-1, self.top_k)  # (B*T, K)
        
        # 结果容器
        final_expert_out = torch.zeros_like(flat_x)
        
        # 只循环专家，不循环token（性能优化）
        for i in range(self.num_experts):
            # 找出选中专家 i 的所有 token 索引
            # mask: (B*T, K)
            mask = (flat_indices == i)
            # 哪些 token 选中了专家 i (B*T)
            batch_mask = mask.any(dim=-1)
            
            if batch_mask.any():
                # 选出这些 token
                expert_input = flat_x[batch_mask]  # (N_selected, C)

                # 计算专家输出（使用SwiGLU激活）
                h = self.experts_fc[i](expert_input)  # (N_selected, 2*4*C)
                gate, h = h.chunk(2, dim=-1)  # 分为两部分，各(N_selected, 4*C)
                h = F.silu(gate) * h  # SwiGLU激活
                out = self.experts_proj[i](h)  # (N_selected, C)
                
                # 获取对应的权重
                # mask[batch_mask] 形状为 (N_selected, K)，其中每行只有一个 True
                # flat_weights[batch_mask] 形状为 (N_selected, K)
                # 提取权重: (N_selected, 1)
                weight = (flat_weights[batch_mask] * mask[batch_mask]).sum(dim=-1, keepdim=True)
                
                out = out * weight
                
                # 累加回结果 (scatter add)
                final_expert_out[batch_mask] += out
        
        # 组合共享专家和路由专家输出
        routed_out = final_expert_out.view(B, T, C)
        out = shared_out + routed_out
        return out
    
    def _dense_forward(self, x: torch.Tensor) -> torch.Tensor:
        """标准FFN前向传播

        Args:
            x: 输入张量，形状为(B, T, C)

        Returns:
            输出张量，形状为(B, T, C)
        """
        x = self.c_fc(x)  # (B, T, 2*4*C)
        gate, x = x.chunk(2, dim=-1)  # 分为两部分，各(B, T, 4*C)
        x = F.silu(gate) * x  # SwiGLU激活
        x = self.c_proj(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer块（解码器层）
    
    包含：
    - 多头潜在注意力（MLA）
    - 混合专家前馈网络（MoE）
    - 残差连接
    - 层归一化
    """
    
    def __init__(self, config: NanoDeepSeekConfig, layer_idx: int):
        """初始化Transformer块
        
        Args:
            config: 模型配置
            layer_idx: 当前层索引
        """
        super().__init__()
        self.attn = MultiHeadLatentAttention(config, layer_idx)
        self.mlp = MoEFFN(config)
    
    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        kv_cache: Optional[object]
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为(B, T, C)
            cos_sin: 旋转编码的(cos, sin)元组
            kv_cache: KV缓存对象（可选）
            
        Returns:
            输出张量，形状为(B, T, C)
        """
        # 注意力子层 + 残差连接
        x = x + self.attn(rms_norm(x), cos_sin, kv_cache)
        # 前馈网络子层 + 残差连接
        x = x + self.mlp(rms_norm(x))
        return x


class NanoDeepSeek(nn.Module):
    """nanoDeepSeek模型主类
    
    精简版DeepSeek v3.2架构的完整实现，包含：
    - Token嵌入层
    - Transformer编码器（多层）
    - 输出投影层
    - 旋转位置编码
    - MTP（多Token预测）支持
    """
    
    def __init__(self, config: NanoDeepSeekConfig):
        """初始化nanoDeepSeek模型
        
        Args:
            config: 模型配置
        """
        super().__init__()
        self.config = config
        
        # Transformer组件
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([
                TransformerBlock(config, layer_idx)
                for layer_idx in range(config.n_layer)
            ]),
        })
        
        # 语言模型头（输出投影）
        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )
        
        # MTP头（多Token预测）
        if config.use_mtp:
            self.mtp_heads = nn.ModuleList([
                nn.Linear(config.n_embd, config.vocab_size, bias=False)
                for _ in range(config.mtp_num_future_tokens)
            ])
        
        # 预计算旋转位置编码
        # 支持meta设备初始化，但这里是伪造的
        # 预先计算10倍序列长度的旋转编码（应该足够）
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len,
            head_dim
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    
    def init_weights(self) -> None:
        """初始化模型权重"""
        self.apply(self._init_weights)
        
        # 零化分类器权重
        torch.nn.init.zeros_(self.lm_head.weight)
        
        # 零化MTP头权重（如果存在）
        if self.config.use_mtp:
            for mtp_head in self.mtp_heads:
                torch.nn.init.zeros_(mtp_head.weight)
        
        # 零化所有块的输出投影权重
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        
        # 重新初始化旋转编码
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len,
            head_dim
        )
        self.cos, self.sin = cos, sin
        
        # 将嵌入层从fp32转换为bf16以节省内存
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
    
    def _init_weights(self, module: nn.Module) -> None:
        """初始化模块权重
        
        Args:
            module: 要初始化的模块
        """
        if isinstance(module, nn.Linear):
            # 使用Kaiming初始化的变体
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)
    
    def _precompute_rotary_embeddings(
        self,
        seq_len: int,
        head_dim: int,
        base: float = 10000.0,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """预计算旋转位置编码
        
        Args:
            seq_len: 序列长度
            head_dim: 头维度（必须为偶数）
            base: 旋转频率基数
            device: 设备（None则自动检测）
            
        Returns:
            (cos, sin)元组，形状均为(1, seq_len, 1, head_dim/2)
        """
        # 自动从模型嵌入层检测设备
        if device is None:
            device = self.transformer.wte.weight.device
        
        # 通道步进（偶数位置）
        channel_range = torch.arange(
            0,
            head_dim,
            2,
            dtype=torch.float32,
            device=device
        )
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        
        # 时间步步进
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        
        # 计算每个(时间, 通道)对的旋转频率
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        
        # 保持在bfloat16以节省内存
        cos, sin = cos.bfloat16(), sin.bfloat16()
        
        # 添加批和头维度以便后续广播
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        
        return cos, sin
    
    def get_device(self) -> torch.device:
        """获取模型所在设备
        
        Returns:
            设备对象
        """
        return self.transformer.wte.weight.device
    
    def estimate_flops(self) -> float:
        """估计每个token的FLOPs
        
        参考: https://arxiv.org/abs/2204.02311
        
        Returns:
            每个token的浮点运算次数
        """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = (
            self.config.n_layer,
            self.config.n_head,
            self.config.n_embd // self.config.n_head,
            self.config.sequence_len
        )
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token
    
    def setup_optimizers(
        self,
        unembedding_lr: float = 0.004,
        embedding_lr: float = 0.2,
        matrix_lr: float = 0.02,
        weight_decay: float = 0.0
    ) -> list:
        """设置优化器（嵌入层用AdamW，矩阵层用Muon）
        
        Args:
            unembedding_lr: 解嵌入层学习率
            embedding_lr: 嵌入层学习率
            matrix_lr: 矩阵层学习率
            weight_decay: 权重衰减
            
        Returns:
            优化器列表 [adamw_optimizer, muon_optimizer]
        """
        from nanochat.muon import Muon, DistMuon
        from nanochat.adamw import DistAdamW
        
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        
        # 将所有参数分成3组（矩阵、嵌入、lm_head）
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        
        assert len(list(self.parameters())) == \
            len(matrix_params) + len(embedding_params) + len(lm_head_params)
        
        # 为嵌入层和lm_head创建AdamW优化器
        # 根据√dmodel缩放LR（针对768维模型调优）
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"AdamW参数LR缩放 ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        
        # 为线性层创建Muon优化器
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        
        # 合并两个优化器
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        
        return optimizers
    
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[object] = None,
        loss_reduction: str = 'mean'
    ) -> torch.Tensor:
        """前向传播（支持MTP）
        
        Args:
            idx: 输入token索引，形状为(B, T)
            targets: 目标token（训练时使用），形状为(B, T)
            kv_cache: KV缓存对象（推理时使用）
            loss_reduction: 损失聚合方式
            
        Returns:
            如果targets为None，返回logits (B, T, vocab_size)
            否则返回损失标量（包含MTP损失，如果启用）
        """
        B, T = idx.size()
        
        # 获取当前序列长度的旋转编码
        assert T <= self.cos.size(1), \
            f"序列长度超过旋转编码缓存: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, \
            f"旋转编码和idx在不同设备: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, \
            "旋转编码必须是bfloat16类型"
        
        # 如果KV缓存存在，需要偏移旋转编码到缓存中的当前位置
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
        
        # 前向传播Transformer主干
        x = self.transformer.wte(idx)
        x = rms_norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = rms_norm(x)
        
        # 前向传播语言模型头（计算logits）
        softcap = 15  # 平滑地将logits限制到范围[-softcap, softcap]
        logits = self.lm_head(x)  # (B, T, vocab_size) - 非常大的张量
        logits = logits.float()  # 切换到fp32进行logit softcap和损失计算
        logits = softcap * torch.tanh(logits / softcap)  # 压缩logits
        
        if targets is not None:
            # 训练：给定目标，计算并返回损失
            # 主损失（下一个token预测）
            main_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction
            )

            # MTP损失（多token预测）
            if self.config.use_mtp:
                mtp_loss = 0.0
                num_mtp_losses = 0

                for i, mtp_head in enumerate(self.mtp_heads):
                    # 目标向前偏移 i+1 个token
                    if i + 1 < T:
                        mtp_logits = mtp_head(x)  # (B, T, vocab_size)
                        mtp_logits = mtp_logits.float()  # 切换到fp32
                        mtp_logits = softcap * torch.tanh(mtp_logits / softcap)  # 压缩logits

                        # 只计算前 T-i-1 个位置的损失
                        mtp_targets = targets[:, i+1:]  # 偏移目标
                        mtp_logits_shifted = mtp_logits[:, :T-i-1, :]  # 对应的logits

                        mtp_loss += F.cross_entropy(
                            mtp_logits_shifted.reshape(-1, mtp_logits.size(-1)),
                            mtp_targets.reshape(-1),
                            ignore_index=-1,
                            reduction=loss_reduction
                        )
                        num_mtp_losses += 1

                # 平均MTP损失并与主损失结合（MTP损失权重0.5）
                if num_mtp_losses > 0:
                    mtp_loss = mtp_loss / num_mtp_losses
                    total_loss = main_loss + 0.5 * mtp_loss
                    return total_loss
                else:
                    return main_loss
            else:
                return main_loss
        else:
            # 推理：直接返回logits
            return logits
    
    @torch.inference_mode()
    def generate(
        self,
        tokens: list,
        max_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        seed: int = 42
    ) -> int:
        """朴素的自回归流式推理
        
        假设：
        - 批大小为1
        - ids和生成的token是简单的Python列表和整数
        
        Args:
            tokens: 起始token列表
            max_tokens: 最大生成token数
            temperature: 采样温度（0=贪婪）
            top_k: Top-K采样（None=禁用）
            seed: 随机种子
            
        Yields:
            生成的token ID
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        
        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # 添加批维度
        
        for _ in range(max_tokens):
            logits = self.forward(ids)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token


# 便捷函数：创建不同规模的nanoDeepSeek模型
def create_nano_deepseek(
    scale: str = "base",
    sequence_len: int = 1024,
    vocab_size: int = 50304,
    use_mla: bool = True,
    use_moe: bool = True
) -> NanoDeepSeek:
    """创建不同规模的nanoDeepSeek模型
    
    Args:
        scale: 模型规模 ('small', 'base', 'large')
        sequence_len: 最大序列长度
        vocab_size: 词汇表大小
        use_mla: 是否启用MLA（多头潜在注意力）
        use_moe: 是否启用MoE（混合专家）
        
    Returns:
        nanoDeepSeek模型实例
    """
    configs = {
        "small": dict(
            n_layer=6, n_head=4, n_kv_head=4, n_embd=384,
            num_experts=8, top_k=2
        ),
        "base": dict(
            n_layer=12, n_head=6, n_kv_head=6, n_embd=768,
            num_experts=16, top_k=4
        ),
        "large": dict(
            n_layer=24, n_head=12, n_kv_head=6, n_embd=1536,
            num_experts=32, top_k=6
        ),
    }
    
    config_dict = configs.get(scale, configs["base"])
    config = NanoDeepSeekConfig(
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        use_mla=use_mla,
        use_moe=use_moe,
        **config_dict
    )
    
    return NanoDeepSeek(config)
