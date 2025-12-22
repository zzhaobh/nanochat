"""
DeepSeek V3.2 完整模型实现
基于 DeepSeek V3.2 的架构设计
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Tuple, Dict, Any

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW
from nanochat.deepseek_config import DeepSeekConfig
from nanochat.deepseek_attention import DeepSeekMLA
from nanochat.deepseek_moe import DeepSeekMLP


def norm(x: torch.Tensor) -> torch.Tensor:
    """纯函数式 RMSNorm"""
    return F.rms_norm(x, (x.size(-1),))


class DeepSeekBlock(nn.Module):
    """DeepSeek V3.2 单个Transformer块"""
    
    def __init__(self, config: DeepSeekConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # 注意力层 (MLA - Multi-head Latent Attention)
        self.attn = DeepSeekMLA(config, layer_idx)
        
        # MLP层（支持MoE）
        self.mlp = DeepSeekMLP(config)
    
    def forward(self, x: torch.Tensor, cos_sin: Tuple[torch.Tensor, torch.Tensor], 
                kv_cache: Optional[dict] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 注意力层
        attn_output = self.attn(norm(x), cos_sin, kv_cache)
        x = x + attn_output
        
        # MLP层
        mlp_output, aux_loss = self.mlp(norm(x))
        x = x + mlp_output
        
        return x, aux_loss


class DeepSeekModel(nn.Module):
    """DeepSeek V3.2 完整模型"""
    
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        
        # 词嵌入
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Transformer层
        self.h = nn.ModuleList([
            DeepSeekBlock(config, layer_idx)
            for layer_idx in range(config.n_layer)
        ])
        
        # 输出层
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 旋转位置编码 (MLA 架构只需要 ROPE 部分)
        self.rotary_seq_len = config.max_seq_len
        rope_dim = getattr(config, 'rope_dim', 64)  # MLA 的 ROPE 维度
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, rope_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """权重初始化"""
        self.apply(self._init_weights)
        
        # 输出层零初始化
        torch.nn.init.zeros_(self.lm_head.weight)
        
        # 将词嵌入转换为bfloat16以节省内存
        if self.wte.weight.device.type == "cuda":
            self.wte.to(dtype=torch.bfloat16)
    
    def _init_weights(self, module):
        """模块权重初始化"""
        if isinstance(module, nn.Linear):
            # DeepSeek 推荐的初始化策略
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)
    
    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, 
                                    base: int = 1000000, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """预计算旋转位置编码"""
        if device is None:
            device = self.wte.weight.device
        
        # 通道范围
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        
        # 时间步长
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        
        # 计算频率
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        
        # 转换为bfloat16
        cos, sin = cos.bfloat16(), sin.bfloat16()
        
        # 添加广播维度 (1, seq_len, 1, head_dim/2)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        
        return cos, sin
    
    def get_device(self) -> torch.device:
        """获取模型设备"""
        return self.wte.weight.device
    
    def estimate_flops(self) -> float:
        """估计每个token的FLOPs (MLA 架构专用)"""
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.wte.weight.numel()
        
        l, h, d = (self.config.n_layer, self.config.n_head, self.config.n_embd)
        
        # MLA 架构的 FLOPs 计算
        # 1. 注意力部分: MLA 的 KV 压缩减少了计算量
        kv_lora_rank = getattr(self.config, 'kv_lora_rank', 512)
        q_lora_rank = getattr(self.config, 'q_lora_rank', 1536)
        
        # MLA 注意力 FLOPs: 压缩投影 + 注意力计算
        mla_attention_flops = (
            # Query 路径: d -> q_lora_rank -> h * (nope + rope)
            2 * d * q_lora_rank + 
            2 * q_lora_rank * h * (d // h) + 
            # KV 路径: d -> kv_lora_rank -> h * (nope + rope + v)
            2 * d * kv_lora_rank + 
            2 * kv_lora_rank * h * (d // h + getattr(self.config, 'v_head_dim', 128)) +
            # 注意力计算 (QK^T + softmax + V)
            2 * h * d // h * d // h + h * d // h * getattr(self.config, 'v_head_dim', 128)
        )
        
        # 2. MLP 部分
        mlp_flops = 2 * d * (4 * d)  # 标准 MLP (SwiGLU)
        
        # 3. 总 FLOPs 每层
        layer_flops = mla_attention_flops + mlp_flops
        
        # 4. 考虑 DeepSeekMoE 的额外计算
        if self.config.moe_enabled:
            num_routed_experts = getattr(self.config, 'num_experts', 8)
            num_shared_experts = getattr(self.config, 'num_shared_experts', 1)
            top_k = getattr(self.config, 'top_k', 2)
            
            # MoE 额外 FLOPs: 共享专家 + 路由专家
            moe_extra_flops = (
                # 共享专家 (所有token都经过)
                num_shared_experts * 2 * d * (getattr(self.config, 'shared_expert_size', 4096)) +
                # 路由专家 (每个token选择 top_k 个)
                top_k * 2 * d * (getattr(self.config, 'routed_expert_size', 2048))
            )
            
            layer_flops += moe_extra_flops
        
        # 5. 总 FLOPs 每 token
        num_flops_per_token = l * layer_flops
        
        return num_flops_per_token
    
    def setup_optimizers(self, unembedding_lr: float = 0.004, embedding_lr: float = 0.2, 
                        matrix_lr: float = 0.02, weight_decay: float = 0.0) -> list:
        """设置优化器"""
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        
        # 分离参数到不同的组
        matrix_params = list(self.h.parameters())
        embedding_params = list(self.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
        
        # AdamW优化器（词嵌入和输出层）
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling AdamW LR by ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        
        # Muon优化器（矩阵参数）
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        
        # 组合优化器
        optimizers = [adamw_optimizer, muon_optimizer]
        
        # 记录初始学习率
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        
        return optimizers
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, 
                kv_cache: Optional[dict] = None, loss_reduction: str = 'mean') -> torch.Tensor:
        """前向传播"""
        B, T = idx.size()
        
        # 验证序列长度
        assert T <= self.cos.size(1), f"序列长度超过旋转编码缓存: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, "输入和旋转编码设备不一致"
        
        # 获取旋转编码
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
        
        # 词嵌入
        x = self.wte(idx)
        x = norm(x)
        
        # 累计辅助损失（MoE）
        total_aux_loss = torch.tensor(0.0, device=x.device)
        
        # Transformer层前向传播
        for block in self.h:
            x, aux_loss = block(x, cos_sin, kv_cache)
            total_aux_loss = total_aux_loss + aux_loss
        
        # 最终归一化
        x = norm(x)
        
        # 输出层
        softcap = 15  # 平滑限制logits范围
        logits = self.lm_head(x).float()  # 转换为fp32
        logits = softcap * torch.tanh(logits / softcap)  # 限制logits
        
        if targets is not None:
            # 训练：计算损失
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                 targets.view(-1), ignore_index=-1, 
                                 reduction=loss_reduction)
            
            # 添加MoE辅助损失
            if self.config.moe_enabled and self.training:
                loss = loss + self.config.aux_loss_coef * total_aux_loss
            
            return loss
        else:
            # 推理：返回logits
            return logits
    
    @torch.inference_mode()
    def generate(self, tokens: list, max_tokens: int, temperature: float = 1.0, 
                 top_k: Optional[int] = None, seed: int = 42) -> list:
        """生成文本"""
        assert isinstance(tokens, list)
        
        device = self.get_device()
        rng = None
        
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        
        for _ in range(max_tokens):
            logits = self.forward(ids)[:, -1, :]  # 最后一个token的logits
            
            # top-k过滤
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # 采样
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            
            ids = torch.cat((ids, next_ids), dim=1)
            yield next_ids.item()


def create_deepseek_model(config: DeepSeekConfig) -> DeepSeekModel:
    """创建DeepSeek模型"""
    return DeepSeekModel(config)


def test_deepseek_model():
    """测试DeepSeek模型"""
    from nanochat.deepseek_config import get_deepseek_config
    
    # 使用小规模配置进行测试
    config = get_deepseek_config("small")
    model = DeepSeekModel(config)
    
    # 测试输入
    batch_size = 2
    seq_len = 128
    vocab_size = config.vocab_size
    
    # 随机输入
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 训练模式前向传播
    model.train()
    loss = model(input_ids, targets)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"FLOPs per token: {model.estimate_flops():.2e}")
    
    # 推理模式测试
    model.eval()
    logits = model(input_ids)
    print(f"Logits shape: {logits.shape}")
    
    print("DeepSeek Model test passed!")
    
    return model, loss


if __name__ == "__main__":
    test_deepseek_model()