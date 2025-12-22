"""
DeepSeek V3.2 模型配置
基于 DeepSeek V3.2 官方架构的配置参数
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DeepSeekConfig:
    """DeepSeek V3.2 模型配置"""
    
    # 基础参数
    vocab_size: int = 50304
    max_seq_len: int = 131072  # 128K 上下文窗口
    n_layer: int = 64
    n_head: int = 64
    n_kv_head: int = 8  # GQA 配置
    n_embd: int = 8192
    
    # MLA (Multi-head Latent Attention) 参数
    kv_lora_rank: int = 512      # KV 压缩维度 (官方 DeepSeek V3.2 使用 512)
    q_lora_rank: int = 1536      # Query 压缩维度 (官方 DeepSeek V3.2 使用 1536)
    rope_dim: int = 64           # RoPE 维度
    v_head_dim: int = 128        # Value head 维度
    
    # DeepSeekMoE 参数
    moe_enabled: bool = True
    num_experts: int = 8         # 路由专家数量
    num_shared_experts: int = 1  # 共享专家数量 (官方 DeepSeek V3.2 使用 1-2个)
    top_k: int = 2               # 每个token选择的前k个路由专家
    
    # 专家规模参数
    routed_expert_size: int = 2048      # 路由专家中间层大小
    shared_expert_size: int = 4096      # 共享专家中间层大小 (通常是路由专家的2倍)
    expert_capacity_factor: float = 1.25  # 专家容量因子
    routed_scaling_factor: float = 1.0   # 路由专家输出缩放因子
    
    # 激活函数和归一化
    activation_function: str = "swiglu"  # DeepSeek 使用 SwiGLU
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-5
    
    # 旋转位置编码
    rotary_base: int = 1000000  # 更大的base支持更长序列
    rotary_scaling: Optional[dict] = None
    
    # 训练相关参数
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    
    def __post_init__(self):
        """验证配置参数"""
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        
        # MLA 参数验证
        assert self.kv_lora_rank > 0, "kv_lora_rank must be positive"
        assert self.q_lora_rank > 0, "q_lora_rank must be positive"
        assert self.rope_dim > 0, "rope_dim must be positive"
        assert self.v_head_dim > 0, "v_head_dim must be positive"
        
        # 计算 NOPE 维度
        nope_head_dim = (self.n_embd // self.n_head) - self.rope_dim
        assert nope_head_dim > 0, f"rope_dim ({self.rope_dim}) too large for head_dim ({self.n_embd // self.n_head})"
        
        if self.moe_enabled:
            assert self.num_experts > 1, "MoE requires at least 2 routed experts"
            assert self.num_shared_experts >= 1, "DeepSeekMoE requires at least 1 shared expert"
            assert self.top_k > 0 and self.top_k <= self.num_experts, "top_k must be between 1 and num_routed_experts"
            assert self.routed_expert_size > 0, "routed_expert_size must be positive"
            assert self.shared_expert_size > 0, "shared_expert_size must be positive"


# 预定义的模型规模配置 (基于官方 DeepSeek V3.2 参数)
DEEPSEEK_V3_2_CONFIGS = {
    "small": DeepSeekConfig(
        n_layer=12,
        n_head=12,
        n_embd=768,
        max_seq_len=8192,
        kv_lora_rank=128,      # 小模型的 MLA 参数
        q_lora_rank=384,       # 小模型的 MLA 参数
        rope_dim=32,           # 小模型的 MLA 参数
        v_head_dim=64,         # 小模型的 MLA 参数
        moe_enabled=False,     # 小模型不使用 MoE
    ),
    "medium": DeepSeekConfig(
        n_layer=24,
        n_head=16,
        n_embd=1024,
        max_seq_len=16384,
        kv_lora_rank=256,      # 中等模型的 MLA 参数
        q_lora_rank=768,       # 中等模型的 MLA 参数
        rope_dim=48,           # 中等模型的 MLA 参数
        v_head_dim=96,         # 中等模型的 MLA 参数
        moe_enabled=True,      # 中等模型启用 MoE（根据用户要求）
        num_experts=4,         # 4个路由专家（用户要求）
        num_shared_experts=1,  # 1个共享专家
        routed_expert_size=2048,
        shared_expert_size=4096,
    ),
    "large": DeepSeekConfig(
        n_layer=32,
        n_head=32,
        n_embd=2048,
        max_seq_len=32768,
        kv_lora_rank=384,      # 大模型的 MLA 参数
        q_lora_rank=1152,      # 大模型的 MLA 参数
        rope_dim=64,           # 大模型的 MLA 参数
        v_head_dim=128,        # 大模型的 MLA 参数
        moe_enabled=True,      # 大模型启用 MoE
        num_experts=4,         # 4个路由专家
        num_shared_experts=1,  # 1个共享专家
        routed_expert_size=2048,
        shared_expert_size=4096,
    ),
    "xlarge": DeepSeekConfig(
        n_layer=48,
        n_head=48,
        n_embd=4096,
        max_seq_len=65536,
        kv_lora_rank=512,      # XL模型的 MLA 参数
        q_lora_rank=1536,      # XL模型的 MLA 参数
        rope_dim=64,           # XL模型的 MLA 参数
        v_head_dim=128,        # XL模型的 MLA 参数
        moe_enabled=True,      # XL模型启用 MoE
        num_experts=8,         # 8个路由专家
        num_shared_experts=1,  # 1个共享专家
        routed_expert_size=2048,
        shared_expert_size=4096,
    ),
    "full": DeepSeekConfig()  # 完整 V3.2 配置 (官方参数)
}


def get_deepseek_config(model_size: str = "medium") -> DeepSeekConfig:
    """获取预定义的 DeepSeek V3.2 配置"""
    if model_size not in DEEPSEEK_V3_2_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(DEEPSEEK_V3_2_CONFIGS.keys())}")
    
    return DEEPSEEK_V3_2_CONFIGS[model_size]