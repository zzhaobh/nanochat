"""
DeepSeek V3.2 DeepSeekMoE 实现
基于 DeepSeek V3.2 官方架构的共享专家+路由专家混合架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from nanochat.common import print0


class Expert(nn.Module):
    """单个专家网络 (支持不同规模的专家)"""
    
    def __init__(self, hidden_size: int, intermediate_size: int, activation: str = "swiglu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation
        
        # 专家网络层 - 使用 SwiGLU 激活
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # SwiGLU 激活函数 (DeepSeek 标准)
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden_states = gate * up
        
        return self.down_proj(hidden_states)


class DeepSeekMoELayer(nn.Module):
    """DeepSeekMoE 层 - 共享专家 + 路由专家的混合架构"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # DeepSeekMoE 核心参数
        self.num_routed_experts = getattr(config, 'num_experts', 8)  # 路由专家数量
        self.num_shared_experts = getattr(config, 'num_shared_experts', 1)  # 共享专家数量
        self.top_k = getattr(config, 'top_k', 2)  # 每个token选择的前k个路由专家
        
        # 专家规模参数
        self.routed_expert_size = getattr(config, 'routed_expert_size', config.n_embd * 4)
        self.shared_expert_size = getattr(config, 'shared_expert_size', config.n_embd * 8)
        
        # 路由缩放因子
        self.routed_scaling_factor = getattr(config, 'routed_scaling_factor', 1.0)
        
        # 创建路由专家网络 (较小的专家)
        self.routed_experts = nn.ModuleList([
            Expert(config.n_embd, self.routed_expert_size, config.activation_function)
            for _ in range(self.num_routed_experts)
        ])
        
        # 创建共享专家网络 (较大的专家)
        self.shared_experts = nn.ModuleList([
            Expert(config.n_embd, self.shared_expert_size, config.activation_function)
            for _ in range(self.num_shared_experts)
        ])
        
        # 路由网络
        self.router = nn.Linear(config.n_embd, self.num_routed_experts, bias=False)
        
        # 专家容量因子
        self.expert_capacity_factor = getattr(config, 'expert_capacity_factor', 1.25)
        
        # 负载均衡相关
        self.expert_usage_history = torch.zeros(self.num_routed_experts)
        self.aux_loss_coef = 0.01  # 辅助损失系数 (DeepSeek V3 使用 Auxiliary-Loss-Free)
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
    
    def _compute_router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """计算路由logits"""
        # 应用 RMSNorm
        hidden_states = F.rms_norm(hidden_states, (hidden_states.size(-1),))
        
        # 路由计算
        router_logits = self.router(hidden_states)
        
        # DeepSeek V3 的 Auxiliary-Loss-Free 负载均衡
        # 通过动态bias调节负载，而不是通过aux loss
        if self.training:
            # 计算专家历史负载
            expert_usage = self.expert_usage_history / (self.expert_usage_history.sum() + 1e-8)
            
            # 应用负载均衡bias (鼓励使用较少使用的专家)
            load_balancing_bias = -torch.log(expert_usage + 1e-8)
            router_logits = router_logits + load_balancing_bias.unsqueeze(0)
        
        return router_logits
    
    def _compute_aux_loss(self, router_probs: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """计算辅助损失（兼容性保留，DeepSeek V3 已弃用）"""
        if not self.training:
            return torch.tensor(0.0, device=router_probs.device)
        
        # DeepSeek V3 使用 Auxiliary-Loss-Free，这里保留兼容性
        # 计算每个专家被选择的概率
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_routed_experts)
        expert_usage = expert_mask.float().mean(dim=0)
        
        # 负载均衡损失
        router_prob_mean = router_probs.mean(dim=0)
        aux_loss = torch.sum(expert_usage * router_prob_mean) * self.num_routed_experts
        
        return aux_loss
    
    def _compute_expert_capacity(self, total_tokens: int) -> int:
        """计算专家容量"""
        capacity = int(self.expert_capacity_factor * total_tokens / self.num_routed_experts)
        return max(capacity, 1)
    
    def _process_routed_experts(self, hidden_states_flat: torch.Tensor, 
                               router_logits: torch.Tensor) -> torch.Tensor:
        """处理路由专家计算"""
        batch_size_seq, hidden_size = hidden_states_flat.shape
        
        # 计算路由概率
        router_probs = F.softmax(router_logits, dim=-1)
        
        # 选择 top-k 路由专家
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # 计算专家容量
        expert_capacity = self._compute_expert_capacity(batch_size_seq)
        
        # 创建专家掩码和路由权重
        expert_mask = torch.zeros_like(router_probs)
        expert_mask.scatter_(1, topk_indices, 1)
        
        # 归一化路由权重
        routing_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        # 初始化输出
        routed_output = torch.zeros_like(hidden_states_flat)
        
        # 更新专家使用历史
        if self.training:
            expert_usage = expert_mask.float().mean(dim=0)
            self.expert_usage_history = (0.9 * self.expert_usage_history + 0.1 * expert_usage)
        
        # 为每个路由专家处理输入
        for expert_idx in range(self.num_routed_experts):
            expert_mask_bool = expert_mask[:, expert_idx].bool()
            
            if not expert_mask_bool.any():
                continue
            
            # 获取分配给当前专家的token索引
            token_indices = torch.where(expert_mask_bool)[0]
            
            # 容量控制
            if len(token_indices) > expert_capacity:
                token_indices = token_indices[:expert_capacity]
            
            # 获取对应的输入和路由权重
            expert_input = hidden_states_flat[token_indices]
            expert_weights = routing_weights[token_indices, expert_mask[token_indices, expert_idx].bool()]
            
            # 专家前向传播
            expert_output = self.routed_experts[expert_idx](expert_input)
            
            # 加权输出
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            routed_output[token_indices] += weighted_output
        
        return routed_output
    
    def _process_shared_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """处理共享专家计算"""
        shared_output = torch.zeros_like(hidden_states)
        
        # 所有共享专家的输出求和
        for shared_expert in self.shared_experts:
            shared_output += shared_expert(hidden_states)
        
        return shared_output
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        if not self.config.moe_enabled:
            # 如果 MoE 被禁用，使用第一个路由专家
            output = self.routed_experts[0](hidden_states)
            return output, torch.tensor(0.0, device=hidden_states.device)
        
        # DeepSeekMoE 核心：共享专家 + 路由专家
        
        # 1. 计算路由logits
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        router_logits = self._compute_router_logits(hidden_states_flat)
        
        # 2. 路由专家计算
        routed_output = self._process_routed_experts(hidden_states_flat, router_logits)
        
        # 3. 共享专家计算 (所有token都经过共享专家)
        shared_output = self._process_shared_experts(hidden_states)
        
        # 4. 合并输出 (DeepSeekMoE 的核心思想)
        routed_output = routed_output.view(batch_size, seq_len, hidden_size)
        final_output = shared_output + self.routed_scaling_factor * routed_output
        
        # 5. 辅助损失 (DeepSeek V3 已弃用，保留兼容性)
        aux_loss = torch.tensor(0.0, device=hidden_states.device)
        
        return final_output, aux_loss


class DeepSeekMLP(nn.Module):
    """DeepSeek V3.2 MLP 层（支持 DeepSeekMoE）"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        if config.moe_enabled:
            # 使用 DeepSeekMoE 层
            self.moe_layer = DeepSeekMoELayer(config)
        else:
            # 使用标准的 MLP (SwiGLU)
            intermediate_size = config.n_embd * 4
            
            # DeepSeek V3.2 使用 SwiGLU 激活
            self.gate_proj = nn.Linear(config.n_embd, intermediate_size, bias=False)
            self.up_proj = nn.Linear(config.n_embd, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, config.n_embd, bias=False)
        
        self.activation_function = config.activation_function
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        if hasattr(self, 'gate_proj'):
            nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)
        
        # DeepSeekMoE 层的初始化在内部处理
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        if self.config.moe_enabled:
            # 使用 DeepSeekMoE
            return self.moe_layer(x)
        else:
            # 使用标准 MLP (SwiGLU)
            gate = F.silu(self.gate_proj(x))
            up = self.up_proj(x)
            hidden_states = gate * up
            output = self.down_proj(hidden_states)
            
            return output, torch.tensor(0.0, device=x.device)


def test_deepseek_moe_layer():
    """测试 DeepSeekMoE 层"""
    from nanochat.deepseek_config import DeepSeekConfig
    
    config = DeepSeekConfig(
        n_embd=256,
        moe_enabled=True,
        num_experts=4,
        num_shared_experts=1,  # DeepSeekMoE 参数
        top_k=2
    )
    
    model = DeepSeekMLP(config)
    
    # 测试输入
    batch_size = 2
    seq_len = 32
    x = torch.randn(batch_size, seq_len, config.n_embd)
    
    # 前向传播
    output, aux_loss = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"共享专家数量: {config.num_shared_experts}")
    print(f"路由专家数量: {config.num_experts}")
    print("DeepSeekMoE Layer test passed!")
    
    return output, aux_loss


if __name__ == "__main__":
    test_deepseek_moe_layer()