"""
nanoDeepSeek评估脚本

用于评估nanoDeepSeek模型的性能，包括：
- 损失评估
- 采样生成
- CORE指标计算
"""

import os
from contextlib import nullcontext

import torch

from nanochat.nanodeepseek import NanoDeepSeek, NanoDeepSeekConfig
from nanochat.common import (
    compute_init,
    print0,
    get_base_dir,
    autodetect_device_type
)
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import load_checkpoint
from nanochat.engine import Engine
from nanochat.loss_eval import evaluate_bpb

print0("=" * 80)
print0("nanoDeepSeek模型评估")
print0("=" * 80)

# -----------------------------------------------------------------------------
# 用户配置
device_type = ""  # cuda|cpu|mps (空=自动检测)
checkpoint_step = -1  # 要加载的检查点步骤（-1=最新）
model_tag = "nanodeepseek_base"  # 模型标签/目录名

# 评估设置
eval_tokens = 10 * 262144  # 评估token数
max_gen_tokens = 64  # 最大生成token数
temperature = 0.8  # 采样温度
top_k = 40  # Top-K采样（None=禁用）
# -----------------------------------------------------------------------------

# 计算初始化
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

# 加载分词器
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
bos_token_id = tokenizer.get_bos_token_id()

# 加载检查点
base_dir = get_base_dir()
checkpoint_dir = os.path.join(base_dir, "nanodeepseek_checkpoints", model_tag)
print0(f"从目录加载检查点: {checkpoint_dir}")

model_data, _, meta_data = load_checkpoint(
    checkpoint_dir,
    checkpoint_step,
    device,
    load_optimizer=False,
    rank=ddp_rank
)

model_config_kwargs = meta_data["model_config"]
model_config = NanoDeepSeekConfig(**model_config_kwargs)
model = NanoDeepSeek(model_config)
model.load_state_dict(model_data, strict=True, assign=True)
del model_data  # 释放内存

print0(f"模型配置:")
print0(f"  - 层数: {model_config.n_layer}")
print0(f"  - 隐藏层维度: {model_config.n_embd}")
print0(f"  - 注意力头数: {model_config.n_head}")
print0(f"  - 键值头数: {model_config.n_kv_head}")
print0(f"  - 序列长度: {model_config.sequence_len}")

num_params = sum(p.numel() for p in model.parameters())
print0(f"参数数量: {num_params:,}")

model.eval()

# -----------------------------------------------------------------------------
# 1. 评估验证集损失
print0("\n" + "=" * 80)
print0("1. 评估验证集损失")
print0("=" * 80)

from nanochat.dataloader import tokenizing_distributed_data_loader
device_batch_size = meta_data["device_batch_size"]
max_seq_len = meta_data["max_seq_len"]

val_loader = tokenizing_distributed_data_loader(
    device_batch_size,
    max_seq_len,
    split="val",
    device=device
)
eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)

with autocast_ctx:
    val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)

print0(f"验证集bpb: {val_bpb:.4f}")

# -----------------------------------------------------------------------------
# 2. 文本生成示例
print0("\n" + "=" * 80)
print0("2. 文本生成示例")
print0("=" * 80)

engine = Engine(model, tokenizer)

prompts = [
    "法国的首都是",
    "金的化学符号是",
    "如果今天是周五，那么明天是",
    "热的反义词是",
    "太阳系中的行星有:",
    "我最喜欢的颜色是",
    "如果5*x + 3 = 13，那么x是",
    "解释什么是机器学习:",
    "写一首关于春天的诗:",
    "Python是一种编程语言，它的特点是",
]

print0(f"\n生成配置:")
print0(f"  - 最大生成token数: {max_gen_tokens}")
print0(f"  - 温度: {temperature}")
print0(f"  - Top-K: {top_k}")

print0("\n生成结果:")
print0("-" * 80)

for i, prompt in enumerate(prompts, 1):
    print0(f"\n[{i}] 提示词: {prompt}")
    tokens = tokenizer(prompt, prepend="<|bos|>")
    with autocast_ctx:
        sample, _ = engine.generate_batch(
            tokens,
            num_samples=1,
            max_tokens=max_gen_tokens,
            temperature=temperature,
            top_k=top_k
        )
    generated_text = tokenizer.decode(sample[0])
    print0(f"生成: {generated_text}")

print0("\n" + "=" * 80)
print0("评估完成！")
print0("=" * 80)
