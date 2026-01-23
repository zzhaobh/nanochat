"""
nanoDeepSeek训练脚本

基于nanoDeepSeek模型的训练脚本，支持：
- 小规模数据集训练
- 分布式训练（DDP）
- 梯度累积
- 学习率调度
- 检查点保存和恢复
- WandB日志记录

使用方法:
  python -m scripts.nanodeepseek_train

或分布式:
  torchrun --nproc_per_node=8 -m scripts.nanodeepseek_train
"""

import os
import time
from contextlib import nullcontext

import wandb
import torch

from nanochat.nanodeepseek import (
    NanoDeepSeek,
    NanoDeepSeekConfig,
    create_nano_deepseek
)
from nanochat.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state
from nanochat.common import (
    compute_init,
    compute_cleanup,
    print0,
    DummyWandb,
    print_banner,
    get_base_dir,
    autodetect_device_type
)
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine

print_banner()


# -----------------------------------------------------------------------------
# 用户配置
run = "nanodeepseek_dummy"  # wandb运行名称
device_type = ""  # cuda|cpu|mps (空=自动检测)

# 模型架构
scale = "base"  # small|base|large
max_seq_len = 1024  # 最大上下文长度

# 训练范围
num_iterations = 1000  # 优化步数（-1=禁用）
target_flops = -1.0  # 计算步数以达到目标FLOPs（-1=禁用）
target_param_data_ratio = 20  # 计算步数以维持固定的数据:参数比

# 优化
device_batch_size = 16  # 每设备批大小
total_batch_size = 262144  # 总期望批大小（token数）
embedding_lr = 0.2  # 嵌入参数学习率
unembedding_lr = 0.004  # 解嵌入参数学习率
weight_decay = 0.0  # 权重衰减
matrix_lr = 0.02  # 矩阵参数学习率
grad_clip = 1.0  # 梯度裁剪值（0.0=禁用）
warmup_ratio = 0.0  # LR预热比例
warmdown_ratio = 0.2  # LR冷却比例
final_lr_frac = 0.0  # 最终LR是初始LR的比例
resume_from_step = -1  # 从该步恢复训练（-1=禁用）

# 评估
eval_every = 100  # 每隔多少步评估验证集bpb
eval_tokens = 10 * 262144  # 评估验证损失的token数
sample_every = 200  # 每隔多少步从模型采样
save_every = -1  # 每隔多少步保存检查点（-1=仅在结束时保存）

# 输出
model_tag = ""  # 可选地覆盖输出检查点目录名的模型标签

# 允许CLI通过configurator覆盖设置
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# 计算初始化
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# WandB日志初始化
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=run, config=user_config)

# 分词器（需要词汇表大小）
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"词汇表大小: {vocab_size:,}")

# 从scale推导模型参数
model = create_nano_deepseek(scale=scale, sequence_len=max_seq_len, vocab_size=vocab_size)
model_config = model.config

print0(f"模型规模: {scale}")
print0(f"层数: {model_config.n_layer}")
print0(f"隐藏层维度: {model_config.n_embd}")
print0(f"注意力头数: {model_config.n_head}")
print0(f"键值头数: {model_config.n_kv_head}")

# 优化器/数据/训练长度相关的超参数
# 计算所需的梯度累积以达到期望的总批大小
tokens_per_fwdbwd = device_batch_size * max_seq_len  # 每个rank每次迭代的token数
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size  # 所有rank的总token数
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"每rank微批token数: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"总微批token数: {world_tokens_per_fwdbwd:,}")
print0(f"总批大小 {total_batch_size:,} => 梯度累积步数: {grad_accum_steps}")

# -----------------------------------------------------------------------------
# 初始化模型

# 创建随机权重的新模型
model_config_kwargs = dict(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=model_config.n_layer,
    n_head=model_config.n_head,
    n_kv_head=model_config.n_kv_head,
    n_embd=model_config.n_embd
)
with torch.device("meta"):
    model_config_obj = NanoDeepSeekConfig(**model_config_kwargs)
    model = NanoDeepSeek(model_config_obj)
model.to_empty(device=device)
model.init_weights()

# 如果恢复，用检查点的参数覆盖模型参数
base_dir = get_base_dir()
output_dirname = model_tag if model_tag else f"nanodeepseek_{scale}"
checkpoint_dir = os.path.join(base_dir, "nanodeepseek_checkpoints", output_dirname)
resuming = resume_from_step != -1
if resuming:
    print0(f"从步骤 {resume_from_step} 恢复优化")
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir,
        resume_from_step,
        device,
        load_optimizer=True,
        rank=ddp_rank
    )
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data  # 复制后释放内存

orig_model = model  # 原始未编译模型，用于保存原始state_dict和推理/评估
model = torch.compile(model, dynamic=False)  # 模型输入不会改变形状，所以dynamic=False是安全的
num_params = sum(p.numel() for p in model.parameters())
print0(f"参数数量: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"估计的每token FLOPs: {num_flops_per_token:e}")

# 计算迭代次数
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"使用用户提供的迭代次数: {num_iterations:,}")
elif target_flops > 0:
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"从目标FLOPs计算迭代次数: {num_iterations:,}")
elif target_param_data_ratio > 0:
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"从目标数据:参数比计算迭代次数: {num_iterations:,}")
else:
    raise ValueError("未指定训练范围")

total_tokens = total_batch_size * num_iterations
print0(f"总训练token数: {total_tokens:,}")
print0(f"Tokens:Params比例: {total_batch_size * num_iterations / num_params:.2f}")
print0(f"总训练FLOPs估计: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# 初始化优化器（线性层用Muon，嵌入和lm_head用AdamW）
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay
)
adamw_optimizer, muon_optimizer = optimizers

if resuming:
    for opt, dat in zip(optimizers, optimizer_data):
        opt.load_state_dict(dat)
    del optimizer_data  # 释放内存

# -----------------------------------------------------------------------------
# 初始化训练/验证数据加载器
tokens_dir = os.path.join(base_dir, "tokenized_data")
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state(
    device_batch_size,
    max_seq_len,
    split="train",
    device=device,
    resume_state_dict=dataloader_resume_state_dict
)
build_val_loader = lambda: tokenizing_distributed_data_loader(
    device_batch_size,
    max_seq_len,
    split="val",
    device=device
)
x, y, dataloader_state_dict = next(train_loader)  # 启动第一个批的数据加载

# -----------------------------------------------------------------------------
# 设置超参数调度器

# 学习率调度器
def get_lr_multiplier(it: int) -> float:
    """获取学习率乘数"""
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

# Muon优化器的动量调度器
def get_muon_momentum(it: int) -> float:
    """获取Muon优化器的动量"""
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# 循环状态（训练循环更新的变量）

if not resuming:
    step = 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0  # 训练损失的EMA
    total_training_time = 0  # 总训练时间
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# -----------------------------------------------------------------------------
# 训练循环
while True:
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step

    # 评估验证集bpb（所有rank参与）
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"步骤 {step:05d} | 验证集bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # 从模型采样（仅在主进程）
    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        model.eval()
        prompts = [
            "法国的首都是",
            "金的化学符号是",
            "如果今天是周五，那么明天是",
            "热的反义词是",
            "太阳系中的行星有:",
            "我最喜欢的颜色是",
            "如果5*x + 3 = 13，那么x是",
        ]
        engine = Engine(orig_model, tokenizer)  # 使用orig_model避免重新编译
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(
                    tokens,
                    num_samples=1,
                    max_tokens=16,
                    temperature=0
                )
            print0(tokenizer.decode(sample[0]))
        model.train()

    # 保存检查点：在运行结束时，或每隔save_every步
    if last_step or (step > 0 and step != resume_from_step and save_every > 0 and step % save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            {  # 元数据保存为json
                "step": step,
                "val_bpb": val_bpb,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": {
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # 终止条件
    if last_step:
        break

    # -------------------------------------------------------------------------
    # 单个训练步骤
    # 评估梯度
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()  # 用于日志记录
        loss = loss / grad_accum_steps  # 每个.backward()是梯度和 => 在这里标准化损失
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader)  # 在GPU忙于前向/后向时预取下一批

    # 梯度裁剪
    grad_clip_enabled = grad_clip > 0.0
    if grad_clip_enabled:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
        grad_norm = grad_norm_tensor.item()  # GPU张量 -> CPU浮点数

    # 步进优化器
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # 日志记录
    ema_beta = 0.9  # EMA衰减因子，用于更平滑的日志记录
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size  # bfloat16 H100 SXM，无2:4稀疏性
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100  # 百分比
    if step > 10:
        total_training_time += dt  # 只计算前10步之后的时间

    print_grad_norm = f" 梯度范数: {grad_norm:.4f} |" if grad_clip_enabled else ""
    print0(
        f"步骤 {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | "
        f"损失: {debiased_smooth_loss:.6f} |{print_grad_norm} "
        f"lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | "
        f"tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | "
        f"总时间: {total_training_time/60:.2f}m"
    )

    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        }
        if grad_clip_enabled:
            log_data["train/grad_norm"] = grad_norm
        wandb_run.log(log_data)

    # 状态更新
    step += 1

# 打印更多统计信息
print0(f"峰值内存使用: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"总训练时间: {total_training_time/60:.2f}m")
print0(f"最小验证集bpb: {min_val_bpb:.4f}")

# 记录到报告
from nanochat.report import get_report
get_report().log(
    section="nanoDeepSeek模型训练",
    data=[
        user_config,  # CLI参数
        {  # 训练设置统计
            "参数数量": num_params,
            "每token FLOPs": f"{num_flops_per_token:e}",
            "计算的迭代次数": num_iterations,
            "训练token数": total_tokens,
            "Tokens:Params比例": total_batch_size * num_iterations / num_params,
            "DDP world size": ddp_world_size,
            "warmup_ratio": warmup_ratio,
            "warmdown_ratio": warmdown_ratio,
            "final_lr_frac": final_lr_frac,
        },
        {  # 训练结果统计
            "最小验证集bpb": min_val_bpb,
            "最终验证集bpb": val_bpb,
            "MFU %": f"{mfu:.2f}%",
            "总训练flops": f"{flops_so_far:e}",
            "总训练时间": f"{total_training_time/60:.2f}m",
            "峰值内存使用": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
        }
    ]
)

# 清理
wandb_run.finish()
compute_cleanup()
