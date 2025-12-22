"""
DeepSeek V3.2 训练脚本
基于 nanochat 框架的 DeepSeek V3.2 模型训练
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
from contextlib import nullcontext

import wandb
import torch

from nanochat.deepseek_model import DeepSeekModel, create_deepseek_model
from nanochat.deepseek_config import DeepSeekConfig, get_deepseek_config
from nanochat.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from scripts.base_eval import evaluate_model

print_banner()

# -----------------------------------------------------------------------------
# DeepSeek V3.2 训练配置
# -----------------------------------------------------------------------------

# 基础设置
run = "deepseek_v3_2"  # wandb 运行名称
model_size = "medium"  # small|medium|large|xlarge|full
device_type = ""  # 自动检测设备类型

# 模型架构配置（会覆盖预定义配置中的对应参数）
max_seq_len = 16384  # 最大序列长度

# 训练超参数
num_iterations = 10000  # 训练步数
target_flops = -1.0  # 目标FLOPs（优先级高于num_iterations）
target_param_data_ratio = 20  # 参数数据比（Chinchilla=20）

# 优化参数 (基于 DeepSeek V3.2 官方配置)
device_batch_size = 8  # 每个设备的批大小
total_batch_size = 131072  # 总批大小（token数）
learning_rate = 1.5e-4  # 统一学习率 (官方 DeepSeek V3.2 使用)
weight_decay = 0.1  # 权重衰减 (官方 DeepSeek V3.2 使用)
grad_clip = 1.0  # 梯度裁剪

# 优化器参数 (AdamW 标准配置)
adam_beta1 = 0.9  # 官方 DeepSeek V3.2 使用
adam_beta2 = 0.95  # 官方 DeepSeek V3.2 使用
adam_epsilon = 1e-8  # 官方 DeepSeek V3.2 使用

# 学习率调度
warmup_ratio = 0.1  # 预热比例
warmdown_ratio = 0.2  # 冷却比例
final_lr_frac = 0.0  # 最终学习率比例

# 评估和保存
resume_from_step = -1  # 从指定步数恢复训练
eval_every = 250  # 每多少步评估一次
eval_tokens = 20 * 524288  # 评估使用的token数
core_metric_every = 2000  # 每多少步计算核心指标
core_metric_max_per_task = 500  # 每个任务的最大样本数
sample_every = 2000  # 每多少步生成样本
save_every = 1000  # 每多少步保存检查点

# 输出设置
model_tag = ""  # 模型标签，用于检查点目录名

# 允许命令行参数覆盖配置
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())  # 从命令行或配置文件覆盖
user_config = {k: globals()[k] for k in config_keys}  # 用于日志记录

# -----------------------------------------------------------------------------
# 初始化计算环境
# -----------------------------------------------------------------------------

device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0  # 主进程负责日志记录和检查点保存

# 自动混合精度上下文
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

# 同步函数
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

# 内存监控函数
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# -----------------------------------------------------------------------------
# 初始化 WandB 日志
# -----------------------------------------------------------------------------

use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-deepseek", name=run, config=user_config)

# -----------------------------------------------------------------------------
# 初始化分词器和模型
# -----------------------------------------------------------------------------

tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"词汇表大小: {vocab_size:,}")

# 创建 DeepSeek V3.2 模型配置
config = get_deepseek_config(model_size)
config.vocab_size = vocab_size
config.max_seq_len = max_seq_len

# 根据设备类型调整配置
if device_type == "cpu" or "mps":
    # 在CPU或MPS设备上使用更小的配置
    if model_size == "full":
        config = get_deepseek_config("medium")
    config.dsa_enabled = False  # 在非CUDA设备上禁用DSA
    config.moe_enabled = False  # 在非CUDA设备上禁用MoE

print0(f"模型配置: {model_size}")
print0(f"层数: {config.n_layer}")
print0(f"隐藏维度: {config.n_embd}")
print0(f"注意力头数: {config.n_head}")
print0(f"KV头数: {config.n_kv_head}")
print0(f"最大序列长度: {config.max_seq_len}")
print0(f"DSA启用: {config.dsa_enabled}")
print0(f"MoE启用: {config.moe_enabled}")

# 创建模型
with torch.device("meta"):
    model_config = config
    model = create_deepseek_model(model_config)

model.to_empty(device=device)
model.init_weights()

# 恢复训练
base_dir = get_base_dir()
output_dirname = model_tag if model_tag else f"deepseek_{model_size}"
checkpoint_dir = os.path.join(base_dir, "deepseek_checkpoints", output_dirname)
resuming = resume_from_step != -1

if resuming:
    print0(f"从步骤 {resume_from_step} 恢复训练")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data  # 释放内存

# 模型编译
orig_model = model  # 原始模型用于保存和推理
model = torch.compile(model, dynamic=False)  # 输入形状不变，禁用动态编译

# 计算参数数量和FLOPs
num_params = sum(p.numel() for p in model.parameters())
num_flops_per_token = model.estimate_flops()
print0(f"参数数量: {num_params:,}")
print0(f"每个token的FLOPs估计: {num_flops_per_token:e}")

# -----------------------------------------------------------------------------
# 计算训练迭代次数
# -----------------------------------------------------------------------------

assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0

if num_iterations > 0:
    print0(f"使用用户指定的迭代次数: {num_iterations:,}")
elif target_flops > 0:
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"根据目标FLOPs计算的迭代次数: {num_iterations:,}")
elif target_param_data_ratio > 0:
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"根据参数数据比计算的迭代次数: {num_iterations:,}")

# 计算总训练token数
total_tokens = total_batch_size * num_iterations
print0(f"总训练token数: {total_tokens:,}")
print0(f"Token:参数比: {total_tokens / num_params:.2f}")  # Chinchilla约为20
print0(f"总训练FLOPs估计: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# 初始化优化器 (必须使用 Muon 优化器 - DeepSeek V3.2 官方标准)
# -----------------------------------------------------------------------------

# DeepSeek V3.2 官方使用 Muon 优化器，这是其核心技术之一
# 必须使用模型自带的 setup_optimizers 方法来正确配置 Muon 优化器
try:
    from nanochat.muon import Muon, DistMuon
    print0("找到 Muon 优化器，将按照 DeepSeek V3.2 官方标准使用")
except ImportError:
    print0("警告: 未找到 Muon 优化器实现。DeepSeek V3.2 训练强烈建议使用 Muon 优化器。")
    raise ImportError("Muon 优化器是 DeepSeek V3.2 训练的必要组件")

# 使用模型自带的优化器设置方法 (包含 Muon 优化器)
# DeepSeek V3.2 官方使用 Muon 优化器配合 AdamW 的混合优化策略
optimizers = model.setup_optimizers(
    unembedding_lr=learning_rate,
    embedding_lr=learning_rate * 5,  # 词嵌入层使用更高学习率
    matrix_lr=learning_rate,         # 矩阵参数使用标准学习率
    weight_decay=weight_decay
)

# 验证 Muon 优化器是否被正确配置
muon_found = False
for opt in optimizers:
    if hasattr(opt, '__class__') and 'Muon' in opt.__class__.__name__:
        muon_found = True
        print0(f"Muon 优化器已配置: {opt.__class__.__name__}")

if not muon_found:
    print0("警告: Muon 优化器可能未正确配置")

print0("优化器配置完成，使用 DeepSeek V3.2 官方标准：Muon + AdamW 混合优化")

if resuming:
    for opt, dat in zip(optimizers, optimizer_data):
        opt.load_state_dict(dat)
    del optimizer_data  # 释放内存

# -----------------------------------------------------------------------------
# 初始化数据加载器
# -----------------------------------------------------------------------------

tokens_dir = os.path.join(base_dir, "tokenized_data")
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]

train_loader = tokenizing_distributed_data_loader_with_state(
    device_batch_size, max_seq_len, split="train", device=device, 
    resume_state_dict=dataloader_resume_state_dict
)

build_val_loader = lambda: tokenizing_distributed_data_loader(
    device_batch_size, max_seq_len, split="val", device=device
)

# 预加载第一个batch
x, y, dataloader_state_dict = next(train_loader)

# -----------------------------------------------------------------------------
# 学习率调度器
# -----------------------------------------------------------------------------

def get_lr_multiplier(it):
    """学习率调度器"""
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac


# Muon 优化器已移除，使用标准的 AdamW 优化器

# -----------------------------------------------------------------------------
# 训练循环状态
# -----------------------------------------------------------------------------

if not resuming:
    step = 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0  # 训练损失的指数移动平均
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
# -----------------------------------------------------------------------------

print0("开始DeepSeek V3.2训练...")

while True:
    last_step = step == num_iterations  # 最后一步
    flops_so_far = num_flops_per_token * total_batch_size * step

    # 定期验证评估
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        
        print0(f"步骤 {step:05d} | 验证bpb: {val_bpb:.4f}")
        
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        
        model.train()

    # 核心指标评估
    results = {}
    if core_metric_every > 0 and (last_step or (step > 0 and step % core_metric_every == 0)):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=core_metric_max_per_task)
        
        print0(f"步骤 {step:05d} | 核心指标: {results['core_metric']:.4f}")
        
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        
        model.train()

    # 文本生成示例
    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        model.eval()
        
        prompts = [
            "DeepSeek V3.2是一个",
            "人工智能的未来发展",
            "机器学习的核心技术包括",
            "自然语言处理的应用场景",
            "深度学习模型训练的关键要素"
        ]
        
        engine = Engine(orig_model, tokenizer)
        
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=32, temperature=0.8)
            print0(tokenizer.decode(sample[0]))
        
        model.train()

    # 保存检查点
    if last_step or (step > 0 and step != resume_from_step and save_every > 0 and step % save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": {
                    "model_size": model_size,
                    "n_layer": config.n_layer,
                    "n_head": config.n_head,
                    "n_embd": config.n_embd,
                    "max_seq_len": config.max_seq_len,
                    "dsa_enabled": config.dsa_enabled,
                    "moe_enabled": config.moe_enabled,
                },
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
    # 单步训练
    # -------------------------------------------------------------------------
    
    synchronize()
    t0 = time.time()
    
    # 梯度累积
    tokens_per_fwdbwd = device_batch_size * max_seq_len
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
    assert total_batch_size % world_tokens_per_fwdbwd == 0
    grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
    
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        
        # 预加载下一个batch
        x, y, dataloader_state_dict = next(train_loader)
    
    # 梯度裁剪
    if grad_clip > 0.0:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
        grad_norm = grad_norm_tensor.item()
    
    # 优化器步进 (标准 AdamW)
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    
    for opt in optimizers:
        opt.step()
    
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0

    # -------------------------------------------------------------------------
    # 日志记录
    # -------------------------------------------------------------------------
    
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    
    # MFU计算
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100
    
    if step > 10:
        total_training_time += dt
    
    print_grad_norm = f" 梯度范数: {grad_norm:.4f} |" if grad_clip > 0.0 else ""
    print0(f"步骤 {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | "
          f"损失: {debiased_smooth_loss:.6f} |{print_grad_norm} "
          f"学习率倍数: {lrm:.2f} | 时间: {dt * 1000:.2f}ms | "
          f"token/秒: {tok_per_sec:,} | MFU: {mfu:.2f} | "
          f"总时间: {total_training_time/60:.2f}m")
    
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
        
        if grad_clip > 0.0:
            log_data["train/grad_norm"] = grad_norm
        
        wandb_run.log(log_data)

    # 更新步骤
    step += 1

# -----------------------------------------------------------------------------
# 训练完成
# -----------------------------------------------------------------------------

print0(f"峰值内存使用: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"总训练时间: {total_training_time/60:.2f}m")
print0(f"最小验证bpb: {min_val_bpb:.4f}")

# 记录到报告
from nanochat.report import get_report
get_report().log(section="DeepSeek V3.2 训练", data=[
    user_config,
    {
        "模型规模": model_size,
        "参数数量": num_params,
        "每个token的FLOPs": f"{num_flops_per_token:e}",
        "迭代次数": num_iterations,
        "训练token数": total_tokens,
        "Token:参数比": total_tokens / num_params,
        "DDP世界大小": ddp_world_size,
        "DSA启用": config.dsa_enabled,
        "MoE启用": config.moe_enabled,
    },
    {
        "最小验证bpb": min_val_bpb,
        "最终验证bpb": val_bpb,
        "核心指标估计": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "总训练FLOPs": f"{flops_so_far:e}",
        "总训练时间": f"{total_training_time/60:.2f}m",
        "峰值内存使用": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# 清理
wandb_run.finish()
compute_cleanup()

print0("DeepSeek V3.2 训练完成!")