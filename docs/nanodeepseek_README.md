# nanoDeepSeek - 精简版DeepSeek v3.2架构实现

## 概述

nanoDeepSeek是基于DeepSeek v3.2架构的精简版实现，保持了与原架构相同的Transformer基础结构，同时针对小规模训练和推理进行了优化。该项目与nanochat代码库完全集成，支持分布式训练、梯度累积、KV缓存推理等高级功能。

## 主要特性

### 模型架构特性
- **精简的多头注意力机制**：减少注意力头数和隐藏层维度，适合小规模训练
- **旋转位置编码**：无需可学习参数，提供相对位置编码
- **QK归一化**：稳定训练，提高模型性能
- **Group-Query Attention (GQA)**：支持高效的键值头共享，减少推理时的内存占用
- **RMSNorm归一化**：无学习参数的归一化方法，计算高效
- **Relu²激活函数**：提升模型表达能力

### 训练特性
- **混合优化器策略**：嵌入层使用AdamW，矩阵层使用Muon优化器
- **梯度累积**：支持小批量训练，模拟大批次训练效果
- **学习率调度**：支持预热和冷却调度
- **分布式训练**：支持多GPU分布式数据并行（DDP）
- **检查点管理**：支持训练中断后的恢复

### 推理特性
- **KV缓存**：高效的自回归推理，显著加速生成过程
- **多种采样策略**：支持贪婪解码、Top-K采样、温度采样
- **批处理推理**：支持并行生成多个样本

## 模型规模

提供三种预设规模的模型配置：

| 规模 | 层数 | 隐藏层维度 | 注意力头数 | 键值头数 | 参数数量（约） |
|------|------|-----------|-----------|---------|---------------|
| small | 6 | 384 | 4 | 4 | 8M |
| base | 12 | 768 | 6 | 6 | 35M |
| large | 24 | 1536 | 12 | 6 | 150M |

## 项目结构

```
nanochat/
├── nanodeepseek.py           # nanoDeepSeek模型核心实现
├── gpt.py                    # 原GPT模型（参考实现）
└── ...

scripts/
├── nanodeepseek_train.py     # nanoDeepSeek训练脚本
├── nanodeepseek_eval.py      # nanoDeepSeek评估脚本
├── nanodeepseek_demo.py      # nanoDeepSeek演示脚本
└── ...
```

## 快速开始

### 1. 环境准备

确保已安装nanochat项目的依赖：

```bash
# 使用uv安装依赖
uv sync

# 或使用pip
pip install -e .
```

### 2. 演示模型

运行演示脚本，快速了解nanoDeepSeek的功能：

```bash
python -m scripts.nanodeepseek_demo
```

该脚本将：
- 创建一个小规模的nanoDeepSeek模型
- 在合成数据上进行简单训练
- 演示文本生成功能
- 展示模型架构细节

### 3. 训练模型

#### CPU/MPS训练（小规模模型）

```bash
python -m scripts.nanodeepseek_train \
  --scale=small \
  --max_seq_len=512 \
  --device_batch_size=2 \
  --total_batch_size=4096 \
  --num_iterations=100 \
  --eval_every=20
```

#### GPU训练（中等规模模型）

```bash
python -m scripts.nanodeepseek_train \
  --scale=base \
  --max_seq_len=1024 \
  --device_batch_size=16 \
  --total_batch_size=262144 \
  --num_iterations=1000 \
  --eval_every=100
```

#### 多GPU分布式训练

```bash
torchrun --nproc_per_node=4 -m scripts.nanodeepseek_train \
  --scale=large \
  --max_seq_len=2048 \
  --device_batch_size=8 \
  --total_batch_size=524288 \
  --num_iterations=5000 \
  --eval_every=200
```

### 4. 评估模型

```bash
python -m scripts.nanodeepseek_eval \
  --model_tag=nanodeepseek_base \
  --checkpoint_step=-1
```

## 模型使用示例

### Python API使用

```python
import torch
from nanochat.nanodeepseek import create_nano_deepseek

# 创建模型
model = create_nano_deepseek(scale="base")
model.to("cuda")
model.init_weights()

# 文本生成
tokens = [1, 2, 3]  # 输入token序列
for token_id in model.generate(tokens, max_tokens=50, temperature=0.8):
    print(f"Generated token: {token_id}")
```

### 训练自定义数据

```python
from torch.utils.data import DataLoader
from nanochat.nanodeepseek import NanoDeepSeek, NanoDeepSeekConfig

# 创建模型
config = NanoDeepSeekConfig(
    n_layer=12,
    n_embd=768,
    n_head=6,
    n_kv_head=6,
    sequence_len=1024,
    vocab_size=50000
)
model = NanoDeepSeek(config)
model.init_weights()

# 设置优化器
optimizers = model.setup_optimizers()
adamw_optimizer, muon_optimizer = optimizers

# 训练循环
for x, y in dataloader:
    loss = model(x, y)
    
    # 反向传播
    for opt in optimizers:
        opt.zero_grad()
    loss.backward()
    for opt in optimizers:
        opt.step()
```

## 配置参数说明

### 模型架构参数
- `scale`: 模型规模（small/base/large）
- `n_layer`: Transformer层数
- `n_embd`: 嵌入维度/隐藏层维度
- `n_head`: 注意力头数
- `n_kv_head`: 键值头数（用于GQA）
- `sequence_len`: 最大序列长度
- `vocab_size`: 词汇表大小

### 训练参数
- `device_batch_size`: 每设备批大小
- `total_batch_size`: 总期望批大小（token数）
- `num_iterations`: 训练步数
- `embedding_lr`: 嵌入层学习率
- `unembedding_lr`: 输出层学习率
- `matrix_lr`: 矩阵层学习率
- `grad_clip`: 梯度裁剪值
- `warmup_ratio`: 学习率预热比例
- `warmdown_ratio`: 学习率冷却比例

### 评估参数
- `eval_every`: 评估间隔步数
- `eval_tokens`: 评估token数
- `sample_every`: 采样间隔步数
- `save_every`: 检查点保存间隔步数

## 与nanochat集成

nanoDeepSeek完全集成到nanochat生态系统中：

1. **数据加载器**：使用`tokenizing_distributed_data_loader`
2. **分词器**：使用项目统一的分词器
3. **检查点管理**：使用`save_checkpoint`和`load_checkpoint`
4. **推理引擎**：与`Engine`类兼容，支持KV缓存推理
5. **评估工具**：使用`evaluate_bpb`等评估函数
6. **日志记录**：集成WandB日志记录

## 技术细节

### 旋转位置编码

nanoDeepSeek使用旋转位置编码，通过将查询和键向量旋转来编码位置信息。这种方法无需可学习参数，且能更好地处理外推到训练中未见过的序列长度。

### QK归一化

在计算注意力分数前，对查询和键进行RMS归一化，这有助于稳定训练，特别是在深层网络中。

### GQA (Group-Query Attention)

通过减少键值头的数量来降低推理时的内存占用和计算量。查询头可以共享键值头，实现高效的多头注意力。

### 混合优化器策略

- **AdamW**：用于嵌入层和输出层，提供稳定的学习
- **Muon**：用于线性层，提供更快的收敛速度和更好的泛化性能

## 性能参考

在小规模数据集上的性能（示例数据）：

| 模型规模 | 训练数据量 | 验证损失 | 训练时间（单GPU） |
|---------|-----------|---------|------------------|
| small | 1M tokens | ~3.2 | ~10分钟 |
| base | 10M tokens | ~2.8 | ~1小时 |
| large | 100M tokens | ~2.5 | ~8小时 |

## 最佳实践

1. **内存管理**：
   - GPU内存不足时，减小`device_batch_size`
   - 使用梯度累积保持总批大小不变
   - 考虑使用bfloat16混合精度训练

2. **训练稳定性**：
   - 使用梯度裁剪（`grad_clip=1.0`）
   - 添加学习率预热（`warmup_ratio=0.01`）
   - 监控验证损失以检测过拟合

3. **推理效率**：
   - 使用KV缓存加速自回归生成
   - 考虑使用GQA减少内存占用
   - 批处理多个生成请求

## 故障排除

### OOM（内存溢出）
- 减小`device_batch_size`
- 减小模型规模
- 减小`max_seq_len`
- 使用梯度累积

### 训练不稳定
- 增加梯度裁剪
- 调整学习率
- 增加学习率预热步数
- 检查数据质量

### 生成质量差
- 增加训练数据量
- 训练更多epoch
- 调整温度参数
- 考虑使用更大的模型

## 扩展方向

nanoDeepSeek的设计允许轻松扩展：

1. **添加新注意力机制**：在`MultiHeadAttention`类中实现
2. **自定义FFN**：修改`FeedForward`类
3. **添加新的归一化方法**：替换`rms_norm`函数
4. **实现MoE**：扩展架构支持混合专家
5. **添加量化支持**：在推理时实现权重量化

## 参考文献

- DeepSeek v3.2 Architecture
- Roformer: Enhanced Transformer with Rotary Position Embedding
- GQA: Training Generalized Query Attention
- Muon: An Orthogonal Matrix-Optimizer
- nanochat: The best ChatGPT that $100 can buy

## 许可证

MIT License（与nanochat项目保持一致）

## 贡献

欢迎提交问题和拉取请求！在贡献时，请：
1. 保持代码风格与nanochat项目一致
2. 添加必要的文档和注释
3. 确保所有测试通过
4. 在PR中说明使用LLM的部分

## 联系方式

如有问题或建议，请在GitHub Issues中提出。
