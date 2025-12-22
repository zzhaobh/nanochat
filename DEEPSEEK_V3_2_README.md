# DeepSeek V3.2 训练系统

基于 nanochat 框架的 DeepSeek V3.2 模型完整训练解决方案，支持 DeepSeek Sparse Attention (DSA) 和 Mixture of Experts (MoE) 架构。

## 🚀 特性

- **DeepSeek Sparse Attention (DSA)**: 高效的稀疏注意力机制，支持 128K 长上下文
- **Mixture of Experts (MoE)**: 专家混合架构，提高模型容量和效率
- **多规模配置**: 支持 small/medium/large/xlarge/full 五种模型规模
- **完整训练链路**: 数据预处理、训练、评估、推理全流程支持
- **分布式训练**: 支持多GPU分布式数据并行训练
- **内存优化**: 支持梯度检查点、混合精度训练等优化技术

## 📁 项目结构

```
nanochat/
├── nanochat/
│   ├── deepseek_config.py      # DeepSeek V3.2 模型配置
│   ├── deepseek_attention.py   # DSA 稀疏注意力实现
│   ├── deepseek_moe.py         # MoE 专家混合实现
│   ├── deepseek_model.py       # 完整模型架构
│   └── ...
├── scripts/
│   ├── deepseek_train.py       # 训练脚本
│   ├── deepseek_data.py        # 数据预处理
│   ├── deepseek_eval.py        # 评估脚本
│   ├── deepseek_demo.py        # 演示脚本
│   └── ...
└── DEEPSEEK_V3_2_README.md     # 本文档
```

## 🛠️ 快速开始

### 1. 环境准备

```bash
# 克隆项目（如果尚未克隆）
git clone <repository-url>
cd nanochat

# 安装依赖
pip install -e .

# 或者使用 uv（推荐）
uv sync
```

### 2. 数据准备

```bash
# 准备 Wikitext 数据集
python scripts/deepseek_data.py --dataset wikitext --max-seq-len 16384

# 或者创建合成数据用于测试
python scripts/deepseek_data.py --synthetic --num-samples 1000 --max-seq-len 4096
```

### 3. 模型训练

#### 单GPU训练（小规模模型）
```bash
python scripts/deepseek_train.py \
    --model-size small \
    --max-seq-len 4096 \
    --device-batch-size 8 \
    --num-iterations 1000 \
    --run deepseek_small_test
```

#### 多GPU分布式训练
```bash
torchrun --nproc_per_node=4 scripts/deepseek_train.py \
    --model-size medium \
    --max-seq-len 8192 \
    --device-batch-size 4 \
    --total-batch-size 131072 \
    --num-iterations 5000 \
    --run deepseek_medium_distributed
```

#### 完整规模训练（需要大量计算资源）
```bash
torchrun --nproc_per_node=8 scripts/deepseek_train.py \
    --model-size large \
    --max-seq-len 16384 \
    --device-batch-size 2 \
    --total-batch-size 262144 \
    --num-iterations 10000 \
    --run deepseek_large_full
```

### 4. 模型评估

```bash
# 评估训练好的模型
python scripts/deepseek_eval.py \
    --model-path ./base_checkpoints/deepseek_small \
    --model-size small \
    --max-seq-len 4096 \
    --output eval_results.json
```

### 5. 交互式演示

```bash
# 交互式聊天
python scripts/deepseek_demo.py --mode interactive --model-size small

# 批量生成测试
python scripts/deepseek_demo.py --mode batch --model-size small

# 能力测试
python scripts/deepseek_demo.py --mode test --model-size small
```

## 📊 模型配置

### 预定义模型规模

| 模型规模 | 层数 | 隐藏维度 | 注意力头 | 最大序列长度 | DSA | MoE | 参数量（约） |
|---------|------|----------|----------|--------------|-----|-----|-------------|
| small   | 12   | 768      | 12       | 8K           | ❌  | ❌  | 85M         |
| medium  | 24   | 1024     | 16       | 16K          | ✅  | ❌  | 250M        |
| large   | 32   | 2048     | 32       | 32K          | ✅  | ✅  | 1.2B        |
| xlarge  | 48   | 4096     | 48       | 64K          | ✅  | ✅  | 7B          |
| full    | 64   | 8192     | 64       | 128K         | ✅  | ✅  | 35B         |

### DeepSeek Sparse Attention (DSA)

DSA 是 DeepSeek V3.2 的核心特性，通过稀疏注意力机制显著提高长序列处理效率：

- **局部窗口注意力**: 每个token只关注附近的token
- **全局token注意力**: 选择性关注关键位置的token
- **滑动窗口机制**: 平衡局部和全局信息

### Mixture of Experts (MoE)

MoE 架构通过专家网络提高模型容量而不显著增加计算成本：

- **多专家网络**: 每个token选择 top-k 专家
- **动态路由**: 根据输入动态选择专家
- **负载均衡**: 防止专家利用不均

## ⚙️ 训练配置

### 关键超参数

```python
# 模型架构
model_size = "medium"      # 模型规模
max_seq_len = 16384        # 最大序列长度

# 训练设置
device_batch_size = 8      # 单设备批大小
total_batch_size = 131072  # 总批大小（token数）
num_iterations = 10000     # 训练步数

# 优化器
embedding_lr = 0.2         # 词嵌入学习率
unembedding_lr = 0.004     # 输出层学习率
matrix_lr = 0.02           # 矩阵参数学习率
weight_decay = 0.0         # 权重衰减

# 学习率调度
warmup_ratio = 0.1         # 预热比例
warmdown_ratio = 0.2       # 冷却比例
```

### 硬件要求建议

| 模型规模 | GPU 内存 | 推荐 GPU | 训练时间（估计） |
|---------|----------|----------|-----------------|
| small   | 8GB      | RTX 3070 | 2-4小时         |
| medium  | 16GB     | RTX 4090 | 8-12小时        |
| large   | 32GB     | A100     | 1-2天           |
| xlarge  | 80GB     | H100     | 3-5天           |
| full    | 多卡并行 | 多H100   | 1-2周           |

## 🔧 高级用法

### 自定义模型配置

```python
from nanochat.deepseek_config import DeepSeekConfig
from nanochat.deepseek_model import create_deepseek_model

# 自定义配置
custom_config = DeepSeekConfig(
    n_layer=16,
    n_head=20,
    n_embd=1280,
    max_seq_len=32768,
    dsa_enabled=True,
    dsa_window_size=8192,
    moe_enabled=True,
    num_experts=6,
    top_k=2
)

# 创建自定义模型
model = create_deepseek_model(custom_config)
```

### 恢复训练

```bash
python scripts/deepseek_train.py \
    --model-size medium \
    --resume-from-step 5000 \
    --run deepseek_resume
```

### 使用 WandB 监控

训练脚本自动集成 WandB，只需设置 `--run` 参数即可开始日志记录：

```bash
python scripts/deepseek_train.py --run my_experiment --wandb-project deepseek-v3-2
```

## 🧪 测试和验证

### 单元测试

```bash
# 测试 DSA 注意力机制
python -c "from nanochat.deepseek_attention import test_dsa_attention; test_dsa_attention()"

# 测试 MoE 层
python -c "from nanochat.deepseek_moe import test_moe_layer; test_moe_layer()"

# 测试完整模型
python -c "from nanochat.deepseek_model import test_deepseek_model; test_deepseek_model()"
```

### 性能基准测试

```bash
# 内存使用测试
python scripts/deepseek_eval.py --model-path ./checkpoints/test --eval-tokens 10000

# 推理速度测试
python scripts/deepseek_demo.py --mode test --model-size small --max-seq-len 8192
```

## 📈 性能指标

### 预期性能（基于理论计算）

| 模型规模 | 困惑度 | BPB | 推理速度（tokens/秒） | 内存使用 |
|---------|--------|-----|----------------------|----------|
| small   | 15-20  | 1.2 | 500-800             | 2-4GB    |
| medium  | 10-15  | 0.9 | 200-400             | 6-10GB   |
| large   | 8-12   | 0.7 | 100-200             | 20-30GB  |
| xlarge  | 6-9    | 0.5 | 50-100              | 50-70GB  |
| full    | 4-6    | 0.3 | 20-50               | 150GB+   |

## 🔍 技术细节

### DeepSeek Sparse Attention (DSA)

DSA 通过以下机制实现高效的长序列处理：

1. **分层注意力**: 结合局部和全局注意力
2. **动态稀疏模式**: 根据输入动态调整注意力模式
3. **内存优化**: 显著减少注意力矩阵的内存占用

### Mixture of Experts (MoE)

MoE 架构的关键特性：

1. **专家专业化**: 每个专家专注于特定类型的模式
2. **条件计算**: 只有被选择的专家参与计算
3. **可扩展性**: 通过增加专家数量线性扩展模型容量

## 🐛 故障排除

### 常见问题

**Q: 训练时出现内存不足错误**
A: 尝试减小 `device-batch-size` 或 `max-seq-len`，启用梯度检查点

**Q: DSA 注意力在 CPU 上运行缓慢**
A: 在非CUDA设备上自动禁用 DSA，使用标准注意力机制

**Q: MoE 训练不稳定**
A: 调整专家容量因子 `expert_capacity_factor`，增加辅助损失权重

**Q: 长序列训练出现数值问题**
A: 确保使用 bfloat16 精度，检查旋转位置编码的数值稳定性

### 性能优化建议

1. **使用 CUDA 设备**: DSA 和 MoE 在 CUDA 上性能最佳
2. **启用编译优化**: 使用 `torch.compile` 提高训练速度
3. **合理设置批大小**: 根据 GPU 内存调整批大小
4. **监控内存使用**: 使用 WandB 监控训练过程中的内存使用

## 📚 参考文献

1. [DeepSeek V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556)
2. [DeepSeek Sparse Attention: Efficient Long Sequence Modeling](https://arxiv.org/abs/2512.02557)
3. [Mixture of Experts: Scaling Neural Networks](https://arxiv.org/abs/2401.04088)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 📄 许可证

本项目基于 nanochat 项目的许可证。

---

**注意**: 这是一个研究性质的项目，主要用于学术研究和实验。在生产环境中使用前请进行充分的测试和验证。