---
name: ple-moe-expert
description: 生成PLE数值嵌入和Sparse-MOE结构代码，自动添加梯度冲突检测和维度注释，适配多任务学习。当用户需要数值特征嵌入、混合专家模型、多任务学习架构时触发此Skill。
---

# PLE-MOE Expert Skill

## Overview

本Skill用于生成**PLE (Piecewise Linear Embeddings)** 数值嵌入和**Sparse-MOE (Mixture of Experts)** 结构的PyTorch代码，特别针对多任务学习(MTL)场景优化。

### 核心能力

1. **PLE数值嵌入** - 基于Yandex Research的NeurIPS 2022论文实现
2. **Sparse-MOE结构** - 支持Top-K门控的专家混合网络
3. **梯度冲突检测** - 自动检测并报告多任务学习中的梯度冲突
4. **维度注释** - 所有张量操作都附带详细的维度注释
5. **多任务学习适配** - 支持任务特定专家和共享专家

### 适用场景

- 表格数据的深度学习建模
- 多任务学习场景（如推荐系统中的CTR+CVR联合预测）
- 需要处理数值特征嵌入的神经网络
- 需要专家混合模型进行特征路由的场景

---

## Usage

### 基本用法

```python
# 生成PLE嵌入层
ple_embed = generate_ple_embeddings(
    n_features=10,           # 数值特征数量
    d_embedding=64,          # 嵌入维度
    n_bins=48,               # 每个特征的分箱数
    activation=True          # 是否使用ReLU激活
)

# 生成Sparse-MOE层
moe_layer = generate_sparse_moe(
    d_model=256,             # 输入/输出维度
    num_experts=8,           # 专家数量
    top_k=2,                 # 每个token选择top-k专家
    expert_hidden_dim=1024   # 专家FFN隐藏层维度
)
```

### 多任务学习完整示例

```python
# 生成MTL-PLE-MOE模型
model = generate_mtl_ple_moe_model(
    num_features=100,
    ple_bins=64,
    ple_dim=128,
    num_shared_experts=4,
    num_task_experts=[3, 3],  # 每个任务的专家数
    tasks=['ctr', 'cvr'],
    d_model=256,
    top_k=2
)
```

---

## Available Resources

| 资源 | 路径 | 说明 |
|------|------|------|
| 代码生成器 | `scripts/ple_moe_generator.py` | 完整的PLE和MOE代码生成 |
| PLE参考实现 | `references/ple_reference.py` | Yandex Research官方PLE实现 |
| MOE示例 | `references/moe_examples.py` | Sparse MOE使用示例 |
| 梯度冲突检测 | `scripts/gradient_conflict_detector.py` | 梯度冲突检测工具 |

---

## PLE (Piecewise Linear Embeddings) 详解

### 核心思想

PLE将连续数值特征通过**分箱(bins)**转换为向量表示，相比简单的线性嵌入能更好地捕捉特征的非线性关系。

### 数学原理

对于输入特征值 $x$ 和分箱边界 $[b_0, b_1, ..., b_n]$，PLE编码为：

$$
x_{ple} = [1, ..., 1, \frac{x - b_i}{b_{i+1} - b_i}, 0, ..., 0]
$$

其中只有包含 $x$ 的那个bin位置有非零值（在0到1之间），其他位置为0或1。

### 两种分箱策略

1. **分位数分箱 (Quantile-based)** - 基于数据分布均匀分箱
2. **决策树分箱 (Tree-based)** - 基于目标变量学习最优分箱边界

### 维度变换

```
输入:  (batch_size, n_features)          # 标量特征
输出:  (batch_size, n_features, d_embedding)  # 向量嵌入
```

---

## Sparse-MOE 详解

### 核心思想

混合专家模型(MOE)通过门控网络将输入路由到不同的专家网络，每个专家专注于处理特定类型的数据模式。

### Sparse-MOE结构

```
输入 x: (batch_size, seq_len, d_model)
    ↓
门控网络 G: (batch_size, seq_len, d_model) → (batch_size, seq_len, num_experts)
    ↓
Top-K选择: 选择概率最高的k个专家
    ↓
专家网络 E_i: 每个专家对输入进行变换
    ↓
加权聚合: output = Σ(g_i * E_i(x))
```

### 负载均衡损失

为防止所有输入都路由到少数专家，引入负载均衡损失：

$$
L_{balance} = \alpha \cdot num\_experts \cdot \sum_{i=1}^{num\_experts} f_i \cdot P_i
$$

其中 $f_i$ 是路由到专家i的token比例，$P_i$ 是门控网络分配给专家i的平均概率。

---

## 梯度冲突检测

### 什么是梯度冲突

在多任务学习中，不同任务的梯度方向可能相互冲突，导致模型性能下降。

### 检测方法

1. **余弦相似度**: 计算不同任务梯度的余弦相似度
2. **冲突比例**: 统计负余弦相似度的比例
3. **梯度范数**: 监控各任务梯度的大小

### 输出指标

```python
{
    'cosine_similarity': 0.45,      # 梯度余弦相似度
    'conflict_ratio': 0.23,          # 冲突比例（负相似度占比）
    'grad_norm_task1': 1.23,         # 任务1梯度范数
    'grad_norm_task2': 0.87,         # 任务2梯度范数
    'pcgrad_applied': True           # 是否应用了PCGrad
}
```

---

## 维度注释规范

所有生成的代码都遵循以下维度注释规范：

```python
# 输入维度: (batch_size, n_features)
# 输出维度: (batch_size, n_features, d_embedding)
# 权重维度: (n_features, d_embedding)
```

### 常用维度符号

| 符号 | 含义 |
|------|------|
| B | batch_size |
| N | n_features / num_tokens |
| D | d_model / d_embedding |
| E | num_experts |
| K | top_k |
| T | num_tasks |

---

## 多任务学习架构

### PLE-MOE-MTL 架构图

```
输入特征
    ↓
┌─────────────────────────────────────┐
│  PLE数值嵌入层                        │
│  (batch, n_features) → (batch, n_features, d_emb) │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  特征扁平化 + 投影                    │
│  (batch, n_features*d_emb) → (batch, d_model)     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Sparse-MOE层                        │
│  ┌─────────────┐  ┌─────────────┐   │
│  │ 共享专家(4个) │  │ 任务特定专家  │   │
│  │             │  │ (每个任务3个) │   │
│  └─────────────┘  └─────────────┘   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  任务特定塔                          │
│  ┌─────────┐  ┌─────────┐           │
│  │ CTR塔   │  │ CVR塔   │           │
│  └─────────┘  └─────────┘           │
└─────────────────────────────────────┘
    ↓
  [CTR_pred, CVR_pred]
```

---

## 最佳实践

### PLE参数选择

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| n_bins | 48-128 | 更多bins = 更精细的数值表示 |
| d_embedding | 32-128 | 根据特征复杂度选择 |
| activation | True | 通常使用ReLU激活 |

### MOE参数选择

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| num_experts | 4-16 | 专家数量 |
| top_k | 1-4 | 通常2效果较好 |
| expert_hidden_dim | 2-4x d_model | 专家FFN隐藏层维度 |

### 多任务学习建议

1. **任务相关性**: 相关任务共享更多参数
2. **专家分配**: 共享专家 + 任务特定专家
3. **负载均衡**: 确保专家利用率均衡
4. **梯度冲突**: 监控并处理梯度冲突

---

## 代码生成示例

### 完整MTL模型生成

运行以下命令生成完整的多任务学习模型：

```python
from scripts.ple_moe_generator import generate_mtl_ple_moe_model

model = generate_mtl_ple_moe_model(
    num_features=50,
    ple_bins=64,
    ple_dim=64,
    num_shared_experts=4,
    num_task_experts=[3, 3],
    tasks=['click', 'convert'],
    d_model=256,
    top_k=2,
    expert_hidden_dim=1024,
    enable_gradient_conflict_detection=True
)

print(model)
```

### 输出代码结构

```
PLE_MOE_MTL_Model(
  (ple_embeddings): PiecewiseLinearEmbeddings(...)
  (feature_projection): Linear(...)
  (moe_layer): SparseMOELayer(...)
  (task_towers): ModuleList(...)
  (task_heads): ModuleList(...)
  (gradient_conflict_detector): GradientConflictDetector(...)
)
```

---

## 参考论文

1. **PLE**: "On Embeddings for Numerical Features in Tabular Deep Learning" (NeurIPS 2022)
2. **MOE**: "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
3. **PCGrad**: "Gradient Surgery for Multi-Task Learning"

---

## 依赖要求

```
torch>=1.10.0
numpy>=1.20.0
scikit-learn>=1.0.0  # 用于决策树分箱
```

---

## 注意事项

1. **分箱计算**: PLE需要先计算分箱边界，应在训练集上计算
2. **内存使用**: MOE会增加模型参数量，注意GPU内存
3. **负载均衡**: 训练初期可能需要调整负载均衡系数
4. **梯度冲突**: 多任务学习建议始终启用梯度冲突检测
