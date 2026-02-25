"""
MoE (Mixture of Experts) 简单示例
==================================
最简化的 Sparse-MOE 实现，适合入门学习

核心概念:
1. 门控网络(Gate): 决定输入应该由哪些专家处理
2. 专家网络(Experts): 多个并行的神经网络
3. Top-K 路由: 每个输入只选择 K 个专家
4. 负载均衡: 防止某些专家过载
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SimpleMOE(nn.Module):
    """
    简化版 Sparse-MOE

    维度说明:
    - B: batch_size
    - D: d_model (特征维度)
    - E: num_experts (专家数量)
    - K: top_k (选择的专家数)
    """

    def __init__(
        self,
        d_model: int = 64,        # 输入/输出维度
        num_experts: int = 4,     # 专家数量
        top_k: int = 2,           # 每个输入选择 top_k 个专家
        hidden_dim: int = 128,    # 专家隐藏层维度
    ):
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        # ========== 门控网络 ==========
        # 作用: 根据输入决定选择哪些专家
        # 输入: (B, D) -> 输出: (B, E)
        self.gate = nn.Linear(d_model, num_experts)

        # ========== 专家网络 ==========
        # 多个简单的 FFN 作为专家
        # 每个专家: (B, D) -> (B, D)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, d_model),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入张量, shape (B, D)

        Returns:
            output: 输出张量, shape (B, D)
            aux_loss: 负载均衡损失 (辅助训练)
        """
        B, D = x.shape

        # ========== Step 1: 计算门控分数 ==========
        # gate_logits: (B, E) - 每个输入对每个专家的分数
        gate_logits = self.gate(x)

        # ========== Step 2: Top-K 选择专家 ==========
        # 只选择分数最高的 K 个专家
        # top_k_values: (B, K) - 选中的 K 个分数
        # top_k_indices: (B, K) - 选中的 K 个专家索引
        top_k_values, top_k_indices = torch.topk(
            gate_logits, self.top_k, dim=-1
        )

        # 对选中的分数做 softmax，得到权重
        # gate_weights: (B, K)
        gate_weights = F.softmax(top_k_values, dim=-1)

        # ========== Step 3: 计算负载均衡损失 ==========
        # 目的: 让所有专家都被均匀使用，避免某些专家闲置
        # router_prob: (B, E) - 所有专家的路由概率
        router_prob = F.softmax(gate_logits, dim=-1)

        # f: 每个专家被选中的频率
        # P: 每个专家的平均路由概率
        expert_mask = torch.zeros(B, self.num_experts, device=x.device)
        expert_mask.scatter_(-1, top_k_indices, 1.0)  # 标记被选中的专家
        f = expert_mask.mean(dim=0)  # (E,)
        P = router_prob.mean(dim=0)  # (E,)

        # 负载均衡损失: 当 f 和 P 差异大时，损失大
        aux_loss = self.num_experts * (f * P).sum()

        # ========== Step 4: 专家计算 + 加权聚合 ==========
        # 初始化输出
        output = torch.zeros_like(x)  # (B, D)

        # 遍历每个专家
        for expert_idx in range(self.num_experts):
            # 找到选择了这个专家的所有输入
            # mask: (B,) - bool 张量，标记哪些输入选择了当前专家
            mask = (top_k_indices == expert_idx).any(dim=-1)

            if mask.any():
                # 选中了当前专家的输入
                expert_input = x[mask]  # (num_selected, D)

                # 通过专家网络
                expert_output = self.experts[expert_idx](expert_input)

                # 获取对应的门控权重
                # 找到每个选中输入在当前专家对应的权重位置
                positions = mask.nonzero(as_tuple=True)[0]  # 在 batch 中的位置

                # 计算权重: 对每个选中的输入，找到它分配给当前专家的权重
                for i, pos in enumerate(positions):
                    # 在这个输入的 top_k 选择中，找到当前专家的索引
                    k_positions = (top_k_indices[pos] == expert_idx).nonzero(as_tuple=True)[0]
                    if len(k_positions) > 0:
                        weight = gate_weights[pos, k_positions[0]]
                        output[pos] += expert_output[i] * weight

        return output, aux_loss


# =============================================================================
# 使用示例
# =============================================================================

def demo_basic_usage():
    """基础使用示例"""
    print("=" * 60)
    print("MoE 基础使用示例")
    print("=" * 60)

    # 创建模型
    moe = SimpleMOE(
        d_model=64,
        num_experts=4,
        top_k=2,
        hidden_dim=128,
    )

    # 随机输入
    batch_size = 16
    x = torch.randn(batch_size, 64)

    # 前向传播
    output, aux_loss = moe(x)

    print(f"\n输入维度:  {x.shape}")
    print(f"输出维度:  {output.shape}")
    print(f"辅助损失:  {aux_loss.item():.6f}")

    # 统计专家使用情况
    gate_logits = moe.gate(x)
    _, selected = torch.topk(gate_logits, moe.top_k, dim=-1)

    print(f"\n专家选择统计 (Top-{moe.top_k}):")
    for i in range(moe.num_experts):
        count = (selected == i).sum().item()
        print(f"  专家 {i}: 被选中 {count} 次")


def demo_comparison():
    """对比: 普通 FFN vs MoE"""
    print("\n" + "=" * 60)
    print("对比: 普通 FFN vs MoE")
    print("=" * 60)

    d_model = 64
    batch_size = 32
    x = torch.randn(batch_size, d_model)

    # 普通 FFN
    simple_ffn = nn.Sequential(
        nn.Linear(d_model, 256),
        nn.ReLU(),
        nn.Linear(256, d_model),
    )

    # MoE (4个专家，每个top-2)
    moe = SimpleMOE(
        d_model=d_model,
        num_experts=4,
        top_k=2,
        hidden_dim=256,
    )

    # 计算参数量
    ffn_params = sum(p.numel() for p in simple_ffn.parameters())
    moe_params = sum(p.numel() for p in moe.parameters())

    print(f"\n普通 FFN 参数量: {ffn_params:,}")
    print(f"MoE 参数量:      {moe_params:,}")
    print(f"参数增加比例:    {moe_params / ffn_params:.2f}x")

    # 前向传播
    ffn_out = simple_ffn(x)
    moe_out, aux_loss = moe(x)

    print(f"\n普通 FFN 输出: {ffn_out.shape}")
    print(f"MoE 输出:      {moe_out.shape}")


def demo_expert_specialization():
    """
    演示专家如何学习不同模式
    每个专家会学习到不同类型的数据模式
    """
    print("\n" + "=" * 60)
    print("专家特化学习演示")
    print("=" * 60)

    # 创建一个简单的分类任务
    # 不同类别的数据由不同专家处理

    torch.manual_seed(42)

    # 数据: 4 种类别，每类有不同的特征模式
    num_samples = 200
    d_model = 32

    # 生成 4 类数据
    data = []
    labels = []
    for i in range(4):
        # 每类数据有不同的均值
        mean = torch.randn(d_model) * 2 + i * 3
        samples = torch.randn(num_samples // 4, d_model) + mean
        data.append(samples)
        labels.extend([i] * (num_samples // 4))

    X = torch.cat(data, dim=0)  # (200, 32)
    y = torch.tensor(labels)    # (200,)

    # 创建 MoE 模型
    class MOEClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.moe = SimpleMOE(d_model=32, num_experts=4, top_k=2, hidden_dim=64)
            self.classifier = nn.Linear(32, 4)

        def forward(self, x):
            moe_out, aux_loss = self.moe(x)
            logits = self.classifier(moe_out)
            return logits, aux_loss

    model = MOEClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # 训练
    print("\n训练模型...")
    for epoch in range(50):
        optimizer.zero_grad()
        logits, aux_loss = model(X)

        # 分类损失 + 负载均衡损失
        cls_loss = criterion(logits, y)
        total_loss = cls_loss + 0.01 * aux_loss

        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            acc = (logits.argmax(dim=-1) == y).float().mean().item()
            print(f"Epoch {epoch+1}: Loss={total_loss.item():.4f}, Acc={acc:.4f}")

    # 查看专家使用情况
    model.eval()
    with torch.no_grad():
        gate_logits = model.moe.gate(X)
        _, selected = torch.topk(gate_logits, model.moe.top_k, dim=-1)

    print(f"\n各类数据的专家选择分布:")
    for cls in range(4):
        mask = (y == cls)
        cls_selected = selected[mask]

        print(f"类别 {cls}:", end=" ")
        for expert in range(4):
            count = (cls_selected == expert).sum().item()
            print(f"专家{expert}={count}", end="  ")
        print()


if __name__ == "__main__":
    demo_basic_usage()
    demo_comparison()
    demo_expert_specialization()

    print("\n" + "=" * 60)
    print("所有演示完成!")
    print("=" * 60)
