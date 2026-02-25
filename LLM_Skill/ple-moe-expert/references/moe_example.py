"""
Sparse-MOE (Mixture of Experts) Examples
========================================
稀疏混合专家模型使用示例

包含:
1. 基础Sparse-MOE实现
2. 多任务学习MOE
3. 负载均衡策略
4. 门控网络变体

参考:
- "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
  https://arxiv.org/abs/1701.06538
- "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
  https://arxiv.org/abs/2101.03961
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# =============================================================================
# 基础Sparse-MOE
# =============================================================================

class Expert(nn.Module):
    """
    专家网络 - 标准FFN结构

    维度变换:
    - 输入: (batch_size, seq_len, d_model)
    - 输出: (batch_size, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SparseMOE(nn.Module):
    """
    基础Sparse-MOE实现

    维度变换:
    - 输入: (batch_size, seq_len, d_model)
    - 输出: (batch_size, seq_len, d_model)
    - 辅助损失: scalar

    示例:
    >>> moe = SparseMOE(d_model=512, num_experts=8, top_k=2)
    >>> x = torch.randn(32, 100, 512)
    >>> out, aux_loss = moe(x)
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        expert_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        load_balance_coef: float = 0.01,
        activation: str = 'gelu',
    ) -> None:
        super().__init__()
        assert top_k <= num_experts

        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_coef = load_balance_coef

        if expert_hidden_dim is None:
            expert_hidden_dim = 4 * d_model

        # 门控网络
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # 专家网络
        self.experts = nn.ModuleList([
            Expert(d_model, expert_hidden_dim, dropout, activation)
            for _ in range(num_experts)
        ])

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        前向传播

        Args:
            x: (B, N, D)

        Returns:
            output: (B, N, D)
            aux_loss: scalar
        """
        B, N, D = x.shape

        # 门控分数
        gate_logits = self.gate(x)  # (B, N, E)

        # Top-K选择
        top_k_logits, top_k_indices = torch.topk(
            gate_logits, self.top_k, dim=-1)
        gate_scores = F.softmax(top_k_logits, dim=-1)  # (B, N, K)

        # 负载均衡损失
        router_prob = F.softmax(gate_logits, dim=-1)  # (B, N, E)

        expert_mask = torch.zeros(B, N, self.num_experts, device=x.device)
        expert_mask.scatter_(-1, top_k_indices, 1.0)
        f = expert_mask.mean(dim=(0, 1))  # (E,)
        P = router_prob.mean(dim=(0, 1))  # (E,)

        aux_loss = self.num_experts * (f * P).sum()
        aux_loss = self.load_balance_coef * aux_loss

        # 专家计算
        output = torch.zeros_like(x)

        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i).any(dim=-1)

            if mask.any():
                expert_input = x[mask]
                expert_output = expert(expert_input)

                # 获取门控分数
                positions = mask.nonzero(as_tuple=True)
                k_idx = (top_k_indices[mask] == i).nonzero(as_tuple=True)[1]
                scores = gate_scores[mask][range(len(k_idx)), k_idx]

                expert_output = expert_output * scores.unsqueeze(-1)
                output[mask] += expert_output

        return output, aux_loss


# =============================================================================
# Switch-MOE (Top-1)
# =============================================================================

class SwitchMOE(nn.Module):
    """
    Switch-MOE: 每个token只路由到1个专家

    比Sparse-MOE更高效，但表达能力稍弱

    参考: "Switch Transformers" (Google, 2022)
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        expert_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        load_balance_coef: float = 0.01,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.load_balance_coef = load_balance_coef

        if expert_hidden_dim is None:
            expert_hidden_dim = 4 * d_model

        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            Expert(d_model, expert_hidden_dim, dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        B, N, D = x.shape

        # 门控
        gate_logits = self.gate(x)  # (B, N, E)

        # Top-1选择
        gate_scores, selected_experts = torch.max(
            F.softmax(gate_logits, dim=-1), dim=-1
        )  # (B, N), (B, N)

        # 负载均衡损失
        router_prob = F.softmax(gate_logits, dim=-1)

        # 专家选择频率
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            expert_counts[i] = (selected_experts == i).sum()
        f = expert_counts / (B * N)

        P = router_prob.mean(dim=(0, 1))
        aux_loss = self.num_experts * (f * P).sum()
        aux_loss = self.load_balance_coef * aux_loss

        # 专家计算
        output = torch.zeros_like(x)

        for i, expert in enumerate(self.experts):
            mask = (selected_experts == i)

            if mask.any():
                expert_input = x[mask]
                expert_output = expert(expert_input)
                scores = gate_scores[mask]
                output[mask] = expert_output * scores.unsqueeze(-1)

        return output, aux_loss


# =============================================================================
# 多任务学习MOE
# =============================================================================

class MultiTaskMOE(nn.Module):
    """
    多任务学习MOE

    包含:
    - 共享专家: 所有任务共享
    - 任务特定专家: 每个任务独有

    维度变换:
    - 输入: (batch_size, seq_len, d_model)
    - 输出: List[(batch_size, seq_len, d_model)]
    """

    def __init__(
        self,
        d_model: int,
        num_shared_experts: int,
        num_task_experts: List[int],
        top_k: int = 2,
        expert_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_tasks = len(num_task_experts)
        self.num_shared_experts = num_shared_experts

        total_experts = num_shared_experts + sum(num_task_experts)

        if expert_hidden_dim is None:
            expert_hidden_dim = 4 * d_model

        # 共享专家
        self.shared_experts = nn.ModuleList([
            Expert(d_model, expert_hidden_dim, dropout)
            for _ in range(num_shared_experts)
        ])

        # 任务特定专家
        self.task_experts = nn.ModuleList()
        for n_experts in num_task_experts:
            self.task_experts.append(nn.ModuleList([
                Expert(d_model, expert_hidden_dim, dropout)
                for _ in range(n_experts)
            ]))

        # 门控网络（每个任务一个）
        self.gates = nn.ModuleList([
            nn.Linear(d_model, total_experts, bias=False)
            for _ in range(self.num_tasks)
        ])

        self.top_k = top_k

    def forward(self, x: Tensor) -> List[Tuple[Tensor, Tensor]]:
        """
        前向传播

        Args:
            x: (B, N, D)

        Returns:
            每个任务的(output, aux_loss)列表
        """
        results = []

        for task_id in range(self.num_tasks):
            # 门控
            gate_logits = self.gates[task_id](x)
            top_k_logits, top_k_indices = torch.topk(
                gate_logits, self.top_k, dim=-1
            )
            gate_scores = F.softmax(top_k_logits, dim=-1)

            # 负载均衡损失
            router_prob = F.softmax(gate_logits, dim=-1)
            expert_mask = torch.zeros(
                x.shape[0], x.shape[1],
                len(self.shared_experts) + sum(len(te) for te in self.task_experts),
                device=x.device
            )
            expert_mask.scatter_(-1, top_k_indices, 1.0)
            f = expert_mask.mean(dim=(0, 1))
            P = router_prob.mean(dim=(0, 1))
            aux_loss = len(self.shared_experts) * (f * P).sum()

            # 专家计算
            output = torch.zeros_like(x)

            # 共享专家
            for i, expert in enumerate(self.shared_experts):
                mask = (top_k_indices == i).any(dim=-1)
                if mask.any():
                    expert_output = expert(x[mask])
                    positions = mask.nonzero(as_tuple=True)
                    k_idx = (
                        top_k_indices[mask] == i).nonzero(
                        as_tuple=True)[1]
                    scores = gate_scores[mask][range(len(k_idx)), k_idx]
                    output[mask] += expert_output * scores.unsqueeze(-1)

            # 任务特定专家
            offset = len(self.shared_experts)
            for i, expert in enumerate(self.task_experts[task_id]):
                expert_idx = offset + i
                mask = (top_k_indices == expert_idx).any(dim=-1)
                if mask.any():
                    expert_output = expert(x[mask])
                    positions = mask.nonzero(as_tuple=True)
                    k_idx = (
                        top_k_indices[mask] == expert_idx).nonzero(
                        as_tuple=True)[1]
                    scores = gate_scores[mask][range(len(k_idx)), k_idx]
                    output[mask] += expert_output * scores.unsqueeze(-1)

            results.append((output, aux_loss))

        return results


# =============================================================================
# 噪声门控 (Noisy Top-K Gating)
# =============================================================================

class NoisyTopKGating(nn.Module):
    """
    噪声Top-K门控

    在训练时添加噪声，增加专家选择的多样性

    参考: "Outrageously Large Neural Networks"
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 1.0,
    ) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std

        # 门控权重
        self.w_gate = nn.Linear(d_model, num_experts, bias=False)
        # 噪声权重
        self.w_noise = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        前向传播

        Args:
            x: (B, N, D)

        Returns:
            gate_scores: (B, N, K)
            selected_indices: (B, N, K)
        """
        # 清洁logits
        clean_logits = self.w_gate(x)  # (B, N, E)

        # 噪声logits（仅在训练时添加）
        if self.training:
            noise_logits = self.w_noise(x)
            noise = torch.randn_like(noise_logits) * self.noise_std
            noisy_logits = clean_logits + noise * F.softplus(noise_logits)
        else:
            noisy_logits = clean_logits

        # Top-K选择
        top_k_logits, top_k_indices = torch.topk(
            noisy_logits, self.top_k, dim=-1
        )

        # 使用清洁logits计算最终分数
        top_k_clean_logits = torch.gather(clean_logits, -1, top_k_indices)
        gate_scores = F.softmax(top_k_clean_logits, dim=-1)

        return gate_scores, top_k_indices


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Sparse-MOE Examples")
    print("=" * 70)

    # 测试1: 基础Sparse-MOE
    print("\n[Test 1] Sparse-MOE")
    print("-" * 40)

    moe = SparseMOE(
        d_model=256,
        num_experts=8,
        top_k=2,
        expert_hidden_dim=1024,
    )

    x = torch.randn(32, 100, 256)
    out, aux_loss = moe(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Aux loss: {aux_loss.item():.6f}")

    # 测试2: Switch-MOE
    print("\n[Test 2] Switch-MOE (Top-1)")
    print("-" * 40)

    switch = SwitchMOE(
        d_model=256,
        num_experts=8,
        expert_hidden_dim=1024,
    )

    out, aux_loss = switch(x)
    print(f"Output shape: {out.shape}")
    print(f"Aux loss: {aux_loss.item():.6f}")

    # 测试3: 多任务MOE
    print("\n[Test 3] Multi-Task MOE")
    print("-" * 40)

    mtl_moe = MultiTaskMOE(
        d_model=256,
        num_shared_experts=4,
        num_task_experts=[3, 3],  # 2个任务，每个3个专家
        top_k=2,
    )

    results = mtl_moe(x)
    for i, (out, aux) in enumerate(results):
        print(f"Task {i}: output={out.shape}, aux_loss={aux.item():.6f}")

    # 测试4: 噪声门控
    print("\n[Test 4] Noisy Top-K Gating")
    print("-" * 40)

    noisy_gate = NoisyTopKGating(
        d_model=256,
        num_experts=8,
        top_k=2,
        noise_std=1.0,
    )

    scores, indices = noisy_gate(x)
    print(f"Gate scores shape: {scores.shape}")
    print(f"Selected indices shape: {indices.shape}")

    # 统计专家选择分布
    expert_counts = torch.zeros(8)
    for i in range(8):
        expert_counts[i] = (indices == i).sum()
    print(f"Expert selection distribution: {expert_counts}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
