# python3
# Create Date: 2026-01-28
# Author: Scc_hy
# Func:  moe 
# =================================================================================================


import torch 
from torch import  nn 
from torch.nn import functional as F


class MOE_basic(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """_summary_
        Expand-then-Contract -> learn complex feature interactions in a higher-dimensional space

        Args:
            input_size (int): input tensor size
            hidden_size (int): hidden layer size 
            output_size (int): output size 
        
        input_size < hidden_size: linealy project to a higher-dimensional 
        output_size < hidden_size: After ReLU for non-linearly projected back to lower-dimensional 
        """
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.active = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.active(self.fc(x)))


class simple_MOE(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            MOE_basic(input_size, hidden_size, output_size) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_size, num_experts)
    
    def forward(self, x):
        experts_weight = self.gate(x) # [b, n]
        # [[b, 1, o] * n] -> [b, n, o]
        experts_out_list = torch.cat(
            [exp_i(x).unsqueeze(1) for exp_i in self.experts],
            dim=1
        )
        # [b, 1, n] @ [b, n, o] -> [b, 1, o]
        out = experts_weight.unsqueeze(1)  @ experts_out_list
        return out.squeeze(1)


class MOE_router(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int, use_noisy_gating: bool=True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k 
        self.gate = nn.Linear(hidden_size, num_experts)
        # 可学习温度参数，控制路由随机性（防止前期坍塌）
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
        if use_noisy_gating:
            # 预测噪声的标准差（softplus确保为正）
            self.noise_linear = nn.Linear(hidden_size, num_experts)

    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states) 
        # === 关键修正：Noisy Top-k Gating ===
        if self.use_noisy_gating and self.training:
            # 生成与 logits 同形状的噪声
            # 噪声强度由 noise_linear 预测，softplus 确保 > 0
            noise_std = F.softplus(self.noise_linear(hidden_states)) # ln(1+e^x)
            noise = torch.randn_like(router_logits) * noise_std
            # 添加噪声（训练时增加探索，测试时 deterministic）
            router_logits = router_logits + noise

        router_logits = router_logits / torch.abs(self.temperature)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        # [b, num_experts] -> [b, topK] 每条样本激活TopK个专家
        router_weights, select_experts = torch.topk(
            router_probs, self.top_k, dim=1
        )
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)        
        return router_logits, router_weights, select_experts


class SparseMOE(nn.Module):
    def __init__(
        self, 
        input_size: int, output_size: int, hidden_size: int, 
        top_k: int, num_experts: int, 
        capacity_factor: float = 1.25,
        use_noisy_gating: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.num_experts = num_experts
        self.router = MOE_router(hidden_size, num_experts, top_k, use_noisy_gating)
        self.experts = nn.ModuleList([
            MOE_basic(input_size, hidden_size, output_size) for _ in range(num_experts)
        ])
        
        # 残差映射（当输入输出维度不一致时）
        self.shortcut = nn.Linear(input_size, output_size) if input_size != output_size else None
        
        # 容量限制（防止OOM，关键优化）
        self.capacity_factor = capacity_factor

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [batch, seq, input_size] 注意：input_size 应等于 hidden_size（router输入）
               或 split projection 后传入
        Returns:
            output: [batch, seq, output_size]
            aux_loss: 负载均衡损失（用于反向传播）
        """
        batch_size, seq_len, input_dim = x.shape
        flat_x = x.view(-1, input_dim)  # [N, input_size], N = B*S
        
        # 路由计算
        router_logits, router_weights, selected_experts = self.router(flat_x)
        
        # 输出缓冲区 [N, output_size]
        expert_output = torch.zeros(flat_x.size(0), self.output_size, device=flat_x.device, dtype=flat_x.dtype)
        
        # 计算容量限制（每个专家最多处理的token数）
        capacity = int(self.capacity_factor * flat_x.size(0) * self.top_k / self.num_experts)
        
        # 专家计算：只遍历实际被选中的专家（优化点：避免遍历所有num_experts）
        for expert_idx in range(self.num_experts):
            # 创建掩码：找出所有选择该专家的token [N, top_k]
            mask = (selected_experts == expert_idx)
            if not mask.any():
                continue
            
            # 获取位置索引和对应的top-k权重
            token_idx, k_pos = torch.where(mask)  # token_idx: [M], k_pos: [M]
            
            # 容量截断（关键：防止专家过载导致OOM）
            if token_idx.size(0) > capacity: # 被截断的 Token 不会由该专家处理 -> 1) 被其他专家处理 2) 完全丢失 可优化
                # 按权重排序，只保留Top-capacity
                topk_vals, topk_pos = torch.topk(router_weights[token_idx, k_pos], capacity)
                token_idx = token_idx[topk_pos]
                k_pos = k_pos[topk_pos]
            
            # 提取对应权重 [M, 1]
            weights = router_weights[token_idx, k_pos].unsqueeze(-1)
            
            # 专家前向 
            expert_in = flat_x[token_idx]  # [M, input_size]
            expert_out = self.experts[expert_idx](expert_in)  # [M, output_size]
            
            # 加权并累加（使用index_add_确保梯度正确）
            weighted_out = expert_out * weights
            expert_output.index_add_(0, token_idx, weighted_out)
        
        # 残差连接（MoE标准做法）
        residual = self.shortcut(flat_x) if self.shortcut is not None else flat_x
        output = expert_output + residual
        
        # 恢复形状
        output = output.view(batch_size, seq_len, self.output_size)
        
        # 计算辅助损失（防止路由坍塌，必须加！）
        aux_loss = self._compute_load_balance_loss(router_probs=torch.softmax(router_logits, dim=-1), 
                                                   expert_indices=selected_experts)
        
        return output, aux_loss
    
    def _compute_load_balance_loss(self, router_probs: torch.Tensor, 
                                   expert_indices: torch.Tensor):
        """
        负载均衡损失：确保每个专家被均匀使用
        参考 Switch Transformer & ST-MoE
        """
        # 平均路由概率（重要性）
        avg_router_prob = router_probs.mean(dim=0)  # [num_experts]
        
        # 实际被分派的频率
        expert_mask = F.one_hot(expert_indices, self.num_experts).sum(dim=1).float()  # [N, top_k] -> [N, num_experts]
        avg_expert_usage = expert_mask.mean(dim=0)  # [num_experts]
        
        # 损失：期望每个专家的 importance × frequency 均匀
        # 目标：avg_router_prob 和 avg_expert_usage 都接近 1/num_experts
        balance_loss = self.num_experts * torch.sum(avg_router_prob * avg_expert_usage)
        
        # 可添加 Zhang et al. 的重要性损失（鼓励专家专业化）
        importance = router_probs.sum(dim=0)  # [num_experts]
        cv_importance = torch.std(importance) / (torch.mean(importance) + 1e-10)
        
        return balance_loss + 0.01 * cv_importance



class SharedExpertMOE(nn.Module):
    """
    标准共享专家 MoE 架构：
    - Shared Experts: 1-2 个，所有 Token 经过（提供基础能力）
    - Routed Experts: num_routed 个，每个 Token 选 top_k（提供专业化能力）
    """
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        hidden_size: int, 
        top_k: int,
        num_routed_experts: int,
        num_shared_experts: int = 1, 
        alpha: float = 1.0,           
        use_gating_fusion: bool = False  
    ):
        """_summary_

        Args:
            input_size (int): _description_
            output_size (int): _description_
            hidden_size (int): _description_
            top_k (int): _description_
            num_routed_experts (int): _description_
            num_shared_experts (int, optional): _description_. Defaults to 1.  # 关键：独立参数，通常 1-2 即可
            alpha: # 共享专家输出缩放系数
            use_gating_fusion: # 是否学习动态融合权重
        """
        super().__init__()
        self.num_shared_experts = num_shared_experts
        self.alpha = alpha
        
        # 路由专家（Sparse）
        self.sparse_moe = SparseMOE(
            input_size, output_size, hidden_size, 
            top_k, num_routed_experts
        )
        
        # 共享专家（Dense）：通常 1-2 个，提供通用知识
        self.shared_experts = nn.ModuleList([
            MOE_basic(input_size, hidden_size, output_size) 
            for _ in range(num_shared_experts)
        ])
        
        # 动态融合门（可选）：学习如何结合共享和路由输出
        if use_gating_fusion:
            self.fusion_gate = nn.Sequential(
                nn.Linear(output_size * 2, 2),
                nn.Softmax(dim=-1)
            )
        
        # 输入输出维度适配（残差用）
        self.shortcut = nn.Linear(input_size, output_size) if input_size != output_size else None

    def forward(self, x: torch.Tensor):
        """
        Returns:
            output: [batch, seq, output_size]
            aux_loss: 稀疏路由的负载均衡损失
        """
        # 1. 稀疏部分（专业化能力）
        sparse_out, aux_loss = self.sparse_moe(x)  # [B,S,O]
        
        # 2. 共享部分（通用能力）：累加而非 stack，节省内存
        # 优化点：避免 torch.stack 创建 [num_shared, B, S, O] 的临时张量
        shared_out = 0
        for expert in self.shared_experts:
            shared_out = shared_out + expert(x)  # 原位累加，内存友好
        
        # 缩放（防止共享专家数量多时起主导作用）
        shared_out = shared_out * (self.alpha / max(self.num_shared_experts, 1))
        
        # 3. 融合策略
        if hasattr(self, 'fusion_gate'):
            # 动态加权融合 [B,S,2]，分别对应共享和路由的权重
            concat_feat = torch.cat([shared_out, sparse_out], dim=-1)
            weights = self.fusion_gate(concat_feat)  # [B,S,2]
            out = weights[..., 0:1] * shared_out + weights[..., 1:2] * sparse_out
        else:
            # 简单相加（标准做法：共享提供基础，路由提供残差修正）
            out = shared_out + sparse_out
        
        # 4. 残差连接（可选，看具体架构需求）
        if self.shortcut is not None:
            residual = self.shortcut(x)
            out = out + residual
            
        return out, aux_loss


# ============================================
# 极端性能优化版本（适合 num_shared_experts > 2）
# ============================================

class SharedExpertMOE_Fused(nn.Module):
    """
    当共享专家数量较多时的优化版本：
    使用向量化计算替代 ModuleList 循环（类似之前 VectorizedDenseMOE 的思路）
    """
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        hidden_size: int,
        top_k: int,
        num_routed_experts: int,
        num_shared_experts: int = 2
    ):
        super().__init__()
        self.num_shared_experts = num_shared_experts
        
        # 路由部分保持不变
        self.sparse_moe = SparseMOE(
            input_size, output_size, hidden_size,
            top_k, num_routed_experts
        )
        
        # 共享专家：堆叠为 3D 参数张量，使用 einsum 一次计算
        # 形状: [num_shared, input_size, hidden_size] 和 [num_shared, hidden_size, output_size]
        self.shared_w1 = nn.Parameter(torch.randn(num_shared_experts, input_size, hidden_size) * 0.02)
        self.shared_b1 = nn.Parameter(torch.zeros(num_shared_experts, 1, hidden_size))
        self.shared_w2 = nn.Parameter(torch.randn(num_shared_experts, hidden_size, output_size) * 0.02)
        self.shared_b2 = nn.Parameter(torch.zeros(num_shared_experts, 1, output_size))
        
        # 可学习融合权重（替代固定求和）
        self.shared_gate = nn.Parameter(torch.ones(num_shared_experts) / num_shared_experts)

    def forward(self, x: torch.Tensor):
        batch, seq, _ = x.shape
        
        # 1. 稀疏部分
        sparse_out, aux_loss = self.sparse_moe(x)  # [B,S,O]
        
        # 2. 共享部分：向量化计算
        flat_x = x.view(-1, 1, self.shared_w1.size(1))  # [N, 1, I]
        
        # 所有共享专家并行计算: [num_shared, N, hidden]
        h = torch.einsum('nji,sih->snh', flat_x, self.shared_w1) + self.shared_b1
        h = F.relu(h)
        expert_outs = torch.einsum('snh,sho->sno', h, self.shared_w2) + self.shared_b2  # [S,N,O]
        
        # 加权求和（可学习权重）
        shared_out = torch.einsum('s,sno->no', F.softmax(self.shared_gate, dim=0), expert_outs)
        shared_out = shared_out.view(batch, seq, -1)
        
        # 3. 融合
        return sparse_out + shared_out, aux_loss
