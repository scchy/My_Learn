"""
PLE-MOE Code Generator
======================
生成PLE数值嵌入和Sparse-MOE结构的PyTorch代码
适配多任务学习，自动添加梯度冲突检测和维度注释

参考: Yandex Research - "On Embeddings for Numerical Features in Tabular Deep Learning" (NeurIPS 2022)
"""

import math
import warnings
from typing import List, Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# 维度注释工具
# =============================================================================

class DimAnnotatedModule(nn.Module):
    """基础模块，所有子类都应实现get_dim_info方法返回维度信息"""
    
    def get_dim_info(self) -> Dict[str, str]:
        """返回维度信息字典"""
        raise NotImplementedError


def annotate_dims(func):
    """装饰器：为forward方法添加维度注释"""
    def wrapper(self, *args, **kwargs):
        # 获取输入维度信息
        input_info = []
        for i, arg in enumerate(args):
            if isinstance(arg, Tensor):
                input_info.append(f"arg{i}: {list(arg.shape)}")
        
        # 执行前向传播
        output = func(self, *args, **kwargs)
        
        # 获取输出维度信息
        output_info = ""
        if isinstance(output, Tensor):
            output_info = f" -> {list(output.shape)}"
        elif isinstance(output, (tuple, list)):
            shapes = [list(o.shape) if isinstance(o, Tensor) else str(type(o)) for o in output]
            output_info = f" -> {shapes}"
        
        # 存储维度信息供调试使用
        if not hasattr(self, '_last_dim_info'):
            self._last_dim_info = {}
        self._last_dim_info = {
            'input': input_info,
            'output': output_info
        }
        
        return output
    return wrapper


# =============================================================================
# PLE (Piecewise Linear Embeddings) 实现
# =============================================================================

class PiecewiseLinearEncodingImpl(nn.Module):
    """
    PLE编码实现 - 内部使用
    
    维度变换:
    - 输入: (B, N) - batch_size, n_features
    - 输出: (B, N, max_n_bins) - batch_size, n_features, max_bins
    """
    
    def __init__(self, bins: List[Tensor]) -> None:
        """
        Args:
            bins: 每个特征的分箱边界列表，每个元素是1D Tensor
        """
        super().__init__()
        assert len(bins) > 0, "bins列表不能为空"
        
        n_features = len(bins)
        n_bins = [len(x) - 1 for x in bins]  # 每个特征的bin数量
        max_n_bins = max(n_bins)
        
        # 注册为buffer（非可训练参数）
        # weight/bias用于线性变换: output = weight * x + bias
        self.register_buffer('weight', torch.zeros(n_features, max_n_bins))
        self.register_buffer('bias', torch.zeros(n_features, max_n_bins))
        
        # 单bin特征的mask
        single_bin_mask = torch.tensor(n_bins) == 1
        self.register_buffer(
            'single_bin_mask', 
            single_bin_mask if single_bin_mask.any() else None
        )
        
        # 有效位置mask（用于处理不同特征不同bin数量的情况）
        self.register_buffer(
            'mask',
            None if all(len(x) == len(bins[0]) for x in bins)
            else torch.row_stack([
                torch.cat([
                    torch.ones((len(x) - 1) - 1, dtype=torch.bool),
                    torch.zeros(max_n_bins - (len(x) - 1), dtype=torch.bool),
                    torch.ones(1, dtype=torch.bool),
                ])
                for x in bins
            ])
        )
        
        # 初始化weight和bias
        for i, bin_edges in enumerate(bins):
            bin_width = bin_edges.diff()  # (n_bins,)
            w = 1.0 / bin_width  # (n_bins,)
            b = -bin_edges[:-1] / bin_width  # (n_bins,)
            
            # 最后一个bin的编码放在最后位置
            self.weight[i, -1] = w[-1]
            self.bias[i, -1] = b[-1]
            
            # 其他bin的编码放在前面
            self.weight[i, :n_bins[i] - 1] = w[:-1]
            self.bias[i, :n_bins[i] - 1] = b[:-1]
    
    def get_max_n_bins(self) -> int:
        """返回最大bin数量"""
        return self.weight.shape[-1]
    
    @annotate_dims
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x: (B, N) - 输入数值特征
        
        Returns:
            (B, N, max_n_bins) - PLE编码
        """
        # 线性变换: x_ple = weight * x + bias
        # x[..., None]: (B, N, 1)
        # weight: (N, max_n_bins)
        # output: (B, N, max_n_bins)
        x = torch.addcmul(self.bias, self.weight, x[..., None])
        
        if x.shape[-1] > 1:
            # 应用clamp激活
            x = torch.cat([
                # 第一个位置: 最大值为1
                x[..., :1].clamp_max(1.0),
                # 中间位置: 限制在[0, 1]
                x[..., 1:-1].clamp(0.0, 1.0),
                # 最后一个位置: 最小值为0
                (
                    x[..., -1:].clamp_min(0.0)
                    if self.single_bin_mask is None
                    else torch.where(
                        self.single_bin_mask[..., None],
                        x[..., -1:],  # 单bin特征: 不clamp
                        x[..., -1:].clamp_min(0.0)  # 多bin特征: clamp
                    )
                ),
            ], dim=-1)
        
        return x


class PiecewiseLinearEmbeddings(DimAnnotatedModule):
    """
    PLE数值嵌入层
    
    维度变换:
    - 输入: (B, N) - batch_size, n_features
    - 输出: (B, N, D) - batch_size, n_features, d_embedding
    
    示例:
    >>> bins = [torch.linspace(0, 1, 49) for _ in range(10)]  # 10个特征，每个48个bins
    >>> ple = PiecewiseLinearEmbeddings(bins, d_embedding=64, activation=True)
    >>> x = torch.randn(32, 10)  # batch=32, 10个特征
    >>> out = ple(x)  # (32, 10, 64)
    """
    
    def __init__(
        self,
        bins: List[Tensor],
        d_embedding: int,
        *,
        activation: bool = True,
        version: str = 'A',
    ) -> None:
        """
        Args:
            bins: 分箱边界列表
            d_embedding: 嵌入维度
            activation: 是否使用ReLU激活
            version: 'A'或'B'，B版本包含额外的线性层
        """
        super().__init__()
        assert d_embedding > 0, f"d_embedding必须为正数，当前: {d_embedding}"
        assert version in ['A', 'B'], f"version必须是'A'或'B'，当前: {version}"
        
        n_features = len(bins)
        is_version_B = version == 'B'
        
        # Version B: 额外的线性层，初始化为0，逐步学习
        if is_version_B:
            self.linear0 = nn.Linear(n_features, d_embedding)
        else:
            self.register_buffer('linear0', None)
        
        # PLE编码实现
        self.ple_impl = PiecewiseLinearEncodingImpl(bins)
        
        # N个独立的线性层，每个特征一个
        self.linear = NLinear(
            n_features,
            self.ple_impl.get_max_n_bins(),
            d_embedding,
            bias=not is_version_B,
        )
        
        # Version B: 初始化PLE部分为0
        if is_version_B:
            nn.init.zeros_(self.linear.weight)
        
        self.activation = nn.ReLU() if activation else None
    
    def get_output_shape(self) -> torch.Size:
        """返回输出形状 (N, D)"""
        n_features = self.linear.weight.shape[0]
        d_embedding = self.linear.weight.shape[2]
        return torch.Size((n_features, d_embedding))
    
    def get_dim_info(self) -> Dict[str, str]:
        """返回维度信息"""
        n_features, max_bins, d_emb = self.linear.weight.shape
        return {
            'input': '(B, N)',
            'output': '(B, N, D)',
            'params': f'n_features={n_features}, max_bins={max_bins}, d_embedding={d_emb}'
        }
    
    @annotate_dims
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x: (B, N) - 输入数值特征
        
        Returns:
            (B, N, D) - PLE嵌入
        """
        assert x.ndim == 2, f"输入必须是2D张量，当前: {x.ndim}D"
        
        # Version B: 额外的线性变换
        x_linear = None
        if self.linear0 is not None:
            x_linear = self.linear0(x)  # (B, N) -> (B, D)
            x_linear = x_linear.unsqueeze(1)  # (B, 1, D)
        
        # PLE编码 + 线性变换
        x_ple = self.ple_impl(x)  # (B, N) -> (B, N, max_bins)
        x_ple = self.linear(x_ple)  # (B, N, max_bins) -> (B, N, D)
        
        # 激活函数
        if self.activation is not None:
            x_ple = self.activation(x_ple)
        
        # Version B: 残差连接
        if x_linear is not None:
            return x_ple + x_linear
        
        return x_ple


class NLinear(nn.Module):
    """
    N个独立的线性层，每个特征一个
    
    维度变换:
    - 输入: (B, N, D_in)
    - 输出: (B, N, D_out)
    """
    
    def __init__(
        self,
        n: int,
        in_features: int,
        out_features: int,
        bias: bool = True
    ) -> None:
        super().__init__()
        # weight: (N, D_in, D_out)
        self.weight = nn.Parameter(torch.empty(n, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(n, out_features)) if bias else None
        self.reset_parameters()
    
    def reset_parameters(self):
        """参数初始化"""
        d_in_rsqrt = self.weight.shape[-2] ** -0.5
        nn.init.uniform_(self.weight, -d_in_rsqrt, d_in_rsqrt)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -d_in_rsqrt, d_in_rsqrt)
    
    @annotate_dims
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x: (B, N, D_in)
        
        Returns:
            (B, N, D_out)
        """
        assert x.ndim == 3, f"输入必须是3D张量，当前: {x.ndim}D"
        
        # x: (B, N, D_in) -> (N, B, D_in)
        x = x.transpose(0, 1)
        # x @ weight: (N, B, D_in) @ (N, D_in, D_out) = (N, B, D_out)
        x = x @ self.weight
        # x: (N, B, D_out) -> (B, N, D_out)
        x = x.transpose(0, 1)
        
        if self.bias is not None:
            x = x + self.bias  # bias: (N, D_out)
        
        return x


# =============================================================================
# Sparse-MOE 实现
# =============================================================================

class Expert(nn.Module):
    """
    专家网络 - 标准FFN结构
    
    维度变换:
    - 输入: (B, N, D)
    - 输出: (B, N, D)
    """
    
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
    
    @annotate_dims
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x: (B, N, D)
        
        Returns:
            (B, N, D)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SparseMOELayer(DimAnnotatedModule):
    """
    Sparse-MOE层 - Top-K门控
    
    维度变换:
    - 输入: (B, N, D)
    - 输出: (B, N, D)
    - 辅助输出: load_balance_loss (标量)
    
    示例:
    >>> moe = SparseMOELayer(d_model=256, num_experts=8, top_k=2)
    >>> x = torch.randn(32, 100, 256)  # batch=32, seq_len=100
    >>> out, loss = moe(x)
    >>> out.shape  # (32, 100, 256)
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
        """
        Args:
            d_model: 输入/输出维度
            num_experts: 专家数量
            top_k: 每个token选择的专家数量
            expert_hidden_dim: 专家隐藏层维度，默认为4*d_model
            dropout: dropout概率
            load_balance_coef: 负载均衡损失系数
            activation: 激活函数类型
        """
        super().__init__()
        assert top_k <= num_experts, f"top_k({top_k})不能大于num_experts({num_experts})"
        
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_coef = load_balance_coef
        
        if expert_hidden_dim is None:
            expert_hidden_dim = 4 * d_model
        
        # 门控网络: (B, N, D) -> (B, N, E)
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # 专家网络列表
        self.experts = nn.ModuleList([
            Expert(d_model, expert_hidden_dim, dropout, activation)
            for _ in range(num_experts)
        ])
    
    def get_dim_info(self) -> Dict[str, str]:
        """返回维度信息"""
        return {
            'input': '(B, N, D)',
            'output': '(B, N, D)',
            'aux_loss': 'scalar',
            'params': f'd_model={self.d_model}, num_experts={self.num_experts}, top_k={self.top_k}'
        }
    
    @annotate_dims
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        前向传播
        
        Args:
            x: (B, N, D)
        
        Returns:
            output: (B, N, D)
            load_balance_loss: scalar
        """
        B, N, D = x.shape
        
        # 门控分数: (B, N, D) -> (B, N, E)
        gate_logits = self.gate(x)  # (B, N, E)
        
        # Top-K选择
        top_k_logits, top_k_indices = torch.topk(
            gate_logits, self.top_k, dim=-1
        )  # (B, N, K), (B, N, K)
        
        # Softmax归一化（只在top-k上）
        gate_scores = F.softmax(top_k_logits, dim=-1)  # (B, N, K)
        
        # 计算负载均衡损失
        # f_i: 路由到专家i的token比例
        # P_i: 门控分配给专家i的平均概率
        router_prob = F.softmax(gate_logits, dim=-1)  # (B, N, E)
        
        # f: (E,) - 每个专家被选择的频率
        expert_mask = torch.zeros(B, N, self.num_experts, device=x.device)
        expert_mask.scatter_(-1, top_k_indices, 1.0)  # (B, N, E)
        f = expert_mask.mean(dim=(0, 1))  # (E,)
        
        # P: (E,) - 每个专家的平均门控概率
        P = router_prob.mean(dim=(0, 1))  # (E,)
        
        # 负载均衡损失
        load_balance_loss = self.num_experts * (f * P).sum()  # scalar
        load_balance_loss = self.load_balance_coef * load_balance_loss
        
        # 专家计算
        output = torch.zeros_like(x)  # (B, N, D)
        
        for i, expert in enumerate(self.experts):
            # 找到选择专家i的所有token
            expert_mask_i = (top_k_indices == i).any(dim=-1)  # (B, N)
            
            if expert_mask_i.any():
                # 获取选择该专家的token
                expert_input = x[expert_mask_i]  # (M, D), M是被选中的token数
                
                # 获取对应的门控分数
                expert_gate_idx = (top_k_indices == i).nonzero(as_tuple=True)
                expert_gate_scores = gate_scores[expert_gate_idx[0], expert_gate_idx[1]]
                # 需要找到对应的k索引
                k_idx = (top_k_indices[expert_gate_idx[0], expert_gate_idx[1]] == i).nonzero(as_tuple=True)[1]
                expert_gate_scores = gate_scores[expert_gate_idx[0], expert_gate_idx[1], k_idx]
                
                # 专家前向传播
                expert_output = expert(expert_input)  # (M, D)
                
                # 加权
                expert_output = expert_output * expert_gate_scores.unsqueeze(-1)  # (M, D)
                
                # 累加到输出
                output[expert_mask_i] += expert_output
        
        return output, load_balance_loss


# =============================================================================
# 梯度冲突检测
# =============================================================================

class GradientConflictDetector(nn.Module):
    """
    梯度冲突检测器 - 用于多任务学习
    
    检测不同任务梯度之间的冲突，支持PCGrad算法
    
    参考: "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020)
    """
    
    def __init__(
        self,
        num_tasks: int,
        conflict_threshold: float = 0.0,
        enable_pcgrad: bool = True,
    ) -> None:
        """
        Args:
            num_tasks: 任务数量
            conflict_threshold: 冲突检测阈值（余弦相似度小于此值视为冲突）
            enable_pcgrad: 是否启用PCGrad梯度投影
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.conflict_threshold = conflict_threshold
        self.enable_pcgrad = enable_pcgrad
        
        # 存储统计信息
        self.register_buffer('conflict_count', torch.tensor(0))
        self.register_buffer('total_count', torch.tensor(0))
    
    def compute_cosine_similarity(self, grad1: Tensor, grad2: Tensor) -> float:
        """
        计算两个梯度的余弦相似度
        
        Args:
            grad1: 梯度1 (flattened)
            grad2: 梯度2 (flattened)
        
        Returns:
            余弦相似度 [-1, 1]
        """
        grad1_flat = grad1.flatten()
        grad2_flat = grad2.flatten()
        
        dot_product = torch.dot(grad1_flat, grad2_flat)
        norm1 = torch.norm(grad1_flat)
        norm2 = torch.norm(grad2_flat)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return (dot_product / (norm1 * norm2)).item()
    
    def pcgrad_projection(
        self,
        grad_task: Tensor,
        grad_other: Tensor,
    ) -> Tensor:
        """
        PCGrad投影: 当两个梯度冲突时，将grad_task投影到grad_other的正交方向
        
        Args:
            grad_task: 需要调整的梯度
            grad_other: 参考梯度
        
        Returns:
            投影后的梯度
        """
        grad_task_flat = grad_task.flatten()
        grad_other_flat = grad_other.flatten()
        
        dot_product = torch.dot(grad_task_flat, grad_other_flat)
        
        # 如果点积为负，说明有冲突
        if dot_product < 0:
            # 投影: g_i = g_i - (g_i·g_j / ||g_j||^2) * g_j
            norm_sq = torch.dot(grad_other_flat, grad_other_flat)
            if norm_sq > 0:
                projection = (dot_product / norm_sq) * grad_other_flat
                grad_task_flat = grad_task_flat - projection
        
        return grad_task_flat.reshape_as(grad_task)
    
    def detect_conflicts(
        self,
        task_gradients: List[Tensor],
    ) -> Dict[str, Any]:
        """
        检测任务间的梯度冲突
        
        Args:
            task_gradients: 每个任务的梯度列表，每个元素是flattened梯度
        
        Returns:
            冲突检测结果字典
        """
        assert len(task_gradients) == self.num_tasks
        
        # 计算所有任务对的余弦相似度
        cosine_matrix = torch.zeros(self.num_tasks, self.num_tasks)
        conflict_matrix = torch.zeros(self.num_tasks, self.num_tasks, dtype=torch.bool)
        
        for i in range(self.num_tasks):
            for j in range(i + 1, self.num_tasks):
                cos_sim = self.compute_cosine_similarity(
                    task_gradients[i], task_gradients[j]
                )
                cosine_matrix[i, j] = cos_sim
                cosine_matrix[j, i] = cos_sim
                
                # 判断冲突
                is_conflict = cos_sim < self.conflict_threshold
                conflict_matrix[i, j] = is_conflict
                conflict_matrix[j, i] = is_conflict
        
        # 统计信息
        total_pairs = self.num_tasks * (self.num_tasks - 1) // 2
        conflict_pairs = conflict_matrix.triu(diagonal=1).sum().item()
        conflict_ratio = conflict_pairs / total_pairs if total_pairs > 0 else 0.0
        
        # 平均余弦相似度
        avg_cosine = cosine_matrix.triu(diagonal=1).sum().item() / total_pairs
        
        # 各任务梯度范数
        grad_norms = [torch.norm(g).item() for g in task_gradients]
        
        # 更新统计
        self.conflict_count += conflict_pairs
        self.total_count += total_pairs
        
        return {
            'cosine_similarity_matrix': cosine_matrix,
            'conflict_matrix': conflict_matrix,
            'conflict_ratio': conflict_ratio,
            'avg_cosine_similarity': avg_cosine,
            'grad_norms': grad_norms,
            'total_conflicts': self.conflict_count.item(),
            'total_pairs': self.total_count.item(),
        }
    
    def apply_pcgrad(
        self,
        task_gradients: List[Tensor],
    ) -> List[Tensor]:
        """
        应用PCGrad算法处理梯度冲突
        
        Args:
            task_gradients: 每个任务的梯度列表
        
        Returns:
            处理后的梯度列表
        """
        if not self.enable_pcgrad:
            return task_gradients
        
        processed_grads = []
        
        for i in range(self.num_tasks):
            grad_i = task_gradients[i].clone()
            
            # 与其他所有任务的梯度进行投影
            for j in range(self.num_tasks):
                if i != j:
                    grad_i = self.pcgrad_projection(grad_i, task_gradients[j])
            
            processed_grads.append(grad_i)
        
        return processed_grads
    
    def forward(self, task_gradients: List[Tensor]) -> Dict[str, Any]:
        """
        前向传播 - 检测冲突并可选应用PCGrad
        
        Args:
            task_gradients: 每个任务的梯度列表
        
        Returns:
            包含冲突信息和处理后梯度的字典
        """
        # 检测冲突
        conflict_info = self.detect_conflicts(task_gradients)
        
        # 应用PCGrad
        processed_grads = self.apply_pcgrad(task_gradients)
        
        return {
            **conflict_info,
            'processed_gradients': processed_grads,
            'pcgrad_applied': self.enable_pcgrad,
        }


# =============================================================================
# 多任务学习模型生成器
# =============================================================================

class MTLPLEMOEModel(DimAnnotatedModule):
    """
    多任务学习PLE-MOE模型
    
    完整架构:
    1. PLE数值嵌入层
    2. 特征投影层
    3. Sparse-MOE层
    4. 任务特定塔
    5. 任务输出头
    
    维度变换:
    - 输入: (B, N) - 数值特征
    - 输出: List[(B, 1)] - 每个任务的预测
    """
    
    def __init__(
        self,
        num_features: int,
        ple_bins: List[Tensor],
        ple_dim: int,
        num_shared_experts: int,
        num_task_experts: List[int],
        task_names: List[str],
        d_model: int,
        top_k: int = 2,
        expert_hidden_dim: Optional[int] = None,
        tower_hidden_dims: List[int] = [256, 128],
        dropout: float = 0.1,
        load_balance_coef: float = 0.01,
        enable_gradient_conflict_detection: bool = True,
    ) -> None:
        """
        Args:
            num_features: 数值特征数量
            ple_bins: PLE分箱边界列表
            ple_dim: PLE嵌入维度
            num_shared_experts: 共享专家数量
            num_task_experts: 每个任务的专家数量列表
            task_names: 任务名称列表
            d_model: 模型隐藏维度
            top_k: MOE top-k
            expert_hidden_dim: 专家隐藏层维度
            tower_hidden_dims: 任务塔隐藏层维度列表
            dropout: dropout概率
            load_balance_coef: 负载均衡损失系数
            enable_gradient_conflict_detection: 是否启用梯度冲突检测
        """
        super().__init__()
        
        self.num_tasks = len(task_names)
        self.task_names = task_names
        self.enable_gradient_conflict_detection = enable_gradient_conflict_detection
        
        # 1. PLE数值嵌入层
        self.ple_embeddings = PiecewiseLinearEmbeddings(
            bins=ple_bins,
            d_embedding=ple_dim,
            activation=True,
        )
        
        # 2. 特征扁平化 + 投影
        ple_output_dim = num_features * ple_dim
        self.feature_projection = nn.Sequential(
            nn.Linear(ple_output_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 3. Sparse-MOE层
        total_experts = num_shared_experts + sum(num_task_experts)
        self.moe_layer = SparseMOELayer(
            d_model=d_model,
            num_experts=total_experts,
            top_k=top_k,
            expert_hidden_dim=expert_hidden_dim,
            dropout=dropout,
            load_balance_coef=load_balance_coef,
        )
        
        # 记录专家分配
        self.expert_allocation = {
            'shared': list(range(num_shared_experts)),
        }
        idx = num_shared_experts
        for i, task_name in enumerate(task_names):
            self.expert_allocation[task_name] = list(range(idx, idx + num_task_experts[i]))
            idx += num_task_experts[i]
        
        # 4. 任务特定塔
        self.task_towers = nn.ModuleList()
        for _ in range(self.num_tasks):
            layers = []
            in_dim = d_model
            for hidden_dim in tower_hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                in_dim = hidden_dim
            self.task_towers.append(nn.Sequential(*layers))
        
        # 5. 任务输出头
        self.task_heads = nn.ModuleList([
            nn.Linear(tower_hidden_dims[-1], 1)
            for _ in range(self.num_tasks)
        ])
        
        # 6. 梯度冲突检测器
        if enable_gradient_conflict_detection:
            self.gradient_conflict_detector = GradientConflictDetector(
                num_tasks=self.num_tasks,
                conflict_threshold=0.0,
                enable_pcgrad=True,
            )
        else:
            self.gradient_conflict_detector = None
    
    def get_dim_info(self) -> Dict[str, str]:
        """返回维度信息"""
        return {
            'input': '(B, N)',
            'output': f'List[(B, 1)] x {self.num_tasks}',
            'tasks': str(self.task_names),
            'expert_allocation': str(self.expert_allocation),
        }
    
    @annotate_dims
    def forward(self, x: Tensor) -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            x: (B, N) - 数值特征
        
        Returns:
            包含预测结果和辅助损失的字典
        """
        # 1. PLE嵌入
        # x: (B, N) -> (B, N, ple_dim)
        x_ple = self.ple_embeddings(x)
        
        # 2. 扁平化 + 投影
        # x_ple: (B, N, ple_dim) -> (B, N*ple_dim)
        x_flat = x_ple.flatten(start_dim=1)
        # x_flat: (B, N*ple_dim) -> (B, d_model)
        x_proj = self.feature_projection(x_flat)
        
        # 3. MOE层
        # x_proj: (B, d_model) -> (B, 1, d_model) 添加序列维度
        x_seq = x_proj.unsqueeze(1)
        # moe_out: (B, 1, d_model)
        moe_out, load_balance_loss = self.moe_layer(x_seq)
        # 移除序列维度: (B, 1, d_model) -> (B, d_model)
        moe_out = moe_out.squeeze(1)
        
        # 4. 任务塔 + 输出头
        predictions = []
        for i in range(self.num_tasks):
            tower_out = self.task_towers[i](moe_out)
            pred = self.task_heads[i](tower_out)
            predictions.append(pred)
        
        return {
            'predictions': predictions,
            'load_balance_loss': load_balance_loss,
            'task_names': self.task_names,
        }
    
    def compute_loss(
        self,
        predictions: List[Tensor],
        targets: List[Tensor],
        task_loss_weights: Optional[List[float]] = None,
    ) -> Dict[str, Tensor]:
        """
        计算多任务损失
        
        Args:
            predictions: 每个任务的预测列表
            targets: 每个任务的目标列表
            task_loss_weights: 每个任务的损失权重
        
        Returns:
            损失字典
        """
        if task_loss_weights is None:
            task_loss_weights = [1.0] * self.num_tasks
        
        task_losses = []
        for i in range(self.num_tasks):
            # 使用BCEWithLogitsLoss（假设是二分类任务）
            loss_fn = nn.BCEWithLogitsLoss()
            task_loss = loss_fn(predictions[i], targets[i])
            task_losses.append(task_loss * task_loss_weights[i])
        
        total_loss = sum(task_losses)
        
        loss_dict = {'total_loss': total_loss}
        for i in range(self.num_tasks):
            loss_dict[f'{self.task_names[i]}_loss'] = task_losses[i]
        
        return loss_dict


# =============================================================================
# 便捷生成函数
# =============================================================================

def generate_ple_embeddings(
    n_features: int,
    d_embedding: int,
    n_bins: int = 48,
    activation: bool = True,
    X_train: Optional[Tensor] = None,
) -> PiecewiseLinearEmbeddings:
    """
    生成PLE嵌入层
    
    Args:
        n_features: 特征数量
        d_embedding: 嵌入维度
        n_bins: 每个特征的bin数量
        activation: 是否使用ReLU激活
        X_train: 训练数据，用于计算分箱边界。如果为None，使用均匀分箱
    
    Returns:
        PiecewiseLinearEmbeddings实例
    
    示例:
    >>> ple = generate_ple_embeddings(n_features=10, d_embedding=64, n_bins=48)
    >>> x = torch.randn(32, 10)
    >>> out = ple(x)  # (32, 10, 64)
    """
    if X_train is not None:
        # 基于训练数据计算分箱边界
        bins = [
            torch.quantile(
                X_train[:, i],
                torch.linspace(0.0, 1.0, n_bins + 1)
            )
            for i in range(n_features)
        ]
    else:
        # 使用均匀分箱 [-3, 3]
        bins = [
            torch.linspace(-3.0, 3.0, n_bins + 1)
            for _ in range(n_features)
        ]
    
    return PiecewiseLinearEmbeddings(
        bins=bins,
        d_embedding=d_embedding,
        activation=activation,
    )


def generate_sparse_moe(
    d_model: int,
    num_experts: int = 8,
    top_k: int = 2,
    expert_hidden_dim: Optional[int] = None,
    dropout: float = 0.1,
    load_balance_coef: float = 0.01,
) -> SparseMOELayer:
    """
    生成Sparse-MOE层
    
    Args:
        d_model: 输入/输出维度
        num_experts: 专家数量
        top_k: 每个token选择的专家数量
        expert_hidden_dim: 专家隐藏层维度
        dropout: dropout概率
        load_balance_coef: 负载均衡损失系数
    
    Returns:
        SparseMOELayer实例
    
    示例:
    >>> moe = generate_sparse_moe(d_model=256, num_experts=8, top_k=2)
    >>> x = torch.randn(32, 100, 256)
    >>> out, loss = moe(x)
    """
    return SparseMOELayer(
        d_model=d_model,
        num_experts=num_experts,
        top_k=top_k,
        expert_hidden_dim=expert_hidden_dim,
        dropout=dropout,
        load_balance_coef=load_balance_coef,
    )


def generate_mtl_ple_moe_model(
    num_features: int,
    ple_bins: Union[int, List[Tensor]],
    ple_dim: int,
    num_shared_experts: int,
    num_task_experts: List[int],
    tasks: List[str],
    d_model: int = 256,
    top_k: int = 2,
    expert_hidden_dim: Optional[int] = None,
    tower_hidden_dims: List[int] = [256, 128],
    dropout: float = 0.1,
    load_balance_coef: float = 0.01,
    enable_gradient_conflict_detection: bool = True,
    X_train: Optional[Tensor] = None,
) -> MTLPLEMOEModel:
    """
    生成完整的多任务学习PLE-MOE模型
    
    Args:
        num_features: 数值特征数量
        ple_bins: 每个特征的bin数量，或预计算的分箱边界列表
        ple_dim: PLE嵌入维度
        num_shared_experts: 共享专家数量
        num_task_experts: 每个任务的专家数量列表
        tasks: 任务名称列表
        d_model: 模型隐藏维度
        top_k: MOE top-k
        expert_hidden_dim: 专家隐藏层维度
        tower_hidden_dims: 任务塔隐藏层维度
        dropout: dropout概率
        load_balance_coef: 负载均衡损失系数
        enable_gradient_conflict_detection: 是否启用梯度冲突检测
        X_train: 训练数据，用于计算分箱边界
    
    Returns:
        MTLPLEMOEModel实例
    
    示例:
    >>> model = generate_mtl_ple_moe_model(
    ...     num_features=50,
    ...     ple_bins=64,
    ...     ple_dim=64,
    ...     num_shared_experts=4,
    ...     num_task_experts=[3, 3],
    ...     tasks=['ctr', 'cvr'],
    ...     d_model=256,
    ... )
    >>> x = torch.randn(32, 50)
    >>> out = model(x)
    """
    # 处理ple_bins参数
    if isinstance(ple_bins, int):
        n_bins = ple_bins
        if X_train is not None:
            # 基于训练数据计算分箱边界
            bins = [
                torch.quantile(
                    X_train[:, i],
                    torch.linspace(0.0, 1.0, n_bins + 1)
                )
                for i in range(num_features)
            ]
        else:
            # 使用均匀分箱
            bins = [
                torch.linspace(-3.0, 3.0, n_bins + 1)
                for _ in range(num_features)
            ]
    else:
        bins = ple_bins
    
    return MTLPLEMOEModel(
        num_features=num_features,
        ple_bins=bins,
        ple_dim=ple_dim,
        num_shared_experts=num_shared_experts,
        num_task_experts=num_task_experts,
        task_names=tasks,
        d_model=d_model,
        top_k=top_k,
        expert_hidden_dim=expert_hidden_dim,
        tower_hidden_dims=tower_hidden_dims,
        dropout=dropout,
        load_balance_coef=load_balance_coef,
        enable_gradient_conflict_detection=enable_gradient_conflict_detection,
    )


# =============================================================================
# 代码生成器 - 输出完整的Python代码字符串
# =============================================================================

class PLEMOECodeGenerator:
    """
    PLE-MOE代码生成器 - 生成完整的可运行Python代码
    """
    
    @staticmethod
    def generate_ple_code(
        n_features: int,
        d_embedding: int,
        n_bins: int = 48,
        with_comments: bool = True,
    ) -> str:
        """生成PLE嵌入层代码"""
        
        code = '''"""
PLE (Piecewise Linear Embeddings) Implementation
================================================
Generated by ple-moe-expert Skill

维度变换:
- 输入: (batch_size, n_features)
- 输出: (batch_size, n_features, d_embedding)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List


class PiecewiseLinearEmbeddings(nn.Module):
    """
    PLE数值嵌入层
    
    参数:
        n_features: {n_features}  # 数值特征数量
        d_embedding: {d_embedding}  # 嵌入维度
        n_bins: {n_bins}  # 每个特征的分箱数
    """
    
    def __init__(self):
        super().__init__()
        
        # 分箱边界 (预计算)
        # 每个特征有 {n_bins}+1 个边界点，形成 {n_bins} 个bins
        self.register_buffer('bins', torch.stack([
            torch.linspace(-3.0, 3.0, {n_bins}+1)
            for _ in range({n_features})
        ]))  # 维度: ({n_features}, {n_bins}+1)
        
        # PLE编码的线性变换参数
        max_n_bins = {n_bins}
        self.register_buffer('ple_weight', torch.zeros({n_features}, max_n_bins))
        self.register_buffer('ple_bias', torch.zeros({n_features}, max_n_bins))
        
        # 初始化PLE参数
        for i in range({n_features}):
            bin_edges = self.bins[i]
            bin_width = bin_edges[1:] - bin_edges[:-1]  # ({n_bins},)
            w = 1.0 / bin_width
            b = -bin_edges[:-1] / bin_width
            self.ple_weight[i] = w
            self.ple_bias[i] = b
        
        # 可训练的线性投影
        self.linear = nn.Linear(max_n_bins, {d_embedding})
        self.activation = nn.ReLU()
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x: (B, {n_features}) - 输入数值特征
        
        Returns:
            (B, {n_features}, {d_embedding}) - PLE嵌入
        """
        B = x.shape[0]
        
        # PLE编码: (B, {n_features}) -> (B, {n_features}, {n_bins})
        x_ple = torch.addcmul(self.ple_bias, self.ple_weight, x.unsqueeze(-1))
        
        # Clamp激活
        x_ple = torch.cat([
            x_ple[..., :1].clamp_max(1.0),
            x_ple[..., 1:-1].clamp(0.0, 1.0),
            x_ple[..., -1:].clamp_min(0.0),
        ], dim=-1)
        
        # 线性投影: (B, {n_features}, {n_bins}) -> (B, {n_features}, {d_embedding})
        x_embed = self.linear(x_ple)
        x_embed = self.activation(x_embed)
        
        return x_embed


# 使用示例
if __name__ == "__main__":
    ple = PiecewiseLinearEmbeddings()
    x = torch.randn(32, {n_features})  # batch=32
    output = ple(x)
    print(f"输入维度: {{x.shape}}")  # (32, {n_features})
    print(f"输出维度: {{output.shape}}")  # (32, {n_features}, {d_embedding})
'''.format(
            n_features=n_features,
            d_embedding=d_embedding,
            n_bins=n_bins,
        )
        
        return code
    
    @staticmethod
    def generate_moe_code(
        d_model: int,
        num_experts: int,
        top_k: int,
        expert_hidden_dim: int,
    ) -> str:
        """生成Sparse-MOE代码"""
        
        code = '''"""
Sparse-MOE (Mixture of Experts) Implementation
==============================================
Generated by ple-moe-expert Skill

维度变换:
- 输入: (batch_size, seq_len, d_model)
- 输出: (batch_size, seq_len, d_model)
- 辅助损失: scalar (负载均衡损失)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class Expert(nn.Module):
    """专家网络 - FFN结构"""
    
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SparseMOELayer(nn.Module):
    """
    Sparse-MOE层 - Top-{top_k}门控
    
    参数:
        d_model: {d_model}  # 输入/输出维度
        num_experts: {num_experts}  # 专家数量
        top_k: {top_k}  # 每个token选择的专家数
        expert_hidden_dim: {expert_hidden_dim}  # 专家隐藏层维度
    """
    
    def __init__(self):
        super().__init__()
        self.d_model = {d_model}
        self.num_experts = {num_experts}
        self.top_k = {top_k}
        
        # 门控网络: (B, N, D) -> (B, N, E)
        self.gate = nn.Linear({d_model}, {num_experts}, bias=False)
        
        # 专家网络
        self.experts = nn.ModuleList([
            Expert({d_model}, {expert_hidden_dim})
            for _ in range({num_experts})
        ])
        
        # 负载均衡损失系数
        self.load_balance_coef = 0.01
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        前向传播
        
        Args:
            x: (B, N, {d_model}) - 输入
        
        Returns:
            output: (B, N, {d_model}) - 输出
            aux_loss: scalar - 负载均衡损失
        """
        B, N, D = x.shape
        
        # 门控分数: (B, N, {num_experts})
        gate_logits = self.gate(x)
        
        # Top-{top_k}选择
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        gate_scores = F.softmax(top_k_logits, dim=-1)  # (B, N, {top_k})
        
        # 计算负载均衡损失
        router_prob = F.softmax(gate_logits, dim=-1)  # (B, N, E)
        
        # f: 每个专家被选择的频率
        expert_mask = torch.zeros(B, N, self.num_experts, device=x.device)
        expert_mask.scatter_(-1, top_k_indices, 1.0)
        f = expert_mask.mean(dim=(0, 1))  # (E,)
        
        # P: 每个专家的平均门控概率
        P = router_prob.mean(dim=(0, 1))  # (E,)
        
        # 负载均衡损失
        aux_loss = self.num_experts * (f * P).sum()
        aux_loss = self.load_balance_coef * aux_loss
        
        # 专家计算
        output = torch.zeros_like(x)
        
        for i, expert in enumerate(self.experts):
            # 找到选择专家i的所有位置
            mask = (top_k_indices == i).any(dim=-1)  # (B, N)
            
            if mask.any():
                # 获取门控分数
                positions = mask.nonzero(as_tuple=True)
                
                # 找到对应的top-k索引
                k_positions = (top_k_indices[mask] == i).nonzero(as_tuple=True)[1]
                scores = gate_scores[mask][range(len(k_positions)), k_positions]
                
                # 专家计算
                expert_input = x[mask]  # (M, D)
                expert_output = expert(expert_input)  # (M, D)
                expert_output = expert_output * scores.unsqueeze(-1)
                
                output[mask] += expert_output
        
        return output, aux_loss


# 使用示例
if __name__ == "__main__":
    moe = SparseMOELayer()
    x = torch.randn(32, 100, {d_model})  # batch=32, seq_len=100
    output, aux_loss = moe(x)
    print(f"输入维度: {{x.shape}}")  # (32, 100, {d_model})
    print(f"输出维度: {{output.shape}}")  # (32, 100, {d_model})
    print(f"辅助损失: {{aux_loss.item()}}")
'''.format(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            expert_hidden_dim=expert_hidden_dim,
        )
        
        return code
    
    @staticmethod
    def generate_gradient_conflict_detector_code(num_tasks: int) -> str:
        """生成梯度冲突检测器代码"""
        
        code = f'''"""
Gradient Conflict Detector
==========================
Generated by ple-moe-expert Skill

检测多任务学习中的梯度冲突，支持PCGrad算法

参考: "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Dict, Any


class GradientConflictDetector(nn.Module):
    """
    梯度冲突检测器
    
    参数:
        num_tasks: {num_tasks}  # 任务数量
    """
    
    def __init__(self):
        super().__init__()
        self.num_tasks = {num_tasks}
        self.conflict_threshold = 0.0
    
    def compute_cosine_similarity(self, grad1: Tensor, grad2: Tensor) -> float:
        """计算两个梯度的余弦相似度"""
        g1 = grad1.flatten()
        g2 = grad2.flatten()
        
        dot = torch.dot(g1, g2)
        norm1 = torch.norm(g1)
        norm2 = torch.norm(g2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return (dot / (norm1 * norm2)).item()
    
    def pcgrad_projection(self, grad_task: Tensor, grad_other: Tensor) -> Tensor:
        """
        PCGrad投影: 将grad_task投影到grad_other的正交方向
        """
        g_task = grad_task.flatten()
        g_other = grad_other.flatten()
        
        dot = torch.dot(g_task, g_other)
        
        if dot < 0:  # 冲突
            norm_sq = torch.dot(g_other, g_other)
            if norm_sq > 0:
                proj = (dot / norm_sq) * g_other
                g_task = g_task - proj
        
        return g_task.reshape_as(grad_task)
    
    def detect_conflicts(self, task_gradients: List[Tensor]) -> Dict[str, Any]:
        """
        检测梯度冲突
        
        Args:
            task_gradients: 每个任务的梯度列表 (flattened)
        
        Returns:
            冲突检测结果
        """
        n = len(task_gradients)
        cosine_matrix = torch.zeros(n, n)
        conflict_matrix = torch.zeros(n, n, dtype=torch.bool)
        
        for i in range(n):
            for j in range(i + 1, n):
                cos_sim = self.compute_cosine_similarity(
                    task_gradients[i], task_gradients[j]
                )
                cosine_matrix[i, j] = cos_sim
                cosine_matrix[j, i] = cos_sim
                
                is_conflict = cos_sim < self.conflict_threshold
                conflict_matrix[i, j] = is_conflict
                conflict_matrix[j, i] = is_conflict
        
        # 统计
        total_pairs = n * (n - 1) // 2
        conflict_pairs = conflict_matrix.triu(diagonal=1).sum().item()
        conflict_ratio = conflict_pairs / total_pairs
        
        avg_cosine = cosine_matrix.triu(diagonal=1).sum().item() / total_pairs
        grad_norms = [torch.norm(g).item() for g in task_gradients]
        
        return dict(
            conflict_ratio=conflict_ratio,
            avg_cosine_similarity=avg_cosine,
            grad_norms=grad_norms,
            cosine_matrix=cosine_matrix,
        )
    
    def apply_pcgrad(self, task_gradients: List[Tensor]) -> List[Tensor]:
        """应用PCGrad处理梯度冲突"""
        processed = []
        
        for i in range(len(task_gradients)):
            grad_i = task_gradients[i].clone()
            
            for j in range(len(task_gradients)):
                if i != j:
                    grad_i = self.pcgrad_projection(grad_i, task_gradients[j])
            
            processed.append(grad_i)
        
        return processed


# 使用示例
if __name__ == "__main__":
    detector = GradientConflictDetector()
    
    # 模拟两个任务的梯度
    grad1 = torch.randn(1000)
    grad2 = torch.randn(1000)
    
    # 检测冲突
    result = detector.detect_conflicts([grad1, grad2])
    print(f"冲突比例: {{result['conflict_ratio']:.2%}}")
    print(f"平均余弦相似度: {{result['avg_cosine_similarity']:.4f}}")
    
    # 应用PCGrad
    processed = detector.apply_pcgrad([grad1, grad2])
    print(f"处理后梯度范数: {{[torch.norm(g).item() for g in processed]}}")
'''
        
        return code


# =============================================================================
# 主函数 - 便捷调用
# =============================================================================

if __name__ == "__main__":
    # 测试PLE
    print("=" * 60)
    print("Testing PLE Embeddings")
    print("=" * 60)
    
    ple = generate_ple_embeddings(
        n_features=10,
        d_embedding=64,
        n_bins=48,
    )
    x = torch.randn(32, 10)
    out = ple(x)
    print(f"PLE Input: {x.shape}")
    print(f"PLE Output: {out.shape}")
    print(f"PLE Dim Info: {ple.get_dim_info()}")
    
    # 测试MOE
    print("\n" + "=" * 60)
    print("Testing Sparse-MOE")
    print("=" * 60)
    
    moe = generate_sparse_moe(
        d_model=256,
        num_experts=8,
        top_k=2,
    )
    x = torch.randn(32, 100, 256)
    out, aux_loss = moe(x)
    print(f"MOE Input: {x.shape}")
    print(f"MOE Output: {out.shape}")
    print(f"MOE Aux Loss: {aux_loss.item():.6f}")
    print(f"MOE Dim Info: {moe.get_dim_info()}")
    
    # 测试梯度冲突检测器
    print("\n" + "=" * 60)
    print("Testing Gradient Conflict Detector")
    print("=" * 60)
    
    detector = GradientConflictDetector(
        num_tasks=2,
        conflict_threshold=0.0,
        enable_pcgrad=True,
    )
    
    grad1 = torch.randn(1000)
    grad2 = -grad1 + torch.randn(1000) * 0.1  # 近似相反方向
    
    result = detector([grad1, grad2])
    print(f"Conflict Ratio: {result['conflict_ratio']:.2%}")
    print(f"Avg Cosine Similarity: {result['avg_cosine_similarity']:.4f}")
    print(f"Grad Norms: {result['grad_norms']}")
    
    # 测试代码生成
    print("\n" + "=" * 60)
    print("Testing Code Generator")
    print("=" * 60)
    
    generator = PLEMOECodeGenerator()
    
    ple_code = generator.generate_ple_code(
        n_features=10,
        d_embedding=64,
        n_bins=48,
    )
    print(f"Generated PLE Code Length: {len(ple_code)} chars")
    
    moe_code = generator.generate_moe_code(
        d_model=256,
        num_experts=8,
        top_k=2,
        expert_hidden_dim=1024,
    )
    print(f"Generated MOE Code Length: {len(moe_code)} chars")
