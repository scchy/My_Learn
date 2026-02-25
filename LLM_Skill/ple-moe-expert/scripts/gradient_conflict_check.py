"""
Gradient Conflict Detector for Multi-Task Learning
==================================================
梯度冲突检测器 - 用于检测和处理多任务学习中的梯度冲突

支持:
1. 梯度余弦相似度计算
2. 冲突比例统计
3. PCGrad (Project Conflicting Gradients) 算法
4. 实时梯度监控

参考论文:
- "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020)
  https://arxiv.org/abs/2001.06782

作者: ple-moe-expert Skill
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import deque


class GradientConflictDetector(nn.Module):
    """
    梯度冲突检测器
    
    用于检测多任务学习中不同任务梯度之间的冲突，并可选应用PCGrad算法进行处理。
    
    维度说明:
    - 输入梯度: List[Tensor]，每个元素是flattened的梯度张量
    - 输出: 包含冲突统计信息和处理后梯度的字典
    
    示例:
    >>> detector = GradientConflictDetector(num_tasks=2)
    >>> grad1 = torch.randn(1000)  # 任务1的梯度
    >>> grad2 = torch.randn(1000)  # 任务2的梯度
    >>> result = detector([grad1, grad2])
    >>> print(f"冲突比例: {result['conflict_ratio']:.2%}")
    """
    
    def __init__(
        self,
        num_tasks: int,
        conflict_threshold: float = 0.0,
        enable_pcgrad: bool = True,
        history_size: int = 100,
    ) -> None:
        """
        Args:
            num_tasks: 任务数量
            conflict_threshold: 冲突检测阈值，余弦相似度小于此值视为冲突
            enable_pcgrad: 是否启用PCGrad梯度投影
            history_size: 历史记录窗口大小
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.conflict_threshold = conflict_threshold
        self.enable_pcgrad = enable_pcgrad
        
        # 历史统计
        self.conflict_history = deque(maxlen=history_size)
        self.cosine_history = deque(maxlen=history_size)
        
        # 累计统计
        self.register_buffer('total_conflicts', torch.tensor(0))
        self.register_buffer('total_pairs', torch.tensor(0))
    
    def compute_cosine_similarity(self, grad1: Tensor, grad2: Tensor) -> float:
        """
        计算两个梯度的余弦相似度
        
        公式: cos(θ) = (g1 · g2) / (||g1|| * ||g2||)
        
        Args:
            grad1: 梯度1 (任意形状，会被flatten)
            grad2: 梯度2 (任意形状，会被flatten)
        
        Returns:
            余弦相似度，范围[-1, 1]
            - 1: 完全同向
            - 0: 正交
            - -1: 完全反向（最大冲突）
        """
        grad1_flat = grad1.flatten()
        grad2_flat = grad2.flatten()
        
        dot_product = torch.dot(grad1_flat, grad2_flat)
        norm1 = torch.norm(grad1_flat)
        norm2 = torch.norm(grad2_flat)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return (dot_product / (norm1 * norm2)).item()
    
    def compute_conflict_score(self, grad1: Tensor, grad2: Tensor) -> float:
        """
        计算冲突分数 (0-1，越大表示冲突越严重)
        
        Args:
            grad1: 梯度1
            grad2: 梯度2
        
        Returns:
            冲突分数 [0, 1]
        """
        cos_sim = self.compute_cosine_similarity(grad1, grad2)
        # 将余弦相似度转换为冲突分数
        # cos_sim = -1 (完全冲突) -> score = 1
        # cos_sim = 1 (完全同向) -> score = 0
        return (1 - cos_sim) / 2
    
    def pcgrad_projection(
        self,
        grad_task: Tensor,
        grad_other: Tensor,
    ) -> Tensor:
        """
        PCGrad投影: 当两个梯度冲突时，将grad_task投影到grad_other的正交方向
        
        算法步骤:
        1. 计算两个梯度的点积
        2. 如果点积 < 0 (存在冲突):
           - 计算grad_task在grad_other方向上的投影
           - 从grad_task中减去该投影
        
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
    
    def apply_pcgrad(
        self,
        task_gradients: List[Tensor],
    ) -> List[Tensor]:
        """
        应用PCGrad算法处理梯度冲突
        
        对每个任务的梯度，依次投影到其他所有任务梯度的正交方向
        
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
    
    def detect_conflicts(
        self,
        task_gradients: List[Tensor],
    ) -> Dict[str, Any]:
        """
        检测任务间的梯度冲突
        
        Args:
            task_gradients: 每个任务的梯度列表，每个元素是flattened梯度
        
        Returns:
            冲突检测结果字典，包含:
            - cosine_similarity_matrix: 余弦相似度矩阵
            - conflict_matrix: 冲突矩阵
            - conflict_ratio: 冲突比例
            - avg_cosine_similarity: 平均余弦相似度
            - grad_norms: 各任务梯度范数
            - conflict_scores: 冲突分数列表
        """
        assert len(task_gradients) == self.num_tasks, \
            f"梯度数量({len(task_gradients)})应与任务数量({self.num_tasks})一致"
        
        # 计算所有任务对的余弦相似度
        cosine_matrix = torch.zeros(self.num_tasks, self.num_tasks)
        conflict_matrix = torch.zeros(self.num_tasks, self.num_tasks, dtype=torch.bool)
        conflict_scores = []
        
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
                
                # 计算冲突分数
                conflict_score = self.compute_conflict_score(
                    task_gradients[i], task_gradients[j]
                )
                conflict_scores.append(conflict_score)
        
        # 统计信息
        total_pairs = self.num_tasks * (self.num_tasks - 1) // 2
        conflict_pairs = conflict_matrix.triu(diagonal=1).sum().item()
        conflict_ratio = conflict_pairs / total_pairs if total_pairs > 0 else 0.0
        
        # 平均余弦相似度
        avg_cosine = cosine_matrix.triu(diagonal=1).sum().item() / total_pairs
        
        # 各任务梯度范数
        grad_norms = [torch.norm(g).item() for g in task_gradients]
        
        # 更新历史
        self.conflict_history.append(conflict_ratio)
        self.cosine_history.append(avg_cosine)
        
        # 更新累计统计
        self.total_conflicts += conflict_pairs
        self.total_total_pairs = total_pairs
        
        return {
            'cosine_similarity_matrix': cosine_matrix,
            'conflict_matrix': conflict_matrix,
            'conflict_ratio': conflict_ratio,
            'avg_cosine_similarity': avg_cosine,
            'grad_norms': grad_norms,
            'conflict_scores': conflict_scores,
            'total_conflicts': self.total_conflicts.item(),
            'total_pairs': getattr(self, 'total_total_pairs', 0),
        }
    
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
    
    def get_statistics(self) -> Dict[str, float]:
        """
        获取历史统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'total_conflicts': self.total_conflicts.item(),
            'total_pairs': getattr(self, 'total_total_pairs', 0),
        }
        
        if len(self.conflict_history) > 0:
            stats['avg_conflict_ratio'] = np.mean(self.conflict_history)
            stats['max_conflict_ratio'] = max(self.conflict_history)
            stats['min_conflict_ratio'] = min(self.conflict_history)
        
        if len(self.cosine_history) > 0:
            stats['avg_cosine_similarity'] = np.mean(self.cosine_history)
        
        return stats
    
    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.conflict_history.clear()
        self.cosine_history.clear()
        self.total_conflicts.zero_()


class MultiTaskGradientHandler:
    """
    多任务梯度处理器
    
    整合梯度冲突检测和PCGrad处理，提供便捷的梯度处理接口
    
    示例:
    >>> handler = MultiTaskGradientHandler(num_tasks=2)
    >>> model = MyMTLModel()
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> 
    >>> for batch in dataloader:
    ...     losses = model(batch)  # 返回每个任务的损失
    ...     handler.backward(losses, model)
    ...     handler.step(optimizer)
    """
    
    def __init__(
        self,
        num_tasks: int,
        conflict_threshold: float = 0.0,
        enable_pcgrad: bool = True,
        log_interval: int = 100,
    ):
        """
        Args:
            num_tasks: 任务数量
            conflict_threshold: 冲突检测阈值
            enable_pcgrad: 是否启用PCGrad
            log_interval: 日志打印间隔
        """
        self.num_tasks = num_tasks
        self.log_interval = log_interval
        self.step_count = 0
        
        self.detector = GradientConflictDetector(
            num_tasks=num_tasks,
            conflict_threshold=conflict_threshold,
            enable_pcgrad=enable_pcgrad,
        )
        
        # 存储当前梯度
        self.current_gradients: List[Optional[Tensor]] = [None] * num_tasks
        self.conflict_info: Optional[Dict[str, Any]] = None
    
    def backward(
        self,
        task_losses: List[Tensor],
        model: nn.Module,
        retain_graph: bool = False,
    ) -> Dict[str, Any]:
        """
        执行多任务反向传播
        
        Args:
            task_losses: 每个任务的损失列表
            model: 模型
            retain_graph: 是否保留计算图
        
        Returns:
            冲突信息字典
        """
        assert len(task_losses) == self.num_tasks
        
        # 清空梯度
        model.zero_grad()
        
        # 分别计算每个任务的梯度
        task_gradients = []
        
        for i, loss in enumerate(task_losses):
            # 反向传播
            loss.backward(retain_graph=(retain_graph or i < self.num_tasks - 1))
            
            # 收集梯度
            grads = []
            for p in model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.flatten())
            
            if len(grads) > 0:
                task_grad = torch.cat(grads)
            else:
                task_grad = torch.tensor(0.0)
            
            task_gradients.append(task_grad)
            
            # 清空梯度（为下一个任务准备）
            model.zero_grad()
        
        # 检测冲突并应用PCGrad
        result = self.detector(task_gradients)
        self.conflict_info = result
        self.current_gradients = result['processed_gradients']
        
        # 应用处理后的梯度
        grad_idx = 0
        for p in model.parameters():
            if p.grad is not None:
                numel = p.numel()
                # 合并所有任务的梯度（简单平均）
                merged_grad = sum(
                    g[grad_idx:grad_idx+numel] 
                    for g in self.current_gradients
                ) / self.num_tasks
                p.grad = merged_grad.reshape(p.shape)
                grad_idx += numel
        
        # 日志
        self.step_count += 1
        if self.step_count % self.log_interval == 0:
            self._log_conflict_info()
        
        return result
    
    def _log_conflict_info(self) -> None:
        """打印冲突信息"""
        if self.conflict_info is None:
            return
        
        print(f"\n[Step {self.step_count}] Gradient Conflict Info:")
        print(f"  Conflict Ratio: {self.conflict_info['conflict_ratio']:.2%}")
        print(f"  Avg Cosine Similarity: {self.conflict_info['avg_cosine_similarity']:.4f}")
        print(f"  Grad Norms: {[f'{n:.4f}' for n in self.conflict_info['grad_norms']]}")
        print(f"  PCGrad Applied: {self.conflict_info['pcgrad_applied']}")
    
    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """执行优化器步骤"""
        optimizer.step()
    
    def get_statistics(self) -> Dict[str, float]:
        """获取统计信息"""
        return self.detector.get_statistics()


def visualize_gradient_conflicts(
    cosine_matrix: Tensor,
    task_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    可视化梯度冲突矩阵
    
    Args:
        cosine_matrix: 余弦相似度矩阵
        task_names: 任务名称列表
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib和seaborn需要安装: pip install matplotlib seaborn")
        return
    
    n = cosine_matrix.shape[0]
    if task_names is None:
        task_names = [f"Task {i+1}" for i in range(n)]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cosine_matrix.cpu().numpy(),
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        xticklabels=task_names,
        yticklabels=task_names,
        vmin=-1,
        vmax=1,
    )
    plt.title("Gradient Cosine Similarity Matrix")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Gradient Conflict Detector Test")
    print("=" * 70)
    
    # 测试1: 无冲突梯度
    print("\n[Test 1] No Conflict Gradients")
    print("-" * 40)
    
    detector = GradientConflictDetector(num_tasks=2)
    
    # 同向梯度
    base_grad = torch.randn(1000)
    grad1 = base_grad + torch.randn(1000) * 0.1
    grad2 = base_grad + torch.randn(1000) * 0.1
    
    result = detector([grad1, grad2])
    print(f"Cosine Similarity: {result['avg_cosine_similarity']:.4f}")
    print(f"Conflict Ratio: {result['conflict_ratio']:.2%}")
    
    # 测试2: 强冲突梯度
    print("\n[Test 2] Strong Conflict Gradients")
    print("-" * 40)
    
    grad1 = torch.randn(1000)
    grad2 = -grad1 + torch.randn(1000) * 0.1  # 近似相反方向
    
    result = detector([grad1, grad2])
    print(f"Cosine Similarity: {result['avg_cosine_similarity']:.4f}")
    print(f"Conflict Ratio: {result['conflict_ratio']:.2%}")
    
    # 测试3: PCGrad效果
    print("\n[Test 3] PCGrad Effect")
    print("-" * 40)
    
    grad1 = torch.randn(1000)
    grad2 = -grad1 + torch.randn(1000) * 0.1
    
    print(f"Before PCGrad:")
    print(f"  Grad1 Norm: {torch.norm(grad1).item():.4f}")
    print(f"  Grad2 Norm: {torch.norm(grad2).item():.4f}")
    print(f"  Cosine Sim: {detector.compute_cosine_similarity(grad1, grad2):.4f}")
    
    processed = detector.apply_pcgrad([grad1, grad2])
    
    print(f"After PCGrad:")
    print(f"  Grad1 Norm: {torch.norm(processed[0]).item():.4f}")
    print(f"  Grad2 Norm: {torch.norm(processed[1]).item():.4f}")
    print(f"  Cosine Sim: {detector.compute_cosine_similarity(processed[0], processed[1]):.4f}")
    
    # 测试4: 多任务 (3个任务)
    print("\n[Test 4] Three Tasks")
    print("-" * 40)
    
    detector3 = GradientConflictDetector(num_tasks=3)
    
    grads = [
        torch.randn(500),
        torch.randn(500),
        torch.randn(500),
    ]
    
    result = detector3(grads)
    print(f"Cosine Similarity Matrix:\n{result['cosine_similarity_matrix']}")
    print(f"Conflict Matrix:\n{result['conflict_matrix']}")
    print(f"Conflict Ratio: {result['conflict_ratio']:.2%}")
    
    # 测试5: 统计信息
    print("\n[Test 5] Statistics")
    print("-" * 40)
    
    stats = detector3.get_statistics()
    print(f"Statistics: {stats}")
    
    print("\n" + "=" * 70)
    print("All Tests Passed!")
    print("=" * 70)
