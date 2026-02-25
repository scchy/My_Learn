"""
PLE (Piecewise Linear Embeddings) Reference Implementation
==========================================================
参考: Yandex Research - "On Embeddings for Numerical Features in Tabular Deep Learning"
        NeurIPS 2022
        https://github.com/yandex-research/rtdl-num-embeddings

本文件包含PLE的官方实现参考，用于理解算法原理和实现细节。
"""

import math
import warnings
from typing import Any, List, Literal, Optional, Union

try:
    import sklearn.tree as sklearn_tree
except ImportError:
    sklearn_tree = None

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter


def _check_bins(bins: list[Tensor]) -> None:
    """验证分箱边界的有效性"""
    if not bins:
        raise ValueError('The list of bins must not be empty')

    for i, feature_bins in enumerate(bins):
        if not isinstance(feature_bins, Tensor):
            raise ValueError(
                f'bins must be a list of PyTorch tensors. '
                f'However, for i={i}: type={type(bins[i])}'
            )
        if feature_bins.ndim != 1:
            raise ValueError(
                f'Each item of the bin list must have exactly one dimension.'
                f' However, for i={i}: ndim={bins[i].ndim}'
            )
        if len(feature_bins) < 2:
            raise ValueError(
                f'All features must have at least two bin edges.'
                f' However, for i={i}: len={len(bins[i])}'
            )
        if not feature_bins.isfinite().all():
            raise ValueError(
                f'Bin edges must not contain nan/inf/-inf.'
                f' However, this is not true for the {i}-th feature'
            )
        if (feature_bins[:-1] >= feature_bins[1:]).any():
            raise ValueError(
                f'Bin edges must be sorted.'
                f' However, the for the {i}-th feature, the bin edges are not sorted')


def compute_bins(
    X: torch.Tensor,
    n_bins: int = 48,
    *,
    tree_kwargs: Optional[dict[str, Any]] = None,
    y: Optional[Tensor] = None,
    regression: Optional[bool] = None,
    verbose: bool = False,
) -> list[Tensor]:
    """
    计算PLE的分箱边界

    支持两种分箱策略:
    1. 分位数分箱 (Quantile-based): 基于数据分布均匀分箱
    2. 决策树分箱 (Tree-based): 基于目标变量学习最优分箱边界

    Args:
        X: 训练特征，形状 (n_samples, n_features)
        n_bins: 分箱数量
        tree_kwargs: 决策树参数（用于树分箱）
        y: 目标变量（用于树分箱）
        regression: 是否为回归任务（用于树分箱）
        verbose: 是否显示进度

    Returns:
        每个特征的分箱边界列表

    示例:
    >>> X_train = torch.randn(10000, 5)
    >>>
    >>> # 分位数分箱
    >>> bins_quantile = compute_bins(X_train, n_bins=48)
    >>>
    >>> # 决策树分箱
    >>> y_train = torch.randn(10000)
    >>> bins_tree = compute_bins(
    ...     X_train,
    ...     y=y_train,
    ...     regression=True,
    ...     tree_kwargs={'min_samples_leaf': 64},
    ... )
    """
    if not isinstance(X, Tensor):
        raise ValueError(
            f'X must be a PyTorch tensor, however: type={type(X)}')
    if X.ndim != 2:
        raise ValueError(
            f'X must have exactly two dimensions, however: ndim={X.ndim}')
    if n_bins <= 1 or n_bins >= len(X):
        raise ValueError(
            f'n_bins must be more than 1, but less than len(X), '
            f'however: n_bins={n_bins}, len(X)={len(X)}'
        )

    # 分位数分箱
    if tree_kwargs is None:
        # 计算分位数
        bins = [
            q.unique()
            for q in torch.quantile(
                X,
                torch.linspace(0.0, 1.0, n_bins + 1).to(X),
                dim=0
            ).T
        ]
        _check_bins(bins)
        return bins

    # 决策树分箱
    else:
        if sklearn_tree is None:
            raise RuntimeError('scikit-learn is required for tree-based bins')
        if y is None or regression is None:
            raise ValueError(
                'y and regression must be provided for tree-based bins')

        X_numpy = X.cpu().numpy()
        y_numpy = y.cpu().numpy()

        bins = []
        for column in X_numpy.T:
            feature_bin_edges = [float(column.min()), float(column.max())]

            # 训练决策树
            tree_cls = (
                sklearn_tree.DecisionTreeRegressor
                if regression
                else sklearn_tree.DecisionTreeClassifier
            )
            tree = tree_cls(max_leaf_nodes=n_bins, **tree_kwargs)
            tree.fit(column.reshape(-1, 1), y_numpy)

            # 提取分割点
            tree_ = tree.tree_
            for node_id in range(tree_.node_count):
                # 只考虑分裂节点
                if tree_.children_left[node_id] != tree_.children_right[node_id]:
                    feature_bin_edges.append(float(tree_.threshold[node_id]))

            bins.append(torch.as_tensor(feature_bin_edges).unique())

        _check_bins(bins)
        return [x.to(device=X.device, dtype=X.dtype) for x in bins]


class _PiecewiseLinearEncodingImpl(nn.Module):
    """
    PLE编码实现（内部类）

    将连续数值特征转换为分段线性编码

    数学原理:
    对于输入x和分箱边界[b_0, b_1, ..., b_n]，PLE编码为:
    x_ple = [1, ..., 1, (x - b_i)/(b_{i+1} - b_i), 0, ..., 0]

    其中只有包含x的那个bin位置有非零值

    维度变换:
    - 输入: (batch_size, n_features)
    - 输出: (batch_size, n_features, max_n_bins)
    """

    weight: Tensor
    bias: Tensor
    single_bin_mask: Optional[Tensor]
    mask: Optional[Tensor]

    def __init__(self, bins: list[Tensor]) -> None:
        super().__init__()
        assert len(bins) > 0

        n_features = len(bins)
        n_bins = [len(x) - 1 for x in bins]
        max_n_bins = max(n_bins)

        # 线性变换参数
        self.register_buffer('weight', torch.zeros(n_features, max_n_bins))
        self.register_buffer('bias', torch.zeros(n_features, max_n_bins))

        # 单bin特征标记
        single_bin_mask = torch.tensor(n_bins) == 1
        self.register_buffer(
            'single_bin_mask',
            single_bin_mask if single_bin_mask.any() else None
        )

        # 有效位置mask
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
            bin_width = bin_edges.diff()
            w = 1.0 / bin_width
            b = -bin_edges[:-1] / bin_width

            self.weight[i, -1] = w[-1]
            self.bias[i, -1] = b[-1]
            self.weight[i, :n_bins[i] - 1] = w[:-1]
            self.bias[i, :n_bins[i] - 1] = b[:-1]

    def get_max_n_bins(self) -> int:
        return self.weight.shape[-1]

    def forward(self, x: Tensor) -> Tensor:
        # 线性变换
        x = torch.addcmul(self.bias, self.weight, x[..., None])

        if x.shape[-1] > 1:
            # Clamp激活
            x = torch.cat([
                x[..., :1].clamp_max(1.0),
                x[..., 1:-1].clamp(0.0, 1.0),
                (
                    x[..., -1:].clamp_min(0.0)
                    if self.single_bin_mask is None
                    else torch.where(
                        self.single_bin_mask[..., None],
                        x[..., -1:],
                        x[..., -1:].clamp_min(0.0),
                    )
                ),
            ], dim=-1)

        return x


class PiecewiseLinearEmbeddings(nn.Module):
    """
    PLE数值嵌入层

    维度变换:
    - 输入: (batch_size, n_features)
    - 输出: (batch_size, n_features, d_embedding)

    示例:
    >>> bins = [torch.linspace(0, 1, 49) for _ in range(10)]
    >>> ple = PiecewiseLinearEmbeddings(bins, d_embedding=64, activation=True)
    >>> x = torch.randn(32, 10)
    >>> out = ple(x)  # (32, 10, 64)
    """

    def __init__(
        self,
        bins: list[Tensor],
        d_embedding: int,
        *,
        activation: bool,
        version: Literal[None, 'A', 'B'] = None,
    ) -> None:
        """
        Args:
            bins: 分箱边界列表
            d_embedding: 嵌入维度
            activation: 是否使用ReLU激活
            version: 'A'或'B'，B版本包含额外的线性层
        """
        if d_embedding <= 0:
            raise ValueError(f'd_embedding must be positive: {d_embedding}')
        _check_bins(bins)

        if version is None:
            warnings.warn(
                'version not provided, using "A" for backward compatibility')
            version = 'A'

        super().__init__()
        n_features = len(bins)
        is_version_B = version == 'B'

        # Version B: 额外的线性层
        if is_version_B:
            self.linear0 = nn.Linear(n_features, d_embedding)
        else:
            self.register_buffer('linear0', None)

        self.impl = _PiecewiseLinearEncodingImpl(bins)

        # N个独立的线性层
        self.linear = _NLinear(
            len(bins),
            self.impl.get_max_n_bins(),
            d_embedding,
            bias=not is_version_B,
        )

        if is_version_B:
            nn.init.zeros_(self.linear.weight)

        self.activation = nn.ReLU() if activation else None

    def get_output_shape(self) -> torch.Size:
        n_features = self.linear.weight.shape[0]
        d_embedding = self.linear.weight.shape[2]
        return torch.Size((n_features, d_embedding))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError('Only 2D inputs are supported')

        x_linear = None if self.linear0 is None else self.linear0(x)
        if x_linear is not None:
            x_linear = x_linear.unsqueeze(1)

        x_ple = self.impl(x)
        x_ple = self.linear(x_ple)

        if self.activation is not None:
            x_ple = self.activation(x_ple)

        return x_ple if x_linear is None else x_linear + x_ple


class _NLinear(nn.Module):
    """N个独立的线性层"""

    def __init__(
        self,
        n: int,
        in_features: int,
        out_features: int,
        bias: bool = True
    ) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(n, in_features, out_features))
        self.bias = Parameter(torch.empty(n, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        d_in_rsqrt = self.weight.shape[-2] ** -0.5
        nn.init.uniform_(self.weight, -d_in_rsqrt, d_in_rsqrt)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -d_in_rsqrt, d_in_rsqrt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0, 1)
        x = x @ self.weight
        x = x.transpose(0, 1)
        if self.bias is not None:
            x = x + self.bias
        return x


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PLE Reference Implementation Demo")
    print("=" * 60)

    # 生成示例数据
    n_samples = 1000
    n_features = 5
    X_train = torch.randn(n_samples, n_features)

    # 1. 分位数分箱
    print("\n1. Quantile-based Bins")
    print("-" * 40)

    bins_quantile = compute_bins(X_train, n_bins=16)
    print(f"Number of features: {len(bins_quantile)}")
    for i, b in enumerate(bins_quantile):
        print(f"  Feature {i}: {len(b)-1} bins")

    # 2. PLE嵌入
    print("\n2. PLE Embeddings")
    print("-" * 40)

    ple = PiecewiseLinearEmbeddings(
        bins=bins_quantile,
        d_embedding=32,
        activation=True,
    )

    X_test = torch.randn(100, n_features)
    X_embedded = ple(X_test)

    print(f"Input shape: {X_test.shape}")
    print(f"Output shape: {X_embedded.shape}")
    print(f"Output shape (per feature): {ple.get_output_shape()}")

    # 3. 决策树分箱（需要sklearn和目标变量）
    print("\n3. Tree-based Bins")
    print("-" * 40)

    try:
        y_train = torch.randn(n_samples)
        bins_tree = compute_bins(
            X_train,
            n_bins=16,
            y=y_train,
            regression=True,
            tree_kwargs={'min_samples_leaf': 50},
        )
        print(f"Tree-based bins computed successfully")
        for i, b in enumerate(bins_tree):
            print(f"  Feature {i}: {len(b)-1} bins")
    except Exception as e:
        print(f"Tree-based bins failed: {e}")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
