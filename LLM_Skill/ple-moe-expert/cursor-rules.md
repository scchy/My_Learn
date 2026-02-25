# PLE-MOE Expert - Cursor 专用规则

## 触发条件
用户提及以下关键词时激活：
- MOE, Sparse-MOE, shareExpert, PLE
- 多任务学习, 梯度冲突, expert routing
- PK成功率/取消率联合建模

## 执行步骤

### Step 1: 参数确认
必须向用户确认或自动推断：
```python
{
    "input_dim": int,           # 输入特征维度
    "num_tasks": int,           # 任务数（如2：PK+取消率）
    "num_experts_per_task": int, # 每个任务的expert数（默认4）
    "use_share_expert": bool,   # 是否共享expert（默认True）
    "expert_dim": int,          # expert输出维度（默认64）
    "routing_dim": int          # routing网络维度（默认64）
}
```

### Step 2: 生成PLE嵌入层


### Step 3: 生成Sparse-MOE层（带维度注释）

```python
class SparseMOELayer(nn.Module):
    def forward(self, x, task_id=None, temperature=1.0):
        # x: (batch_size, input_dim)
        # gates: (batch_size, num_experts)
        gates = F.softmax(self.routing(x) / temperature, dim=-1)
        
        # expert_inputs: (batch_size, num_experts, input_dim)
        expert_inputs = x.unsqueeze(1).expand(-1, self.num_experts, -1)
        
        # expert_outputs: (batch_size, num_experts, expert_dim)
        expert_outputs = torch.stack([
            expert(expert_inputs[:, i, :]) 
            for i, expert in enumerate(self.experts)
        ], dim=1)
        
        # output: (batch_size, expert_dim)
        # b: batch, n: num_experts, d: expert_dim
        output = torch.einsum('bnd,bn->bd', expert_outputs, gates)
        
        # 必须返回auxiliary_loss用于监控expert利用率
        aux_loss = self.compute_load_balancing_loss(gates)
        return output, aux_loss
```

### Step 4: 强制附加代码

必须同时生成以下代码块：

A. Temperature退火
```python
def get_temperature(self, epoch, init_temp=1.0, min_temp=0.1, decay_rate=0.95):
    """必须实现，防止routing collapse"""
    return max(min_temp, init_temp * (decay_rate ** epoch))
```


B. Expert利用率监控
```python
def compute_load_balancing_loss(self, gates, importance=None):
    """计算auxiliary loss，确保expert负载均衡"""
    # gates: (batch_size, num_experts)
    router_prob = gates.mean(dim=0)  # (num_experts,)
    if importance is None:
        importance = gates.sum(dim=0)  # (num_experts,)
    
    # 系数 of variation，目标是最小化
    balance_loss = self.num_experts * (router_prob * importance).sum()
    return balance_loss
```

C. 梯度冲突检测（调用脚本）

```python
# 引用 scripts/gradient_conflict_check.py
from gradient_conflict_check import PCGrad, compute_conflict_score

# 训练循环中必须包含
pc_grad = PCGrad(optimizer)
conflict_score = pc_grad.step(losses)  # losses: list of task losses
print(f"Gradient conflict score: {conflict_score:.4f}")
```


## 代码规范
- 所有函数必须类型注解（`def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:`）
- 随机种子固定：torch.manual_seed(42)在初始化中
- einsum必须带维度注释（格式：# b: batch, n: num_experts, d: dim）
- 禁止省略expert利用率监控
- 禁止没有temperature退火的routing


## 禁止事项

- 禁止生成没有置信区间的"显著提升"结论（这是实验分析Skill的范畴，但代码中不得硬编码性能声明）
- 禁止在routing中使用固定temperature
- 禁止省略task-specific和shared expert的区分（当`use_share_expert=True`时）


## 可用资源路径

- 模板：.cursor-skills/ple-moe-expert/assets/moe-template.py
- 脚本：.cursor-skills/ple-moe-expert/scripts/
- 文档：.cursor-skills/ple-moe-expert/references/


## 输出模板

生成代码后必须按以下结构组织

```markdown 
## 1. 模型结构图
[ASCII图或层级注释]

## 2. 核心代码
[包含PLE+MOE的完整nn.Module]

## 3. 使用示例
[含输入输出shape的示例]

## 4. 训练注意事项
- temperature退火策略
- auxiliary loss权重设置（建议0.01-0.1）
- 梯度冲突检测接入点
```




