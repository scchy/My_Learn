# Agent Skills 练习题

**来源**: Why Use Skills & Skills与其他组件对比  
**说明**: 请独立完成以下练习，答案请参考讲义内容。

---

## 第一部分：判断题 (True/False)

1. **Agent Skills 采用渐进式加载机制，其中 SKILL.md 的元数据（name + description）始终占用上下文窗口。**

2. **MCP 的核心目的是教智能体如何处理数据，而 Skills 的核心目的是提供访问外部系统的权限。**

3. **根据渐进式披露机制，references/、scripts/、assets/ 目录中的文件在 Skill 触发时就会自动加载。**

4. **Subagents 可以拥有独立的上下文和工具权限，能够并行执行任务，结果返回给主智能体。**

5. **Skills 和 Prompts 的主要区别在于：Skills 是跨对话保持的，而 Prompts 仅在单个对话中有效。**

---
答： 1 √, 2 X, 3 X, 4 √, 5 √

## 第二部分：解释题

1. **解释什么是 Skills 的"渐进式披露机制"，并说明三个层级的加载时机和典型大小。**

2. **对比 Skills 和 Tools 在上下文管理方面的差异，并说明为什么这种差异很重要。**

3. **什么是 Skills 的"可组合性"？请用一个实际的例子说明多个 Skills 如何组合构建复杂工作流。**

4. **在使用 Skills 的架构中，通用智能体相比专用智能体有什么优势？结合认知演进图说明。**

5. **解释 Skills、MCP、Tools 和 Subagents 在 Agent 能力栈中的各自角色定位，并说明它们之间的关系。**

---
答：


## 第三部分：编程题

### 目标
实现一个 Skill 配置解析器，能够读取 Agent Skill 的 YAML frontmatter 并返回关键信息。

### 任务
编写一个 Python 函数 `parse_skill_metadata(skill_content: str) -> dict`，该函数：

1. 接收一个包含 YAML frontmatter 的 SKILL.md 内容字符串
2. 解析出以下字段：`name`, `description`
3. 返回一个字典，包含解析后的元数据

### 输入示例
```python
skill_content = '''---
name: generating-practice-questions
description: Generate educational practice questions from lecture notes to test student understanding.
---

# Practice Question Generator

Generate comprehensive practice questions...
'''
```

### 预期输出
```python
{
    "name": "generating-practice-questions",
    "description": "Generate educational practice questions from lecture notes to test student understanding."
}
```

### 要求
- 使用 Python 标准库实现
- 处理 YAML frontmatter 分隔符 `---`
- 如果字段不存在，返回空字符串

### 提示
- YAML frontmatter 位于文件开头，由 `---` 包围
- 可以考虑使用字符串分割方法
- 每行格式为 `key: value`

---

## 第四部分：应用案例

### 场景
你是一家电商公司的 AI 工程负责人，需要设计一个"智能客服助手"系统来处理客户咨询。系统需要：

1. **访问订单数据库** - 查询客户订单状态、物流信息
2. **遵循客服标准话术** - 按照公司规定的服务流程和语气回复客户
3. **处理退款申请** - 根据退款政策判断是否符合条件
4. **并行处理多个客户** - 同时服务多个客户不互相干扰

### 数据描述
假设有以下数据表结构：
```sql
-- orders 表
CREATE TABLE orders (
    order_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20),
    status ENUM('pending', 'shipped', 'delivered', 'cancelled'),
    amount DECIMAL(10,2),
    created_at TIMESTAMP
);

-- refunds 表
CREATE TABLE refunds (
    refund_id VARCHAR(20) PRIMARY KEY,
    order_id VARCHAR(20),
    reason TEXT,
    status ENUM('pending', 'approved', 'rejected'),
    requested_at TIMESTAMP
);
```

### 任务
基于 Skills、MCP、Tools、Subagents 的能力栈，设计系统架构：

1. **选择合适的组件** - 说明每个场景应该使用哪种组件（MCP/Skills/Tools/Subagents）
2. **说明理由** - 解释为什么选择该组件
3. **画出架构图** - 用文字描述组件之间的关系和数据流向

### 提示
- 考虑"数据访问 → 做事方法 → 具体执行"的流程
- 哪些部分需要专业知识？哪些需要外部连接？哪些需要并行处理？
- 参考讲义中"客户洞察分析器"的架构示例

### 约束
- 必须使用至少两种不同类型的组件
- 架构需要支持同时处理 3 个以上客户

---

**完成时间建议**: 60-90 分钟
