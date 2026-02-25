

# Cursor With Skill


用 Cursor Project Rules + 自定义脚本模拟 Skill 结构



## 一、项目目录结构

推荐放在 `~/.cursor/skills/` 或项目内 `.cursor-skills/`


```bash
project-root/
├── .cursorrules                    # Cursor 主配置文件
└── .cursor-skills/                 # Skill 目录（可 Git 管理）
    ├── ple-moe-expert/
    │   ├── SKILL.md                # 标准格式（给 Kimi 预留）
    │   ├── cursor-rules.md         # Cursor 优化格式
    │   ├── scripts/
    │   │   ├── generate_moe_layer.py
    │   │   └── gradient_conflict_check.py
    │   ├── references/
    │   │   ├── ple-embedding-guide.md
    │   │   └── moe-routing-best-practices.md
    │   └── assets/
    │       └── moe-template.py
    │
    └── experiment-analysis/
        ├── SKILL.md
        ├── cursor-rules.md
        ├── scripts/
        │   ├── ab_test_calculator.py
        │   └── confidence_interval.py
        └── assets/
            └── report-template.md
```

核心markdown格式差异总结

| 元素       | SKILL.md（标准）                | cursor-rules.md（Cursor） | .cursorrules（主配置） |
| -------- | --------------------------- | ----------------------- | ----------------- |
| **头部**   | YAML frontmatter            | 无 frontmatter，纯标题       | 无 frontmatter     |
| **触发条件** | `description` 隐含            | 显式 `## 触发条件`            | 显式列出关键词映射         |
| **执行步骤** | `## Usage` 分步说明             | `## 执行步骤` 更详细           | 简化为路径引用           |
| **资源引用** | `## Available Resources` 列表 | `## 可用资源路径` 绝对路径        | 仅声明 Skill 名称和路径   |
| **代码示例** | 可包含                         | 必须包含具体规范                | 不包含，引用 Skill      |


### 1.1 skill.md 格式
> 标准skill格式

```markdown
---
name: ple-moe-expert
description: 生成 PLE 数值嵌入和 Sparse-MOE 结构代码，自动添加梯度冲突检测
---

# PLE-MOE Expert

## Overview
生成多任务学习场景下的 PLE 数值嵌入层和 Sparse-MOE 结构，包含：
- torch.einsum 维度注释
- 梯度冲突检测（PCGrad/GradNorm）
- Expert 利用率监控

## Usage
1. 确认参数：input_dim, num_tasks, num_experts, shareExpert 开关
2. 加载模板 `assets/moe-template.py` 生成基础结构
3. 调用 `scripts/generate_moe_layer.py` 生成动态部分
4. 强制附加 `scripts/gradient_conflict_check.py`
5. 输出：结构图 → 核心代码 → 使用示例

## Available Resources
- `scripts/generate_moe_layer.py`
- `scripts/gradient_conflict_check.py`
- `assets/moe-template.py`
- `references/ple-embedding-guide.md`
```

**命名规范**：
- ✅ 使用kebab-case：`notion-project-setup`
- ❌ 不使用空格：`Notion Project Setup`
- ❌ 不使用下划线：`notion_project_setup`
- ❌ 不使用驼峰：`NotionProjectSetup`

---

### 1.2 cursor-rules.md 格式

```markdown 
# PLE-MOE Expert - Cursor 专用规则

## 触发条件
用户提及：MOE, PLE, Sparse-MOE, shareExpert, 梯度冲突, expert routing

## 执行步骤
1. 检查必要参数：input_dim, num_tasks, num_experts, use_share_expert
2. 读取模板文件 `.cursor-skills/ple-moe-expert/assets/moe-template.py`
3. 生成代码时必须包含：
   - PLE 数值嵌入层（参考 2025 订单量预测项目）
   - torch.einsum 维度注释（格式：# b: batch, n: num_experts, d: dim）
   - temperature 退火的 routing 机制
   - expert 利用率监控（load balancing loss）
4. 自动附加梯度冲突检测代码（PCGrad 实现）
5. 输出顺序：结构图 → 核心代码 → 使用示例

## 代码规范
- 所有函数必须类型注解
- 随机种子固定：torch.manual_seed(42)
- 禁止生成没有置信区间的"显著提升"结论

## 可用资源路径
- 模板：`.cursor-skills/ple-moe-expert/assets/moe-template.py`
- 脚本：`.cursor-skills/ple-moe-expert/scripts/`
- 文档：`.cursor-skills/ple-moe-expert/references/`
```

### 1.3  .cursorrules（项目根目录主配置）

```markdown
# 算法项目 Cursor 配置

## 全局规范
- Python 代码必须类型注解
- 模型训练必须固定随机种子
- 指标结论必须含 p-value 和置信区间

## Skill 系统
当用户输入匹配以下关键词时，激活对应 Skill：

### @ple-moe-expert
触发词：MOE, PLE, Sparse-MOE, shareExpert, 梯度冲突, expert routing
规则来源：`.cursor-skills/ple-moe-expert/cursor-rules.md`

### @experiment-analysis  
触发词：AB测试, 显著性检验, 置信区间, p-value, 实验报告
规则来源：`.cursor-skills/experiment-analysis/cursor-rules.md`

## 命名约定
- 特征变量：feat_{业务含义}_{加工方式}
- 实验对照组：control_v{日期}
- 指标命名：metric_{指标名}_{时间窗口}
```


## 二、快速启动模板

### 2.1 创建目录结构

```bash
mkdir -p .cursor-skills/{ple-moe-expert,experiment-analysis}/{scripts,references,assets}
touch .cursor-skills/ple-moe-expert/{SKILL.md,cursor-rules.md}
touch .cursorrules
```


### 2.2 填充内容
复制上述 cursor-rules.md 模板到对应 Skill  
复制上述 .cursorrules 模板到项目根目录  

### 2.3验证生效
在 Cursor 中输入：  
```text
帮我写一个 shareExpert-MOE 层，输入维度 64，3 个任务
```
检查：
- 是否自动带 einsum 维度注释
- 是否包含梯度冲突检测代码
- 是否有 expert 利用率监控



