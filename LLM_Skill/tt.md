# Agent Skills with Anthropic 学习记录

> **课程来源**: [DeepLearning.AI - Agent Skills with Anthropic](https://www.deeplearning.ai/short-courses/agent-skills-with-anthropic/)  
> **讲师**: Elie Schoppik (Anthropic 技术教育负责人)  
> **合作方**: DeepLearning.AI × Anthropic  
> **课程时长**: 约2小时19分钟 (10个视频课程)  
> **难度级别**: 初级  

---

## 课程概述

本课程系统讲解了 **Agent Skills** —— 一种遵循开放标准的指令文件夹格式，可赋予AI Agent专业能力并自动化工作流。Skills是包含指令的文件夹，能够扩展Agent的能力，为其提供专业领域知识。开发者可以将特定工作流程打包成独立的技能模块，实现"一次构建，多处部署"。

### 核心学习目标

- ✅ 使用开放标准格式和最佳实践创建可复用的Skills
- ✅ 构建代码生成与审查、数据分析、研究等自定义Skills
- ✅ 将Skills与MCP和子代理结合，创建具有专业知识的强大Agent系统
- ✅ 在Claude.ai、Claude Code、Claude API和Claude Agent SDK中部署相同的Skills

---

## 课程大纲

| 序号 | 课程名称 | 时长 | 主要内容 |
|------|----------|------|----------|
| 1 | Introduction | 2分钟 | 课程介绍与概览 |
| 2 | Course Materials | 1分钟 | 课程资料说明 |
| 3 | Why Use Skills - Part I | 11分钟 | 为什么使用Skills（上） |
| 4 | Why Use Skills - Part II | 8分钟 | 为什么使用Skills（下） |
| 5 | Skills vs Tools, MCP, and Subagents | 7分钟 | Skills与Tools、MCP、子代理的对比 |
| 6 | Exploring Pre-Built Skills | 18分钟 | 探索预置Skills |
| 7 | Creating Custom Skills | 16分钟 | 创建自定义Skills |
| 8 | Skills with the Claude API | 17分钟 | 在Claude API中使用Skills |
| 9 | Skills with Claude Code | 24分钟 | 在Claude Code中使用Skills |
| 10 | Skills with the Claude Agent SDK | 20分钟 | 在Claude Agent SDK中使用Skills |
| 11 | Conclusion | 1分钟 | 课程总结 |

---

## 核心概念详解

### 1. 什么是Agent Skills？

**Skills** 是包含指令的文件夹，用于扩展Agent的能力，为其提供专业领域知识。与传统将所有指令一次性塞入Prompt的做法不同，Skills采用标准化的"技能文件夹"结构。

#### Skills的核心组成

```
your-skill-name/
├── SKILL.md              # 必需 - 主要技能文件（YAML frontmatter + Markdown）
├── scripts/              # 可选 - 可执行代码（Python、Bash等）
│   ├── process_data.py
│   └── validate.sh
├── references/           # 可选 - 参考文档
│   ├── api-guide.md
│   └── examples/
└── assets/               # 可选 - 模板等资源
    └── report-template.md
```

#### SKILL.md 格式规范

```markdown
---
name: my-skill                    # 技能名称（kebab-case）
description: 简要描述技能功能和触发时机  # 关键：说明"做什么"和"何时使用"
---

# My Skill

## Overview
解释此技能使AI能够做什么

## Usage
AI代理的分步使用说明...

## Available Resources
- `scripts/process_data.py` - 处理输入数据
- `assets/report_template.md` - 输出模板
```

**命名规范**：
- ✅ 使用kebab-case：`notion-project-setup`
- ❌ 不使用空格：`Notion Project Setup`
- ❌ 不使用下划线：`notion_project_setup`
- ❌ 不使用驼峰：`NotionProjectSetup`

---

### 2. 渐进式披露（Progressive Disclosure）

这是Skills最关键的技术特性，用于高效管理上下文窗口。

#### 三级加载机制

| 级别 | 加载时机 | 内容 | Token消耗 |
|------|----------|------|-----------|
| **Level 1 - Metadata** | 启动时始终加载 | YAML frontmatter（name + description） | ~100 tokens/技能 |
| **Level 2 - SKILL.md Body** | 技能被触发时 | 完整的指令和指导 | ~5k tokens |
| **Level 3 - Linked Files** | 需要时按需加载 | references/、scripts/、assets/中的文件 | 按需 |

#### 为什么这很重要？

- **节省上下文窗口**：安装20个Skills，启动仅占用约2000 tokens（metadata）
- **按需加载**：真正用到某个Skill时才加载详细内容
- **用完释放**：不持续占用上下文空间
- **Claude Code上下文窗口为200k tokens**，渐进式加载让你可以安装几十个Skills

> 💡 **监控技巧**：使用 `/context` 命令随时查看上下文使用情况

---

### 3. Skills vs 其他概念对比

#### Skills vs Tools

| 特性 | Skills | Tools |
|------|--------|-------|
| **定义** | 告诉AI "如何" 做事 | 为AI "提供" 做事的能力 |
| **内容** | 工作流程、最佳实践、领域知识 | 函数调用、API访问、代码执行 |
| **示例** | "如何按照规范组织Vuex模块" | 读取文件、执行代码、搜索网络 |
| **关系** | 知识层 | 执行层 |

#### Skills vs MCP (Model Context Protocol)

| 特性 | Skills | MCP |
|------|--------|-----|
| **类比** | 食谱（Recipes） | 专业厨房（Kitchen） |
| **功能** | 提供步骤指导和最佳实践 | 提供工具、食材、设备访问 |
| **关系** | 知识层，告诉AI如何使用工具 | 连接层，连接外部工具和API |
| **组合** | **Skills + MCP = 完整的Agent能力** | |

**厨房类比**：
- MCP提供专业的厨房：工具、食材、设备
- Skills提供食谱：如何创造有价值的东西的逐步指导
- 两者结合使用户能够完成复杂任务，无需专业知识

#### Skills vs Subagents

| 特性 | Skills | Subagents |
|------|--------|-----------|
| **定义** | 为当前对话中的Claude添加知识 | 创建一个新的、隔离的Claude实例 |
| **上下文** | 共享主对话上下文 | 独立的上下文窗口 |
| **用途** | 教Claude如何做特定任务 | 委派大型任务，并行处理 |
| **示例** | 教Claude如何写更好的computed属性 | 将"为整个项目添加单元测试"委派出去 |

---

### 4. Anthropic预置Skills

Anthropic官方提供了一系列开箱即用的Skills，托管在 [github.com/anthropic/skills](https://github.com/anthropic/skills)

#### 文档处理Skills（始终启用）

| 技能名称 | 功能描述 |
|----------|----------|
| **Excel Skill** | 创建、编辑、分析Excel电子表格 |
| **PowerPoint Skill** | 创建、编辑、分析PowerPoint演示文稿 |
| **Word Skill** | 创建、编辑Word文档 |
| **PDF Skill** | 处理PDF文件 |

#### 示例Skills（可手动启用）

| 技能名称 | 功能描述 |
|----------|----------|
| **skill-creator** | 帮助创建新的Skills |
| **webapp-building** | Web应用开发指导 |
| **data-analysis** | 数据分析工作流 |

#### 启用方式

在Claude Desktop中：
1. 点击用户头像 → Settings
2. 进入 Capabilities 面板
3. 在 Skills 部分点击 "Turn on"
4. 选择需要启用的Skills

---

## 实战案例

### 案例1：营销活动分析工作流

**场景**：使用Excel Skill分析营销活动数据

**工作流程**：
1. 用户上传营销数据
2. Claude自动加载Excel Skill
3. 创建包含分析结果的电子表格
4. 生成可视化图表

**效果**：无需手动解释如何处理Excel文件，Skill自动提供专业级数据分析能力。

---

### 案例2：代码审查工作流

**场景**：创建自定义代码审查Skill

**SKILL.md 结构**：
```markdown
---
name: code-review
description: 按照团队规范进行代码审查，检查代码质量、安全性和最佳实践
---

# Code Review Skill

## 审查清单
- [ ] 代码是否符合项目编码规范
- [ ] 是否存在安全风险
- [ ] 是否有足够的单元测试
- [ ] 性能是否优化

## 输出格式
使用标准模板生成审查报告...
```

**效果**：每次代码审查都遵循统一标准，确保质量一致性。

---

### 案例3：研究代理构建

**场景**：使用Claude Agent SDK构建研究代理

**工作流程**：
1. 创建研究Skill，包含信息收集、来源验证、报告生成等步骤
2. 结合Web Search MCP获取实时信息
3. 使用子代理并行处理多个研究任务
4. 生成综合研究报告

**效果**：自动化复杂的研究流程，提高信息收集和分析效率。

---

## 多平台部署

Skills遵循开放标准，可以跨平台部署：

### 1. Claude.ai / Claude Desktop

- 在Settings → Capabilities中启用Skills
- 支持代码执行和文件创建功能
- 预置Skills开箱即用

### 2. Claude Code

Skills自动从以下位置加载：
- **全局**：`~/.claude/skills/` (Windows: `C:\Users\<YourUsername>\.claude\skills\`)
- **项目级**：项目文件夹中的 `.claude/skills/`

**验证加载**：
```bash
# 查看已加载的Skills
列出所有可用的技能
```

### 3. Claude API

需要手动配置：
- 使用代码执行工具（Code Execution Tool）
- 使用Files API提供文件系统访问
- 提供bash执行能力

**注意**：Claude AI/Desktop中创建的Skills不会自动共享到Claude API或Claude Code。

### 4. Claude Agent SDK

- 在Agent配置中引用Skills文件夹
- 支持多Agent协作
- 可以结合MCP Servers使用

---

## 最佳实践

### 创建高质量Skills的10条原则

1. **描述要清晰**：必须包含"做什么"和"何时使用"
2. **保持SKILL.md简洁**：<500行，详细内容放到references/
3. **使用渐进式披露**：避免一次性加载过多内容
4. **一致的术语**：比丰富的表达更重要
5. **具体示例胜过抽象描述**：提供before/after对比
6. **可组合性设计**：Skill应该能与其他Skill协同工作
7. **可移植性考虑**：确保跨平台兼容
8. **定义成功标准**：如何验证Skill是否正常工作
9. **提供错误处理**：常见问题的解决方案
10. **版本控制友好**：使用Git管理Skills

### 安全注意事项

⚠️ **Skills能执行代码，务必小心**：

- ✅ 只用自己创建或官方的Skills
- ✅ 从GitHub/插件市场安装前先检查代码
- ❌ 不使用来路不明的Skills
- ❌ Skills中不放API密钥

**安全检查清单**：
- 检查SKILL.md中的所有指令
- 审查scripts/目录下的所有代码
- 确认没有硬编码的敏感信息

---

## 生态系统与资源

### 官方资源

- [Anthropic Skills GitHub](https://github.com/anthropic/skills) - 官方Skills仓库
- [Claude Skills Documentation](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) - 官方文档
- [The Complete Guide to Building Skills for Claude](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf) - 完整指南

### 社区资源

- [cognitive-toolworks](https://github.com/williamzujkowski/cognitive-toolworks) - 61个生产就绪Skills
- [awesome-claude-code-workflows](https://github.com/luandro/awesome-claude-code-workflows) - 工作流合集
- [MCP Market Skills](https://mcpmarket.com/tools/skills) - Skills市场

### 相关课程

- [Claude Code: A Highly Agentic Coding Assistant](https://www.deeplearning.ai/short-courses/claude-code-a-highly-agentic-coding-assistant/) - 深入Claude Code使用

---

## 学习心得与总结

### 核心收获

1. **Skills是AI工程化的重要一步**：将Prompt工程升级为结构化的知识管理
2. **渐进式披露是关键技术**：解决了上下文窗口限制的核心问题
3. **开放标准带来可移植性**：一次构建，到处使用
4. **与MCP结合威力倍增**：知识层 + 执行层 = 完整解决方案

### 适用场景

- ✅ 需要标准化、可重复的工作流程
- ✅ 团队协作需要统一标准
- ✅ 复杂的领域知识需要封装
- ✅ 需要跨项目复用的能力

### 不适用场景

- ❌ 一次性、临时性的任务
- ❌ 上下文窗口极小（<50k）的环境
- ❌ 不需要专业知识的基础任务

### 实践建议

1. **从简单开始**：先创建1-2个简单的Skills熟悉流程
2. **迭代优化**：根据实际使用情况不断调整
3. **团队共享**：将高质量的Skills分享给团队成员
4. **持续学习**：关注社区新Skills和最佳实践

---

## 参考链接

- 课程主页：https://www.deeplearning.ai/short-courses/agent-skills-with-anthropic/
- 课程材料：https://learn.deeplearning.ai/courses/agent-skills-with-anthropic/lesson/53z7ulp0/course-materials
- Anthropic官方文档：https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview
- GitHub Skills仓库：https://github.com/anthropic/skills

---

> 📅 **学习日期**: 2026年2月  
> 📝 **文档版本**: v1.0  
> ✍️ **整理者**: AI Assistant
