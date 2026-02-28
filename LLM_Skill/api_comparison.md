# Kimi API 使用方式对比

## 三种调用方式的区别

| 特性 | OpenAI API | kimi-cli (subprocess) | KimiClient (封装类) |
|------|-----------|----------------------|-------------------|
| **依赖** | `openai` 库 | `kimi` CLI 已安装 | `kimi` CLI 已安装 |
| **调用方式** | HTTP API | 本地 subprocess | 本地 subprocess |
| **Skill 支持** | ❌ 不支持 | ✅ 支持 | ✅ 支持 |
| **工具调用** | ✅ API 内置 | ✅ CLI 自动执行 | ✅ CLI 自动执行 |
| **流式输出** | ✅ 支持 | ✅ 支持 | ✅ 支持 |
| **速度** | ⭐⭐⭐ 快 | ⭐⭐ 中等 | ⭐⭐ 中等 |
| **使用场景** | 生产环境 | 本地自动化 | 本地开发 |

---

## 1. OpenAI API 方式（无 Skill）

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="https://api.moonshot.cn/v1"
)

# 标准对话
response = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[{"role": "user", "content": "你好"}]
)
print(response.choices[0].message.content)

# 流式输出
for chunk in client.chat.completions.create(
    model="kimi-k2.5",
    messages=[{"role": "user", "content": "你好"}],
    stream=True
):
    print(chunk.choices[0].delta.content or "", end="")
```

**⚠️ 限制：无法使用 Skill、无法自动执行本地工具**

---

## 2. kimi-cli 方式（有 Skill）

```python
import subprocess

def ask_kimi_with_skills(prompt, skills_dir=None):
    cmd = ['kimi', '--print', '--yes']
    
    if skills_dir:
        cmd.extend(['--skills-dir', skills_dir])
    
    cmd.extend(['--command', prompt])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

# 使用 Skill 分析代码
response = ask_kimi_with_skills(
    "分析这个项目的代码质量",
    skills_dir="~/.config/agents/skills/code-analyzer"
)
print(response)
```

**✅ 优势：可以使用 Skill、自动执行本地命令、读写文件**

---

## 3. KimiClient 封装类（推荐）

```python
from kimi_client import KimiClient

client = KimiClient(skills_dir="~/.config/agents/skills")

# 使用 Skill 的对话
response = client.messages.create(
    messages=[{"role": "user", "content": "分析项目结构"}]
)
print(response.content)
```

---

## 为什么 API 无法使用 Skill？

```
┌─────────────────────────────────────────────────────────────┐
│                         Skill 架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐      ┌──────────────┐      ┌────────────┐ │
│  │  SKILL.md   │─────→│  kimi-cli    │─────→│  LLM API   │ │
│  │  (本地文件)  │      │  (本地进程)   │      │  (远程服务) │ │
│  └─────────────┘      └──────────────┘      └────────────┘ │
│                              │                              │
│                         ┌────┴────┐                         │
│                         ↓         ↓                         │
│                    ┌────────┐ ┌────────┐                    │
│                    │本地工具 │ │本地文件 │                    │
│                    └────────┘ └────────┘                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Skill 的核心功能：**
1. **本地工具执行** - 运行 shell 命令、Python 脚本
2. **本地文件访问** - 读写项目文件
3. **本地知识库** - 读取 SKILL.md 中的领域知识

这些都需要**本地运行时环境**，而 HTTP API 是远程服务，无法访问你的本地文件系统。

---

## 选择建议

| 场景 | 推荐方式 |
|------|---------|
| 生产服务部署 | OpenAI API |
| 需要 Skill 增强 | kimi-cli 或 KimiClient |
| 自动化脚本 | KimiClient |
| Jupyter Notebook | KimiClient |
| 批量处理文件 | kimi-cli + Skill |
