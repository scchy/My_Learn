# Pi Agent 最小核心 Python 重构执行计划

> **目标**：1 周（每天 1-2 小时）手敲 ~700 行 Python，覆盖 Pi Agent 全部核心机制  
> **交付**：可运行的 Agent CLI，支持交互/非交互双模式  
> **时间**：2026-07-21 起，共 7 天

---

## 一、为什么重构

### 直接收益

| 维度 | 读源码/文档 | 手敲重构 |
|:---|:---|:---|
| Agent Loop 边界条件 | 知道"有重试" | 亲手处理流式 chunk 中断、工具超时、malformed JSON |
| 上下文压缩策略 | 知道"三层压缩" | 亲手实现 Pi-style Compaction：从后往前找合法切割点、Turn 边界保护、双摘要、增量更新、文件操作追踪 |
| 工具安全边界 | 知道"要确认" | 亲手设计 `dangerous` 标记、用户确认流程、沙箱逃逸防护 |
| 扩展系统 | 知道"可插件化" | 亲手用 `importlib` 实现动态加载，理解热更新与状态隔离 |

### 对当前项目的帮助

1. **HyperTuneAgent / AutoTuneAgent**：Agent Loop（推理→执行→观察→再推理）与调参的搜索-评估-反馈循环**同构**，手敲后对状态管理、并发策略、早停机制的体感直接迁移
2. **Agent 评估体系**：从"评估者"变成"被评估者"，对"末段停滞"、"可靠性"等指标的权重分配有直觉
3. **工业项目（数据集构建）**："采集—筛选—标注—审核—回流"与 Agent 数据流设计**完全同构**

---

## 二、技术栈

| 用途 | 库 | 说明 |
|:---|:---|:---|
| 异步 HTTP | `httpx` | 流式 SSE 请求、重试、超时 |
| TUI 渲染 | `rich` | Live 面板、Markdown、Confirm/Prompt |
| CLI 解析 | `typer` | 命令、参数、子命令 |
| 配置管理 | `pyyaml` | `~/.pi-agent/config.yaml` |
| 持久化 | `jsonlines` | 会话树存储 |

```bash
pip install httpx rich typer pyyaml jsonlines
```

---

## 三、参考仓库

| 仓库 | 地址 | 作用 | 对应 Day |
|:---|:---|:---|:---:|
| **earendil-works/pi** | [github.com/earendil-works/pi](https://github.com/earendil-works/pi) | Pi Agent 官方主仓库，全部核心包源码 | 全部 |
| **cellinlab/how-pi-agent-works** | [github.com/cellinlab/how-pi-agent-works](https://github.com/cellinlab/how-pi-agent-works) | 中文教学版，含 4 个渐进式 Demo + React/Node 教学版 | 全部 |
| **pi.dev** | [pi.dev](https://pi.dev/) | 官方文档与演示站点 | 全部 |
| **dimetron/pi-go** | [github.com/dimetron/pi-go](https://github.com/dimetron/pi-go) | Go 语言重新实现，含沙箱、LSP、子 Agent | Day 3 |
| **joelreymont/pz** | [github.com/joelreymont/pz](https://github.com/joelreymont/pz) | Zig 语言安全优先实现，单二进制 1.7MB | Day 2 |
| **he-yufeng/CoreCoder** | [github.com/he-yufeng/CoreCoder](https://github.com/he-yufeng/CoreCoder) | Python 教学版，1,081 行引擎 + 8 篇双语源码解读 | 全部 |

---

## 四、目录结构

```
pi-agent-mini/
├── pi_agent/
│   ├── __init__.py          # 包入口
│   ├── llm.py               # Day 1: 统一 LLM 客户端 (~80行)
│   ├── tools.py             # Day 2: 工具注册与执行 (~100行)
│   ├── agent.py             # Day 3: Agent Loop 核心 (~250行)
│   ├── context.py           # Day 4: 上下文压缩 (~80行)
│   ├── session.py           # Day 5: 会话持久化 (~180行)
│   └── cli.py               # Day 6: CLI 入口 (~380行)
├── tests/
│   ├── test_llm.py          # Day 7: LLM 模块测试
│   ├── test_tools.py        # Day 7: 工具模块测试
│   ├── test_agent.py        # Day 7: Agent 循环测试
│   └── test_context.py      # Day 7: 上下文压缩测试
├── config.yaml              # 用户配置模板
└── README.md
```

**总代码量**：~1,400 行核心代码 + ~100 行测试代码

> **参考实现**：`/home/scc/sccWork/myGitHub/My_Learn/PI_Agent/pi_agent_reference/` 下的 7 个文件
> 是修正后的完整实现，可直接运行。修正要点见附录 [十、关键修正记录]。

---

## 五、Day-by-Day 执行计划

### Day 1：统一 LLM 客户端
> Done 2026-07-22-01:06
> 学会了sse流式数据捕捉与中断，组合返回

**文件**：`pi_agent/llm.py`  
**代码量**：~80 行  
**核心机制**：流式 SSE 解析、指数退避重试、Token 估算、LLM 自身摘要

**关键设计**：
- `Message` dataclass：统一消息格式（system/user/assistant/tool）
- `LLMClient.chat()`：返回 `AsyncIterator[str]`，每个 chunk 是增量文本
- 重试策略：`max_retries=3`，退避间隔 `2^attempt` 秒
- Token 估算：保守按字符数计算（中文 1 字 ≈ 1 token）
- `summarize()`：调用 LLM 自身做摘要，用于上下文压缩

**参考源码**：
- Pi 官方 `pi-ai` 包：[packages/ai/src](https://github.com/earendil-works/pi/tree/main/packages/ai/src)（统一 LLM API）
- CoreCoder LLM 客户端：[corecoder/llm.py](https://github.com/he-yufeng/CoreCoder/blob/main/corecoder/llm.py)
- 教学版 Demo：[examples/demos/01](https://github.com/cellinlab/how-pi-agent-works/tree/main/examples/demos)

**验收标准**：
- [x] 能连接任意 OpenAI 兼容接口（Kimi、DeepSeek、本地 vLLM）
- [x] 流式输出不丢字、不卡顿
- [x] 网络异常时自动重试 3 次（指数退避）
- [x] Token 估算误差在 ±20% 以内
- [x] 摘要功能可用

**预期踩坑**：SSE 流式解析漏判 `data: [DONE]` 和空行

---

### Day 2：工具系统
> Done 2026-07-22-20:03
> 学会了一些最为基本的工具
> 
**文件**：`pi_agent/tools.py`  
**代码量**：~100 行  
**核心机制**：装饰器注册、Schema 自动生成、危险操作确认、超时截断

**关键设计**：
- `@tool` 装饰器：自动提取函数签名生成 JSON Schema
- `ToolRegistry`：全局注册中心，提供 `to_openai_schema()` 转换
- 危险标记：`@tool(dangerous=True)` 执行前弹出 `rich.Confirm` 确认
- 超时控制：`asyncio.wait_for(asyncio.to_thread(...), timeout=30.0)`
- 输出截断：工具返回超过 4000 字符自动截断

**7 个基础工具**：

| 工具 | 功能 | 危险 |
|:---|:---|:---:|
| `read` | 读取文件内容（支持 offset/limit）| ✅ |
| `bash` | 执行 shell 命令 | ✅ |
| `edit` | 文件内字符串替换 | ✅ |
| `write` | 写入文件（自动创建目录）| ✅ |
| `grep` | 文件内容搜索 | ✅ |
| `find` | 文件查找（通配符）| ✅ |
| `ls` | 目录列表 | ✅ |

**参考源码**：
- Pi 官方 `pi-coding-agent` 工具集：[packages/coding-agent/src/tools](https://github.com/earendil-works/pi/tree/main/packages/coding-agent/src)
- CoreCoder 工具系统：[corecoder/tools.py](https://github.com/he-yufeng/CoreCoder/blob/main/corecoder/tools.py)
- Pi 工具 Schema 定义：[packages/agent-core/src](https://github.com/earendil-works/pi/tree/main/packages/agent-core/src)

**验收标准**：
- [x] 7 个工具全部可用，Schema 自动生成
- [x] `bash` 等危险工具执行前必须确认
- [x] 工具输出超长自动截断（4000 字符）
- [x] 工具超时 30 秒自动终止
- [x] 错误处理：文件不存在、字符串未找到等

**预期踩坑**：`inspect.signature` 对复杂类型处理不好，先只用 `str` 类型

---

### Day 3：Agent Loop 核心
> Done 2026-07-23-15:45
> 学会了ReAct循环 工具执行
> 
**文件**：`pi_agent/agent.py`  
**代码量**：~150 行  
**核心机制**：ReAct 循环、并行工具执行、流式实时渲染、steering 中断

**关键设计**：
- `AgentConfig`：max_turns、context_limit、temperature、system_prompt
- `Agent.run()`：主循环，每轮自动触发上下文压缩检查
- `_stream_response()`：流式获取 LLM 响应，实时 `rich.Live` 渲染
- `_execute_tools()`：`asyncio.gather()` 并行执行多个工具调用
- `steer()`：用户中途输入通过 `asyncio.Queue` 投递，打断当前流
- 循环终止条件：
  - LLM 不返回 tool_calls → 任务完成
  - 达到 max_turns → 强制停止
  - 用户 steering → 中断当前轮

**参考源码**：
- Pi 官方 `pi-agent-core` 运行时：[packages/agent-core/src](https://github.com/earendil-works/pi/tree/main/packages/agent-core/src)（Agent Loop、状态管理）
- CoreCoder Agent 引擎：[corecoder/agent.py](https://github.com/he-yufeng/CoreCoder/blob/main/corecoder/agent.py)
- Pi 流式事件处理：[packages/agent-core/src/events](https://github.com/earendil-works/pi/tree/main/packages/agent-core/src)
- Pi steering 机制：[packages/coding-agent/src/commands](https://github.com/earendil-works/pi/tree/main/packages/coding-agent/src/commands)

**验收标准**：
- [x] ReAct 循环完整：推理 → 工具 → 观察 → 再推理
- [x] 多工具并行执行（`asyncio.gather`）
- [x] 流式输出实时渲染（`rich.Live`）
- [x] 支持 steering 中断当前工具链
- [x] max_turns 保护死循环
- [x] 每轮自动触发上下文压缩

**预期踩坑**：
- `rich.Live` 内不能嵌套 `Live`，工具结果用 `console.print(Panel)`
- steering 的 race condition，必须用 `asyncio.Queue` 做线程安全投递

---

### Day 4：上下文压缩
> Done 2026-07-24-17:13 | New Fix 2026-07-25-00:24
> 学会了Pi-agent的简单压缩: 基于保留的token数找切割点，压缩切割点前的messages（tool使用的完整块）摘要中保留文件操作轨迹 + 剩余messages
> 
**文件**：`pi_agent/context.py`  
**代码量**：~80 行  
**核心机制**：Pi-style Compaction —— 绝对 Token 预算驱动、从后往前找合法切割点、Turn 边界保护、双摘要、增量更新、文件操作追踪

**整体架构：5 阶段流程**

```
Stage 1: 判断是否触发压缩
   └── context_tokens > context_limit - reserve_tokens ?
Stage 2: find_cut_point()
   └── 从最新消息倒推累积 token，超过 keep_recent_tokens 后向前对齐合法切割点
Stage 3: generate_summary()
   └── 压缩切割点之前的历史（首次用 SUMMARIZATION_PROMPT，增量用 UPDATE）
Stage 4: generate_turn_prefix_summary() [可选]
   └── 当切割点落在 Turn 中间时，并行生成当前 Turn 前缀摘要
Stage 5: compact()
   └── 合并双摘要 + 文件操作记录，生成 [上下文摘要] 消息插入历史
```

**关键设计**：
- `CompactionConfig`：`reserve_tokens`（生成预留）、`keep_recent_tokens`（保留最近原始消息预算）、摘要长度限制
- `pi_agent/compaction/cutpoint.py`：`find_cut_point()` / `find_turn_start_index()` / `is_valid_cut_point()`
- `pi_agent/compaction/token_utils.py`：`estimate_tokens()` chars/4 启发式 + `calculate_context_tokens()` 从 usage 计算
- `pi_agent/compaction/summary.py`：结构化摘要 Prompt + 双摘要合并
- `pi_agent/compaction/fileops.py`：从 tool_calls 提取 `read_files` / `modified_files`
- `ContextCompactor.compress_if_needed()`：每轮 Agent Loop 调用，自动决策
- `get_stats()`：返回上下文统计

**参考源码**：
- Pi 上下文压缩（Compaction）：[packages/coding-agent/src/core/compaction/compaction.ts](https://github.com/earendil-works/pi/tree/main/packages/coding-agent/src/core/compaction)
- Pi 配置系统：[packages/coding-agent/src/config](https://github.com/earendil-works/pi/tree/main/packages/coding-agent/src/config)

**验收标准**：
- [x] 按绝对 token 预算触发压缩：`estimated > context_limit - reserve_tokens`
- [x] 保留最近 `keep_recent_tokens` 的原始消息
- [x] 不单独切割 toolResult，保证 assistant + toolResult 配对
- [x] Split Turn 时生成双摘要（历史摘要 + Turn 前缀摘要）
- [x] 摘要输出结构化：Goal / Progress / Next Steps / Critical Context
- [x] 摘要中保留文件操作轨迹（read / modified）
- [x] 系统提示始终保留
- [x] 提供 stats 接口供外部监控

**预期踩坑**：
- LLM 摘要可能丢失关键信息，需观察实际效果调 `summary_max_tokens`
- `keep_recent_tokens` 与 `reserve_tokens` 的比例决定压缩频率和保留上下文多少

**设计演进**

最初实现是简单的“三级阈值压缩”：

```
ratio < 0.50  → 不压缩
0.50 ≤ ratio < 0.70 → Tier 1：丢弃最旧块
0.70 ≤ ratio < 0.90 → Tier 2：摘要旧块
ratio ≥ 0.90 → Tier 3：紧急压缩
```

问题：
1. 单一阈值导致反复触发 Tier 1；
2. 固定保留块数在 256K 大窗口下过度压缩；
3. 没有 Turn 边界保护，可能切断 tool-call/tool-result；
4. 不追踪文件操作，工程上下文易丢失。

**最终方案：Pi-style Compaction**

参考 Pi Agent 的 `compaction.ts`，改为**绝对 token 预算 + 双摘要 + 文件追踪**：

1. **触发条件**：`estimated_tokens > context_limit - reserve_tokens`
   - 默认 `reserve_tokens = 16384`，给模型生成留足空间。
2. **保留策略**：始终保留最近 `keep_recent_tokens`（默认 20000）的原始消息，不压缩。
3. **切割点算法**：从后往前累积 token，超过 `keep_recent_tokens` 后向前对齐合法切割点。
   - 合法切割点：`user`、`assistant`（含后续 tool results）
   - 非法切割点：单独出现的 `toolResult`（避免孤儿）
4. **Turn 边界保护**：
   - 若切割点落在 Turn 中间（非 user 起点），标记 `is_split_turn = true`。
   - 并行生成两个摘要：历史摘要 + 当前 Turn 前缀摘要。
5. **增量摘要**：后续压缩在上一次摘要基础上更新（当前实现保留接口，首次实现为重新生成）。
6. **文件操作追踪**：从 `read`/`edit`/`write`/`bash` 等 tool_calls 中提取文件路径，附加到摘要中。

**相关文件**：
- `pi_agent/context.py`
- `pi_agent/agent.py`
- `pi_agent/compaction/cutpoint.py`
- `pi_agent/compaction/token_utils.py`
- `pi_agent/compaction/summary.py`
- `pi_agent/compaction/fileops.py`
- `tests/test_context.py`

---

### Day 5：会话持久化
> Done 2026-07-27-13:53 | PI-AGENT Review & Fix 2026-07-27-13:59
> 学会了全部保留到一个JSONL文件，会话树

**文件**：`pi_agent/session.py`
**代码量**：~180 行
**核心机制**：JSONL 存储、会话树（parent_id）、书签标记

**关键设计**：
- `SessionNode`：会话节点，含 id、parent_id、messages、bookmark、created_at
- `SessionStore`：全局存储，JSONL 持久化到 `~/.pi-agent/sessions.jsonl`
- `create_node()`：创建新节点（可指定 parent_id 实现分支）
- `get_branch()`：从根到指定节点的完整路径
- `bookmark()` / `list_bookmarks()`：书签标记和检索

**参考源码**：
- Pi 会话树实现：[packages/agent-core/src/session](https://github.com/earendil-works/pi/tree/main/packages/agent-core/src/session)
- Pi 历史管理：[packages/coding-agent/src/history](https://github.com/earendil-works/pi/tree/main/packages/coding-agent/src/history)
- Pi `/tree` 命令实现：[packages/coding-agent/src/commands/tree](https://github.com/earendil-works/pi/tree/main/packages/coding-agent/src/commands/tree)

**验收标准**：
- [x] 会话保存/加载完整，数据不丢失
- [x] 支持分支（指定 parent_id 创建新节点）
- [x] 书签标记和检索正常
- [x] 自动保存（每轮后或手动触发）—— CLI 层实现

**预期踩坑**：JSONL 并发写会损坏文件，单进程 + `tempfile` 原子写

---

### Day 6：CLI 外壳
> Done 2026-07-28-14:36 | Review & Optimize 2026-07-28

**文件**：`pi_agent/cli.py`  
**代码量**：~380 行  
**核心机制**：typer 命令解析、rich 交互、配置优先级、/save /exit 命令、会话 metadata 存取

**关键设计**：

#### 命令入口
- `chat` 命令：支持交互模式（无 prompt 参数）和非交互模式（带 prompt 参数）
- `resume` 子命令：按书签名恢复会话，支持 `--max-turns` 覆盖

#### 配置优先级
```
CLI 参数 > 配置文件 (~/.pi-agent/config.yaml) > 环境变量 > 硬编码默认值
```
- `resolve_kwargs()` 按优先级逐层合并 api_key / base_url / model / max_turns 等参数
- API Key：`--api-key` > `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` > config.yaml
- Base URL：`--base-url` > `OPENAI_BASE_URL` > config.yaml > `https://api.deepseek.com`
- Model：`--model` > config.yaml > `deepseek-chat`

#### 共享初始化工厂
- `_build_agent_and_store(kwargs)`：统一创建 LLMClient → ToolRegistry → AgentConfig → Agent → SessionStore 的完整链路，`chat` 和 `resume` 复用，消除重复代码
- `_agent_metadata(agent)`：提取 `model` / `max_turns` / `context_limit` 作为会话节点元数据，随 `/save` 和 `/exit` 持久化到 SessionNode.metadata

#### 交互模式命令

| 命令 | 功能 |
|:---|:---|
| `/save <name>` | 保存当前会话为书签节点（附带 agent metadata） |
| `/bookmarks` | 列出所有书签，当前节点标记 ★ |
| `/stats` | 查看上下文统计：轮次、消息数、Token 估算、使用率 |
| `/exit` 或 `/quit` | 保存当前状态为新节点，打印 node_id + 文件路径，退出 |
| `/help` | 显示帮助 |

#### 会话 metadata 存取
- **保存时**：`/save`、`/exit`、非交互模式自动保存均通过 `_agent_metadata(agent)` 将运行配置写入 `SessionNode.metadata`
- **恢复时**：`resume` 命令从 `node.metadata` 读取 `max_turns` / `context_limit` / `model`，优先级为：**CLI 显式传入 > 存储 metadata > config 文件 > 默认值**

#### 异常处理
- API Key 缺失：红色提示 + 三种配置方式说明
- 书签不存在：列出所有可用书签供选择
- 交互模式 `agent.run()` 异常：捕获后打印错误并 `continue`，不退出循环
- `KeyboardInterrupt` / `EOFError`：友好退出

#### 优化点（相比初版计划）
- `agent.turn_count` property 替代 `agent._turn_count` 私有属性访问
- `SessionStore._dirty` 标记：`/save` 后 `/exit` 只触发一次全量写回
- `resume` 新增 `--max-turns` CLI 参数，补全 `--model` `--api-key` `--base-url` `--no-confirm`

**参考源码**：
- Pi CLI 入口：[packages/coding-agent/src/cli](https://github.com/earendil-works/pi/tree/main/packages/coding-agent/src/cli)
- Pi 命令系统：[packages/coding-agent/src/commands](https://github.com/earendil-works/pi/tree/main/packages/coding-agent/src/commands)
- Pi TUI 渲染：[packages/tui/src](https://github.com/earendil-works/pi/tree/main/packages/tui/src)（参考设计，用 rich 替代实现）

**运行命令**：

```bash
# 交互模式
python -m pi_agent.cli chat --model deepseek-chat

# 非交互模式（脚本自动化）
python -m pi_agent.cli chat "请帮我写一个快速排序的 Python 代码"

# 恢复会话（按书签）
python -m pi_agent.cli resume my-session

# 恢复时覆盖 max_turns
python -m pi_agent.cli resume my-session --max-turns 100
```

**验收标准**：
- [x] 交互/非交互双模式正常
- [x] 配置加载优先级正确（CLI > 配置文件 > 环境变量）
- [x] `/save` 命令可保存会话并打书签
- [x] `/exit` 退出时保存 + 打印 node_id 和文件路径
- [x] `resume` 子命令可按书签恢复，并从 metadata 还原 max_turns
- [x] 异常处理：API Key 缺失、书签不存在、agent.run() 异常等友好提示
- [x] 会话 metadata（model / max_turns / context_limit）随节点存取
- [x] _dirty 标记避免重复全量写回

**预期踩坑**：typer 同步入口 + `asyncio.run()` 包装异步 Agent 运行

---

### Day 7：整合测试

**文件**：`tests/*.py`（4 个文件）  
**代码量**：~100 行测试代码  
**核心机制**：单元测试 + 集成测试 + 边界测试

**参考源码**：
- Pi 测试套件：[test.sh](https://github.com/earendil-works/pi/blob/main/test.sh)、[pi-test.sh](https://github.com/earendil-works/pi/blob/main/pi-test.sh)
- Pi CI 配置：[.github/workflows](https://github.com/earendil-works/pi/tree/main/.github/workflows)
- CoreCoder 测试：[tests/](https://github.com/he-yufeng/CoreCoder/tree/main/tests)

**测试清单**：

#### 1. 单元测试（Mock）

| 测试文件 | 覆盖内容 | Mock 方式 |
|:---|:---|:---|
| `test_llm.py` | 流式输出、重试、token 估算 | Mock SSE 流，模拟 503 错误 |
| `test_tools.py` | 7 个工具、危险确认、超时 | 直接调用，mock `Confirm.ask` |
| `test_agent.py` | ReAct 循环、并行工具、steering | Mock LLM 返回 tool_calls |
| `test_context.py` | 三层压缩触发 | 构造大消息列表，mock LLM 摘要 |

#### 2. 集成测试（真实 API）

- [ ] 用 `gpt-4o-mini` 或等效小模型跑简单任务
- [ ] 测试："请读取当前目录的 README.md 并总结"
- [ ] 测试："找出所有 .py 文件，搜索包含 'def main' 的文件"

#### 3. 边界测试

- [ ] 流式中断：模拟网络断开，确认重试
- [ ] 工具超时：`bash sleep 60`，确认 30s 超时返回
- [ ] 上下文溢出：构造 1000 条消息，确认压缩不崩溃
- [ ] 空输入：用户输入空字符串，确认不崩溃

#### 4. 性能测试（可选）

- [ ] 100 轮循环，确认内存不泄漏
- [ ] 大文件 `read`（10MB），确认截断到 4000 字符

**验收标准**：
- [ ] 全部单元测试通过（Mock）
- [ ] 集成测试用真实 API 跑通简单编码任务
- [ ] 边界测试无崩溃

---

## 六、与 Pi 原版的差异对比

| 特性 | Pi 原版 (TypeScript, ~15,000 行) | 你的 Python 版 (~700 行) |
|:---|:---|:---|
| **TUI** | 自研 ANSI 差分渲染 | `rich` Live + Panel |
| **扩展系统** | npm 包动态加载 | 暂不实现（后续 `importlib`）|
| **多提供商** | 15+ 提供商原生适配 | OpenAI 兼容接口（覆盖 80%）|
| **会话树 UI** | 完整树形导航（`/tree` 命令）| `parent_id` + 书签列表 |
| **上下文压缩** | 可扩展（自定义扩展）| 三层固定策略（可扩展）|
| **核心机制** | Agent Loop / 流式 LLM / 工具调用 / 上下文压缩 | **100% 等价** |

**关键设计决策**：

1. **不做自研 TUI**：Pi 的 ANSI 差分渲染是工程复杂度大头，用 `rich` 替代省 2000+ 行，聚焦核心机制
2. **不做扩展系统**：Python 用 `importlib` 更简单，先实现核心，扩展是后续迭代
3. **不做 OAuth / 多提供商适配**：OpenAI 兼容接口覆盖 80% 场景（Kimi、DeepSeek、本地 vLLM），减少无关复杂度
4. **会话树简化**：完整树形 UI 是 TUI 层复杂功能，用 `parent_id` + 列表保留分支能力
5. **工具用同步函数 + `asyncio.to_thread`**：文件 IO、子进程都是阻塞的，`to_thread` 是最简洁的异步化方案

---

## 七、快速启动

```bash
# 1. 创建项目
mkdir pi-agent-mini && cd pi-agent-mini
python -m venv .venv && source .venv/bin/activate

# 2. 安装依赖
pip install httpx rich typer pyyaml jsonlines

# 3. 创建目录
mkdir -p pi_agent tests

# 4. 按 Day 1-6 创建文件，Day 7 写测试

# 5. 运行
python -m pi_agent.cli chat --model gpt-4o-mini --api-key sk-xxx
```

---

## 八、执行原则

1. **每天只写一个文件**，当天验收通过再进入下一天
2. **不要复制粘贴**，理解每一行代码的作用
3. **遇到坑先自己 debug 30 分钟**，再查文档或参考 Pi 源码
4. **每天结束时写一行总结**：今天理解了什么机制，有什么新发现
5. **如果被打断，随时恢复**：每个文件独立，不依赖后续文件

---

## 十、关键修正记录（相比初版计划的重要修正）

| # | 问题 | 严重度 | 修正方案 | 影响文件 |
|---|------|--------|---------|---------|
| 1 | `_last_tool_calls` 副作用：Agent 并发调用时互相覆盖 | 🔴 | 改为 `Delta` 类型的流式协议，`chat()` 返回 `AsyncIterator[Delta]`，调用方自行累积 | `llm.py` |
| 2 | Steering 中断缺少取消令牌 | 🔴 | `chat()` 接受 `cancel_event: asyncio.Event`，每行 SSE 检查 `is_set()`，触发即抛 `StreamCancelledError` | `llm.py`, `agent.py` |
| 3 | 工具异常未反馈给 LLM（会导致 Agent 死循环） | 🔴 | `_execute_tools()` 用 `asyncio.gather(return_exceptions=True)`，异常包装为 `ToolResult(is_error=True)` 以 tool message 注入对话 | `agent.py`, `tools.py` |
| 4 | 上下文压缩采用简单三级阈值，无 Turn 边界保护 | 🔴 | 重写为 Pi-style Compaction：绝对 token 预算驱动、从后往前找合法切割点、双摘要保护 Split Turn、文件操作追踪 | `context.py`, `pi_agent/compaction/*.py` |
| 5 | CLI 缺少 --base-url，只能连 DeepSeek | 🟡 | CLI 增加 `--base-url` / `-b` 参数，支持任意 OpenAI 兼容端点 | `cli.py` |
| 6 | JSONL 树遍历 O(n)，会话量大时变慢 | 🟡 | 启动时全量载入 `dict[str, SessionNode]`，退出时 tempfile 原子写回 | `session.py` |
| 7 | `messages_from_node` 缺少 `ensure_loaded()`，未加载时永远返回 None | 🟡 | 补上 `self.ensure_loaded()` 调用 | `session.py` |
| 8 | `save()` 未加载时调用会空写覆盖文件，数据全部丢失 | 🔴 | 加 `_loaded` guard，未加载时抛 `RuntimeError` | `session.py` |
| 9 | `load()` 静默吞异常，损坏行无声丢失 | 🟡 | 改用 `logger.warning` 记录跳过原因和行内容 | `session.py` |
| 10 | `bookmark_node()` 书签名去重限制过严，需允许多节点同名 | 🟢 | 移除去重检查，`get_by_bookmark` 返回首个匹配 | `session.py` |
| 11 | CLI 有错误 import（`from re import L`、`from torch.optim...`）和拼写错误（`bold grean`、`\b`） | 🟡 | 清理 IDE 自动导入残留，修正 Rich 标记拼写 | `cli.py` |
| 12 | `resume` 命令丢失 `max_turns` / `context_limit` 参数 | 🟡 | 补全 AgentConfig 参数，新增 `--max-turns` CLI 选项 | `cli.py` |
| 13 | `/stats` 使用不存在的 `ratio` key，始终显示 `?` | 🟡 | 改为从 `estimated_tokens / context_limit` 动态计算百分比 | `cli.py` |
| 14 | 交互模式 `agent.run()` 异常导致循环崩溃 | 🟡 | 加 try/except 捕获异常，打印错误并 continue | `cli.py` |
| 15 | `/save` 和 `/exit` 每次触发全量写回 JSONL | 🟡 | SessionStore 新增 `_dirty` 标记，无变更跳过写入 | `session.py`, `cli.py` |
| 16 | `resume` 无法还原保存时的 `max_turns` 配置 | 🟡 | SessionNode 新增 `metadata` 字段，存取 model/max_turns/context_limit | `session.py`, `cli.py` |
| 17 | CLI 直接访问 `agent._turn_count` 私有属性 | 🟢 | Agent 新增 `turn_count` property，CLI 改用公开接口 | `agent.py`, `cli.py` |

## 十一、最终交付物

- [ ] 7 个 Python 文件，~900 行核心代码
- [ ] 可运行的 CLI（交互 + 非交互）
- [ ] 完整的测试套件（单元 + 集成 + 边界）
- [ ] 对 Agent Loop、流式 LLM、工具调用、上下文压缩的"肌肉记忆"级理解
- [ ] 可直接迁移到 HyperTuneAgent / 评估体系设计中的工程经验
