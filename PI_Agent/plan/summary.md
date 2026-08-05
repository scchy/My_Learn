# 从零手写一个 Pi Agent（7 天复盘）


> 我花 7 天从零手写了一个 Pi Agent，理解了所有 Agent 的通用骨架：
> **循环 + 工具 + 上下文管理**。

整体的实现地址 [mini-PI-Agent](https://github.com/scchy/My_Learn/tree/master/PI_Agent/pi_agent)

---

## 为什么写这篇博客

- **动机**：不想只当 API 调用者，想彻底搞懂 Agent 内部机制，以帮助我更好的进行agent开发和RL相关研究
- **成果**：~2000 行代码、205 单测全过
- **读者**：想理解 Agent 原理的开发者
- **预告**：按"每个模块 = 一个工程问题 + 我的解法"组织，最后提炼通用骨架

---

## 先看整体：Agent 系统架构图

可以把 Agent 类比成一台计算机：LLM + Tools 是 CPU，近期对话是 RAM，压缩摘要是虚拟内存，会话树是磁盘，CLI 是操作系统。把这些部件'固定'成一个系统，就能跑不同的 APP——写代码、查资料、分析数据。换 APP 不改骨架，这就是泛化到任何 Agent 的本质。


Agent 系统架构图
```text
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐  │
│   │   OS / CLI  (操作系统 + 外壳)                                  │  │
│   │   ── Agent 术语：Harness / Agent 循环调度器                    │  │
│   │   调度 · 配置优先级 · 交互/steering · 元命令(/compact /status)  │  │
│   └─────────────────────────────┬──────────────────────────────────┘  │
│                                 │ 组装（固定成可复用系统）              │
│                                 ▼                                      │
│   ┌────────────────────────────────────────────────────────────────┐  │
│   │                                                              │  │
│   │   ┌─────────────── Agent Loop（循环 = Agent 的心脏）────────┐ │  │
│   │   │                                                       │ │  │
│   │   │    ┌──────────┐      ┌──────────────┐      ┌────────┐ │ │  │
│   │   │    │  推理     │─────►│   行动        │─────►│  观察   │ │ │  │
│   │   │    │ (LLM)    │      │ (调用 Tools)  │      │ (结果)  │ │ │  │
│   │   │    └────┬─────┘      └──────┬───────┘      └───┬────┘ │ │  │
│   │   │         │                   │                  │      │ │  │
│   │   │         └───────────────────┴──────────────────┘      │ │  │
│   │   │                     ▲ 循环直到终止                      │ │  │
│   │   │         (无 tool_call / max_turns / 用户中断)          │ │  │
│   │   └───────────────────────────────────────────────────────┘ │  │
│   │                                                              │  │
│   │   ┌────────────────────────┐   ┌──────────────────────────┐  │  │
│   │   │  CPU (计算单元)         │   │  RAM (工作内存)           │  │  │
│   │   │  ── Agent: LLM + Tools │◄─►│  ── Agent: Context        │  │  │
│   │   │  推理引擎 + 工具集      │   │  近期对话 + 状态 + 文件汇总│  │  │
│   │   └────────────────────────┘   └────────────┬─────────────┘  │  │
│   │                                             │ 满了换出        │  │
│   │                                             ▼                │  │
│   │   ┌──────────────────────────────────────────────────────┐  │  │
│   │   │  虚拟内存 (swap)                                    │  │  │
│   │   │  ── Agent: Compaction / 检索                        │  │  │
│   │   │  压缩摘要 + 检索 (换出/换回)                        │  │  │
│   │   └──────────────────────────────┬───────────────────────┘  │  │
│   │                                  │ 持久化                    │  │
│   │   ┌──────────────────────────────▼───────────────────────┐  │  │
│   │   │  Device (持久存储)                                   │  │  │
│   │   │  ── Agent: Session / 文件追踪                        │  │  │
│   │   │  会话树 / JSONL / 文件追踪                           │  │  │
│   │   └─────────────────────────────────────────────────────┘  │  │
│   │                                                              │  │
│   └────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐  │
│   │   APP (具体任务，跑在系统上)                                   │  │
│   │   ── Agent: 具体 Agent 应用                                   │  │
│   │   写代码 · 查资料 · 数据分析 · 文件操作 ...                    │  │
│   │   ── 换 APP 不改骨架 = 泛化到任何 Agent ──                    │  │
│   └────────────────────────────────────────────────────────────────┘  │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

**双语对照表**：

| 计算机 | Agent | 作用 |
|:---|:---|:---|
| CPU | LLM + Tools | 计算单元 |
| RAM | Context | 工作内存 |
| 虚拟内存 | Compaction / 检索 | 换出/换回 |
| Device | Session / 文件追踪 | 持久存储 |
| OS | Harness / CLI | 调度 + 配置 + 交互 |
| APP | 具体 Agent 应用 | 跑在系统上的任务 |


## 模块 1：LLM 客户端（`llm.py`，538 行）

### 问题

如何稳定地调用一个`OpenAI Chat Completions API`的流式LLM接口？


**难点**：流式接口不是一次返回完整结果，而是**边生成边推送**；网络抖动、限流（429）随时可能中断。稳定调用 = 正确解析流 + 优雅处理失败。

**模块定位**：LLM 客户端是**通用层**——任何 Agent 都要调 LLM。所以这里只放"机制"（怎么解析、怎么重试、怎么估算），不放"策略"（何时压缩、保留多少），后者属于 Agent 特有的上下文管理模块。


### 解法
- [x] SSE 流式解析（`async for` 逐块读取）
  - data: `<JSON>` + 空行分隔
  - `[DONE]` 是流的结束标记
  - delta 是"增量"，不是"全量"

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM 服务端（如 OpenAI API）                       │
│                                                                     │
│   生成内容： "今天天气很好，我们"  →  "去公园散步吧"                  │
│                                                                     │
│   服务端把内容切成多个 chunk，逐个通过 SSE 推送：                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP 响应流（SSE）
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SSE 原始字节流（网络层）                           │
│                                                                     │
│   data: {"choices":[{"delta":{"content":"今天"}}]}                  │
│                                                                     │
│   data: {"choices":[{"delta":{"content":"天气"}}]}                  │
│                                                                     │
│   data: {"choices":[{"delta":{"content":"很好"}}]}                  │
│                                                                     │
│   data: {"choices":[{"delta":{"content":"，我们"}}]}                │
│                                                                     │
│   data: {"choices":[{"delta":{"content":"去公园"}}]}                │
│                                                                     │
│   data: {"choices":[{"delta":{"content":"散步吧"}}]}                │
│                                                                     │
│   data: [DONE]                                                      │
│                                                                     │
│   （每行以 \n\n 分隔，data: 前缀 + JSON 内容）                       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ 逐行读取（async for）
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    客户端解析（ llm.py）                         │
│                                                                     │
│   async for line in response:          # 逐行读 SSE                 │
│       if line.startswith("data:"):     # 只处理 data: 行            │
│           data = line[5:].strip()      # 去掉 "data:" 前缀          │
│           if data == "[DONE]":         # 流结束标记                  │
│               break                    # 终止循环                    │
│           chunk = json.loads(data)     # 解析 JSON                   │
│           text = chunk["choices"][0]["delta"]["content"]            │
│           yield text                   # 产出文本片段                │
│                                                                     │
│   产出序列： "今天" → "天气" → "很好" → "，我们" → "去公园" → "散步吧" │
│                                                                     │
│   客户端把这些片段拼接起来： "今天天气很好，我们去公园散步吧"          │
└─────────────────────────────────────────────────────────────────────┘
```
- [x] 指数退避重试（网络不稳定, 或者出现TRM-429的情况）
  - 指数退避借鉴了 CSMA/CD 的"退避"思想，但应用场景不同——一个是"发送前避免冲突"，一个是"失败后等待恢复"。
    - 指数退避：第 1 次失败等 1s，第 2 次等 2s，第 3 次等 4s，第 4 次等 8s……
    - CSMA/CD（以太网）：冲突后等待 $2^n × slot\_time$，n 是冲突次数
```
┌─────────────────────────────────────────────────────────────────────┐
│                    客户端（ llm.py）                             │
│                                                                     │
│   ┌─────────────┐                                                   │
│   │ 第 1 次请求  │───► 失败（网络抖动 / 429）                        │
│   └─────────────┘        │                                          │
│                          ▼                                          │
│                   等待 1s（2^0 × base）                              │
│                          │                                          │
│   ┌─────────────┐        │                                          │
│   │ 第 2 次请求  │◄───────┘                                          │
│   └─────────────┘───► 失败（仍 429）                                │
│                          │                                          │
│                          ▼                                          │
│                   等待 2s（2^1 × base）                              │
│                          │                                          │
│   ┌─────────────┐        │                                          │
│   │ 第 3 次请求  │◄───────┘                                          │
│   └─────────────┘───► 失败（仍 429）                                │
│                          │                                          │
│                          ▼                                          │
│                   等待 4s（2^2 × base）                              │
│                          │                                          │
│   ┌─────────────┐        │                                          │
│   │ 第 4 次请求  │◄───────┘                                          │
│   └─────────────┘───► 成功！返回数据                                 │
│                          │                                          │
│                          ▼                                          │
│                   重置退避计数（下次从 1s 开始）                      │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  等待时间 = 2^(n-1) × base + 随机抖动 (jitter)              │   │
│   │  n = 重试次数，base = 初始间隔（如 1s）                      │   │
│   │  上限：max_retries（如 5 次）或 max_wait（如 60s）           │   │
│   └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
**Python 实现**：

```python
async def _request_with_retry(self, func, *args, **kwargs):
    """指数退避重试：网络抖动 / 429 时自动重试"""
    max_retries = 5
    base_delay = 1.0          # 初始等待 1s
    max_delay = 60.0          # 上限 60s

    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except (NetworkError, RateLimitError) as e:
            if attempt == max_retries - 1:
                raise  # 最后一次失败，直接抛出

            # 指数退避：2^attempt × base + 随机抖动
            delay = min(base_delay * (2 ** attempt), max_delay)
            delay += random.uniform(0, 0.5)   # jitter，防止"惊群"
            await asyncio.sleep(delay)
```
- [x] **token 估算**：按字符类型分类——CJK 1 字 ≈ 1 token、其他 4 字符 ≈ 1 token，向上取整并加每条消息 ~4 token 的元数据开销，保守估计避免溢出
- [x] **LLM 自摘要**：提供摘要用的 prompt 模板和调用封装
  - 机制（怎么总结）： + LLM 调用 → 放 `llm.py`，可复用
  - 策略（何时总结、保留多少、怎么接回）：→ 放 `context.py`，是 Agent 特有的

### 坑

**流式解析的边界情况**（写 SSE 解析必踩的坑）

- **半截 JSON**：网络抖动时，一个 chunk 可能只收到一半 JSON，`json.loads` 直接抛异常。
  - 处理：捕获 `json.JSONDecodeError`，跳过该 chunk，等下一个 chunk 补全。
- **断流**：服务端中途断开，`async for` 提前结束，但没收到 `[DONE]`。
  - 处理：把"流提前结束"当作异常处理，触发重试（配合指数退避）。
- **空行 / 注释行**：SSE 协议里可能有空行或 `:` 开头的注释行。
  - 处理：跳过，不能当数据解析。
- **`delta` 为空**：有些 chunk 的 `delta` 里没有 `content`（比如只有 `role` 或 `tool_calls`）。
  - 处理：判空再取，否则 `KeyError`。
- **`data:` 前缀剥离**：`data:` 是协议的一部分，不是数据本身，要 `line[5:]` 去掉。
  - 坑：如果直接 `json.loads(line)`，会把 `data:` 也当 JSON 解析，直接报错。

**思考**：SSE 解析的难点不在"正常情况"，而在"边界情况"。正常流谁都会写，但半截 JSON、断流、空 delta 这些坑，只有真实跑过才会遇到。**防御性编程**（判空、捕获异常、把断流当异常）是流式解析的必修课。

### 反思

LLM 客户端是所有 Agent 的地基。没有稳定的流式解析和重试，上层全白搭。而"机制 vs 策略"的分层，是让 LLM 客户端可复用的关键——它不关心 Agent 怎么用上下文，只负责稳定地拿到结果。

---

## 模块 2：工具系统（`tools.py`，418 行）

### 问题
如何让 LLM 安全地调用函数？

**难点**：LLM 只能输出文本，不能直接执行代码。工具系统要做两件事——**把函数暴露给 LLM**（让它"知道"有什么工具、怎么调），**保护系统安全**（防止 LLM 乱调危险操作）。前者靠 Schema，后者靠安全防线。

**模块定位**：工具协议是 Agent 的"手脚"。它决定了 Agent 能做什么、不能做什么。这里的核心设计是**"声明式"**——开发者用 `@tool` 装饰器声明工具，系统自动生成 Schema，LLM 按 Schema 调用。


### 解法

- [x] **装饰器注册（`@tool`）**
  - 开发者用 `@tool` 装饰一个普通函数，系统自动把它注册进工具表
  - 好处：**声明式**，开发者不用手动维护工具列表，加一个工具就是加一个函数, 有点类似`@mcp.tool`
```python
@tool
def get_weather(city: str) -> str:
    """查询城市天气"""
    return f"{city} 今天晴，25°C"
```
- [x] Schema 自动生成（从函数签名生成 JSON Schema）
  - 用 inspect.signature 读函数签名，把参数类型、默认值、docstring 转成 JSON Schema 
  - 好处：单一数据源——函数签名是唯一真相，Schema 自动生成，不会"签名改了 Schema 忘了改"
- [x] 危险确认（危险操作需人工确认）
    - 给工具打 dangerous=True 标记，调用前弹确认框，人工点头才执行
    - 好处：LLM 可能误判或被骗（prompt injection），危险操作必须有人把关
```python
@tool(dangerous=True)
def delete_file(path: str) -> str:
    """删除文件（危险操作，需人工确认）"""
    os.remove(path)
    return f"已删除 {path}"
```
- [x] 超时截断（防止工具无限运行）
  - 给工具调用加超时（如 30s），超时强制中断，防止工具卡死拖垮整个 Agent
  - 好处：单个工具失败不能拖垮整个循环，超时后 Agent 可以继续或报错


### 坑

**1. Schema 生成：所有参数统一视为 `string`（当前实现的简化）**：所有参数统一视为 `string`（当前实现的简化）**
- LLM 按 Schema 传参时，会把 int 参数当字符串传（如 "30" 而不是 30）
- 所以内置工具里到处是 int(offset)、float(timeout_sec) 这种手动类型转换——这是"Schema 简化"的连锁代价
```python
# 所有参数统一视为string
prop: dict[str, Any] = {"type": "string", "description": param_name}
```
> 坑的本质：Schema 是 LLM 和函数之间的"契约"。契约不精确（全标 string），LLM 就会传错类型，函数就得自己兜底转换。契约的精度决定了兜底代码的量。

**2. 危险标记的判定标准（什么算"危险"？）**（什么算"危险"？）
- 当前实现里只有 bash 标了 dangerous=True，其他工具（write、edit、delete）都没标。但：
  - write / edit 是写操作，会改文件——算不算危险？
  - bash 能执行任意命令，包括 rm -rf——必须危险

> 坑的本质：危险判定没有绝对标准，是工程判断。原则是"宁可多标，不可漏标"——把不确定的标成危险让用户确认，比漏标导致误删强。当前只标 bash 是偏保守的选择，但 write/edit 是否也该标，值得商榷。

**3. 超时截断：`asyncio.wait_for` + `to_thread` 的配合**

```python
result = await asyncio.wait_for(
    asyncio.to_thread(spec._func, **arguments),
    timeout=spec.timeout
)
```  
- bash 工具额外用了 subprocess.run(timeout=...) 来真正杀死子进程（注释里写了"妥善杀死超时子进程"）
- 但其他工具（如 read 大文件）超时后，线程可能还在后台跑

> 坑的本质：asyncio.wait_for 只能"放弃等待"，不能"杀死线程"。要真正终止，得靠工具内部自己支持超时（如 subprocess.run(timeout=)）。超时是"协作式"的，不是"强制式"的。


**4. 输出截断：防止工具输出撑爆上下文**

```python
if len(output) > 4000:
    output = output[:3900] + f"\n\n... [截断，原输出 {len(output)} 字符]"
```
- 工具输出（如 bash 跑 `ls -R`）可能非常长，直接塞进上下文会撑爆 token 预算。所以截断到 4000 字符，并标注原长度。

> 坑的本质：工具输出是上下文的重要来源，但不受控。截断是"上下文管理"的第一道防线——在工具层就限制输出大小，比在上下文层压缩更省。

### 反思
工具系统是 Agent 的"手脚"，四道防线（`危险标记` / `确认` / `超时` / `错误包装` / `输出截断`）是工程化标配。但更深一层，从代码里能看到几个设计哲学：

"契约"决定一切：Schema 是 LLM 和函数之间的契约。契约不精确（全标 string），LLM 就传错类型，函数就得兜底转换。契约的精度 = 兜底代码的量。

错误是"数据"不是"异常"：ToolResult 把错误包装成返回值，让 LLM 能看到错误、自己调整。这是 Agent 场景特有的设计——错误应该交给 LLM 判断，而不是中断循环。

超时是"协作式"的：asyncio.wait_for 只能放弃等待，不能杀死线程。真正终止得靠工具内部支持。超时不是银弹，工具自己要能"优雅退出"。

安全是"分层"的：危险标记（静态声明）+ 人工确认（运行时把关）+ 超时（防卡死）+ 输出截断（防撑爆）。单层防线不够，要层层叠加。

---

## 模块 3：Agent 循环（`agent.py`，432 行）

### 问题
如何让 LLM 自主行动？

**难点**：LLM 本身是"无状态"的——它只根据当前输入生成输出，不会自己"做事"。Agent 循环要做的，是**把"思考"和"行动"串起来**：LLM 想一步 → 执行工具 → 看到结果 → 再想一步……直到任务完成。这个"想→做→看→再想"的循环，就是 Agent 的心脏。

**模块定位**：Agent 循环是**所有 Agent 的骨架**。不管上层是 ReAct、Plan-then-Execute 还是 Reflexion，底层都是这个循环。它的核心设计决策是**终止条件**——什么时候停下来。


### 解法
- [x] **ReAct 循环（推理→行动→观察）**
  - 每轮：LLM 流式输出"思考"（text）+ "行动"（tool_call）→ 执行工具 → 把结果（tool message）喂回 LLM → 下一轮
  - 直到 LLM 不再调用工具（直接给最终答案），或达到 `max_turns`（默认 50）
```python
for turn in range(self.config.max_turns):
    # 1. 检查 steering（用户中途插话）
    steering_msg = await self._drain_steering()
    if steering_msg:
        self.messages.append(Message(role='user', content=f'[用户中途指示]\n{steering_msg}'))
    
    # 2. 上下文压缩（每轮自动检查 token 用量）
    self.messages = await self.compressor.compress_if_needed(...)
    
    # 3. 流式调用 LLM，收集 text + tool_calls
    accumulated = await self._stream_and_collect()
    
    # 4. 无 tool_calls → 任务完成
    if not tool_calls:
        final_text = text
        break
    
    # 5. 有 tool_calls → 并行执行，结果回注
    tool_msgs, tool_outputs = await self._execute_tools(tool_calls)
    self.messages.extend(tool_msgs)
```
- [x] 并行工具调用（`asyncio.gather`）
    - 当 LLM 一次输出多个 tool_call 时，用 asyncio.gather 并发执行，而不是串行
    - 关键：`return_exceptions=True`——保证所有工具都执行完毕，一个失败不拖垮其他
```python
tasks = [asyncio.create_task(self.tools.execute(name, args)) for ...]
results = await asyncio.gather(*tasks, return_exceptions=True)
```
- [x] 流式渲染
    - LLM 的回复边生成边显示（打字机效果），用户不用干等
    - 关键：单面板 + 工具结果延迟打印——避免嵌套 `Live` 冲突 
```python
with Live("", console=self.console, refresh_per_second=10) as live:
    async for delta in stream:
        if delta.kind == 'text':
            live_text += delta.text
            live.update(Markdown(live_text + "▌"))
```
- [x] **Steering 中断**（`asyncio.Queue` 注入用户消息）
  - 用户可以在 Agent 运行中插话，改变方向
  - 实现：`steer()` 把消息放进 `_steer_queue`，同时 `_cancel_event.set()` 打断当前 LLM 流；下一轮循环 `_drain_steering()` 取出消息注入
```python
async def steer(self, message: str) -> None:
    await self._steer_queue.put(message)
    self._cancel_event.set()  # 打断当前 LLM 流
```

### 坑

**1. 循环终止条件的设计（最容易忽略、最关键的决策）**

Agent 循环必须能停下来，否则会无限循环烧钱。代码里有四层终止：

- 无 `tool_call` ： LLM不再调用工具，直接给最终答案 → 正常结束
- max_turns：达到最大轮数（默认 50）→ 强制结束，防止死循环
- 用户中断：StreamCancelledError 或 _aborted → 立即结束
- LLM 错误：LLMError → 记录错误，结束本轮

> 坑的本质：终止条件不是"一个"，是"多层叠加"。只靠"无 `tool_call`"不够——LLM 可能一直调用工具（比如反复读同一个文件），必须用 max_turns 兜底。终止条件是循环设计里最容易忽略、也最关键的决策。

**2. 并行工具的结果回注顺序**

`asyncio.gather` 并发执行多个工具，但结果回注到 `messages` 时顺序必须和 `tool_call` 对应。代码里的关键做法：
```python
# 用 zip 把 task_info 和 results 一一对应
for (call_id, name, args), result in zip(task_info, results):
    tool_msgs.append(Message(role="tool", content=tr.output, tool_call_id=call_id))
```

> 坑的本质：`asyncio.gather` 返回的结果按传入顺序排列（不是按完成顺序），所以只要按 `tool_call_id` 对应回注，顺序就不会乱。关键是要显式维护 `tool_call` 和结果的对应关系，不能假设"谁先完成谁先回注"。

**3. 工具结果要回注给正确的 `tool_call`**

OpenAI 协议要求：工具结果必须带 `tool_call_id`，且每个 `tool_call` 都要有对应结果。漏掉一个，LLM 会报错或行为异常。

> 坑的本质：工具结果不是"随便塞进 messages"，而是按 `tool_call_id` 精确对应。这是协议要求，也是 LLM 正确推理的前提。

**4. 工具错误要"反馈"给 LLM，而不是崩溃**

代码里 `return_exceptions=True + ToolResult(is_error=True) `的设计（注释里写了"工具错误以 `isError` 语义反馈给 LLM（而非崩溃）"）：

```python
if isinstance(result, Exception):
    tr = ToolResult.error(f'{type(result).__name__}: {result}')
```
> 坑的本质：工具错误有两种处理哲学——抛异常（中断流程）vs 返回错误标记（让 LLM 自己处理）。当前实现选后者，因为 Agent 场景下，错误应该交给 LLM 判断，让它自己调整重试。代码里还有一句 `all_failed` 检查——所有工具都失败时提示"LLM 将尝试恢复"。

**5. 危险工具确认（`confirm_dangerous`）**

代码里危险工具（如 bash）执行前会打印警告，但非交互模式下自动确认：
```python
if self.confirm_dangerous and name in self.tools.get_dangerous_tools():
    self.console.print(f"[yellow]⚠ 执行危险操作: {name}...[/yellow]")
    self.console.print("[dim]自动确认危险操作[/dim]")  # 非交互模式自动确认
```

> 坑的本质：危险确认在交互模式下应该弹窗让人点头，但在非交互模式（如脚本、CI）下没法等人，只能自动确认。"确认"这个动作本身要分模式处理——交互模式人工把关，非交互模式自动放行（或直接拒绝）。


### 反思
Agent 循环是 Agent 的心脏，但它的设计哲学是<font color=darkred>**"循环 + 终止 + 容错"**</font>：

1. 循环是"想→做→看"的重复：LLM 每轮只做一步，靠循环把多步串起来。这是 ReAct 的核心——把复杂任务拆成多轮简单决策。

2. 终止条件是循环的灵魂：没有终止条件，循环就是死循环。四层终止（无 `tool_call` / `max_turns` / `用户中断` / `LLM 错误`）是工程化标配——既要能正常结束，也要能强制兜底，还要能被人打断。

3. 容错是"反馈"不是"崩溃"：工具错误包装成 `ToolResult(is_error=True)` 反馈给 LLM，让它自己恢复。这是 Agent 场景特有的设计——错误是"数据"，交给 LLM 判断，而不是中断循环。

4. 并行是"省时间"不是"必须"：并行工具调用是优化，不是核心。**核心是正确回注结果——顺序、tool_call_id 对应**，这些细节决定 LLM 能不能正确推理。

5. Steering 是"人机协作"的关键：Agent 不是"全自动黑盒"，用户要能随时插话、改变方向。`asyncio.Queue` + `cancel_event` 是"人在回路"（`human-in-the-loop`）的实现——既能注入新指令，也能打断当前动作。


---

## 模块 4：上下文管理（`context.py` + `compaction/`，~550 行）

### 问题
如何不让长对话"失忆"？

**难点**：LLM 的上下文窗口是有限的（128K 或更少），而一次编程任务可能产生上千条消息——读文件、改代码、跑测试、修 bug，来回几十轮。当消息的总 token 数逼近窗口上限时，你必须**把旧内容"换出"，把空间留给最近的对话**。但换出不是简单丢弃——丢掉的上下文里可能有关键决策、文件路径、错误信息，丢了就"失忆"。

换出的难点在于三个冲突：
1. **保留 vs 丢弃**：哪些消息该换出？凭什么？
2. **摘要 vs 原文**：换出时是直接丢弃还是压缩成摘要？摘要丢失多少细节？
3. **首次 vs 多次**：第一次压缩写摘要还行，但长对话可能被压缩 5~10 次——每次都全量重写旧摘要，还是增量更新？

**模块定位**：上下文管理是 Agent 的"虚拟内存"——RAM（近期对话）满了，就把旧页换出到磁盘（压缩成摘要），需要时再换回（注入摘要消息）。它决定了 Agent 的"记忆跨度"——能记住 50 轮前的决策，还是只能记住最近 5 轮。


### 解法

整体流程：每轮 Agent 循环开始时，`compress_if_needed()` 被调用，做 5 件事：

```
compress_if_needed(messages, context_limit)
  │
  ├── Step 1: 估算 token → 超过阈值才触发
  │     threshold = context_limit - reserve_tokens
  │     例: 128K - 16K = 112K，当前用了 115K → 触发
  │
  ├── Step 2: 切分系统消息/非系统消息
  │     system 消息（提示词）→ 永远保留，不压缩
  │     non_system → 进入后续处理
  │
  ├── Step 3: 从后往前找合法切割点
  │     保留最近 keep_recent_tokens 的原始消息
  │     切割点必须合法（user / tool_call 起点 / 无孤儿的 assistant）
  │     ↓
  │     判断 is_split_turn：切割点是否落在 Turn 中间？
  │
  ├── Step 4: 生成摘要
  │     ├── 正常切割（Turn 边界）→ 单摘要（全量 / 增量）
  │     └── Split Turn → 双摘要（历史摘要 + Turn 前缀摘要）
  │
  └── Step 5: 合并摘要 + 文件操作追踪 → 重构消息列表
        system + [摘要消息] + 保留的最近消息
```

- [x] **绝对 token 预算（`reserve_tokens` + `keep_recent_tokens`）**
  - 两个预算各管各的，不混在一起：
    - `reserve_tokens = 16384`：给模型生成预留空间 + 安全余量。`context_limit - reserve_tokens` 是触发阈值——超过这个值才启动压缩。
    - `keep_recent_tokens = 20000`：保留最近多少 token 的原始消息（不被压缩）。
  - 关键区别：`reserve_tokens` 决定"何时触发"，`keep_recent_tokens` 决定"保留多少"。两者独立，不能用同一个值。
```python
@dataclass
class CompactionConfig:
    reserve_tokens: int = 16384       # 触发阈值 = context_limit - reserve_tokens
    keep_recent_tokens: int = 20000   # 从后往前保留最近 N token 的原始消息
    summary_max_tokens: int = 4096    # 历史摘要长度上限
    turn_prefix_summary_max_tokens: int = 2048  # Turn 前缀摘要长度上限
```

- [x] **切割点查找（从后往前倒推 + Turn 边界保护）**
  - 思路：从最新一条消息开始，往回累加 token 数，直到超过 `keep_recent_tokens` → 在当前位置向后找第一个合法切割点。
  - 合法切割点的判定规则（`is_valid_cut_point`）：

```
消息角色           │ 可作为切割点？ │ 原因
───────────────────┼──────────────┼─────────────────────────────
user               │ ✅ 是         │ Turn 起点，天然边界
assistant + tool_calls │ ✅ 是      │ tool 调用单元起点，后续 tool result 跟随
assistant (纯文本)  │ ⚠️ 看前一条    │ 如果前面是 tool，不能切（会切断 tool 单元）
tool                │ ❌ 否         │ 单独出现会形成孤儿 tool result
system              │ ❌ 否         │ 在最开头，不参与切割
```

```python
def is_valid_cut_point(messages, index):
    msg = messages[index]
    if msg.role == "tool":    # tool 不能单独保留（孤儿）
        return False
    if msg.role == "user":    # user 是 Turn 起点
        return True
    if msg.tool_calls:        # assistant 带 tool_calls = 调用单元起点
        return True
    # 纯文本 assistant：前面不能是 tool（否则切断 tool 单元）
    if index > 0 and messages[index - 1].role == "tool":
        return False
    return True
```

   - 切割点找到后，判断是否为 **split turn**：切割点往前找最近的 `user` 消息作为 Turn 起点，如果切割点 ≠ Turn 起点 → split turn → 双摘要。

```
消息序列（从旧到新）：
[user: "读文件"] [assistant: tool_call(read)] [tool: file content] [assistant: "文件内容是..."]
←─── Turn 起点                               ↑ 切割点落在这里 → split turn!
```

- [x] **双摘要（全量 `SUMMARIZATION_PROMPT` + 增量 `UPDATE_SUMMARIZATION_PROMPT`）**

  两种 prompt 模板，覆盖两种场景：

  - **全量摘要**（`SUMMARIZATION_PROMPT`）：首次压缩，没有旧摘要。要求 LLM 从零输出结构化摘要：Goal → Constraints → Progress（Done / In Progress / Blocked）→ Key Decisions → Next Steps → Critical Context。
  - **增量摘要**（`UPDATE_SUMMARIZATION_PROMPT`）：第 2+ 次压缩，已有旧摘要。把旧摘要文本通过 `{previous_summary}` 注入 prompt，要求 LLM 在旧摘要基础上追加新进展、更新状态、保留原有关键信息。

  调用时的关键逻辑——检测是否有旧摘要：
```python
def _extract_previous_summary(messages, cut_index):
    """从待压缩消息中扫描 [上下文摘要] 标记，提取旧摘要文本。"""
    for i in range(cut_index - 1, -1, -1):
        if messages[i].content.startswith("[上下文摘要]"):
            summary = messages[i].content.replace("[上下文摘要]", "", 1).strip()
            return summary, i   # 返回旧摘要文本 + 索引位置
    return None, 0              # 无旧摘要 → 全量模式
```

  有了旧摘要后，只把**摘要之后的新消息**传给 LLM（`history_start = prev_summary_idx + 1`），而不是重读全部历史：
```python
# 增量模式：只传新消息 + 旧摘要
previous_summary, prev_idx = _extract_previous_summary(non_system, cut_index)
history_start = prev_idx + 1 if previous_summary else 0
new_messages = non_system[history_start:cut_index]  # 只取新消息

summary = await generate_summary(
    client, new_messages,
    previous_summary=previous_summary,  # 非 None → 走 UPDATE_SUMMARIZATION_PROMPT
)
```

  - **Turn 前缀摘要**（`TURN_PREFIX_SUMMARIZATION_PROMPT`）：split turn 时的特有产物。当一个 Turn 被切分成两半——前半段被压缩、后半段保留——Turn 前缀摘要负责把前半段的"上下文"传递给后半段。
```
Split Turn 示意图：
┌──────────────────────────────────────────────────────┐
│  被压缩部分                     │  保留部分          │
│                                  │                    │
│  [user: 任务请求]                │  [assistant: 方案] │
│  [assistant: tool_call(read)]   │  [assistant: 修改] │
│  [tool: file content]           │  ← 后半段需要知道  │
│                                  │    前半段读到什么  │
│  ← 生成 Turn 前缀摘要 ──────────→│                    │
│  "用户请求读 X 文件，文件内容    │                    │
│   显示 Y 配置有问题..."          │                    │
└──────────────────────────────────────────────────────┘
```

- [x] **文件操作追踪（`fileops.py`）**
  - 从被压缩消息中的所有 `tool_calls` 提取 `read` / `grep` / `find` / `ls`（读取类）和 `write` / `edit` / `bash`（修改类）操作的文件路径。
  - 把文件列表格式化成 Markdown 区块，追加到摘下面，帮助 LLM 知道"历史上读过/改过哪些文件"。
```python
def extract_file_operations(messages):
    reads, modified = set(), set()
    for msg in messages:
        if msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc["function"]["name"]
                path = json.loads(tc["function"]["arguments"]).get("path", "")
                if name in {"read", "grep", "find", "ls"}:
                    reads.add(path)
                elif name in {"write", "edit", "bash"}:
                    modified.add(path)
    return {"read_files": sorted(reads), "modified_files": sorted(modified)}
```

  最终摘要消息格式：
```markdown
[上下文摘要]

## Goal
修复 config.py 中的数据库连接错误

## Progress
### Done
- [x] 读取 config.py，发现 db_host 指向 localhost:3306
...

## Files Read
- /home/user/project/config.py
- /home/user/project/db.py

## Files Modified
- /home/user/project/config.py
```

- [x] **Turn 边界保护（不破坏 tool-call 配对）**
  - 切割点的合法性检查（`is_valid_cut_point`）是整个压缩系统的"约束基石"——如果切错了位置，LLM 收到的上下文里就会出现孤立的 `tool` 消息（没有对应的 `assistant` tool_call），导致协议错误。
  - 规则总结为一句话：**只能切在"对话的自然边界"上**——要么是 Turn 起点（user），要么是 tool 调用单元的起点（assistant + tool_calls），绝对不能切在 tool 调用链条的中间。


### 坑

**1. 增量摘要的"接上"问题——最该补的坑**

`UPDATE_SUMMARIZATION_PROMPT` 和 `generate_summary(previous_summary=...)` 参数从一开始就写好了，但 `compress_if_needed()` 调用时永远传 `None`——导致每次压缩都全量重写旧摘要。长对话被压缩 5 次 = 5 次全量重写，每次都要把历史摘要当输入再读一遍，既浪费 token 又可能丢失细节。

修复的核心是 `_extract_previous_summary()`——向前扫描 `[上下文摘要]` 标记消息，提取旧文本，然后只把**摘要之后的新消息**传给 LLM：
```python
# 修复前：永远全量重写
history_summary = await generate_summary(client, history_messages, previous_summary=None)

# 修复后：检测旧摘要，增量更新
previous_summary, prev_idx = _extract_previous_summary(non_system, cut_index)
history_start = prev_idx + 1 if previous_summary else 0
history_summary = await generate_summary(
    client,
    non_system[history_start:cut_index],  # 只传新消息
    previous_summary=previous_summary,    # 旧摘要 → UPDATE_SUMMARIZATION_PROMPT
)
```

> 坑的本质：增量摘要的"接口"写好了，但"调用"没接上。这是典型的"代码到位、逻辑未通"——prompt 模板和参数定义都是对的，缺少的只是从消息列表中"发现"旧摘要的那一步。**接口的正确性不等于功能的完整性**。

**2. 切割点"向后搜索"的必要性**

从后往前累加 token 找到位置 i 后，不能直接切在 i。因为 i 可能是不合法的切割点（比如切在 tool 消息上）。必须从 i **向后**搜索第一个合法点——这会扩大保留范围，但保证了消息的完整性。
```python
for i in range(len(messages) - 1, -1, -1):
    accumulated += estimate_message_tokens(messages[i])
    if accumulated >= keep_recent_tokens:
        for j in range(i, len(messages)):    # 向后找合法点
            if is_valid_cut_point(messages, j):
                cut_index = j
                break
        break
```

> 坑的本质："找到阈值位置"和"找到合法切割点"是两个步骤。向后搜索意味着保留的消息比预算多——这是有意为之的 trade-off：**宁可多用一点 token 保留完整消息，也不能为了精确预算切断 tool 调用链**。

**3. Split Turn 时双摘要的拼接逻辑**

当切割点落在 Turn 中间时，需要两份摘要：
- 历史摘要：所有完整 Turn 的压缩（切割点之前 + Turn 起点之前的历史）
- Turn 前缀摘要：当前 Turn 被切掉的前半段

但这两份摘要的拼接不是简单 `+`——合并后的消息里需要明确区分"历史上下文"和"当前 Turn 的前半段"，否则 LLM 会困惑。
```python
def merge_summaries(history_summary, turn_prefix_summary):
    return f"""{history_summary}

---

**Turn Context (split turn):**

{turn_prefix_summary}"""
```

> 坑的本质：Split turn 是压缩的"边界情况"，但长对话里大概率出现。双摘要的核心是**用格式区分上下文层次**——历史是"背景"，Turn 前缀是"当前任务的延续"。混淆层次会导致 LLM 把历史当当前任务执行。

**4. 压缩摘要消息的"身份识别"**

压缩摘要在消息列表中是特殊的——它不是用户发的，也不是 LLM 的正常回复，而是系统自动注入的。`[上下文摘要]` 前缀就是为了让它可被识别：
- `_is_compaction_summary()` 用它判断是否为摘要消息
- `_extract_previous_summary()` 用它找到旧摘要
- 它还是"增量 vs 全量"决策的分水岭

> 坑的本质：在消息流中注入"元消息"（摘要），必须给它打上可识别的标记。否则后续的压缩逻辑不知道哪些消息是摘要、哪些是原始对话，增量更新就无从谈起。**标记是元数据的载体**。


### 反思

上下文管理是工程化 Agent **最难的环节**——不是因为代码复杂（总共 ~550 行），而是因为**策略选择直接影响长对话的质量和成本**。

1. **预算驱动 vs 轮次驱动**：Pi 用的是绝对 token 预算（`reserve_tokens` / `keep_recent_tokens`），而不是"保留最近 N 轮"。轮次驱动的问题在于——每轮长度差异巨大（短到 10 token 的"继续"，长到 5000 token 的代码 review），固定轮次要么浪费空间，要么撑爆窗口。**token 预算对模型窗口更精准**。

2. **切割点的"向后搜索"是一种 trade-off**：找到阈值位置后向后找合法点，意味着保留的消息超过预算。这是对"消息完整性"的妥协——**多花一点 token 保住 tool-call 配对，比精确预算但产生孤儿消息强一万倍**。

3. **增量摘要的"接口 + 调用"两层设计**：`UPDATE_SUMMARIZATION_PROMPT` 是接口层（"怎么增量更新"），`_extract_previous_summary` 是调用层（"什么时候走增量"）。接口写对了不等于功能完成了——**代码到位和逻辑贯通之间，差一个"发现旧摘要"的步骤**。这是最容易忽视的工程细节。

4. **Split turn 是对"原子性"的妥协**：理想情况下压缩应该切在 Turn 边界，但长 Turn 里可能一个 user 请求后 LLM 读了 20 个文件——20 轮 tool-call 可能远超 `keep_recent_tokens`。Split turn 就是承认"一个 Turn 太长了，必须切在中间"。双摘要用格式区分上下文层次，是对这种妥协的补偿。

5. **文件追踪是"长期记忆"的第一块积木**：`fileops.py` 从历史 tool_calls 中提取 read/modified 文件列表，让摘要不仅包含"说了什么"，还包含"碰了什么文件"。这块信息在摘要之外独立存在——即使摘要丢失了文件路径（LLM 摘要不可控），文件列表仍然保留。**双重编码（摘要 + 结构化列表）比单靠 LLM 摘要可靠**。

计算机类比在这里特别贴切：
- **Context = RAM**：近期对话是工作内存，快速但有限。
- **Compaction = 虚拟内存**：旧对话被"换出"成摘要，需要时通过摘要消息"换回"。
- **增量摘要 = 增量 checkpoint**：不是每次都 dump 全部 RAM 到磁盘，而是只写 dirty pages。
- **文件追踪 = page table**：维护"哪些文件被碰过"的元数据，独立于内容本身。

上下文管理的本质就是一句话：**用有限的空间记住无限长的对话**。压缩策略（全量 vs 增量）、切割点选择（turn 边界 vs split turn）、元数据保留（文件追踪），都是这个目标的具体实现。

---

## 模块 5：会话持久化（`session.py`，307 行）

### 问题
如何保存和恢复对话？

**难点**：Agent 的一次对话可能持续数小时、跨越多个 session。今天写的代码，明天回来要继续——你不能每次都从零开始。所以必须把对话**持久化到磁盘**，下次启动时**恢复**。但持久化不只是"把消息存成文件"——它涉及三个更深层的问题：

1. **结构问题**：对话不是一条直线。用户可能随时 `/save` 一个检查点，然后从那个点分叉出不同的探索方向。"保存"的不是一个文件，而是一棵**会话树**。
2. **性能问题**：如果每次 `save()` 都全量重写整个文件（可能几百 KB），频繁保存会很慢。有没有办法只追加新数据？
3. **一致性问题**：如果程序在写入过程中崩溃（比如写了一半断电），文件就会损坏。如何保证**原子写入**？

**模块定位**：会话持久化是 Agent 的"磁盘"（Device）。它对应计算机类比中的最底层——RAM 的东西最终要落到磁盘。没有它，Agent 就是"鱼的记忆"——每次启动都是全新开始。


### 解法

- [x] **JSONL 存储 + 全量载入内存**
  - 每行一个 JSON 对象（一个会话节点），方便追加和逐行解析。
  - 启动时全量载入到 `dict[str, SessionNode]`，O(1) 查找。
  - 退出时写回。

```python
class SessionStore:
    def __init__(self, filepath="~/.pi-agent/sessions.jsonl"):
        self._nodes: dict[str, SessionNode] = {}   # 内存索引
        self._loaded = False
        self._dirty = False
        self._new_ids: set[str] = set()             # 新增节点
        self._modified_ids: set[str] = set()         # 修改节点

    def load(self):
        """启动时全量载入。"""
        with open(self.filepath) as f:
            for line in f:
                data = json.loads(line.strip())
                node = SessionNode(id=data["id"], parent_id=data.get("parent_id"),
                                   messages=data["messages"], bookmark=data.get("bookmark"))
                self._nodes[node.id] = node
        self._loaded = True
```

- [x] **会话树（`parent_id` 链表）**
  - 每个节点通过 `parent_id` 指向父节点，形成树状结构。
  - 分叉：同一个父节点可以创建多个子节点 → 用户可以从检查点探索不同方向。
  - 查询：`get_branch(node_id)` 从叶子回溯到根，`get_children(parent_id)` 列出所有分叉。

```
会话树示意图：
                    ┌──────────────┐
                    │  root (abc1) │  ← 第一次对话
                    │  "帮我写个API"│
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  node (abc2) │  ← 用户 /save checkpoint1
                    │  "API完成"   │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
     ┌────────▼───┐  ┌─────▼──────┐  ┌─▼──────────┐
     │ node (abc3)│  │ node (abc4)│  │ node (abc5) │  ← 三个分叉
     │ 添加缓存    │  │ 改为异步   │  │ 加单元测试  │
     └────────────┘  └────────────┘  └────────────┘
```

- [x] **书签（恢复点）**
  - 用户可以给任意节点打书签：`store.bookmark_node(node_id, "before-refactor")`
  - 恢复时按书签查找：`store.get_by_bookmark("before-refactor")`
  - 书签是"人类可读的恢复点"——比记忆 `abc123def456` 友好得多。

- [x] **原子写入 + append-only 优化**
  - 两种写入策略，自动选择：
    - **仅新增节点**（`_new_ids` 不为空，`_modified_ids` 为空）→ **append-only**：只把新行追加到文件末尾，O(新节点数)，零重写开销。
    - **有修改节点**（`_modified_ids` 不为空）→ **全量原子重写**：写入临时文件 → `os.replace(tmp, target)`。`os.replace` 在 POSIX 上是原子操作，要么全部成功，要么原文件不变。

```python
def save(self):
    if self._modified_ids:
        self._full_rewrite()        # 全量原子重写
    elif self._new_ids:
        self._append_new()          # 快速追加

def _full_rewrite(self):
    """写入临时文件 → 原子 rename，保证不损坏原文件。"""
    fd, tmp = tempfile.mkstemp(dir=str(self.filepath.parent), prefix='.sessions_')
    with os.fdopen(fd, 'w') as f:
        for node in self._nodes.values():
            f.write(json.dumps({...}) + '\n')
    os.replace(tmp, str(self.filepath))  # 原子替换
```

- [x] **消息序列化/反序列化**
  - `Message` ↔ `dict` 互转：`_message_to_dict()` / `_dict_to_message()`。
  - 会话节点存储的是 `list[dict]`（JSON 兼容），恢复时还原为 `list[Message]`。


### 坑

**1. 树 vs 线：何时需要分支？**

JSONL 每行一个节点 + `parent_id` 链接 = 天然支持树。但"树"带来了复杂度：
- 恢复时，你恢复的是**一个节点**（一条路径），而不是整棵树。
- 分叉的子节点存在但不影响当前路径——这既是优点（隔离），也可能导致困惑（"为什么我找不到刚才的修改？因为那是另一个分叉"）。

> 坑的本质：树结构是"探索"的建模——用户可以回溯到检查点、尝试不同方案。但**大多数对话不需要分叉**——用户只是在一条直线上推进。树是能力，但线性是默认。**默认线性，需要时分支**，是最务实的策略。

**2. 恢复时的状态重建（压缩器状态、token 计数）**

当前实现只恢复了 `messages` 列表，但 Agent 还有运行时状态：
- `ContextCompactor._last_stats`：压缩统计数据
- `Agent._turn_count`：当前轮次计数（恢复后从 0 开始）
- `Agent._steer_queue` / `_cancel_event`：运行时控制信号

这些状态不持久化——恢复后的 Agent 是"部分重建"的。压缩器状态丢失意味着恢复后第一次压缩可能不如预期精确（因为没有上一次的统计参考）。

> 坑的本质：持久化的粒度决定恢复的"完整度"。当前是"消息级持久化"（只存 messages），不是"状态级持久化"（存所有运行时状态）。如果要完整恢复，需要把 `Agent` 全部状态序列化——复杂度会翻倍，但完整度也会翻倍。**这是一个 trade-off：简单 vs 完整**。

**3. 空写保护：`ensure_loaded()` 的必要性**

如果用户直接调用 `save()` 而没先 `load()`，会导致**空写覆盖**——用空 `_nodes` 覆盖已有的 JSONL 文件。解决：
```python
def save(self):
    if not self._loaded:
        raise RuntimeError("SessionStore 尚未加载数据，禁止 save()")
```

> 坑的本质：持久化操作需要**显式的状态机**——"已加载"是一个前提条件。不检查状态就写入，后果是灾难性的。**防御性编程在 I/O 操作中不是可选项，是必须项**。

**4. append-only 的"脏数据"风险**

如果同一个节点被 append 了两次（比如 `create_node` 后没有清 `_new_ids`，再次 `save()` 会重复追加），JSONL 文件就会出现重复行。下次 `load()` 时，后面的行会覆盖前面的——虽然结果没问题，但文件会膨胀。

> 坑的本质：Append-only 性能好但有状态依赖——必须精确跟踪"哪些已经写过"。`_new_ids` 和 `_modified_ids` 的清理时机（`save()` 结束时）是关键。清早清晚都会出问题。


### 反思

会话持久化是 Agent 的"三层记忆"中最持久的那层：

1. **短期记忆 = `agent.messages`**（RAM）：当前对话的完整消息列表，随程序退出而消失。
2. **中期记忆 = 会话树**（Disk）：JSONL 持久化的对话历史，跨 session 存在。可以分叉、回溯、恢复。
3. **长期记忆 = 文件追踪 + 摘要**（档案）：`fileops.py` 提取的文件操作记录和 compaction 摘要——不依赖具体对话，可以在完全不同的 session 中被引用。

设计哲学：
- **JSONL > SQLite**：对于"几百个节点、偶尔查询"的场景，JSONL + 全量载入内存的简单性远胜于 SQLite 的复杂度。不需要 SQL 查询、不需要 ORM、不需要迁移脚本。**简单场景用简单方案**。
- **原子写入 > 直接写入**：`tempfile + os.replace` 是经典 POSIX 原子写入模式。成本多一次文件复制，收益是"永远不会损坏原文件"。**I/O 安全比 I/O 性能重要**。
- **append-only 是"常见路径"优化**：大多数 save 操作只需要追加新节点（用户正常对话 → 退出 → 保存），而不需要修改已有节点。只优化常见路径，不优化罕见路径——**二八定律在工程优化中同样适用**。


---

## 模块 6：CLI/Harness（`cli.py`，449 行）

### 问题
如何把 Agent 变成可用产品？

**难点**：前面 5 个模块把 Agent 的内核搭好了——LLM 客户端、工具系统、循环、上下文管理、会话持久化。但用户不可能每次都写 Python 脚本来调用 Agent。需要一个**"外壳"（Harness）** 把内核包起来，提供命令行入口、配置管理、交互体验。

Harness 的核心挑战是**"多场景适配"**：
1. **一次性 vs 持续对话**：只问一个问题就走，还是持续交互？
2. **配置来源的优先级**：用户可能从 CLI 参数、环境变量、配置文件三个地方提供 API Key / model / base_url，怎么合并？
3. **中断恢复**：上次对话中断了，怎么快速恢复？

**模块定位**：Harness 是 Agent 的"操作系统"——它负责启动、配置、调度 Agent，提供用户界面。Agent 内核是"引擎"，Harness 是"方向盘 + 仪表盘"。


### 解法

- [x] **typer 命令（`chat` + `resume`）**
  - `pi-agent chat`：启动新对话。可带 `prompt` 参数做一次性问答，省略则进入交互模式。
  - `pi-agent resume <bookmark>`：按书签恢复之前的对话，从保存点继续。

```python
@app.command()
def chat(
    prompt: Optional[str] = typer.Argument(None),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k"),
    base_url: Optional[str] = typer.Option(None, "--base-url", "-b"),
    max_turns: Optional[int] = typer.Option(None, "--max-turns"),
    no_confirm: bool = typer.Option(False, "--no-confirm"),
):
    ...
```

- [x] **配置优先级（CLI > 配置文件 > 环境变量）**

  三个来源，严格排序。`resolve_kwargs()` 用 `or` 链实现递推 fallback：

```python
def resolve_kwargs(cli_kwargs):
    config = load_config()  # 从 ~/.pi-agent/config.yaml 加载
    return {
        "api_key":  cli_kwargs["api_key"]
                    or os.environ.get("OPENAI_API_KEY")
                    or config.get("api_key"),
        "base_url": cli_kwargs["base_url"]
                    or os.environ.get("OPENAI_BASE_URL")
                    or config.get("base_url")
                    or "https://api.deepseek.com",
        "model":    cli_kwargs["model"]
                    or config.get("model")
                    or "deepseek-chat",
        ...
    }
```

  优先级链条：
```
CLI 参数  >  环境变量  >  配置文件 (~/.pi-agent/config.yaml)  >  硬编码默认值
  --model      OPENAI_        model: deepseek-chat            "deepseek-chat"
  deepseek-v4  API_KEY        base_url: https://...
```

- [x] **交互 / 非交互双模式**
  - **非交互**（`chat "帮我写个函数"`）：一句话进去，结果出来，自动保存，退出。适合脚本/CI。
  - **交互**（`chat` 不带参数）：进入 REPL 循环，持续对话，支持元命令。适合开发调试。

```python
if prompt:
    asyncio.run(_run_non_interactive(agent, store, prompt))
else:
    asyncio.run(_run_interactive(agent, store))
```

- [x] **元命令系统（`/save` `/exit` `/stats` `/bookmarks` `/help`）**

  交互模式下，以 `/` 开头的输入被解释为元命令而不是发给 LLM：

```
/exit        保存当前会话并退出
/save <name> 打书签（不创建新节点，只给当前节点命名）
/bookmarks   列出所有书签（★ 标记当前节点）
/stats       查看上下文统计（轮次、消息数、token 使用率）
/help        显示帮助信息
```

  元命令的实现：在 `_run_interactive` 的主循环中，先检查 `user_input.startswith("/")`，匹配命令后 `continue` 跳过 LLM 调用。

- [x] **Rich 终端渲染**
  - 启动信息：`Panel.fit()` 展示模型、API 地址、最大轮次、最近书签。
  - Agent 输出：`Markdown()` 渲染代码块高亮。
  - 工具结果：`Panel(body, title=f"tool: {name}")` 彩色面板区分成功/失败。
  - 恢复提示：显示书签名和对应的节点 ID。

- [x] **`resume` 的智能恢复**
  - 先按书签名查找，再按节点 ID 查找——书签是人类友好的名字，优先匹配。
  - 从节点 `metadata` 恢复运行参数（model、max_turns），但如果 CLI 显式传入则覆盖。
  - 恢复后直接进入交互模式，用户不用重新配置。


### 坑

**1. 配置优先级的 `or` 链陷阱**

`or` 链实现 fallback 看起来很优雅，但有一个坑——**空字符串和 None 的处理**。如果环境变量 `OPENAI_API_KEY=""`（被设了但为空），`or` 会把它当 falsy 跳过。但如果用户传 `--api-key ""`（typer 给的是空字符串），也会被跳过——用户可能以为自己在清空 key，实际走了 fallback。

> 坑的本质：`or` 链分不清"用户没传"和"用户传了空值"。对于 API Key 这种"必须有值"的字段还好，但对于"允许为空"的字段（如 `base_url`），空字符串可能是有意为之。Python 的 `or` 是简洁但不精确的 fallback 方案——**如果需要区分"未设置"和"设为空"，应该用 `is None` 判断**。

**2. 交互循环的异常处理**

`_run_interactive` 里有一个 `try/except` 包裹 `agent.run()`：
```python
try:
    result = await agent.run(user_input)
except Exception as exc:
    console.print(f"[red]✗ 运行错误: {exc}[/red]")
    continue
```
但如果 `agent.run()` 内部抛了未捕获的异常（比如 LLM API 挂了），这个 `except` 会兜底——但 Agent 的内部状态（`messages`、`_turn_count`）可能已经部分修改。下次 `run()` 会从半截状态开始。

> 坑的本质：异常不崩溃 = 交互体验好（用户不会丢失对话），但状态一致性 = 可能被破坏。当前实现选前者——**"容错优先于一致性"**，对于交互式工具是合理的选择，但对于生产级 Agent 需要更细粒度的状态回滚。

**3. `resume` 的 partial 状态恢复**

恢复时只恢复了 `agent.messages` 和部分配置，但以下状态丢失：
- 压缩器状态（`compressor._last_stats`）
- 轮次计数（从 0 重新开始）
- 当前 `current_node_id` 绑定关系（如果是新对话恢复旧节点）

这些丢失导致恢复后的行为与中断前不完全一致——比如压缩阈值判断可能不准（因为上次压缩统计丢失了）。

> 坑的本质：恢复的"完整度"是一个 sliding scale——从"只恢复消息"（最简）到"完整快照恢复"（最全）。当前实现选择了最简方案，因为完整快照需要序列化 Agent 全部内部状态，且不同版本的 Agent 快照不兼容。**"恢复 80% 的状态"比"恢复 100% 但版本不兼容"更实用**。

**4. 配置文件的位置和格式**

支持两个位置：`./config.yaml`（项目级）和 `~/.pi-agent/config.yaml`（用户级）。当前实现用 `config.update()` 合并，后加载的覆盖先加载的（即用户级覆盖项目级）。但 YAML 的嵌套结构可能产生意料之外的合并结果。

```yaml
# ~/.pi-agent/config.yaml
api_key: sk-xxx
model: deepseek-chat
base_url: https://api.deepseek.com
max_turns: 50
```

> 坑的本质：配置文件简单时一切 OK，但一旦支持嵌套配置（如 `compaction.reserve_tokens: 8192`），`dict.update` 的浅合并就会出问题。**浅合并对平面配置够用，嵌套配置需要 deep merge**。


### 反思

Harness 是 Agent 的"最后一公里"——内核写得再好，没有好用的 CLI，用户也不会用。几个设计哲学：

1. **双模式 = 双用户**：一次性模式给脚本/CI（可组合、可自动化），交互模式给人（可探索、可中断）。两个模式的代码共用同一套 Agent 初始化逻辑（`_build_agent_and_store`），只是在"输入循环"上分叉。**共用核心、分叉界面**。

2. **配置优先级是"用户自由度"的体现**：CLI 覆盖最多（每次调用可不同），配置文件覆盖最少（全局默认）。优先级链条越长，用户自由度越大，但调试也越难——"到底哪个值生效了？"是配置优先级的经典问题。当前实现没有 `--show-config` 命令来打印最终生效的配置，是个 UX 缺口。

3. **元命令 vs 自然语言**：`/` 前缀是明确的"命令 vs 对话"分界。为什么不直接用自然语言（如"请保存会话"）？因为：
   - 确定性：`/save` 100% 触发保存，自然语言可能被 LLM 误解。
   - 效率：一个字符 `/` 就能区分，不需要额外往返 LLM。
   - 隐私：元命令不经过 LLM（不消耗 token、不泄露对话）。

4. **Harness 是"可以替换的外壳"**：同样的 Agent 内核，可以换不同的 Harness——CLI、Web UI、IDE 插件、API 服务。Harness 决定了"怎么交互"，Agent 决定了"能做什么"。**解耦 Harness 和 Agent，是产品化的第一步**。

5. **`resume` 是"人机信任"的关键功能**：用户花 30 分钟和 Agent 调试一个 bug，中途退出或崩溃，下次回来能无缝继续——这比任何新功能都能建立信任。**中断恢复不是"nice to have"，是"must have"**。

---

## 结尾：从 Pi Agent 泛化到任何 Agent

### 1. Agent 通用骨架

> 循环 + 工具 + 上下文管理 + 记忆 + 控制

### 2. 可替换部件（泛化的关键）

| 部件 | Pi 的实现 | 其他变体 |
|:---|:---|:---|
| 决策范式 | ReAct | Plan-then-Execute / Reflexion |
| 工具协议 | 本地函数 | MCP |
| 上下文压缩 | 双摘要 | 5 层渐进管道 / RAG |
| 记忆 | 会话树 | 向量库 |
| 多 Agent | 单 Agent | 子 Agent 委派 / 编排 |

### 3. 计算机类比（点睛之笔）
> 把 Agent 类比成一台计算机：LLM + Tools 是 CPU，Context 是 RAM，
> Compaction 是虚拟内存，Session 是磁盘，CLI 是操作系统。
> 把这些部件"固定"成一个系统，就能跑不同的 APP——写代码、查资料、分析数据。
> **换 APP 不改骨架，这就是泛化到任何 Agent 的本质。**

### 4. 下一步预告
- 子 Agent 委派（spawn 独立上下文的子 Agent）
- 元命令系统（`/compact`、`/status`、`/undo` 由 Agent 层处理）
- MCP 工具协议支持（替代本地函数工具）



