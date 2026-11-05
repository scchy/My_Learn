
> # 安装HelloAgents框架（第10章版本）pip install "hello-agents[protocol]==0.2.2"


## 10.1 智能体通信协议基础


## 10.2 MCP
其实MCP本质还是function call 的"生态标准化协议"， 最终都是以 {... tools: [{...}, {...}]} 去请求 LLM API  
- 最大的优势是执行环境隔离 和 独立部署-可热更新 （MCP更新, 用户无需重启agent）


核心功能
1. Tools: 主动的（执行操作）
2. Resource: 被动的 （提供数据）
3. Prompts: 指导性的 （提供模板）


### 使用 MCP 客户端

1- 连接到MCP服务器
```python
import asyncio 
from hello_agents.protocols import MCPClient

async def connect_to_server():
    # 方式1：连接到社区提供的文件系统服务器
    # npx会自动下载并运行@modelcontextprotocol/server-filesystem包
    client = MCPClient([
        "npx", "-y",
        "@modelcontextprotocol/server-filesystem",
        "."  # 指定根目录
    ])

    # 使用async with确保连接正确关闭
    async with client:
        # 在这里使用client
        tools = await client.list_tools()
        print(f"可用工具: {[t['name'] for t in tools]}")

    # 方式2：连接到自定义的Python MCP服务器
    client = MCPClient(["python", "my_mcp_server.py"])
    async with client:
        # 使用client...
        pass

# 运行异步函数
asyncio.run(connect_to_server())

```

2- 发现可用工具 
连接成功后，第一步通常是查询服务器提供了哪些工具
```python

async def discover_tools():
    client = MCPClient(["npx", "-y", "@modelcontextprotocol/server-filesystem", "."])

    async with client:
        # 获取所有可用工具
        tools = await client.list_tools()

        print(f"服务器提供了 {len(tools)} 个工具：")
        for tool in tools:
            print(f"\n工具名称: {tool['name']}")
            print(f"描述: {tool.get('description', '无描述')}")

            # 打印参数信息
            if 'inputSchema' in tool:
                schema = tool['inputSchema']
                if 'properties' in schema:
                    print("参数:")
                    for param_name, param_info in schema['properties'].items():
                        param_type = param_info.get('type', 'any')
                        param_desc = param_info.get('description', '')
                        print(f"  - {param_name} ({param_type}): {param_desc}")

asyncio.run(discover_tools())
```
3- 调用工具
调用工具时，只需提供工具名称和符合JSON Schema的参数

```python
async def safe_tool_call():
    client = MCPClient(["npx", "-y", "@modelcontextprotocol/server-filesystem", "."])

    async with client:
        try:
            # 尝试读取可能不存在的文件
            result = await client.call_tool("read_file", {"path": "nonexistent.txt"})
            print(result)
        except Exception as e:
            print(f"工具调用失败: {e}")
            # 可以选择重试、使用默认值或向用户报告错误

asyncio.run(safe_tool_call())
```

4- 访问资源
除了工具，MCP 服务器还可以提供资源（Resources）

```python
# 列出可用资源
resources = client.list_resources()
print(f"可用资源：{[r['uri'] for r in resources]}")

# 读取资源
resource_content = client.read_resource("file:///path/to/resource")
print(f"资源内容：{resource_content}")
```

5- 使用提示模板

```python
# 列出可用提示
prompts = client.list_prompts()
print(f"可用提示：{[p['name'] for p in prompts]}")

# 获取提示内容
prompt = client.get_prompt("code_review", {"language": "python"})
print(f"提示内容：{prompt}")
```

6- 完整示例：使用 GitHub MCP 服务

```python
"""
GitHub MCP 服务示例

注意：需要设置环境变量
    Windows: $env:GITHUB_PERSONAL_ACCESS_TOKEN="your_token_here"
    Linux/macOS: export GITHUB_PERSONAL_ACCESS_TOKEN="your_token_here"
"""

from hello_agents.tools import MCPTool

# 创建 GitHub MCP 工具
github_tool = MCPTool(
    server_command=["npx", "-y", "@modelcontextprotocol/server-github"]
)

# 1. 列出可用工具
print("📋 可用工具：")
result = github_tool.run({"action": "list_tools"})
print(result)

# 2. 搜索仓库
print("\n🔍 搜索仓库：")
result = github_tool.run({
    "action": "call_tool",
    "tool_name": "search_repositories",
    "arguments": {
        "query": "AI agents language:python",
        "page": 1,
        "perPage": 3
    }
})
print(result)

```
### MCP 传输方式详解

> 本地 → stdio，远程/生产 → StreamableHTTP（新版为无状态模式）
>
> 
| 传输方式 | 适用场景 | 优点 | 缺点 |
| --- | --- | --- | --- |
| Memory | 单元测试、快速原型 | 最快、无网络开销 | 仅限同进程 |
| Stdio | 本地开发、命令行工具 | 简单、无需网络配置 | 仅限本地、可能有兼容性问题 |
| HTTP | 生产环境、远程服务 | 通用、防火墙友好 | 无流式支持、延迟较高 |
| SSE | 实时通信、流式响应 | 支持服务器推送 | 单向通信、需要HTTP服务器 |
| StreamableHTTP | 流式HTTP通信 | 双向流式、HTTP兼容 | 需要特定服务器支持 |

```python
from hello_agents.tools import MCPTool

# 1. Memory Transport - 内存传输（用于测试）
# 不指定任何参数，使用内置演示服务器
mcp_tool = MCPTool()

# 2. Stdio Transport - 标准输入输出传输（本地开发）
# 使用命令列表启动本地服务器
mcp_tool = MCPTool(server_command=["python", "examples/mcp_example_server.py"])

# 3. Stdio Transport with Args - 带参数的命令传输
# 可以传递额外参数
mcp_tool = MCPTool(server_command=["python", "examples/mcp_example_server.py", "--debug"])

# 4. Stdio Transport - 社区服务器（npx方式）
# 使用npx启动社区MCP服务器
mcp_tool = MCPTool(server_command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "."])

# 5. HTTP/SSE/StreamableHTTP Transport
# 注意：MCPTool主要用于Stdio和Memory传输
# 对于HTTP/SSE等远程传输，建议直接使用MCPClient

import asyncio
from hello_agents.protocols import MCPClient

async def test_sse_transport():
    # 连接到 SSE MCP 服务器
    client = MCPClient(
        "http://localhost:8080/sse",
        transport_type="sse"
    )

    async with client:
        # SSE 特别适合流式处理
        result = await client.call_tool("stream_process", {
            "input": "大量数据处理请求",
            "stream": True
        })
        print(f"流式处理结果: {result}")

# 注意：需要支持 SSE 的 MCP 服务器
# asyncio.run(test_sse_transport())


# 注意：MCPTool 主要用于 Stdio 和 Memory 传输
# 对于 StreamableHTTP 传输，建议使用底层的 MCPClient

import asyncio
from hello_agents.protocols import MCPClient

async def test_streamable_http_transport():
    # 连接到 StreamableHTTP MCP 服务器
    client = MCPClient(
        "http://localhost:8080/mcp",
        transport_type="streamable_http"
    )

    async with client:
        # 支持双向流式通信
        tools = await client.list_tools()
        print(f"StreamableHTTP 服务器工具: {len(tools)} 个")

# 注意：需要支持 StreamableHTTP 的 MCP 服务器
# asyncio.run(test_streamable_http_transport())
```

### 实战案例：智能文档助手

[mcp_try](./mcp_try.py)

### MCP 社区生态
> https://hello-agents.datawhale.cc/#/./chapter10


| 服务器名称 | 功能 | NPM包名 | 使用场景 |
| --- | --- | --- | --- |
| filesystem | 文件系统访问 | @modelcontextprotocol/server-filesystem | 读写本地文件、目录操作 |
| github | GitHub API | @modelcontextprotocol/server-github | 搜索仓库、读取代码 |
| postgres | PostgreSQL数据库 | @modelcontextprotocol/server-postgres | 数据库查询、数据分析 |
| sqlite | SQLite数据库 | @modelcontextprotocol/server-sqlite | 轻量级数据库操作 |
| slack | Slack消息 | @modelcontextprotocol/server-slack | 发送消息、读取频道 |
| google-drive | Google Drive | @modelcontextprotocol/server-google-drive | 访问云端文件 |
| brave-search | Brave搜索 | @modelcontextprotocol/server-brave-search | 网页搜索、实时信息获取 |
| fetch | 网页抓取 | @modelcontextprotocol/server-fetch | 获取网页内容、提取数据 |


| 服务器名称 | 功能 | 包名/仓库 | 炫酷特性 |
| --- | --- | --- | --- |
| Playwright | 浏览器自动化 | @playwright/mcp | 自动化网页交互、截图、填表单 |
| Puppeteer | 浏览器控制 | mcp-server-puppeteer | 网页爬取、PDF生成 |
| Screenpipe | 屏幕录制 | mediar-ai/screenpipe | 本地屏幕/音频捕获、时间戳索引、语义搜索 |
| Obsidian | 笔记管理 | calclavia/mcp-obsidian | 读取和搜索Markdown笔记、知识库管理 |
| Notion | 协作文档 | Badhansen/notion-mcp | 管理待办事项、数据库操作 |
| Jira | 项目管理 | nguyenvanduocit/jira-mcp | Issue管理、Sprint规划、工作流 |
| Tavily | AI搜索 | kshern/mcp-tavily | 专为AI优化的搜索API |
| YouTube | 视频处理 | anaisbetts/mcp-youtube | 获取字幕、视频信息、转录内容 |
| Spotify | 音乐控制 | marcelmarais/Spotify | 播放控制、播放列表管理 |
| Wolfram Alpha | 计算知识 | ricocf/mcp-wolframalpha | 数学计算、科学数据、实时知识 |
| Sentry | 错误追踪 | getsentry/sentry-mcp | 错误监控、性能分析 |
| Grafana | 可视化监控 | grafana/mcp-grafana | 查询仪表板、数据源查询 |


「平台厂商做原生工具，长尾生态留给 MCP」——这是一个分工而非替代关系。对 Agent 产品方来说，核心路径（搜索、代码、文件、浏览器）必须自研以保质量和数据飞轮；对企业用户来说，MCP 是把内部系统接进 Agent 的最优解。社区服务器数量多（几千个），但头部产品 90% 的价值集中在它自研的那十几个工具上，这就是结构如此的原因。



## 10.3  A2A

> 实现智能体之间的点对点通信

### 10.3.1 协议设计动机

A2A 协议采用点对点（P2P）架构（网状拓扑），允许智能体直接通信，从根本上解决了上述问题。它的核心是任务（Task）和工件（Artifact）这两个抽象概念，这是它与 MCP 最大的区别 。

| 概念           | 说明         | 与MCP的区别       | 示例            |
| ------------ | ---------- | ------------- | ------------- |
| Task（任务）     | 智能体之间委托的单元 | 比Tool更高层次的抽象  | “撰写一篇关于AI的文章” |
| Artifact（工件） | 任务执行产生的结果  | 比Resource更结构化 | 文章文本、分析报告     |
| Message（消息）  | 智能体间的通信载体  | 包含任务状态信息      | “任务已完成50%”    |
| Part（部分）     | 消息的组成部分    | 支持多模态内容       | 文本、图片、文件      |
| Agent Card   | 智能体描述文档    | 类似MCP的工具描述    | JSON格式的能力声明   |

### 10.3.2 使用 A2A 协议实战

通过 A2A-SDK 来继承部分功能实现

> 实践建议：单 agent 用 MCP 就够了；只有当你的 agent 要对接别人写的、不同团队的、不同厂商的 agent 时才上 A2A——它不是死协议，而是"边界协议"，日常感知不到恰恰说明大部分系统还没到需要跨边界协作的成熟度。




## 10.4 ANP (Agent Network Protocol)

ANP协议专注于解决大规模、开放网络环境下的智能体管理问题。


### 10.4.1 协议目标

- 服务发现：单新任务达到时，如何快速找到能够处理该任务的智能体
- 智能路由：如果多个智能体都能处理同一任务，如何选择最合适的一个(如负载、成本等) 并向其分派任务？
- 动态扩展：如何让新加入网络的智能体被其他成员发现和调用？


| 概念            | 说明                       | 示例                                      |
| ------------- | ------------------------ | --------------------------------------- |
| ANP Discovery | 服务发现中心，用于注册和查询网络中的智能体服务。 | 一个中央服务器或一个P2P的DHT网络。                    |
| Service Info  | 描述智能体服务的信息，包括其能力、地址和元数据。 | `{"agent_id": "nlp-agent-01", ...}`     |
| ANP Network   | 对智能体网络的抽象，管理节点间的连接与通信。   | 整个智能体集群的拓扑视图。                           |
| Capability    | 描述智能体功能的能力标签，用于服务发现时的匹配。 | `"text_analysis"`, `"image_processing"` |
| Metadata      | 服务的动态或静态元数据，用于路由决策。      | 负载情况、服务价格、软件版本等。                        |


## 10.6 本章总结

协议定位：

MCP (Model Context Protocol): 作为智能体与工具之间的桥梁，提供统一的工具访问接口，适用于增强单个智能体的能力。
A2A (Agent-to-Agent Protocol): 作为智能体之间的对话系统，支持直接通信与任务协商，适用于小规模团队的紧密协作。
ANP (Agent Network Protocol): 作为智能体的“互联网”，提供服务发现、路由与负载均衡机制，适用于构建大规模、开放的智能体网络。


深入学习：
- 阅读 MCP 官方文档：https://modelcontextprotocol.io
- 阅读 A2A 官方文档：https://a2a-protocol.org/latest/
- 阅读 ANP 官方文档：https://agent-network-protocol.com/guide/










