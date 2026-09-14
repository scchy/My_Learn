# p_mem_and_rag.py 运行记录（Chap08 记忆与 RAG）

> 记录时间：2026-09-10（初版） / 2026-09-14（复跑验证 + 补记 RAG 段、gitignore）
> 目标：在本地 `LLM` conda 环境下跑通 `chap08/p_mem_and_rag.py`（记忆写入/检索 + 30 秒上手 RAG）
> 结论：**已跑通**，记忆与 RAG 全流程检索均正确召回（复跑 23s）。下面把踩到的坑、脚本改动和最终可用命令沉淀下来。

---

## 1. 环境基线

| 项 | 值 |
|---|---|
| Python | `/home/scc/anaconda3/envs/LLM/bin/python3`（3.10） |
| hello-agents | 0.2.9，editable 安装 → `/home/scc/sccWork/openProject/hello_agent_learn_version/HelloAgents`（**自改版 fork**，非 PyPI 官方版） |
| LLM | DeepSeek，`LLM_MODEL_ID=deepseek-v4-flash`（连通性已验证，`invoke` 正常返回） |
| Qdrant | 客户端 1.19.0；**服务端未启动**（`localhost:6333` 无监听） |
| Neo4j | 本机 `~/neo4j-local` 已在运行，`bolt://localhost:7687` 可连 |
| Embedding | `dashscope` SDK 1.25.15 可用；`sentence-transformers` 2.2.2 存在但 `transformers` 安装不完整（见坑 3） |
| 配置 | `chap08/.env` 已固化「嵌入式 Qdrant + DashScope embedding」，可直接裸跑（见第 2 节） |

> ⚠️ 关键：本 fork 与官方书里的 `hello-agents` API **不一致**，照抄书上的代码会报错，见第 4 节。

---

## 2. 运行命令（已固化，可直接裸跑）

配置已写进 `chap08/.env`，脚本会自动 `chdir` 到自己的目录，现在**从任意目录直接裸跑即可**（下面仍给出在 chap08 下的常规写法）：

```bash
cd /home/scc/sccWork/myGitHub/My_Learn/Hello-Agent/chap08
/home/scc/anaconda3/envs/LLM/bin/python3 p_mem_and_rag.py
```

`.env` 里固化的关键项：

```bash
# QDRANT_URL=http://localhost:6333   # 注释掉：让 url 为空才会回退嵌入式存储（坑 1）
QDRANT_COLLECTION=chapter08_demo
QDRANT_VECTOR_SIZE=1024              # 与 text-embedding-v3 维度一致（坑 2）

EMBED_MODEL_TYPE=dashscope
EMBED_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBED_API_KEY=${ALI_API_KEY}          # 复用 shell 环境变量，python-dotenv 支持 ${VAR} 展开
```

> 临时覆盖配置时，命令行写法仍然有效（`load_dotenv()` 默认 `override=False`，不会覆盖已有环境变量）：
>
> ```bash
> cd chap08 && EMBED_MODEL_TYPE=tfidf python3 p_mem_and_rag.py
> ```

要点：
- **目录由脚本自己锁定**：脚本开头已 `os.chdir(CHAP_DIR)`（且在 `import hello_agents` 之前——该库的 `core/database_config.py` 在导入时就会 `load_dotenv()`），所以从任意目录执行都会把 `.env` / `./memory_data/` / `./knowledge_base/` 落在 `chap08` 下；但**手工开 REPL / 直接 import 跑**时仍是相对 cwd 落盘（`./memory_data/memory.db` + `./memory_data/qdrant_local/`），换目录会在别处生成一份新的。
- **`EMBED_API_KEY` 依赖 shell 里已导出 `ALI_API_KEY`**；如果换到 IDE / 定时任务等没有该变量的环境，要么先 `export ALI_API_KEY=...`，要么把该行改成明文 key。
- **维度必须一致**：`.env` 的 `QDRANT_VECTOR_SIZE` 与 `EMBED_MODEL_NAME` 的维度要对齐，否则嵌入式 Qdrant 会因维度不匹配写入失败；对不上时删掉 `./memory_data/qdrant_local/` 重建，或直接换一个 `QDRANT_COLLECTION` 名。

### 预期输出（2026-09-14 复跑，节选）

```
=== 添加多个记忆 ===
记忆1: ✅ 记忆已添加 (ID: 1106a883...)
记忆2: ✅ 记忆已添加 (ID: 4eb5eb12...)

=== 检索记忆 ===
🔍 找到 1 条相关记忆:
1. [语义记忆] 用户张三是一名Python开发者，专注于机器学习和数据分析 (重要性: 0.80)

=== 搜索特定记忆 ===
🔍 找到 1 条相关记忆:
1. [语义记忆] 李四是前端工程师，擅长React和Vue.js开发 (重要性: 0.70)

=== 记忆统计 ===
📈 记忆系统统计
总记忆数: 2
启用的记忆类型: working, episodic, semantic
会话ID: session_20260914_103456
对话轮次: 0

=== 记忆摘要 ===
📊 记忆系统摘要
  • 语义记忆: 2 条 (平均重要性: 0.75)

=== 检索知识 ===
1. 文档: **./knowledge_base/rag_concept.md** (相似度: 0.932)
2. 文档: **./knowledge_base/ml_basics.md** (相似度: 0.486)
3. 文档: **./knowledge_base/python_intro.md** (相似度: 0.412)

=== 知识库问答 ===
🤖 RAG 是“检索增强生成”… 其核心机制是“通过检索相关知识来增强大语言模型的生成能力”。
📚 参考来源  🟢 [1] rag_concept.md (0.914)  🔵 [2] ml_basics.md (0.487)  🔵 [3] python_intro.md (0.438)
⚡ 检索: 6532ms | 生成: 2720ms | 平均相似度: 0.613

=== 知识库统计 ===
📝 命名空间: test
📋 集合名称: test_collection
📂 存储根路径: ./knowledge_base
📊 文档分块数: 3
🔢 向量维度: 1024   📎 距离度量: cosine
✅ RAG 管道: 正常   ✅ LLM 连接: 正常
```

> 记忆检索的条数会随库里已有数据变化：这次 `张三 技术方向` 只命中 1 条（精确命中张三，未误召李四），`前端工程师` 精确命中第 2 条。

初始化阶段日志里出现这些是**正常**的：

```
⚠️ 本地Qdrant服务不可用（[Errno 111] Connection refused），回退到嵌入式本地存储   ← 预期，走嵌入式模式
⚠️ 没有可用的spaCy模型进行实体识别                                  ← 图谱实体为空，不影响向量检索
```

---

## 3. 踩坑记录（现象 → 原因 → 解法）

### 坑 1：MemoryTool 初始化直接抛 `ConnectionRefused`

**现象**

```
INFO:...qdrant_store:✅ 成功连接到Qdrant服务: http://localhost:6333
ERROR:...qdrant_store:❌ Qdrant连接失败: [Errno 111] Connection refused
qdrant_client.http.exceptions.ResponseHandlingException: [Errno 111] Connection refused
```

**原因**

`chap08/.env` 里设了 `QDRANT_URL=http://localhost:6333`，但本机没有 Qdrant 服务。fork 里虽然加了「本地不可用就回退嵌入式存储」的逻辑（`hello_agents/memory/storage/qdrant_store.py` 的 `_connect_local_or_embedded()`），但该回退**只在 `url` 为空时才会走到**：

```python
if self.url and self.api_key:
    ...  # 云服务分支，失败直接 raise
elif self.url:
    ...  # 自定义 URL 分支，失败直接 raise
else:
    self.client = self._connect_local_or_embedded()  # 只有这里才有回退
```

**解法（三选一）**

- 把 `chap08/.env` 里 `QDRANT_URL` 注释掉（已在 `.env` 中固化，推荐）；
- 或命令行 `QDRANT_URL=` 置空 → 走嵌入式本地存储（无需 Docker）；
- 或真起一个服务：`docker run -d -p 6333:6333 qdrant/qdrant`。

### 坑 2：检索恒定「未找到相关记忆」

**现象**：`add` 都成功，但 `search` 返回 0 条；直接调 `search_similar` 能看到 3 条结果，但 **score 全是 0.0**，混合排序阶段被过滤掉。

**原因**：没有配置 embedding，走的兜底哈希向量化（fork 里把原来的 TF-IDF 改成了 `HashingVectorizer`，dim=1000）。它默认按「单词」切词，中文整句被当成一个 token，查询词与库内文本零重叠 → 余弦相似度 0：

```
嵌入模型就绪，维度: 1000      ← 哈希兜底
=== 未找到与 '张三 技术方向' 相关的记忆
```

**解法**：给真正的向量模型。验证过可用的组合是 DashScope `text-embedding-v3`（dim=1024）：

```bash
EMBED_MODEL_TYPE=dashscope
EMBED_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBED_API_KEY="$ALI_API_KEY"
```

**注意维度冲突**：嵌入式 Qdrant 的 collection 是持久化的，旧 collection（dim=1000）不能直接写 dim=1024 的向量，所以要么换 `QDRANT_COLLECTION` 名，要么删掉 `chap08/memory_data/qdrant_local/` 重来。当前 `.env` 用的是 `QDRANT_COLLECTION=chapter08_demo` + `QDRANT_VECTOR_SIZE=1024` 的干净组合。

> 环境判定优先级（`hello_agents/memory/embedding.py::_build_embedder`）：
> `EMBED_MODEL_TYPE` 默认 `dashscope`，回退链为 `dashscope → local → tfidf/hashing`。

### 坑 3：`EMBED_MODEL_TYPE=local` 走不通（transformers 装坏了）

**现象**

```
ImportError: cannot import name 'GenerationMixin' from 'transformers.generation'
```

**原因**：`LLM` 环境里的 transformers 4.57.6 是**不完整安装**，`site-packages/transformers/generation/` 下缺 `utils.py` 等文件：

```bash
$ ls /home/scc/anaconda3/envs/LLM/lib/python3.10/site-packages/transformers/generation/
beam_constraints.py  beam_search.py  candidate_generator.py  configuration_utils.py
continuous_batching  flax_logits_process.py  flax_utils.py  __init__.py
logits_process.py  __pycache__        # ← 没有 utils.py
```

**影响范围**：所有 `from transformers import AutoTokenizer / AutoModel` 的代码都会挂，不止本章——`chap03` 加载 Qwen 大概率同样失败。

**最小验证**

```bash
/home/scc/anaconda3/envs/LLM/bin/python3 -c "from transformers import AutoTokenizer"
```

**解法（未执行，待确认）**

```bash
/home/scc/anaconda3/envs/LLM/bin/pip install --force-reinstall transformers==4.57.6
```

### 坑 4（补记）：`qdrant-client` 1.19 移除了 `vectors_count` 字段 —— 已修

**现象**（`stats` / `summary` / `rag stats` 时刷错误日志，功能不挂）

```
ERROR:hello_agents.memory.storage.qdrant_store:❌ 获取集合信息失败: 'CollectionInfo' object has no attribute 'vectors_count'
```

**原因**：`qdrant_store.py::get_collection_info()` 直接访问 `collection_info.vectors_count`，而 qdrant-client 1.19 的 `CollectionInfo` 只剩下 `points_count` / `indexed_vectors_count`：

```python
>>> sorted(CollectionInfo.model_fields.keys())
['config', 'indexed_vectors_count', 'optimizer_status', 'payload_schema',
 'points_count', 'segments_count', 'status', 'update_queue', 'warnings']
```

写入本身是成功的，只是取集合元信息时抛异常并被 `except` 吞掉 → 返回 `{}`，所以只是噪声（但 `rag_tool` 的统计里拿不到向量数）。

**解法（已改 `hello_agents/memory/storage/qdrant_store.py::get_collection_info()`）**

```python
# 兼容 qdrant-client 新旧版本：<1.19 有 vectors_count；>=1.19 已移除，用 points_count 兜底
points_count = getattr(collection_info, "points_count", None)
vectors_count = getattr(collection_info, "vectors_count", None)
if vectors_count is None:
    vectors_count = points_count or 0

info = {
    "name": self.collection_name,
    "vectors_count": vectors_count,
    "indexed_vectors_count": getattr(collection_info, "indexed_vectors_count", None),
    "points_count": points_count,
    "segments_count": getattr(collection_info, "segments_count", None),
    "config": {"vector_size": self.vector_size, "distance": self.distance.value.lower()},
}
```

顺手把 `"distance"` 从枚举值 `'Cosine'` 改成小写 `'cosine'`（仅展示用）。

**验证**：

```
get_collection_info -> {'name': 'chapter08_demo', 'vectors_count': 4, 'indexed_vectors_count': 0,
                        'points_count': 4, 'segments_count': 1,
                        'config': {'vector_size': 1024, 'distance': 'cosine'}}
```

> ⚠️ 已在 REPL 里跑过的进程需**重启 Python** 才能加载新代码；且 `QdrantVectorStore` 是单例，改配置/改代码同样要重启。

### 坑 5（补记）：嵌入式 Qdrant 目录独占 —— memory_tool 与 rag_tool 不能各开一个 client —— 已修

**现象**（同一个脚本里先用 `MemoryTool` 再用 `RAGTool`）

```
❌ RAG工具初始化失败: Storage folder ./memory_data/qdrant_local is already accessed by another instance of Qdrant client.
✅ 工具 'rag' 已注册。
知识[python_intro]: ❌ RAG工具未正确初始化，请检查配置: Storage folder ... already accessed ...
```

**原因**：`QdrantClient(path=...)` 是**嵌入式模式，以磁盘目录为锁**，同一目录不能被两个 client 打开。`MemoryTool` 已经把它占了（语义/情景记忆），而 `RAGTool._init_components()` → `create_rag_pipeline()` → 又 `QdrantVectorStore(...)` → 又 `QdrantClient(path=...)` → 撞锁。

> 注意 `QdrantConnectionManager` 的单例 key 是 `(url, collection)`，memory 与 RAG 的 collection 不同（`chapter08_demo` vs `test_collection`），所以会各建一个 store，单例机制救不了。

**解法（已改 `hello_agents/memory/storage/qdrant_store.py`）**：对同一路径做进程级 client 复用，不同 collection 共用一个 client（server 模式本来就是这用法）：

```python
_embedded_clients: Dict[str, Any] = {}
_embedded_clients_lock = threading.Lock()

# _connect_local_or_embedded() 末尾：
key = os.path.abspath(local_path)
with _embedded_clients_lock:
    cached = _embedded_clients.get(key)
    if cached is not None:
        return cached                      # ♻️ 复用，避免第二次打开同一目录
    client = QdrantClient(path=local_path)
    _embedded_clients[key] = client
```

> 彻底解法：起 Docker Qdrant（`docker run -d -p 6333:6333 qdrant/qdrant`）并把 `QDRANT_URL` 配上，多进程/多实例都不冲突。

**验证**：memory（add/search/stats/summary）+ RAG（add_text/search/ask/stats）在同一脚本内先后执行全部正常，RAG 检索相似度 0.936。

**同进程外的提醒**：嵌入式模式是**跨进程**锁目录的，所以 REPL 里开着 `memory_tool` 同时再跑脚本会报同样的错 —— 先退出 REPL。

### 补记 2：RAGTool 不传 namespace 时硬编码 `"default"` —— 已修

`run()` 里 6 处 `parameters.get("namespace", "default")` 改为 `parameters.get("namespace", self.rag_namespace)`。

否则 `RAGTool(rag_namespace="test")` 建好了管道，但 `run({"action": "stats"})` 不传 namespace 时会去 `_get_pipeline("default")` —— 为一个不存在的 namespace 新建管道（collection 相同、namespace 标签不同），造成 `📝 命名空间: default` 与实际写入的 `test` 不一致。修后显示正确，也少建一个重复管道。

---

## 4. 脚本改动说明（`chap08/p_mem_and_rag.py`）

脚本现已整理为「一、记忆功能 + 二、RAG 功能」两段，从**任意目录**执行都会先 `chdir` 到脚本所在目录（`.env` / `memory_data` / `knowledge_base` 因此始终落在 `chap08`），可直接：

```bash
/home/scc/anaconda3/envs/LLM/bin/python3 /home/scc/sccWork/myGitHub/My_Learn/Hello-Agent/chap08/p_mem_and_rag.py
```

官方书里的写法在这个 fork 上跑不了，共 6 处改动（前 4 处为我改的，后 2 处为自加的调试代码）：

| 位置 | 书/原稿写法 | 本 fork 正确写法 |
|---|---|---|
| 入口 | 无 | `load_dotenv()`，模型名改读 `os.getenv("LLM_MODEL_ID")` |
| 记忆写入 | `memory_tool.run("add", content=..., memory_type=..., importance=...)` | `memory_tool.run({"action": "add", "content": ..., "memory_type": ..., "importance": ...})` |
| 检索 | 无 | 新增 `{"action": "search", "query": "张三 技术方向", "memory_type": "semantic", "limit": 3}` |
| 统计 | 无 | 新增 `{"action": "stats"}` |
| 定向检索 | 无 | 新增 `{"action": "search", "query": "前端工程师", "limit": 3}`（验证能精准命中第 2 条） |
| 摘要 | 无 | 新增 `{"action": "summary"}` |

> `memory_tool.run({"action": "add", content=...})` 这种「位置参数 + 关键字混用」在 Python 里也会直接报 `TypeError`，别踩。

核心差异：本 fork 的 `Tool.run(self, parameters: Dict[str, Any])` 只吃**一个字典参数**（`hello_agents/tools/base.py:70`、`memory_tool.py:53`），书上是 `run(action, **kwargs)` 风格。以后写本章代码一律按字典传参。

### 脚本里的 RAG 段（定稿）

第二段接在记忆之后，同一个进程内再建 `RAGTool`：

```python
rag_tool = RAGTool(
    knowledge_base_path="./knowledge_base",   # 不是书上的参数名风格，且必须有
    collection_name="test_collection",
    rag_namespace="test",
)
```

- **与 MemoryTool 共用同一个嵌入式 Qdrant 目录**（`./memory_data/qdrant_local`）：靠坑 5 的进程级 client 复用才能同进程共存，实跑日志里能看到 `♻️ 复用嵌入式Qdrant本地存储: ./memory_data/qdrant_local`；
- **namespace 别漏传**：脚本走的是 `rag_namespace="test"`，`stats` / `search` 不显式传 namespace 时会用实例的 `self.rag_namespace`（补记 2 的修复），所以输出里是 `命名空间: test`；
- **collection 名**：脚本用 `test_collection`，本文第 4 节附录的 REPL 实验用的是 `rag_knowledge_base` —— 都是同一个嵌入式库下的不同 collection，互不影响；
- 写入的 3 条知识（`python_intro` / `ml_basics` / `rag_concept`）是 `add_text`，原文文件由 loader 临时生成后清理，`knowledge_base/` 目录平时是空的。

### 附：REPL / 交互式调试对照

| 书上写法（官方版） | 本 fork 写法 |
|---|---|
| `memory_tool.run("search", query="x", limit=3)` | `memory_tool.run({"action": "search", "query": "x", "limit": 3})` |
| `memory_tool.execute("summary", limit=10)` | `memory_tool.run({"action": "summary", "limit": 10})`（本 fork **没有** `execute`） |

嫌字典麻烦时，可在 REPL 里包一层，之后就能继续用「书上的手感」：

```python
>>> mem = lambda action, **kw: memory_tool.run({"action": action, **kw})
>>> print(mem("search", query="前端工程师", limit=3))
>>> print(mem("add", content="王五做数据分析", memory_type="semantic", importance=0.6))
```

想彻底去掉 `"add"` / `"search"` 这种字符串参数、让 LLM 以 function calling 方式自主调用，可以建实例时开展开模式：`MemoryTool(user_id="scc", expandable=True)`，再用 `memory_tool.get_expanded_tools()` 拿到 `memory_add` / `memory_search` / `memory_summary` 等子工具注册进 `ToolRegistry`（子工具内部同样按字典传参）。

### 附：RAGTool 用法（同样没有 `execute`，只有 `run(dict)`）

`RAGTool` 的坑和 `MemoryTool` 一样：基类 `Tool` 只定义了 `run(self, parameters: Dict)`，fork 里**不存在** `execute` 方法。

| 书上/官方写法 | 本 fork 写法 |
|---|---|
| `rag_tool.execute("add_text", text=..., document_id=...)` | `rag_tool.run({"action": "add_text", "text": ..., "document_id": ...})` |
| `rag_tool.execute("add_document", file_path=...)` | `rag_tool.run({"action": "add_document", "file_path": ..., "document_id": ...})` |
| `rag_tool.execute("search", query=...)` | `rag_tool.run({"action": "search", "query": ..., "limit": 3})` |
| `rag_tool.execute("ask", question=...)` | `rag_tool.run({"action": "ask", "question": ...})` |
| `rag_tool.execute("stats")` | `rag_tool.run({"action": "stats"})` |
| `rag_tool.execute("clear", confirm=True)` | `rag_tool.run({"action": "clear", "confirm": True})` |

实测片段（在 `chap08` 下，沿用已固化的 `.env`，2026-09-10 全部跑通）：

```python
>>> from hello_agents.tools import RAGTool
>>> rag = RAGTool(knowledge_base_path="./rag_demo_kb")
>>> print(rag.initialized)                     # True，False 时看 rag.init_error
>>> rag.run({"action": "add_text",
...          "text": "机器学习是人工智能的一个分支……主要包括监督学习、无监督学习和强化学习三种类型。",
...          "document_id": "ml_basics"})
✅ 文本已添加到知识库: ml_basics  📊 分块数量: 1  ⏱️ 594ms

>>> rag.run({"action": "search", "query": "监督学习", "limit": 3})
1. 文档: **./rag_demo_kb/ml_basics.md** (相似度: 0.739)

>>> rag.run({"action": "ask", "question": "机器学习主要有哪些类型？", "limit": 3})
🤖 根据提供的上下文，机器学习主要包括以下三种类型：1. 监督学习 2. 无监督学习 3. 强化学习
📚 参考来源 🟢 [1] ml_basics.md (相似度: 0.893)   ⚡ 检索: 5853ms | 生成: 1997ms

>>> rag.run({"action": "stats"})
📋 集合名称: rag_knowledge_base  📦 存储类型: qdrant  🔢 向量维度: 1024  📎 距离度量: cosine
✅ RAG 管道: 正常   ✅ LLM 连接: 正常
```

要点：
- `RAGTool.__init__(knowledge_base_path=..., collection_name="rag_knowledge_base", rag_namespace="default")`，**没有** `embedding_model` 参数（书上的示例里有，本 fork 不要传）；维度来自全局 embedder（`create_rag_pipeline()` 里的 `get_dimension()`），所以 `.env` 的 embedding 配置对它同样生效；
- `qdrant_url` 默认取 `os.getenv("QDRANT_URL")`，`.env` 里已注释 → 同样走嵌入式本地存储；
- 知识库数据落在 `knowledge_base_path` 目录（如 `./rag_demo_kb/`，里面还有原文 `.md`），collection 名为 `rag_knowledge_base`；
- `ask` 会调 LLM（默认取 `HelloAgentsLLM()`，即 `.env` 里的 DeepSeek）；`search` 不调 LLM；
- 进程退出时可能出现 `Exception ignored in: <function QdrantClient.__del__> ... sys.meta_path is None` —— 嵌入式 Qdrant 关闭时的噪声，无害。

### 附：Embedding 用的是云 API

当前方案走的是**云端 API**（DashScope `text-embedding-v3`，OpenAI 兼容 REST，dim=1024），不是本地模型：

- 调用点：`hello_agents/memory/embedding.py::DashScopeEmbedding`（设了 `EMBED_BASE_URL` 故走 `requests.post(.../embeddings)` 分支，不依赖 dashscope SDK）；
- 触发时机：每次 `add` 编码 1 次内容、每次 `search` 编码 1 次查询，即每条操作 1 个 HTTP 请求；
- 成本/依赖：文本短、调用次数少，费用可忽略；但需要联网 + `ALI_API_KEY` 有效。

自查当前生效的是哪种 embedding：

```python
from hello_agents.memory.embedding import get_text_embedder
m = get_text_embedder()
print(type(m).__name__, m.dimension)   # DashScopeEmbedding 1024=用API；TFIDFEmbedding 1000=掉兜底
```

---

## 5. 待办 / 可选优化

- [x] **固化配置**：`EMBED_MODEL_TYPE=dashscope`、`EMBED_BASE_URL=...compatible-mode/v1`、`EMBED_API_KEY=${ALI_API_KEY}` 已写进 `chap08/.env`，并注释掉 `QDRANT_URL`、`QDRANT_COLLECTION=chapter08_demo`、`QDRANT_VECTOR_SIZE=1024`。已实测裸跑通过（维度 1024）。
- [x] **修掉 Qdrant 统计报错**（已修，见坑 4）：`qdrant_store.py::get_collection_info()` 改用 `getattr` + `points_count` 兜底，兼容 qdrant-client <1.19 / >=1.19。
- [ ] **补 spaCy 中文模型**：`python -m spacy download zh_core_web_sm`，否则语义记忆的实体/关系恒为 0，图谱检索分支形同虚设。
- [ ] **修复 transformers 安装**（见坑 3），解锁 `EMBED_MODEL_TYPE=local` 与 chap03 的本地模型；修好后可用 `BAAI/bge-small-zh-v1.5`（512 维，免费离线）。
- [ ] 想彻底离线：给 `embedding.py::TFIDFEmbedding` 的 `HashingVectorizer` 加 `analyzer="char", ngram_range=(1,2)`（中文按字切分），否则它对本任务等于不可用。
- [x] **运行时产物已加 gitignore**（2026-09-14）：`.gitignore` 新增 `Hello-Agent/chap08/memory_data/` 和 `Hello-Agent/chap08/memory_data.bak_*/`，`git status` 里不再出现这两个目录。清理时直接 `rm -rf chap08/memory_data` 即可，首次运行会自动重建。
- [ ] 旧的 1000 维数据备份在 `chap08/memory_data.bak_20260910/`，已忽略但文件仍在，确认无用即可删。另发现 `My_Learn/memory_data/`（仓库根目录）存在一份早期数据，非本次生成，尚未加入 ignore，按需处理。

---

## 6. 复跑验证（2026-09-14）

同一脚本、同一 `.env` 重跑一次，全流程通过（`real 0m22.9s`）：

| 环节 | 结果 |
|---|---|
| 记忆 add / search / stats / summary | ✅ `张三 技术方向` 精确命中张三 1 条，`前端工程师` 精确命中李四 1 条 |
| 嵌入式 Qdrant 复用（坑 5） | ✅ 日志出现 `♻️ 复用嵌入式Qdrant本地存储`，MemoryTool 与 RAGTool 同进程共存 |
| RAG `add_text` ×3 | ✅ 3 条知识写入 `test_collection`（namespace=test），单个分块 243~693ms |
| RAG `search` | ✅ `什么是RAG？` → `rag_concept.md` 相似度 0.932 |
| RAG `ask` | ✅ 回答正确引用 3 条来源，检索 6532ms / 生成 2720ms |
| Qdrant 集合统计（坑 4） | ✅ 无 `vectors_count` 报错，`文档分块数: 3`、`向量维度: 1024` 正常 |
| embedding | ✅ `嵌入模型就绪，维度: 1024`（DashScope 云端，未掉哈希兜底） |

日志里剩余的**无害噪声**（可忽略，或按需关掉 qdrant/onnx 的 logger）：

```
⚠️ 本地Qdrant服务不可用（[Errno 111] Connection refused），回退到嵌入式本地存储   ← 预期
⚠️ 无可用spaCy模型，实体提取将受限 / ⚠️ 没有可用的spaCy模型进行实体识别          ← 见待办
UserWarning: Payload indexes have no effect in the local Qdrant                    ← 嵌入式不支持 payload 索引
UserWarning: Local mode performs exact (brute-force) search                        ← 嵌入式走暴力检索
onnxruntime ... GPU device discovery failed ... /sys/class/drm/card0/...           ← 无独显，可忽略
```

---

## 7. 一句话总结

这个 fork 的 API 与书上不一致（只能字典传参）；Qdrant 没起服务时必须让 `QDRANT_URL` 为空才会回退嵌入式；**决定检索效果的是 embedding**——兜底哈希对中文等于不可用，配 DashScope embedding 后一切正常；嵌入式 Qdrant 一个目录只允许一个 client，靠 fork 里的进程级 client 复用才能让 MemoryTool 和 RAGTool 同进程跑。配置现已固化进 `chap08/.env`，脚本自带 `chdir`，任意目录下 `python .../chap08/p_mem_and_rag.py` 即可复现（2026-09-14 验证通过）。
