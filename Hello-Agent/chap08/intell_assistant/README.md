# 问答助手 · PDF 学习助手（Chap08 记忆与 RAG）

一个本地跑的教学型"文档问答"应用：

```
上传 PDF/MD/TXT ──► RAGTool 解析分块 + embedding 入库(Qdrant)
                        │
提问 ──► 记忆(working) ──┴──► 检索 + LLM 生成 ──► 记忆(episodic 存回答摘要)
                                                     └──► 摘要/统计/报告可导出
```

- `bot.py`：后端（`PDFLearningAssistant`），**已按当前 fork 0.2.9 的接口适配**
- `app.py`：前端页面（Streamlit），单文件交互界面
- 数据落盘在上一级 `chap08/` 目录：`chap08/memory_data/`（记忆 + 嵌入式 Qdrant）、`chap08/knowledge_base/`（知识库）

## 快速开始

```bash
cd /home/scc/sccWork/myGitHub/My_Learn/Hello-Agent/chap08/intell_assistant

# 1) 自检（确认 .env / Qdrant / embedding 都通）
/home/scc/anaconda3/envs/LLM/bin/python3 bot.py

# 2) 起前端
/home/scc/anaconda3/envs/LLM/bin/streamlit run app.py
#   浏览器打开 http://localhost:8501
```

前置条件：上一级 `chap08/.env` 已配置好（DeepSeek LLM + DashScope embedding + 注释掉 `QDRANT_URL` 走嵌入式存储）。
详细的环境坑位见 `../p_mem_and_rag_运行记录.md`。

## 页面功能

| 区域 | 功能 |
|---|---|
| 左侧 ① | 用户 ID（按用户隔离记忆与知识库 namespace）、清空对话 |
| 左侧 ② | 上传文档 → 解析/分块/向量化入库，可调 `chunk_size` / `chunk_overlap` |
| 左侧 ③ | 直接补录一段文本知识（`add_text`，免上传文件） |
| 左侧 ④ | 高级检索开关（MQE + HyDE）、召回条数 |
| 左侧 ⑤ | 记忆摘要 / 记忆统计 / 知识库统计 / 召回测试 / 导出报告 JSON |
| 主区域 | 对话流：提问 → 检索增强生成 → 自动沉淀记忆 |

## 后端 API（本 fork 的正确姿势）

```python
from bot import PDFLearningAssistant

bot = PDFLearningAssistant(user_id="scc")
bot.load_document("xx.pdf", chunk_size=1000, chunk_overlap=200)   # -> {"success", "message", ...}
bot.add_text("RAG 是……", "rag_intro")                            # 免文件补录
bot.ask("RAG 是什么？", enable_advanced_search=True, limit=5)      # -> {"success", "answer", "elapsed"}
bot.recall("前端工程师", limit=5, memory_type="episodic")          # 记忆召回（不调 LLM）
bot.memory_summary(); bot.memory_stats(); bot.rag_stats()
bot.generate_report("report.json")
```

## 与书稿（官方 hello-agents）的接口差异 ⚠️

书稿第 8 章的示例代码在当前 fork 上跑不了，`bot.py` 里已全部适配并加了注释：

| 书稿写法 | 本 fork 正确写法 | 说明 |
|---|---|---|
| `tool.execute("add", content=...)` | `tool.run({"action": "add", "content": ...})` | 基类只有 `run(parameters: Dict)`，**没有 `execute`** |
| `MemoryTool(user_id=..., session_id=...)` | `MemoryTool(user_id=...)` 后设 `current_session_id` | 构造参数里没有 `session_id` |
| `run("add", content=...)` | 只能传一个 dict | 位置参数 + 关键字会 `TypeError` |
| 传 `event_type` / `concept` 等自定义字段 | 会被**静默丢弃**；要落库得走 `memory_manager.add_memory(metadata=...)`，且 episodic 只持久化 `session_id/context/outcome/participants/tags` | `run()` 按固定字段取参 |
| `rag_tool.execute("add_document", ...)` 后 `result.get("success")` | `run({...})`，返回的是**字符串**（`✅`/`❌` 开头） | 不是 dict |
| `enable_mqe` / `enable_hyde` | 只有 `enable_advanced_search` | 另两个参数已不存在 |
| `chunck_overlap` | `chunk_overlap` | 拼错会被忽略，静默用默认 100 |

## 常见问题

- **`Storage folder ... is already accessed by another instance`**：嵌入式 Qdrant 是"目录锁"。
  Streamlit 里已用 `@st.cache_resource` 保证单实例；若同时在 REPL 里开着 `MemoryTool`，先退出 REPL。
  彻底解决：起 Docker `qdrant/qdrant` 并在 `.env` 配 `QDRANT_URL`。
- **检索结果恒为空 / 相似度 0.0**：embedding 掉到了哈希兜底（中文不可用），检查 `chap08/.env` 的 `EMBED_*`，
  并确认 `QDRANT_VECTOR_SIZE` 与 embedding 维度一致（DashScope `text-embedding-v3` = 1024）。
  维度对不上时删掉 `chap08/memory_data/qdrant_local/` 重建。
- **大 PDF 很慢**：`add_document` 会按 `chunk_size` 逐块 embedding，170 页的书稿建议先调大 `chunk_size` 或分段入库。
- **换了用户名后检索不到旧文档**：`rag_namespace = pdf_{user_id}`，换 ID 即换命名空间，这是有意隔离。
