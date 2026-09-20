# python3
# Create Date: 2026-09-14
# Author: Scc_hy
# Func: 问答助手 · 前端页面（Streamlit）
# Run : cd Hello-Agent/chap08/intell_assistant && /home/scc/anaconda3/envs/LLM/bin/streamlit run app.py
# ===========================================================================================
"""问答助手前端（Streamlit）。

后端逻辑全部在 bot.PDFLearningAssistant 里，这里只做交互：
    左侧栏：会话 / 文档入库 / 文本补录 / 检索开关 / 记忆与统计
    主区域：对话流（提问 -> 检索增强生成 -> 沉淀记忆）

注：会话状态一律用 ``ss()`` 读取（``st.session_state.get``），
这样即使没有 Streamlit runtime（例如直接 ``python app.py`` 冒烟测试）也不会 KeyError。
"""

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from bot import ROUTES, SUPPORTED_SUFFIXES, PDFLearningAssistant

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
ACCEPT_TYPES = sorted(s.lstrip(".") for s in SUPPORTED_SUFFIXES)

ROUTE_LABELS = {
    "auto": "🤖 自动路由",
    "rag": "📚 仅知识库 RAG",
    "memory": "🧠 仅记忆",
    "hybrid": "🔀 混合",
}
ISO_LABELS = {
    "namespace": "共用库 + 命名空间过滤（默认）",
    "collection": "每用户独立库（强隔离）",
}

st.set_page_config(page_title="问答助手 · PDF 学习助手", page_icon="📚", layout="wide")


def ss(key: str, default=None):
    """读取会话状态（不依赖 key 已初始化）。"""
    return st.session_state.get(key, default)


def put(key: str, value) -> None:
    st.session_state[key] = value


def render_report_md(report: dict) -> str:
    """习题 4.2：把智能学习报告 dict 渲染成 markdown（轨迹 / 盲点 / 推荐 / LLM 叙事）。"""
    lines = [f"### 📊 学习报告 · {report['user_id']} ｜ {report['generated_at']}"]

    lines.append("\n**学习轨迹**（来自 episodic 记忆，按会话还原）")
    if report["trajectory"]:
        for seg in report["trajectory"]:
            events = "；".join(
                f"{_fmt_time(e.get('time'))} " + (e.get("question") or e.get("document") or "")
                for e in seg["events"][-5:])   # 每个会话最多展示 5 条，避免过长
            lines.append(f"- `{seg['session']}`：{events}")
    else:
        lines.append("- 暂无学习记录")

    lines.append("\n**主题与盲点**（反复问 + 疑难词 + 无笔记 三信号）")
    if report["topics"]:
        lines.append("| 主题 | 提问次数 | 疑难信号 | 已记笔记 | 盲点分 |")
        lines.append("|---|---|---|---|---|")
        for t in report["topics"][:8]:
            lines.append(f"| {t['topic'][:24]} | {t['ask_count']} | "
                         f"{'⚠️' if t['confused'] else '—'} | {'✅' if t['noted'] else '❌'} | {t['blind_score']} |")
    else:
        lines.append("- 还没有提问记录")

    lines.append("\n**下一步建议**")
    for i, rec in enumerate(report["recommendations"], 1):
        lines.append(f"{i}. {rec}")

    if report.get("llm_summary"):
        lines.append(f"\n**教练点评（LLM）**\n\n> {report['llm_summary'].replace(chr(10), chr(10) + '> ')}")
    return "\n".join(lines)


def _fmt_time(t) -> str:
    try:
        return t.strftime("%m-%d %H:%M")
    except Exception:
        return str(t or "")


@st.cache_resource(show_spinner="正在初始化 MemoryTool + RAGTool（首次会连 Qdrant / 建集合）...")
def get_assistant(user_id: str, isolation_mode: str = "namespace") -> PDFLearningAssistant:
    """按 (user_id, 隔离模式) 缓存实例：Streamlit 每次交互都会重跑脚本，不能每次新建（会撞嵌入式 Qdrant 目录锁）。"""
    return PDFLearningAssistant(user_id=user_id, isolation_mode=isolation_mode)


# --------------------------------------------------------------------------- 侧栏
with st.sidebar:
    st.header("① 会话")
    user_id = st.text_input("用户 ID", value="scc", help="记忆与知识库都按用户隔离（namespace=pdf_{id}）")
    iso_mode = st.radio(
        "数据隔离模式（习题 4.3）",
        options=list(ISO_LABELS), format_func=lambda m: ISO_LABELS[m],
        help="namespace=共用 collection 按载荷过滤（省资源）；collection=每用户独立 collection（强隔离，便于整库删除/迁移）。两种模式的数据互不可见。",
    )

    if st.button("🧹 清空对话", use_container_width=True):
        put("messages", [])
        st.experimental_rerun()

    bot = get_assistant(user_id, iso_mode)
    if bot.ready:
        st.success(f"RAG 就绪 · {bot.isolation_mode} 模式 · namespace = {bot.rag_tool.rag_namespace}")
    else:
        st.error(f"RAG 初始化失败：{bot.init_error}")

    st.markdown("---")
    st.header("② 文档入库")
    uploaded = st.file_uploader("上传文档（pdf / md / txt / docx ...）", type=ACCEPT_TYPES)
    col1, col2 = st.columns(2)
    chunk_size = col1.number_input("chunk_size", 200, 4000, 1000, 100)
    chunk_overlap = col2.number_input("chunk_overlap", 0, 1000, 200, 50)

    if st.button("📥 加载到知识库", use_container_width=True, disabled=uploaded is None):
        save_path = UPLOAD_DIR / uploaded.name
        save_path.write_bytes(uploaded.getbuffer())
        with st.spinner(f"解析 / 分块 / 向量化 {uploaded.name} ...（大 PDF 会比较慢）"):
            res = bot.load_document(str(save_path), int(chunk_size), int(chunk_overlap))
        if res["success"]:
            st.success(res["message"])
            docs = list(ss("loaded_docs", []))
            if res.get("document") and res["document"] not in docs:
                docs.append(res["document"])
            put("loaded_docs", docs)
        else:
            st.error(res["message"])

    if ss("loaded_docs"):
        st.caption("已加载：" + "、".join(ss("loaded_docs")))

    st.markdown("---")
    with st.expander("③ 补录文本知识（可选）"):
        doc_id = st.text_input("document_id", value=f"note_{datetime.now():%H%M%S}")
        text = st.text_area("内容", height=120, placeholder="直接粘贴一段要进知识库的文本")
        if st.button("➕ 添加文本", use_container_width=True):
            res = bot.add_text(text, doc_id)
            (st.success if res["success"] else st.error)(res["message"])

    st.markdown("---")
    st.header("④ 检索设置")
    route = st.radio(
        "检索路由（习题 4.1）",
        options=list(ROUTES), format_func=lambda r: ROUTE_LABELS[r], horizontal=True,
        help="auto=意图规则+双路探测自动选 rag/memory/hybrid；也可手动锁定某一路",
    )
    advanced = st.checkbox(
        "启用高级检索（MQE + HyDE）",
        value=True,
        help="仅 rag 路由生效；本 fork 只有这一个开关，书稿里的 enable_mqe / enable_hyde 参数已不存在",
    )
    top_k = st.slider("召回条数 limit", 1, 10, 5)

    st.markdown("---")
    st.header("⑤ 记忆与报告")
    b1, b2, b3 = st.columns(3)
    if b1.button("摘要", use_container_width=True):
        put("panel", ("记忆摘要", bot.memory_summary(), "text"))
    if b2.button("记忆统计", use_container_width=True):
        put("panel", ("记忆统计", bot.memory_stats(), "text"))
    if b3.button("知识库统计", use_container_width=True):
        put("panel", ("知识库统计", bot.rag_stats(), "text"))

    b4, b5 = st.columns(2)
    if b4.button("📊 智能报告", use_container_width=True,
                 help="习题 4.2：学习轨迹 + 知识盲点 + 下一步建议（含一次 LLM 叙事）"):
        with st.spinner("生成学习报告（检索记忆 + LLM 叙事）..."):
            report = bot.generate_report(with_llm=True)
        put("panel", ("智能学习报告", render_report_md(report), "md"))
    if b5.button("🛡️ 隔离自检", use_container_width=True,
                 help="习题 4.3：随机探针用户尝试越权检索你的数据，应全部落空"):
        with st.spinner("探测跨用户泄漏..."):
            iso = bot.verify_isolation()
        lines = [f"**{iso['message']}**",
                 f"- 探针用户：`{iso['probe_user']}` ｜ 标记词：`{iso['marker']}`",
                 f"- RAG 知识库：我方命中={iso['self_rag_hit']}，探针命中={iso['probe_rag_hit']} → 隔离={'✅' if iso['rag_isolated'] else '❌'}",
                 f"- 记忆系统：我方命中={iso['self_memory_hit']}，探针命中={iso['probe_memory_hit']} → 隔离={'✅' if iso['memory_isolated'] else '❌'}"]
        put("panel", ("多用户隔离自检", "\n\n".join(lines), "md"))

    st.text_input("召回测试（不调 LLM）", key="recall_query", placeholder="输入关键词后点下面按钮")
    c1, c2 = st.columns(2)
    if c1.button("🔎 召回记忆", use_container_width=True):
        q = ss("recall_query", "")
        if q:
            put("panel", ("召回记忆", bot.recall(q, limit=int(top_k)), "text"))
    if c2.button("🔍 召回知识", use_container_width=True):
        q = ss("recall_query", "")
        if q:
            put("panel", ("召回知识", bot.search_knowledge(q, limit=int(top_k))["message"], "text"))

    st.download_button(
        "⬇️ 导出会话报告 (JSON)",
        data=json.dumps(bot.generate_report(with_llm=False), ensure_ascii=False, indent=4, default=str),
        file_name=f"learning_report_{bot.session_id}.json",
        mime="application/json",
        use_container_width=True,
        help="纯结构化数据（不调 LLM，随时可点）；带叙事版用上面的「智能报告」",
    )

# --------------------------------------------------------------------------- 主区域
st.title("📚 问答助手 · PDF 学习助手")
st.caption(
    "上传文档 → 检索增强生成（RAG）→ 沉淀记忆（working / episodic / semantic）。"
    f"当前文档：**{bot.current_document or '未加载'}** ｜ 会话：`{bot.session_id}`"
)

if not bot.current_document:
    st.info(
        "👈 先在左侧「② 文档入库」上传并加载一份文档；"
        "也可以直接问「我之前学了什么」这类记忆问题（习题 4.1 路由会自动走记忆侧，不依赖文档）。"
    )

messages = list(ss("messages", []))
for msg in messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

question = st.chat_input("提问：文档内容走 RAG；「我之前学了什么」走记忆；auto 自动路由")

if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        placeholder.markdown("🔍 检索中 ...")
        try:
            res = bot.ask(question, enable_advanced_search=advanced,
                          limit=int(top_k), route=route)
        except Exception as e:  # 网络 / LLM 异常不该把页面打挂
            res = {"success": False, "message": f"调用失败: {type(e).__name__}: {e}"}

        if res["success"]:
            placeholder.markdown(res["answer"])
            st.caption(
                f"⚡ {res['elapsed']:.1f}s ｜ 路由: {res['route']}（{res['reason']}）"
                f"｜ 高级检索: {'开' if advanced else '关'} ｜ limit={top_k}"
            )
            messages.append({"role": "assistant", "content": res["answer"]})
        else:
            placeholder.error(res["message"])
            messages.append({"role": "assistant", "content": f"❌ {res['message']}"})

put("messages", messages)

# 侧栏「⑤ 记忆与报告」的展示区（放主区域底部，避免侧栏过长）
panel = ss("panel")
if panel:
    kind = panel[2] if len(panel) > 2 else "text"
    with st.expander(f"📌 {panel[0]}", expanded=(kind == "md")):
        (st.markdown if kind == "md" else st.text)(panel[1])
