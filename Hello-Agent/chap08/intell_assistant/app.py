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

from bot import SUPPORTED_SUFFIXES, PDFLearningAssistant

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
ACCEPT_TYPES = sorted(s.lstrip(".") for s in SUPPORTED_SUFFIXES)

st.set_page_config(page_title="问答助手 · PDF 学习助手", page_icon="📚", layout="wide")


def ss(key: str, default=None):
    """读取会话状态（不依赖 key 已初始化）。"""
    return st.session_state.get(key, default)


def put(key: str, value) -> None:
    st.session_state[key] = value


@st.cache_resource(show_spinner="正在初始化 MemoryTool + RAGTool（首次会连 Qdrant / 建集合）...")
def get_assistant(user_id: str) -> PDFLearningAssistant:
    """按 user_id 缓存实例：Streamlit 每次交互都会重跑脚本，不能每次新建（会撞嵌入式 Qdrant 目录锁）。"""
    return PDFLearningAssistant(user_id=user_id)


# --------------------------------------------------------------------------- 侧栏
with st.sidebar:
    st.header("① 会话")
    user_id = st.text_input("用户 ID", value="scc", help="记忆与知识库都按用户隔离（namespace=pdf_{id}）")

    if st.button("🧹 清空对话", use_container_width=True):
        put("messages", [])
        st.experimental_rerun()

    bot = get_assistant(user_id)
    if bot.ready:
        st.success(f"RAG 就绪 · namespace = pdf_{user_id}")
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
    advanced = st.checkbox(
        "启用高级检索（MQE + HyDE）",
        value=True,
        help="本 fork 只有这一个开关；书稿里的 enable_mqe / enable_hyde 参数已不存在，传了会被忽略",
    )
    top_k = st.slider("召回条数 limit", 1, 10, 5)

    st.markdown("---")
    st.header("⑤ 记忆与统计")
    b1, b2, b3 = st.columns(3)
    if b1.button("摘要", use_container_width=True):
        put("panel", ("记忆摘要", bot.memory_summary()))
    if b2.button("记忆统计", use_container_width=True):
        put("panel", ("记忆统计", bot.memory_stats()))
    if b3.button("知识库统计", use_container_width=True):
        put("panel", ("知识库统计", bot.rag_stats()))

    st.text_input("召回测试（不调 LLM）", key="recall_query", placeholder="输入关键词后点下面按钮")
    c1, c2 = st.columns(2)
    if c1.button("🔎 召回记忆", use_container_width=True):
        q = ss("recall_query", "")
        if q:
            put("panel", ("召回记忆", bot.recall(q, limit=int(top_k))))
    if c2.button("🔍 召回知识", use_container_width=True):
        q = ss("recall_query", "")
        if q:
            put("panel", ("召回知识", bot.search_knowledge(q, limit=int(top_k))["message"]))

    st.download_button(
        "⬇️ 导出会话报告 (JSON)",
        data=json.dumps(bot.generate_report(), ensure_ascii=False, indent=4),
        file_name=f"learning_report_{bot.session_id}.json",
        mime="application/json",
        use_container_width=True,
    )

# --------------------------------------------------------------------------- 主区域
st.title("📚 问答助手 · PDF 学习助手")
st.caption(
    "上传文档 → 检索增强生成（RAG）→ 沉淀记忆（working / episodic / semantic）。"
    f"当前文档：**{bot.current_document or '未加载'}** ｜ 会话：`{bot.session_id}`"
)

if not bot.current_document:
    st.info("👈 先在左侧「② 文档入库」上传并加载一份文档，或「③ 补录文本知识」贴一段文字，然后就可以提问了。")

messages = list(ss("messages", []))
for msg in messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

question = st.chat_input("针对已加载的文档提问，例如：RAG 是什么？它解决什么问题？")

if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        placeholder.markdown("🔍 检索中 ...")
        try:
            res = bot.ask(question, enable_advanced_search=advanced, limit=int(top_k))
        except Exception as e:  # 网络 / LLM 异常不该把页面打挂
            res = {"success": False, "message": f"调用失败: {type(e).__name__}: {e}"}

        if res["success"]:
            placeholder.markdown(res["answer"])
            st.caption(f"⚡ 耗时 {res['elapsed']:.1f}s ｜ 高级检索: {'开' if advanced else '关'} ｜ limit={top_k}")
            messages.append({"role": "assistant", "content": res["answer"]})
        else:
            placeholder.error(res["message"])
            messages.append({"role": "assistant", "content": f"❌ {res['message']}"})

put("messages", messages)

# 侧栏「⑤ 记忆与统计」的展示区（放主区域底部，避免侧栏过长）
panel = ss("panel")
if panel:
    with st.expander(f"📌 {panel[0]}", expanded=False):
        st.text(panel[1])
