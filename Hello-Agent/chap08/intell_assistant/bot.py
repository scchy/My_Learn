# python3
# Create Date: 2026-09-14
# Author: Scc_hy
# Func: 智能问答助手（PDF 学习助手）—— RAG 检索问答 + 三层记忆
# Run : /home/scc/anaconda3/envs/LLM/bin/python3 -c "from bot import PDFLearningAssistant as A; print(A('scc').ask('x'))"
# UI  : streamlit run app.py   （同目录，见 README.md）
# ===========================================================================================
"""PDF/文档学习助手：加载文档 -> 检索问答 -> 记忆沉淀。

与书稿（官方 hello-agents）的接口差异，本文件已按当前 fork 0.2.9 适配，勿照抄书稿：

1. 工具只有 `tool.run({"action": ...})`，**没有** `execute()`；
2. `MemoryTool` 构造参数没有 `session_id`，会话 id 靠 `memory_tool.current_session_id` 注入；
3. `run()` 按固定字段取参，多传的 key（如 event_type/concept）会被**静默丢弃**；
4. `RAGTool.add_document()` 返回的是**字符串**，不是 dict（原写法 result.get("success") 会炸）；
5. `ask` 只认 enable_advanced_search，没有 enable_mqe/enable_hyde 开关。

详见同章 `p_mem_and_rag_运行记录.md`。
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# 统一切到 chap08 目录：.env / ./memory_data / ./knowledge_base 都相对这里落盘，
# 这样从任意目录（含 `streamlit run`）启动都不会把数据写到别处。
# 必须在 import hello_agents 之前做：它的 core/database_config.py 在导入时就会 load_dotenv()。
CHAP_DIR = Path(__file__).resolve().parent.parent
os.chdir(CHAP_DIR)
load_dotenv(CHAP_DIR / ".env")

from hello_agents.tools import MemoryTool, RAGTool  # noqa: E402

# markitdown(loader) 支持的常见格式，PDF 走 enhanced pdf 分支
SUPPORTED_SUFFIXES = {
    ".pdf", ".md", ".markdown", ".txt", ".doc", ".docx",
    ".ppt", ".pptx", ".xls", ".xlsx", ".csv", ".html", ".htm",
}


class PDFLearningAssistant:
    """文档学习助手。

    组成：
      * ``MemoryTool``  三层记忆：working(提问) / episodic(问答、加载事件) / semantic(笔记)
      * ``RAGTool``     文档入库(分块+embedding) 与检索问答(MQE+HyDE 高级检索)
    """

    def __init__(
        self,
        user_id: str = "default_user",
        collection_name: str = "rag_knowledge_base",
        memory_types: Optional[List[str]] = None,
    ) -> None:
        self.user_id = user_id
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 记忆工具：user_id 走构造参数，session_id 只能构造后注入
        # （run()->_add_memory() 仅在 current_session_id 为 None 时自动生成，所以这里注入有效）
        self.memory_tool = MemoryTool(user_id=self.user_id, memory_types=memory_types)
        self.memory_tool.current_session_id = self.session_id

        # RAG 工具：没有 embedding_model 参数，向量维度取自全局 embedder（chap08/.env 里的 DashScope）
        self.rag_tool = RAGTool(
            knowledge_base_path="./knowledge_base",
            collection_name=collection_name,
            rag_namespace=f"pdf_{user_id}",  # 按用户隔离，避免互相检索到对方的文档
        )

        self.stats: Dict[str, Any] = {
            "session_start": datetime.now(),
            "documents_loaded": 0,
            "questions_asked": 0,
            "concepts_learned": 0,
        }
        self.current_document: Optional[str] = None

    # ------------------------------------------------------------------ utils
    @property
    def ready(self) -> bool:
        """RAG 管道是否初始化成功（失败时看 ``init_error``）。"""
        return bool(getattr(self.rag_tool, "initialized", False))

    @property
    def init_error(self) -> str:
        return str(getattr(self.rag_tool, "init_error", ""))

    def _remember(
        self,
        content: str,
        memory_type: str = "working",
        importance: float = 0.6,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """写记忆的统一入口。

        ``run()`` 只转发 content / memory_type / importance（+file_path/modality），
        其余字段会被静默丢弃；要留下额外信息只能绕过 run() 直接进 memory_manager，
        且 episodic 仅持久化 metadata 中的 session_id/context/outcome/participants/tags，
        所以自定义字段要包在 ``context`` 里才会落库。
        """
        if context:
            memory_id = self.memory_tool.memory_manager.add_memory(
                content=content,
                memory_type=memory_type,
                importance=importance,
                metadata={"session_id": self.session_id, "context": context},
                auto_classify=False,
            )
            return f"✅ 记忆已添加 (ID: {memory_id[:8]}...)"

        return self.memory_tool.run({
            "action": "add",
            "content": content,
            "memory_type": memory_type,
            "importance": importance,
        })

    @staticmethod
    def _ok(msg: str, **extra: Any) -> Dict[str, Any]:
        return {"success": True, "message": msg, **extra}

    @staticmethod
    def _fail(msg: str, **extra: Any) -> Dict[str, Any]:
        return {"success": False, "message": msg, **extra}

    # ------------------------------------------------------------- load / ask
    def load_document(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> Dict[str, Any]:
        """把文档切块 + 向量化写入知识库（pdf/docx/md/txt... 均走 markitdown）。"""
        if not self.ready:
            return self._fail(f"RAG 工具未初始化: {self.init_error}")

        path = Path(file_path)
        if not path.exists():
            return self._fail(f"文件不存在: {file_path}")
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            return self._fail(f"不支持的格式 {path.suffix}，支持: {sorted(SUPPORTED_SUFFIXES)}")

        t0 = time.perf_counter()
        result = self.rag_tool.run({
            "action": "add_document",
            "file_path": str(path),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,   # 注意拼写：chunk_overlap，写错会被静默忽略(默认 100)
        })
        elapsed = time.perf_counter() - t0

        # add_document 返回字符串（成功以 ✅ 开头），不是 dict
        success = result.lstrip().startswith("✅")
        if not success:
            return self._fail(result, elapsed=elapsed)

        self.current_document = path.name
        self.stats["documents_loaded"] += 1
        self._remember(
            f"Loaded document: {self.current_document}",
            memory_type="episodic",
            importance=0.9,
            context={"event_type": "document_load", "file": str(path)},
        )
        return self._ok(result, document=self.current_document, elapsed=elapsed)

    def add_text(self, text: str, document_id: str) -> Dict[str, Any]:
        """直接向知识库补一段文本（无需先有文件，便于快速验证/补充资料）。"""
        if not self.ready:
            return self._fail(f"RAG 工具未初始化: {self.init_error}")
        if not text or not text.strip():
            return self._fail("文本内容不能为空")

        t0 = time.perf_counter()
        result = self.rag_tool.run({
            "action": "add_text",
            "text": text,
            "document_id": document_id or f"text_{int(time.time())}",
        })
        elapsed = time.perf_counter() - t0
        success = result.lstrip().startswith("✅")
        if success:
            self.current_document = self.current_document or document_id
        return (self._ok(result, elapsed=elapsed) if success
                else self._fail(result, elapsed=elapsed))

    def ask(
        self,
        question: str,
        enable_advanced_search: bool = True,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """检索 + 生成。问题记 working 记忆，问答记 episodic 记忆（带 event_type）。"""
        if not self.current_document:
            return self._fail("⚠️ 请先加载文档或添加文本知识！")
        if not question or not question.strip():
            return self._fail("问题不能为空")

        # 1) 工作记忆：当前会话正在问什么
        self._remember(f"用户提问: {question}", memory_type="working", importance=0.6)

        # 2) RAG 问答（enable_mqe/enable_hyde 在当前版本里不存在，只有 enable_advanced_search）
        t0 = time.perf_counter()
        answer = self.rag_tool.run({
            "action": "ask",
            "question": question,
            "enable_advanced_search": enable_advanced_search,
            "include_citations": True,
            "limit": limit,
        })
        elapsed = time.perf_counter() - t0

        if not isinstance(answer, str) or answer.lstrip().startswith("❌"):
            return self._fail(str(answer), elapsed=elapsed)

        # 3) 情景记忆：这一轮问答本身（回答留摘要，太长没意义）
        summary = answer[:150] + ("..." if len(answer) > 150 else "")
        self._remember(
            f"关于'{question}'的学习: {summary}",
            memory_type="episodic",
            importance=0.7,
            context={"event_type": "qa_interaction", "document": self.current_document},
        )

        self.stats["questions_asked"] += 1
        return self._ok("ok", answer=answer, elapsed=elapsed, question=question)

    def search_knowledge(self, query: str, limit: int = 3) -> Dict[str, Any]:
        """只检索、不调 LLM（省 token，适合快速看召回效果）。"""
        if not self.ready:
            return self._fail(f"RAG 工具未初始化: {self.init_error}")
        result = self.rag_tool.run({"action": "search", "query": query, "limit": limit})
        return self._ok(result, raw=result)

    # ---------------------------------------------------------------- memory
    def add_note(self, content: str, concept: Optional[str] = None) -> Dict[str, Any]:
        """把一条笔记写入语义记忆（concept 当前不会被持久化，故拼进正文）。"""
        if not content or not content.strip():
            return self._fail("笔记内容不能为空")
        text = f"[概念: {concept}] {content}" if concept else content
        msg = self._remember(text, memory_type="semantic", importance=0.8)
        self.stats["concepts_learned"] += 1
        return self._ok(msg)

    def recall(self, query: str, limit: int = 5, memory_type: Optional[str] = None) -> str:
        """按语义召回记忆（working/episodic/semantic，None=全部）。"""
        params: Dict[str, Any] = {"action": "search", "query": query, "limit": limit}
        if memory_type:
            params["memory_type"] = memory_type
        return self.memory_tool.run(params)

    def memory_summary(self, limit: int = 10) -> str:
        return self.memory_tool.run({"action": "summary", "limit": limit})

    def memory_stats(self) -> str:
        return self.memory_tool.run({"action": "stats"})

    def rag_stats(self) -> str:
        return self.rag_tool.run({"action": "stats"})

    # ---------------------------------------------------------------- report
    def get_stats(self) -> Dict[str, Any]:
        duration = (datetime.now() - self.stats["session_start"]).total_seconds()
        return {
            "session_id": self.session_id,
            "session_duration": f"{duration:.1f}s",
            "documents_loaded": self.stats["documents_loaded"],
            "questions_asked": self.stats["questions_asked"],
            "concepts_learned": self.stats["concepts_learned"],
            "current_document": self.current_document or "Not loaded",
            "rag_ready": self.ready,
            "rag_namespace": f"pdf_{self.user_id}",
        }

    def generate_report(self, save_to_file: Optional[str] = None) -> Dict[str, Any]:
        report = {
            "user_id": self.user_id,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "stats": self.get_stats(),
            "memory_summary": self.memory_summary(),
        }
        if save_to_file:
            with open(save_to_file, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=4)
        return report


if __name__ == "__main__":
    bot = PDFLearningAssistant(user_id="scc")
    print("rag ready:", bot.ready, "| init_error:", bot.init_error)
    print(bot.get_stats())
