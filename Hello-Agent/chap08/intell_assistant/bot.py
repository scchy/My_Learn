# python3
# Create Date: 2026-09-14
# Author: Scc_hy
# Func: 智能问答助手（PDF 学习助手）—— RAG 检索问答 + 三层记忆
# Run : /home/scc/anaconda3/envs/LLM/bin/python3 -c "from bot import PDFLearningAssistant as A; print(A('scc').ask('x'))"
# UI  : streamlit run app.py   （同目录，见 README.md）
# ===========================================================================================
"""PDF/文档学习助手：加载文档 -> 检索问答 -> 记忆沉淀。

与书稿描述的接口差异，本文件已按官方 hello-agents 0.2.8 适配：

1. 工具统一走 `tool.run({"action": ...})` 字典传参（Tool 基类标准接口；
   `execute(action, **kwargs)` 也存在，但 run() 才是对外的统一入口）；
2. `MemoryTool` 构造参数没有 `session_id`，会话 id 靠 `memory_tool.current_session_id` 注入；
3. `run()` 按固定字段取参，多传的 key（如 event_type/concept）会被**静默丢弃**；
4. `RAGTool.add_document()` 返回的是**字符串**，不是 dict（原写法 result.get("success") 会炸）；
5. `ask` 只认 enable_advanced_search，没有 enable_mqe/enable_hyde 开关。

详见同章 `p_mem_and_rag_运行记录.md`。

习题 4 扩展（见 ../习题.md 第 4 节）：

* 4.1 检索智能路由  `route_question()` 规则引擎 + 双路轻量探测，`ask(route=...)` 支持
                    rag / memory / hybrid 三路分发（memory、hybrid 不走 RAGTool.ask，
                    而是取上下文后单次 `rag_tool.llm.invoke`，省一次检索）；
* 4.2 智能学习报告  `generate_report()` 从 episodic 记忆还原学习轨迹，按主题聚类
                    挖掘知识盲点（反复问/疑难词/无笔记三信号），规则推荐 + 可选 LLM 叙事；
* 4.3 多用户隔离    `isolation_mode`（namespace 载荷过滤 / collection 独立集合），
                    `verify_isolation()` 做越权探测自检。
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# 统一切到 chap08 目录：.env / ./memory_data / ./knowledge_base 都相对这里落盘，
# 这样从任意目录（含 `streamlit run`）启动都不会把数据写到别处。
# 必须在 import hello_agents 之前做：它的 core/database_config.py 在导入时就会 load_dotenv()。
CHAP_DIR = Path(__file__).resolve().parent.parent
os.chdir(CHAP_DIR)
load_dotenv(CHAP_DIR / ".env")

# shim 在 chap08 目录而本文件在 intell_assistant/ 下，补一下搜索路径
sys.path.insert(0, str(CHAP_DIR))
import _qdrant_embedded  # noqa: E402,F401  # 0.2.8 补丁：QDRANT_PATH 设置时走嵌入式本地模式

from hello_agents.tools import MemoryTool, RAGTool  # noqa: E402

# markitdown(loader) 支持的常见格式，PDF 走 enhanced pdf 分支
SUPPORTED_SUFFIXES = {
    ".pdf", ".md", ".markdown", ".txt", ".doc", ".docx",
    ".ppt", ".pptx", ".xls", ".xlsx", ".csv", ".html", ".htm",
}

# ================================================================= 4.1 路由规则
# 第一层：意图硬规则（正则命中即定向，省掉探测开销）。
# 设计原则：规则只捕捉「明确指向某一侧」的措辞，模糊问题一律交给第二层探测仲裁。
_RE_MEMORY_INTENT = re.compile(
    r"我(之前|刚才|上次|前面|最近|们)|之前(问|聊|说|学)|上次|刚才"
    r"|继续(刚才|上面|之前|上个)|学了(什么|哪些)|问过(什么|哪些)"
    r"|学习(记录|进度|历史|报告|轨迹)|掌握得(怎么样|如何)|盲点|薄弱"
)
_RE_RAG_INTENT = re.compile(
    r"文档|这篇|这份|资料|原文|章节|书里|pdf|材料"
    r"|根据(文档|资料|所学|材料)|总结(一下|这篇|这份)"
)
# 第二层：疑难信号（用于 4.2 盲点识别，不参与路由方向）
_RE_CONFUSED = re.compile(
    r"没(懂|明白|理解)|不懂|不明白|不理解|还是不|再(讲|解释)一遍|有什么区别|区别在哪"
)

# 路由决策的可调参数（余弦分数来自 DashScope text-embedding-v3，经验值）
RAG_STRONG_SCORE = 0.45     # RAG 探测 top-1 相似度超过它视为知识库有干货
MEMORY_STRONG_HITS = 2      # 记忆探测命中条数达到它视为记忆相关（retrieve 不返回相似度，用命中数近似）

# 路由决策与理由（给 UI 展示，也写进 episodic 记忆便于报告回溯）
ROUTES = ("auto", "rag", "memory", "hybrid")


class PDFLearningAssistant:
    """文档学习助手。

    组成：
      * ``MemoryTool``  三层记忆：working(提问) / episodic(问答、加载事件) / semantic(笔记)
      * ``RAGTool``     文档入库(分块+embedding) 与检索问答(MQE+HyDE 高级检索)
    """

    def __init__(
        self,
        user_id: str = "default_user",
        collection_name: Optional[str] = None,
        memory_types: Optional[List[str]] = None,
        isolation_mode: str = "namespace",
    ) -> None:
        """
        Args:
            isolation_mode: 4.3 多用户隔离策略——
                * "namespace"  （默认）所有用户共用一个 collection，检索时按
                  ``rag_namespace = pdf_{user_id}`` 载荷过滤。省资源、建索引快，
                  适合用户多、单用户数据量小的教学/轻量场景；
                * "collection" 每用户独立 collection（``rag_kb_{user_id}``）。
                  强隔离：单用户可整库删除/备份/迁移，互不影响；用户多了管理成本高。
                  注意：与 namespace 模式的数据互不可见，切换模式等于换库。
        """
        self.user_id = user_id
        self.isolation_mode = isolation_mode
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 记忆工具：user_id 走构造参数（记忆侧隔离 = 同一 memories 集合内 user_id 载荷过滤，
        # library 层没有按用户分 collection 的选项）；session_id 只能构造后注入
        # （run()->_add_memory() 仅在 current_session_id 为 None 时自动生成，所以这里注入有效）
        self.memory_tool = MemoryTool(user_id=self.user_id, memory_types=memory_types)
        self.memory_tool.current_session_id = self.session_id

        # RAG 工具：没有 embedding_model 参数，向量维度取自全局 embedder（chap08/.env 里的 DashScope）
        if isolation_mode == "collection" and collection_name is None:
            collection_name = f"rag_kb_{user_id}"
            rag_namespace = "default"
        else:
            collection_name = collection_name or "rag_knowledge_base"
            rag_namespace = f"pdf_{user_id}"  # 按用户隔离，避免互相检索到对方的文档
        self.rag_tool = RAGTool(
            knowledge_base_path="./knowledge_base",
            collection_name=collection_name,
            rag_namespace=rag_namespace,
        )

        self.stats: Dict[str, Any] = {
            "session_start": datetime.now(),
            "documents_loaded": 0,
            "questions_asked": 0,
            "concepts_learned": 0,
            "route_counts": {},  # 4.1 各路由被采用次数（auto 决策结果也计入）
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

    # ------------------------------------------------- 4.1 检索智能路由
    def _probe_rag(self, question: str, limit: int = 3) -> Dict[str, Any]:
        """知识库探测：一次向量检索拿 top 分数（不调 LLM，成本可忽略）。"""
        if not self.ready:
            return {"ok": False, "top_score": 0.0, "hits": 0, "note": "RAG 未初始化"}
        try:
            pipeline = self.rag_tool._get_pipeline()
            results = pipeline["search"](query=question, top_k=limit) or []
            scores = [float(r.get("score", 0.0)) for r in results]
            return {
                "ok": True,
                "top_score": max(scores, default=0.0),
                "hits": len(scores),
                "note": "",
            }
        except Exception as e:  # 探测失败不应阻断提问，按无信号处理
            return {"ok": False, "top_score": 0.0, "hits": 0, "note": str(e)}

    def _probe_memory(self, question: str, limit: int = 5) -> Dict[str, Any]:
        """记忆探测。注意：retrieve_memories 不返回相似度（只按重要性排序），
        因此用命中条数近似记忆相关度——这是 library 接口限制下的务实取舍。"""
        try:
            mems = self.memory_tool.memory_manager.retrieve_memories(
                query=question, limit=limit, min_importance=0.3)
            return {
                "ok": True,
                "hits": len(mems),
                "top_importance": max((m.importance for m in mems), default=0.0),
            }
        except Exception as e:
            return {"ok": False, "hits": 0, "top_importance": 0.0, "note": str(e)}

    def route_question(self, question: str) -> Dict[str, Any]:
        """路由决策（不调 LLM）：意图硬规则 -> 双路探测仲裁。

        决策表（探测层）::

            RAG强(top分≥RAG_STRONG_SCORE)  记忆强(命中≥MEMORY_STRONG_HITS)  -> hybrid
            RAG强                          记忆弱                          -> rag
            RAG弱                          记忆强                          -> memory
            双弱                           -> 有数据的一侧兜底（避免硬拒答）

        Returns: {"route": rag|memory|hybrid, "reason": str, "probe": {...}}
        """
        q = question.strip()

        # 第一层：意图硬规则。规则只捕捉「明确指向某一侧」的措辞，避免误伤普通问题
        if _RE_MEMORY_INTENT.search(q):
            return {"route": "memory", "reason": "意图规则：问题指向个人学习历史/记忆", "probe": {}}
        if _RE_RAG_INTENT.search(q) and self.ready:
            return {"route": "rag", "reason": "意图规则：问题明确指向文档内容", "probe": {}}

        # 第二层：双路探测仲裁（各一次向量检索，零 LLM 调用）
        rag_p = self._probe_rag(q)
        mem_p = self._probe_memory(q)
        rag_strong = rag_p["ok"] and rag_p["top_score"] >= RAG_STRONG_SCORE
        mem_strong = mem_p["ok"] and mem_p["hits"] >= MEMORY_STRONG_HITS

        if rag_strong and mem_strong:
            route = "hybrid"
            reason = (f"探测仲裁：知识库 top 相似度 {rag_p['top_score']:.2f}、"
                      f"记忆命中 {mem_p['hits']} 条，双侧均有依据")
        elif rag_strong:
            route, reason = "rag", f"探测仲裁：知识库 top 相似度 {rag_p['top_score']:.2f}"
        elif mem_strong:
            route = "memory"
            reason = (f"探测仲裁：记忆命中 {mem_p['hits']} 条，"
                      f"知识库相似度 {rag_p['top_score']:.2f} 低于阈值")
        elif rag_p["ok"] and rag_p["hits"] > 0:
            route, reason = "rag", "探测仲裁：仅知识库弱命中，记忆无相关记录"
        elif mem_p["ok"] and mem_p["hits"] > 0:
            route, reason = "memory", "探测仲裁：仅记忆弱命中，知识库无相关内容"
        else:
            route = "rag" if self.ready else "memory"
            reason = "双侧均无强证据，回退到可用检索路径（可能无法回答）"
        return {"route": route, "reason": reason, "probe": {"rag": rag_p, "memory": mem_p}}

    def _answer_with_context(
        self, question: str, rag_ctx: str, mem_ctx: str,
    ) -> str:
        """memory / hybrid 路径的作答：拼上下文后单次 LLM 调用。

        与 RAGTool.ask 的差别：上下文里除了知识库资料还能注入用户记忆
        （学习历史、笔记），这是「个性化回答」的来源；代价是放弃了 RAGTool
        内置的 MQE/HyDE 高级检索（对记忆类问题本就用不上）。
        """
        system = (
            "你是个人学习助手。仅依据下面提供的【知识库资料】和【用户记忆】回答问题；"
            "资料不足以回答时明确说明，不要编造资料中不存在的内容。"
            "若用户记忆与资料冲突，以更新的为准并指出差异。"
        )
        parts = []
        if rag_ctx:
            parts.append(f"【知识库资料】\n{rag_ctx}")
        if mem_ctx:
            parts.append(f"【用户记忆】\n{mem_ctx}")
        user = question.strip() + ("\n\n" + "\n\n".join(parts) if parts else "")
        return self.rag_tool.llm.invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])

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
        route: str = "auto",
    ) -> Dict[str, Any]:
        """4.1 路由化检索 + 生成。

        route: "auto" 智能路由 | "rag" 仅知识库 | "memory" 仅记忆 | "hybrid" 混合。
        问题记 working 记忆；问答记 episodic 记忆（event_type + route 供 4.2 报告回溯）。
        """
        if not question or not question.strip():
            return self._fail("问题不能为空")
        if route not in ROUTES:
            return self._fail(f"未知 route: {route}，可选 {ROUTES}")

        # 1) 路由决策（auto 走规则+探测；手动指定则直接采用）
        decision = (self.route_question(question) if route == "auto"
                    else {"route": route, "reason": f"手动指定 ({route})", "probe": {}})
        final_route = decision["route"]

        # 2) 工作记忆：当前会话正在问什么（无论哪条路径，提问本身都值得暂存）
        self._remember(f"用户提问: {question}", memory_type="working", importance=0.6)

        # 3) 按路径生成
        t0 = time.perf_counter()
        if final_route == "rag":
            if not self.current_document:
                return self._fail("⚠️ 知识库为空：请先加载文档/添加文本，或改用 memory / hybrid 路由！")
            answer = self.rag_tool.run({   # enable_mqe/enable_hyde 不存在，只有 enable_advanced_search
                "action": "ask",
                "question": question,
                "enable_advanced_search": enable_advanced_search,
                "include_citations": True,
                "limit": limit,
            })
            if not isinstance(answer, str) or answer.lstrip().startswith("❌"):
                return self._fail(str(answer), elapsed=time.perf_counter() - t0)
        else:
            # memory / hybrid：取上下文后单次 LLM 调用（不走 RAGTool.ask，那里塞不进记忆）
            mem_ctx = self.memory_tool.get_context_for_query(question, limit=limit)
            rag_ctx = ""
            if final_route == "hybrid" and self.ready and self.current_document:
                rag_ctx = self.rag_tool.get_relevant_context(question, limit=limit)
            if not mem_ctx and not rag_ctx:
                return self._fail("⚠️ 记忆与知识库均无相关内容，暂时无法回答这个问题")
            try:
                answer = self._answer_with_context(question, rag_ctx, mem_ctx)
            except Exception as e:
                return self._fail(f"LLM 调用失败: {type(e).__name__}: {e}",
                                  elapsed=time.perf_counter() - t0)
        elapsed = time.perf_counter() - t0

        # 4) 情景记忆：这一轮问答本身（回答留摘要，route 进 context 供报告统计）
        summary = answer[:150] + ("..." if len(answer) > 150 else "")
        self._remember(
            f"关于'{question}'的学习: {summary}",
            memory_type="episodic",
            importance=0.7,
            context={
                "event_type": "qa_interaction",
                "document": self.current_document,
                "route": final_route,
            },
        )

        self.stats["questions_asked"] += 1
        self.stats["route_counts"][final_route] = self.stats["route_counts"].get(final_route, 0) + 1
        return self._ok("ok", answer=answer, elapsed=elapsed, question=question,
                        route=final_route, reason=decision["reason"])

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
            "route_counts": dict(self.stats["route_counts"]),
            "current_document": self.current_document or "Not loaded",
            "rag_ready": self.ready,
            "rag_namespace": self.rag_tool.rag_namespace,
            "isolation_mode": self.isolation_mode,
        }

    # ------------------------------------------------- 4.2 智能学习报告
    # episodic 记忆内容的三种固定前缀（本类写入时的格式，报告按它反解析）
    _QA_PAT = re.compile(r"^关于'(.+?)'的学习")
    _DOC_PAT = re.compile(r"^Loaded document: (.+)$")
    _NOTE_PAT = re.compile(r"^\[概念: (.+?)\]")
    # 主题聚类：两个问题 SequenceMatcher 相似度超过它归入同主题（短中文问题经验值）
    _TOPIC_SIM = 0.45

    def _collect_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """从记忆里还原学习历史（跨会话持久，重启不丢）。

        数据源：
          * episodic —— SQLite 权威存储（doc_store.search_memories，带 user_id 过滤，
            进程重启后依然可读）。注意不能用 EpisodicMemory.get_all()：它只返回
            进程内 episodes 列表，且构造 MemoryItem 时访问了 Episode 不存在的
            metadata 属性，实际会抛 AttributeError。
          * semantic —— 内存 semantic_memories（library 未提供按过滤批量读取接口，
            跨进程笔记覆盖检测是已知局限，重启后以本会话笔记为准）。
        """
        out: Dict[str, List[Dict[str, Any]]] = {"qa": [], "docs": [], "notes": []}

        epi = self.memory_tool.memory_manager.memory_types.get("episodic")
        rows: List[Dict[str, Any]] = []
        if epi is not None:
            try:
                rows = epi.doc_store.search_memories(
                    user_id=self.user_id, memory_type="episodic", limit=1000)
            except Exception:
                rows = []
        for r in rows:
            content = r.get("content") or ""
            props = r.get("properties") or {}
            ts = (datetime.fromtimestamp(r["timestamp"])
                  if r.get("timestamp") else None)
            if m := self._QA_PAT.match(content):
                out["qa"].append({
                    "question": m.group(1),
                    "time": ts,
                    "session": props.get("session_id", "未知会话"),
                    "route": (props.get("context") or {}).get("route", "rag"),
                    "answer_preview": content[len(m.group(0)):].strip(": "),
                })
            elif m := self._DOC_PAT.match(content):
                out["docs"].append({"document": m.group(1), "time": ts})

        sem = self.memory_tool.memory_manager.memory_types.get("semantic")
        sem_items = []
        if sem is not None:
            try:
                sem_items = sem.get_all()
            except Exception:
                sem_items = []
        for item in sem_items:
            note = {"note": (getattr(item, "content", "") or "").strip(),
                    "time": getattr(item, "timestamp", None)}
            if hit := self._NOTE_PAT.match(note["note"]):
                note["concept"] = hit.group(1)
            out["notes"].append(note)

        for key in out:
            out[key].sort(key=lambda x: x["time"] or datetime.min)
        return out

    @classmethod
    def _group_topics(cls, questions: List[str]) -> List[List[str]]:
        """问题按语义近似度聚成主题组（平均链接贪心聚类）。

        不再调 embedding：报告要离线可跑，SequenceMatcher 对「同一问题的
        换种问法」足够灵敏；两两比较 O(Q²·len²)，Q 是问题数（<几百），毫秒级。
        """
        groups: List[List[str]] = []
        for q in questions:
            for g in groups:
                if max(SequenceMatcher(None, q, member).ratio() for member in g) >= cls._TOPIC_SIM:
                    g.append(q)
                    break
            else:
                groups.append([q])
        groups.sort(key=len, reverse=True)   # 问得最多的主题排前面
        return groups

    def _find_blind_spots(self, history: Dict[str, List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """三信号挖盲点：反复问 + 疑难词 + 无笔记沉淀。

        盲点分 = 0.5*反复问 + 0.3*含疑难词 + 0.2*未沉淀笔记。
        权重含义：行为证据（重复提问）最可信，措辞信号次之，笔记缺失只是弱证据。
        """
        notes_text = "\n".join(n["note"] for n in history["notes"])
        topics, blind_spots = [], []
        for group in self._group_topics([qa["question"] for qa in history["qa"]]):
            representative = max(group, key=len)
            confused = any(_RE_CONFUSED.search(q) for q in group)
            covered = any(
                SequenceMatcher(None, representative, note).ratio() >= 0.35
                for note in notes_text.split("\n") if note
            ) or (representative[:6] in notes_text if representative else False)
            repeat_score = min(1.0, (len(group) - 1) / 2)      # 问 3 次即满分
            score = 0.5 * repeat_score + 0.3 * confused + 0.2 * (not covered)
            topic = {
                "topic": representative,
                "ask_count": len(group),
                "questions": group,
                "confused": confused,
                "noted": covered,
                "blind_score": round(score, 2),
            }
            topics.append(topic)
            if score >= 0.3:
                blind_spots.append(topic)
        blind_spots.sort(key=lambda t: t["blind_score"], reverse=True)
        return topics, blind_spots[:3]

    @staticmethod
    def _recommend(blind_spots: List[Dict[str, Any]], history: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """规则推荐：盲点驱动，无盲点时给推进建议（不调 LLM，可解释）。"""
        if not history["qa"]:
            return ["还没有提问记录：上传一份文档开始学习，问答会自动沉淀为学习轨迹"]
        recs = []
        for spot in blind_spots:
            why = []
            if spot["ask_count"] >= 2:
                why.append(f"已问 {spot['ask_count']} 次")
            if spot["confused"]:
                why.append("提问中含'没懂/区别'等疑难信号")
            if not spot["noted"]:
                why.append("尚无笔记沉淀")
            recs.append(f"重学「{spot['topic']}」（{'、'.join(why) or '综合信号'}）："
                        f"重读对应章节后用 add_note 沉淀一条笔记")
        if not recs:
            if history["docs"]:
                latest = history["docs"][-1]["document"]
                recs.append(f"近期提问掌握良好：建议继续《{latest}》的后续章节，或上传下一份文档")
            else:
                recs.append("近期提问掌握良好：建议上传新文档拓展学习范围")
        recs.append("下次会话可直接问「我之前学了什么」，路由会自动走记忆侧回顾学习轨迹")
        return recs

    def _llm_narrative(self, report_core: Dict[str, Any]) -> Optional[str]:
        """可选的一次 LLM 叙事：把结构化数据转成自然语言报告（失败返回 None 不阻断）。"""
        try:
            payload = {
                "学习轨迹": report_core["trajectory"],
                "主题掌握": [{"主题": t["topic"], "提问次数": t["ask_count"],
                              "已记笔记": t["noted"]} for t in report_core["topics"][:8]],
                "盲点": [{"主题": s["topic"], "信号": s["questions"][:3]} for s in report_core["blind_spots"]],
                "推荐": report_core["recommendations"],
            }
            return self.rag_tool.llm.invoke([
                {"role": "system", "content": (
                    "你是学习教练。基于给定的结构化学习数据写一份简短的中文学习报告，"
                    "分「学习轨迹」「掌握情况」「盲点」「下一步建议」四段，每段不超过 3 句话，"
                    "语气具体、可直接执行，不要编造数据里没有的信息。")},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
            ])
        except Exception:
            return None

    def generate_report(
        self,
        save_to_file: Optional[str] = None,
        with_llm: bool = True,
    ) -> Dict[str, Any]:
        """4.2 智能学习报告：轨迹（episodic）+ 盲点（三信号）+ 推荐（规则/LLM）。

        用到的记忆与检索：
          * episodic 记忆    —— 学习轨迹的事实来源（qa_interaction / document_load 事件）；
          * semantic 记忆    —— 笔记，作为「已掌握/已沉淀」的证据参与盲点判定；
          * working 记忆     —— 不参与报告（会话级 TTL 数据对长期报告无意义）；
          * 检索策略         —— 轨迹还原用 event_type 过滤 + 时间排序（结构化检索），
                                主题聚类用文本相似度（离线，不依赖 embedding API）。
        """
        history = self._collect_history()
        topics, blind_spots = self._find_blind_spots(history)
        recommendations = self._recommend(blind_spots, history)

        # 轨迹：按会话分组还原时间线（episodic 的 metadata.session_id 已持久化，跨会话可辨）
        sessions: Dict[str, List[Dict[str, Any]]] = {}
        for qa in history["qa"]:
            sessions.setdefault(qa["session"], []).append(
                {"time": qa["time"], "question": qa["question"], "route": qa["route"]})
        trajectory = [{"session": name, "events": events} for name, events in sessions.items()]
        for d in history["docs"]:
            trajectory.append({"session": "文档加载",
                               "events": [{"time": d["time"], "document": d["document"]}]})

        report = {
            "user_id": self.user_id,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "stats": self.get_stats(),
            "memory_summary": self.memory_summary(),
            "trajectory": trajectory,
            "topics": topics,
            "blind_spots": blind_spots,
            "recommendations": recommendations,
            "llm_summary": self._llm_narrative({
                "trajectory": trajectory, "topics": topics,
                "blind_spots": blind_spots, "recommendations": recommendations,
            }) if with_llm else None,
        }
        if save_to_file:
            with open(save_to_file, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=4, default=str)
        return report

    # 兼容别名：习题 4.2 语义上叫「智能报告」
    generate_smart_report = generate_report

    # ------------------------------------------------- 4.3 多用户隔离自检
    _RESULT_LINE = re.compile(r"^\s*\d+\.")   # 工具检索结果的条目行以 "1." 编号

    @staticmethod
    def _result_text(result: Any) -> str:
        """把检索调用结果统一转成可解析的文本（search_knowledge 返回 dict，
        recall 返回 str；str(dict) 的换行是字面 \\n，块解析会失配）。"""
        if isinstance(result, dict):
            return str(result.get("message") or result.get("raw") or result)
        return str(result)

    @classmethod
    def _search_hit(cls, formatted: str, key: str) -> bool:
        """判断格式化检索结果里是否真命中了含 key 的条目。

        两个坑：
          1. 「未找到与 'xxx' 相关的内容」会回显查询词，直接 `key in text`
             会把"未命中"误判成命中；
          2. 条目是跨行的——"1. 文档: ..." 后面一行才是内容预览，只查编号行
             会把"命中"误判成未命中。
        因此按编号行切块，key 出现在任一条目块里才算命中（miss 消息没有编号行）。
        """
        items: List[str] = []
        buf: Optional[List[str]] = None
        for line in (formatted or "").splitlines():
            if cls._RESULT_LINE.match(line):
                if buf is not None:
                    items.append(" ".join(buf))
                buf = [line]
            elif buf is not None:
                buf.append(line)
        if buf is not None:
            items.append(" ".join(buf))
        return any(key in item for item in items)

    def verify_isolation(self, probe_user_id: Optional[str] = None) -> Dict[str, Any]:
        """越权探测自检：确认「我的数据，别的用户检索不到」。

        做法（教学级渗透测试，非破坏式）：
          1. 以随机探针用户建对端实例（新用户保证其库为空）；
          2. 给当前用户写入带唯一标记的知识（RAG）+ 语义记忆；
          3. 两端检索同一标记——自己应命中、探针端应全空，否则隔离失效。

        两种隔离模式的验证点相同：namespace 模式验证的是 rag_namespace 载荷
        过滤没有串库；collection 模式验证的是探针用户根本看不到我的集合。
        注：自检会给当前用户留下两条带标记的数据（无害，留在自己库里）。
        嵌入式 Qdrant 在本 fork 已做进程级 client 复用，同进程建第二个实例
        不会触发目录锁；服务端模式（QDRANT_URL）同样适用。
        """
        probe_user_id = probe_user_id or f"probe_{datetime.now().strftime('%H%M%S')}"
        marker = f"隔离自检标记{int(time.time())}"
        res: Dict[str, Any] = {"probe_user": probe_user_id, "marker": marker}

        # a) 当前用户写入标记（RAG 文本 + 语义记忆）
        rag_written = True
        if self.ready:
            add = self.add_text(
                f"{marker}：这是 {self.user_id} 的私有知识，含独特检索词 zebra{int(time.time())}。",
                document_id=f"iso_check_{int(time.time())}")
            rag_written = bool(add.get("success"))
        else:
            rag_written = False
        self._remember(f"{marker}：{self.user_id} 的私有记忆", memory_type="semantic", importance=0.8)

        # b) 我方检索应命中（search_knowledge 返回 dict，必须取 message 再判，
        #    str(dict) 会把换行变成字面 \n，块解析会全部失配）
        key = marker[:6]
        if rag_written:
            mine_rag = self._result_text(self.search_knowledge(marker, limit=3))
            res["self_rag_hit"] = self._search_hit(mine_rag, key)
        else:
            res["self_rag_hit"] = None   # RAG 不可用时跳过该项
        mine_mem = self._result_text(self.recall(marker, limit=3))
        res["self_memory_hit"] = self._search_hit(mine_mem, key)

        # c) 探针用户检索同一标记，应全部落空
        try:
            probe_bot = PDFLearningAssistant(user_id=probe_user_id, isolation_mode=self.isolation_mode)
        except Exception as e:
            # 嵌入式 Qdrant 的共享 client 被上游析构关闭时，同进程再建实例会失败
            #（library 的 __del__ 会关掉进程级共享 client），此时自检降级为提示
            return self._fail(f"探针实例创建失败（隔离模式={self.isolation_mode}）：{e}",
                              **res, rag_isolated=None, memory_isolated=None, passed=None)
        if res["self_rag_hit"]:
            probe_rag = self._result_text(probe_bot.search_knowledge(marker, limit=3))
            res["probe_rag_hit"] = self._search_hit(probe_rag, key)
        else:
            res["probe_rag_hit"] = None
        probe_mem = self._result_text(probe_bot.recall(marker, limit=3))
        res["probe_memory_hit"] = self._search_hit(probe_mem, key)

        res["rag_isolated"] = (None if res["self_rag_hit"] is None
                               else (res["self_rag_hit"] and not res["probe_rag_hit"]))
        res["memory_isolated"] = res["self_memory_hit"] and not res["probe_memory_hit"]
        res["passed"] = all(v is True or v is None for v in
                            (res["rag_isolated"], res["memory_isolated"]))
        verdict = "✅ 隔离有效" if res["passed"] else "❌ 检测到跨用户数据泄漏！"
        return self._ok(verdict, **res)


if __name__ == "__main__":
    bot = PDFLearningAssistant(user_id="scc")
    print("rag ready:", bot.ready, "| init_error:", bot.init_error)
    print(bot.get_stats())
