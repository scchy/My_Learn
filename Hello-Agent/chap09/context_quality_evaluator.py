# -*- coding: utf-8 -*-
# Author: scc_hy
# Func: 9.3.4 ContextBuilder + 上下文质量评估（规则引擎 + LLM 增强层）+ 习题 2.3 混合级联压缩
# Run : /home/scc/anaconda3/envs/LLM/bin/python3 context_quality_evaluator.py
#       加 --no-llm 可关闭 LLM，退回纯规则模式（零 API 成本）
"""
context_quality_evaluator.py
============================
基于 DataWhale《Hello-Agents》第 9 章 9.3.4 的 ContextBuilder，
增加「上下文质量评估」功能：每次 build() 后自动评估
信息密度 / 相关性 / 完整性，并给出优化建议。

设计要点
--------
1. 通过子类化 + 重写 `_select` / `_compress` 捕获构建过程轨迹（Trace），
   不侵入原有 GSSC 流水线逻辑。
2. 评估完全基于 Trace 中记录的候选/选中/丢弃信息，可解释性强：
   每个分数都能追溯到具体的信息包（ContextPacket）。
3. 两层评估体系：
   - 规则层（零成本、确定性）：阈值命中即触发诊断与建议；
   - LLM 增强层（可选，conda 环境 hello_agents.HelloAgentsLLM，读同目录 .env）：
     a) 相关性打分：一次批量调用替代逐包关键词 Jaccard（弥补中文/同义表述失配）；
     b) 建议生成：基于 Trace+指标复核，发现规则覆盖不到的内容级问题。
   LLM 不可用或调用失败时自动回退纯规则，脚本永远可独立运行。
4. HybridCascadeContextBuilder（习题 2.3）：把 Compress 升级为 S0-S5 级联——
   无损去重 → 价值驱逐 → 分通道滑窗 → LLM 摘要回收 → 存活率校验 → 硬截断兜底；
   Trace 记录每级触发情况与查询覆盖率存活率（比 token 差值更真实的压缩保真度量）。
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
import json
import math
import os
import re
import sys

# ----------------------------------------------------------------------
# 9.3.2 核心数据结构（与书中一致）
# ----------------------------------------------------------------------

@dataclass
class ContextPacket:
    """候选信息包（与 9.3.2 节一致）"""
    content: str
    timestamp: datetime
    token_count: int
    relevance_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.relevance_score = max(0.0, min(1.0, self.relevance_score))


@dataclass
class ContextConfig:
    """上下文构建配置（9.3.2 参数 + 评估功能新增开关）"""
    max_tokens: int = 3000
    reserve_ratio: float = 0.2
    min_relevance: float = 0.1
    enable_compression: bool = True
    recency_weight: float = 0.3
    relevance_weight: float = 0.7
    # --- 新增：质量评估 ---
    enable_quality_eval: bool = True
    # --- 新增：LLM 增强（不可用时自动回退纯规则） ---
    llm_relevance: bool = True      # 相关性打分用 LLM 批量评估（替代关键词 Jaccard）
    llm_suggestions: bool = True    # 规则建议之上叠加 LLM 复核建议
    llm_compression: bool = True    # 级联压缩 S3 用 LLM 摘要回收（关闭则退化为确定性 S0-S2+S5）

    def __post_init__(self):
        assert 0.0 <= self.reserve_ratio <= 1.0, "reserve_ratio 必须在 [0, 1] 范围内"
        assert 0.0 <= self.min_relevance <= 1.0, "min_relevance 必须在 [0, 1] 范围内"
        assert abs(self.recency_weight + self.relevance_weight - 1.0) < 1e-6, \
            "recency_weight + relevance_weight 必须等于 1.0"


# ----------------------------------------------------------------------
# 9.3.3 GSSC 流水线（按书中代码精简复刻，保证本文件可独立运行）
# ----------------------------------------------------------------------


class ContextBuilder:
    def __init__(self, memory_tool=None, rag_tool=None, config: Optional[ContextConfig] = None):
        self.memory_tool = memory_tool
        self.rag_tool = rag_tool
        self.config = config or ContextConfig()

    def build(self, user_query: str, conversation_history=None,
              system_instructions: Optional[str] = None,
              custom_packets: Optional[List[ContextPacket]] = None) -> str:
        packets = self._gather(user_query, conversation_history, system_instructions, custom_packets)
        selected = self._select(packets, user_query, self._available_tokens())
        context = self._structure(selected, user_query)
        if self.config.enable_compression:
            context = self._compress(context, self.config.max_tokens)
        return context

    def _available_tokens(self) -> int:
        return int(self.config.max_tokens * (1 - self.config.reserve_ratio))

    def _count_tokens(self, text: str) -> int:
        chinese_chars = sum(1 for ch in text if '一' <= ch <= '鿿')
        english_words = len([w for w in text.split() if w])
        return int(chinese_chars + english_words * 1.3)

    # ---- Gather ----
    def _gather(self, user_query, conversation_history=None,
                system_instructions=None, custom_packets=None) -> List[ContextPacket]:
        packets = []
        if system_instructions:
            packets.append(ContextPacket(
                content=system_instructions, timestamp=datetime.now(),
                token_count=self._count_tokens(system_instructions),
                relevance_score=1.0,
                metadata={"type": "system_instruction", "priority": "high"}))
        if self.memory_tool:
            try:
                results = self.memory_tool.run({"action": "search", "query": user_query, "limit": 10})
                for r in results:
                    packets.append(ContextPacket(
                        content=r["content"], timestamp=datetime.now(),
                        token_count=self._count_tokens(r["content"]),
                        relevance_score=r.get("score", 0.5),
                        metadata={"type": "memory"}))
            except Exception as e:
                print(f"[WARNING] 记忆检索失败: {e}")
        if self.rag_tool:
            try:
                results = self.rag_tool.run({"action": "search", "query": user_query, "limit": 5})
                for r in results:
                    packets.append(ContextPacket(
                        content=r["content"], timestamp=datetime.now(),
                        token_count=self._count_tokens(r["content"]),
                        relevance_score=r.get("score", 0.5),
                        metadata={"type": "rag_result"}))
            except Exception as e:
                print(f"[WARNING] RAG 检索失败: {e}")
        if conversation_history:
            for msg in conversation_history[-5:]:
                packets.append(ContextPacket(
                    content=f"{msg['role']}: {msg['content']}", timestamp=datetime.now(),
                    token_count=self._count_tokens(msg["content"]),
                    relevance_score=0.6,
                    metadata={"type": "conversation_history", "role": msg["role"]}))
        if custom_packets:
            packets.extend(custom_packets)
        return packets

    # ---- Select ----
    def _select(self, packets, user_query, available_tokens):
        system_packets = [p for p in packets if p.metadata.get("type") == "system_instruction"]
        others = [p for p in packets if p.metadata.get("type") != "system_instruction"]
        system_tokens = sum(p.token_count for p in system_packets)
        if available_tokens - system_tokens <= 0:
            return system_packets
        scored = []
        for p in others:
            if p.relevance_score == 0.5:
                p.relevance_score = self._calculate_relevance(p.content, user_query)
            if p.relevance_score >= self.config.min_relevance:
                scored.append((self._combined_score(p), p))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected, used = list(system_packets), system_tokens
        for s, p in scored:
            if used + p.token_count <= available_tokens:
                selected.append(p)
                used += p.token_count
            else:
                break
        return selected

    def _combined_score(self, p: ContextPacket) -> float:
        return (self.config.relevance_weight * p.relevance_score +
                self.config.recency_weight * self._calculate_recency(p.timestamp))

    def _calculate_relevance(self, content: str, query: str) -> float:
        content_words = set(content.lower().split())
        query_words = set(query.lower().split())
        if not query_words:
            return 0.0
        intersection = content_words & query_words
        union = content_words | query_words
        return len(intersection) / len(union) if union else 0.0

    def _calculate_recency(self, timestamp: datetime) -> float:
        age_hours = (datetime.now() - timestamp).total_seconds() / 3600
        return max(0.1, min(1.0, math.exp(-0.1 * age_hours / 24)))

    # ---- Structure ----
    def _structure(self, selected, user_query):
        groups = {"system": [], "evidence": [], "context": []}
        for p in selected:
            t = p.metadata.get("type", "general")
            if t == "system_instruction":
                groups["system"].append(p.content)
            elif t in ("rag_result", "knowledge"):
                groups["evidence"].append(p.content)
            else:
                groups["context"].append(p.content)
        sections = []
        if groups["system"]:
            sections.append("[Role & Policies]\n" + "\n".join(groups["system"]))
        sections.append(f"[Task]\n{user_query}")
        if groups["evidence"]:
            sections.append("[Evidence]\n" + "\n---\n".join(groups["evidence"]))
        if groups["context"]:
            sections.append("[Context]\n" + "\n".join(groups["context"]))
        sections.append("[Output]\n请基于以上信息,提供准确、有据的回答。")
        return "\n\n".join(sections)

    # ---- Compress ----
    def _compress(self, context: str, max_tokens: int) -> str:
        if self._count_tokens(context) <= max_tokens:
            return context
        sections = context.split("\n\n")
        out, total = [], 0
        for sec in sections:
            n = self._count_tokens(sec)
            if total + n <= max_tokens:
                out.append(sec)
                total += n
            else:
                remain = max_tokens - total
                if remain > 50:
                    out.append(self._truncate_text(sec, remain) + "\n[... 内容已压缩 ...]")
                break
        return "\n\n".join(out)

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        n = max(self._count_tokens(text), 1)
        return text[:int(max_tokens * len(text) / n)]
    

@dataclass
class SelectionTrace:
    """一次构建过程的完整轨迹，是所有评估指标的数据来源。
    核心思想：不只记录'选了什么'，还要记录'没选什么、为什么没选'。"""
    candidates: List[ContextPacket] = field(default_factory=list) # Gather 的全部候选
    scored: List[Tuple[float, ContextPacket]] = field(default_factory=list) # 通过阈值、带综合分
    selected: List[ContextPacket] = field(default_factory=list)            # 最终选中的包
    dropped_by_threshold: List[ContextPacket] = field(default_factory=list)  # 被 min_relevance 过滤
    dropped_by_budget: List[Tuple[float, ContextPacket]] = field(default_factory=list)  # 预算装不下
    budget: int = 0                                                        # 可用 token 预算
    used_tokens: int = 0                                                   # 实际占用
    tokens_before_compress: int = 0
    tokens_after_compress: int = 0
    compressed: bool = False
    compressed_loss_tokens: int = 0
    relevance_engine: str = "keyword"   # 相关性打分方式：keyword（Jaccard）| llm
    # ---- 级联压缩（HybridCascadeContextBuilder，习题 2.3） ----
    compression_stages: List[str] = field(default_factory=list)  # 实际触发的级：S0..S5
    dedup_removed: int = 0            # S0 无损去重移除的包数
    summarized_packets: int = 0       # S3 摘要回收的包数
    query_cov_survival: float = 1.0   # S4 压缩前后查询覆盖率之比（信息存活率）


@dataclass
class ContextQualityReport:
    density: float          # 信息密度 0-1：信号强度 × (1 - 冗余度)
    relevance: float        # 相关性 0-1：入选信息平均相关性 + 查询覆盖率
    completeness: float     # 完整性 0-1：来源覆盖 + 入选率 + 无截断
    overall: float          # 综合分
    metrics: Dict[str, float]              # 全部细项指标
    diagnostics: List[str]                 # 可读诊断信息
    suggestions: List[str]                 # 优化建议
    timestamp: datetime = field(default_factory=datetime.now)

    @staticmethod
    def _grade(score: float) -> str:
        return "A" if score >= 0.85 else "B" if score >= 0.7 else "C" if score >= 0.5 else "D"

    def __str__(self):
        m = self.metrics
        lines = [
            "=" * 60,
            "           上下文质量评估报告",
            "=" * 60,
            f"  综合质量: {self._grade(self.overall)} ({self.overall:.2f})",
            f"  信息密度: {self.density:.2f}  "
            f"[信号强度 {m['signal_strength']:.2f} | 冗余度 {m['redundancy']:.2f} | 预算利用 {m['budget_utilization']:.2f}]",
            f"  相关性  : {self.relevance:.2f}  "
            f"[平均相关性 {m['mean_relevance']:.2f} | 查询覆盖率 {m['query_coverage']:.2f}]",
            f"  完整性  : {self.completeness:.2f}  "
            f"[来源覆盖 {m['source_coverage']:.2f} | 入选率 {m['selection_yield']:.2f} | "
            f"预算截断 {'是' if m['truncated'] else '否'} | 压缩损失 {int(m['compressed_loss_tokens'])} tokens | "
            f"覆盖存活 {m['query_cov_survival']:.2f}]",
            "-" * 60,
            "诊断:",
        ]
        lines += [f"  • {d}" for d in self.diagnostics]
        lines.append("建议:")
        lines += [f"  {i+1}. {s}" for i, s in enumerate(self.suggestions)]
        lines.append("=" * 60)
        return "\n".join(lines)


_STOPWORDS = {
    "的", "了", "和", "是", "我", "你", "在", "有", "什么", "怎么", "如何", "一个",
    "a", "an", "the", "is", "are", "to", "of", "in", "on", "and", "or", "how", "what",
    "do", "does", "did", "i", "you", "my", "your", "can", "should",
}


def _tokenize(text: str) -> set:
    """统一分词：英文单词 + 中文二元组，去停用词。
    查询覆盖率与冗余度共用（中文无空格分隔，按空格切分会丢掉全部中文信号）"""
    terms = set()
    for w in text.lower().split():
        w2 = "".join(ch for ch in w if ch.isalnum() or '一' <= ch <= '鿿')
        if len(w2) >= 2 and w2 not in _STOPWORDS:
            terms.add(w2)
    cjk = "".join(ch for ch in text if '一' <= ch <= '鿿')
    terms.update(cjk[i:i + 2] for i in range(len(cjk) - 1) if cjk[i:i + 2] not in _STOPWORDS)
    return terms


def _query_terms(query: str) -> set:
    """提取查询词项，供覆盖率计算"""
    return _tokenize(query)


def _mean_pairwise_overlap(contents: List[str], max_pairs: int = 500) -> float:
    """入选信息包之间的平均两两词重叠度（Jaccard），衡量冗余"""
    sets = [_tokenize(c) for c in contents if c.strip()]
    if len(sets) < 2:
        return 0.0
    overlaps, pairs = 0.0, 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            ov = len(sets[i] & sets[j]) / len(union) if union else 0.0
            overlaps += ov
            pairs += 1
            if pairs >= max_pairs:
                return overlaps / pairs
    return overlaps / pairs if pairs else 0.0


class QualityAwareContextBuilder(ContextBuilder):
    """带质量评估的上下文构建器。

    使用方式与原 ContextBuilder 完全一致，build() 后自动输出评估报告，
    报告同时保存在 self.last_report，历史报告保存在 self.history。
    """

    # ---- 阈值（规则引擎，可按需调整） ----
    T_DENSITY_LOW = 0.40
    T_REDUNDANCY_HIGH = 0.40
    T_RELEVANCE_LOW = 0.40
    T_UTILIZATION_LOW = 0.50

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_trace: Optional[SelectionTrace] = None
        self.last_report: Optional[ContextQualityReport] = None
        self.history: List[ContextQualityReport] = []
        self._llm: Optional[Any] = None    # 懒加载的 HelloAgentsLLM
        self._llm_tried: bool = False      # 只尝试初始化一次，失败永久回退规则

    # ================= LLM 增强层（conda: hello_agents） =================

    def _get_llm(self) -> Optional[Any]:
        """懒加载 HelloAgentsLLM（读同目录 .env 的 LLM_* 配置，与 944 脚本同一套）；
        任何失败只警告一次并永久回退纯规则，保证脚本可独立运行"""
        if self._llm_tried:
            return self._llm
        self._llm_tried = True
        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
            from hello_agents import HelloAgentsLLM
            self._llm = HelloAgentsLLM(temperature=0.0)
        except Exception as e:
            print(f"[WARNING] LLM 不可用，回退纯规则模式: {e}")
        return self._llm

    def _llm_relevance_scores(self, user_query: str,
                              packets: List[ContextPacket]) -> Optional[List[float]]:
        """一次批量调用为所有待打分包评估相关性（替代逐包关键词 Jaccard，
        可识别同义表述、中文语义匹配）；失败返回 None，由调用方回退 Jaccard"""
        llm = self._get_llm()
        if llm is None or not packets:
            return None
        items = "\n".join(f"[{i}] {p.content[:200]}" for i, p in enumerate(packets, 1))
        prompt = (
            "你是信息检索相关性评估器。给定用户查询和 N 条候选信息，"
            "逐条评估每条信息对回答该查询的有用程度。\n"
            "评分标准：1.0=直接回答查询；0.7以上=关键支撑信息；"
            "0.4以上=背景相关；0.1以下=无关。\n"
            '只输出 JSON（不要任何解释）：{"scores": [s1, ..., sN]}，'
            "长度必须等于 N，每个分数在 [0,1] 之间。\n\n"
            f"用户查询：{user_query}\n\n候选信息：\n{items}"
        )
        try:
            raw = llm.invoke([{"role": "user", "content": prompt}], max_tokens=1024)
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m is None:
                raise ValueError("响应中未找到 JSON 对象")
            scores = json.loads(m.group(0))["scores"]
            if len(scores) != len(packets):
                raise ValueError(f"返回 {len(scores)} 个分数 != {len(packets)} 个候选")
            return [max(0.0, min(1.0, float(s))) for s in scores]
        except Exception as e:
            print(f"[WARNING] LLM 相关性打分失败，回退关键词 Jaccard: {e}")
            return None

    def _llm_suggest(self, user_query: str, t: SelectionTrace,
                     metrics: Dict[str, float], rule_suggestions: List[str]) -> List[str]:
        """LLM 复核：基于 Trace + 指标给出规则引擎覆盖不到的内容级建议；失败返回 []"""
        llm = self._get_llm()
        if llm is None:
            return []
        payload = json.dumps({
            "query": user_query,
            "candidates": len(t.candidates),
            "selected": len(t.selected),
            "dropped_by_threshold": len(t.dropped_by_threshold),
            "dropped_by_budget": len(t.dropped_by_budget),
            "top_dropped_content": t.dropped_by_budget[0][1].content[:60] if t.dropped_by_budget else "",
            "metrics": metrics,
            "rule_suggestions": rule_suggestions,
        }, ensure_ascii=False)
        prompt = (
            "你是上下文工程（Context Engineering）优化专家。以下 JSON 是一次 Agent 上下文"
            "构建过程的评估数据，rule_suggestions 是规则引擎已给出的建议（不要重复它们）。\n"
            "请从『入选内容是否真正支撑回答该查询』的角度，指出数据中规则未覆盖的问题，"
            "给出至多 3 条具体可执行的优化建议。\n"
            "输出格式：每行一条建议（中文，一句话），不要编号、不要多余解释。\n\n" + payload
        )
        try:
            # 不设 max_tokens 上限：flash 模型会先内部推理（reasoning 计入补全额度），
            # 上限太小会把最终内容整段截没（返回空串）
            raw = llm.invoke([{"role": "user", "content": prompt}], temperature=0.3)
            if not raw.strip():
                raise ValueError("LLM 返回空内容（可能被补全额度截断）")
            lines = []
            for ln in raw.strip().splitlines():
                ln = re.sub(r"^[\s\-\*•·\d\.\、)]+", "", ln).strip()
                if ln:
                    lines.append(ln)
            return [f"[LLM] {ln[:160]}" for ln in lines[:3]]
        except Exception as e:
            print(f"[WARNING] LLM 建议生成失败，仅保留规则建议: {e}")
            return []

    def build(self, user_query: str, **kwargs) -> str:
        context = super().build(user_query, **kwargs)
        if self.config.enable_quality_eval and self.last_trace is not None:
            self.last_report = self.evaluate(user_query, context)
            self.history.append(self.last_report)
            print(self.last_report)
        return context

    # ================= Trace 捕获 =================

    def _select(self, packets, user_query, available_tokens):
        """重写 Select：完整记录'谁被选中、谁被过滤、谁因预算被丢弃'"""
        trace = SelectionTrace(candidates=list(packets), budget=available_tokens)

        system = [p for p in packets if p.metadata.get("type") == "system_instruction"]
        others = [p for p in packets if p.metadata.get("type") != "system_instruction"]
        used = sum(p.token_count for p in system)

        if available_tokens - used <= 0:
            trace.selected, trace.used_tokens = system, used
            self.last_trace = trace
            return system

        # LLM 批量打分：一次调用评估所有未打分包（0.5 为默认哨兵值），
        # 失败则静默回退，下方循环照旧逐包用关键词 Jaccard
        if self.config.llm_relevance:
            pending = [p for p in others if p.relevance_score == 0.5]
            scores = self._llm_relevance_scores(user_query, pending)
            if scores is not None:
                for p, s in zip(pending, scores):
                    p.relevance_score = s
                    p.metadata["_llm_scored"] = True  # 防止恰好 0.5 时被误判为未打分
                trace.relevance_engine = "llm"

        for p in others:
            if p.relevance_score == 0.5 and not p.metadata.get("_llm_scored"):
                p.relevance_score = self._calculate_relevance(p.content, user_query)
            if p.relevance_score >= self.config.min_relevance:
                trace.scored.append((self._combined_score(p), p))
            else:
                trace.dropped_by_threshold.append(p)

        trace.scored.sort(key=lambda x: x[0], reverse=True)

        selected = list(system)
        for idx, (s, p) in enumerate(trace.scored):
            if used + p.token_count <= available_tokens:
                selected.append(p)
                used += p.token_count
            else:
                trace.dropped_by_budget = trace.scored[idx:]  # 记录被预算抛弃的高分信息
                break

        trace.selected, trace.used_tokens = selected, used
        self.last_trace = trace
        return selected

    def _compress(self, context: str, max_tokens: int) -> str:
        """重写 Compress：记录压缩造成的 token 损失"""
        before = self._count_tokens(context)
        out = super()._compress(context, max_tokens)
        if self.last_trace is not None:
            after = self._count_tokens(out)
            self.last_trace.tokens_before_compress = before
            self.last_trace.tokens_after_compress = after
            self.last_trace.compressed = after < before
            self.last_trace.compressed_loss_tokens = max(0, before - after)
        return out

    # ================= 指标计算 =================

    def evaluate(self, user_query: str, context_str: str) -> ContextQualityReport:
        t = self.last_trace
        assert t is not None, "evaluate() 必须在 build() 之后调用（Trace 由 _select 记录）"
        non_sys_sel = [p for p in t.selected if p.metadata.get("type") != "system_instruction"]
        non_sys_cand = [p for p in t.candidates if p.metadata.get("type") != "system_instruction"]
        total_tokens = sum(p.token_count for p in t.selected)

        # ---------- 相关性 ----------
        mean_rel = (sum(p.relevance_score for p in non_sys_sel) / len(non_sys_sel)) if non_sys_sel else 0.0
        terms = _query_terms(user_query)
        ctx_lower = context_str.lower()
        query_cov = (len({term for term in terms if term in ctx_lower}) / len(terms)) if terms else 1.0
        relevance = 0.6 * mean_rel + 0.4 * query_cov

        # ---------- 信息密度 ----------
        score_map = {id(p): s for s, p in t.scored}
        signal = (sum(score_map.get(id(p), p.relevance_score) * p.token_count
                      for p in non_sys_sel) / total_tokens) if total_tokens else 0.0
        redundancy = _mean_pairwise_overlap([p.content for p in non_sys_sel])
        density = max(0.0, min(1.0, signal * (1.0 - redundancy)))
        budget_util = min(1.0, t.used_tokens / t.budget) if t.budget else 0.0

        # ---------- 完整性 ----------
        src_candidates = {p.metadata.get("type", "general") for p in t.candidates
                        if p.metadata.get("type") != "system_instruction"}
        src_selected = {p.metadata.get("type", "general") for p in non_sys_sel}
        source_cov = (len(src_selected & src_candidates) / len(src_candidates)) if src_candidates else 1.0
        yield_ratio = (len(non_sys_sel) / len(non_sys_cand)) if non_sys_cand else 1.0
        truncated = len(t.dropped_by_budget) > 0
        retention = 1.0 - (t.compressed_loss_tokens / max(t.tokens_before_compress, 1))
        completeness = max(0.0, min(1.0,
            0.4 * source_cov + 0.3 * yield_ratio + 0.3 * (0.3 if truncated else retention)))

        overall = max(0.0, min(1.0, 0.4 * relevance + 0.25 * density + 0.35 * completeness))

        metrics = {
            "mean_relevance": round(mean_rel, 4),
            "query_coverage": round(query_cov, 4),
            "signal_strength": round(signal, 4),
            "redundancy": round(redundancy, 4),
            "budget_utilization": round(budget_util, 4),
            "source_coverage": round(source_cov, 4),
            "selection_yield": round(yield_ratio, 4),
            "truncated": 1.0 if truncated else 0.0,
            "compressed_loss_tokens": float(t.compressed_loss_tokens),
            "query_cov_survival": round(t.query_cov_survival, 4),
        }
        diagnostics = self._diagnostics(t, metrics)
        suggestions = self._suggest(t, metrics)
        if self.config.llm_suggestions:
            suggestions += self._llm_suggest(user_query, t, metrics, suggestions)

        return ContextQualityReport(
            density=round(density, 4), relevance=round(relevance, 4),
            completeness=round(completeness, 4), overall=round(overall, 4),
            metrics=metrics, diagnostics=diagnostics, suggestions=suggestions)


    # ================= 诊断与建议 =================

    def _diagnostics(self, t: SelectionTrace, m: Dict[str, float]) -> List[str]:
        d = [
            f"候选 {len(t.candidates)} 个 → 入选 {len(t.selected)} 个"
            f"（阈值过滤 {len(t.dropped_by_threshold)}，预算丢弃 {len(t.dropped_by_budget)}）",
            f"token 预算 {t.budget}，实际使用 {t.used_tokens}"
            f"（利用率 {m['budget_utilization']:.0%}）",
            f"相关性打分引擎：{t.relevance_engine}",
        ]
        if t.compressed:
            d.append(f"压缩触发：{t.tokens_before_compress} → {t.tokens_after_compress} tokens，"
                     f"损失 {t.compressed_loss_tokens} tokens")
        if t.compression_stages:
            d.append(f"级联压缩: {' → '.join(t.compression_stages)}"
                     f"（去重移除 {t.dedup_removed} 包，摘要回收 {t.summarized_packets} 包，"
                     f"覆盖率存活 {t.query_cov_survival:.2f}）")
        if t.dropped_by_budget:
            top = t.dropped_by_budget[0][1]
            d.append(f"被预算丢弃的最高分信息: {top.content[:40]}...")
        return d

    def _suggest(self, t: SelectionTrace, m: Dict[str, float]) -> List[str]:
        s = []
        # 1) 预算截断：信息根本没进上下文，完整性受损最严重
        if m["truncated"]:
            n = len(t.dropped_by_budget)
            s.append(f"预算内未装下 {n} 个高分信息包 → 增大 max_tokens；或预压缩对话历史/工具输出；"
                     f"或检查 reserve_ratio 是否预留过多")
        # 2) 压缩损失：先保召回
        if t.compressed_loss_tokens > 0:
            s.append(f"压缩丢失 {t.compressed_loss_tokens} tokens → 优先保召回："
                     f"对 [Context] 分区用 LLM 摘要替代硬截断；缓存静态内容避免重复占用预算")
        # 2b) 信息存活率：比 token 差值更真实的保真度量（摘要式压缩下 token 差值会虚高）
        if m.get("query_cov_survival", 1.0) < 0.8 and t.compressed:
            s.append(f"压缩后查询覆盖率存活率 {m['query_cov_survival']:.2f} 偏低 → "
                     f"回答查询所需的关键概念在压缩中丢失，检查摘要质量或降低压缩强度")
        # 3) 密度与冗余
        if m["redundancy"] > self.T_REDUNDANCY_HIGH:
            s.append(f"入选信息冗余度 {m['redundancy']:.2f} 偏高 → 在 Gather 阶段按内容哈希去重，"
                     f"或降低 recency_weight 减少相似历史消息入选")
        if m["signal_strength"] < self.T_DENSITY_LOW:
            s.append(f"信号强度 {m['signal_strength']:.2f} 偏低 → 提高 min_relevance，"
                     f"或增大 relevance_weight（当前 {self.config.relevance_weight}）")
        # 4) 相关性
        if m["mean_relevance"] < self.T_RELEVANCE_LOW:
            s.append("入选信息平均相关性偏低 → 开启 llm_relevance 用 LLM 语义打分，"
                     "或生产环境改用向量相似度（embedding）检索")
        # 5) 来源覆盖
        src_cand = {p.metadata.get("type", "") for p in t.candidates}
        src_sel = {p.metadata.get("type", "") for p in t.selected}
        missing = src_cand - src_sel - {"system_instruction"}
        if missing:
            s.append(f"来源 {sorted(missing)} 未贡献任何入选信息 → 检查其检索阈值"
                     f"（如 min_importance / min_score）或知识库内容是否过期")
        # 6) 成本
        if m["budget_utilization"] < self.T_UTILIZATION_LOW and not m["truncated"]:
            s.append(f"预算利用率仅 {m['budget_utilization']:.0%} → 可降低 max_tokens 节省推理成本，"
                     f"或放宽 min_relevance 让更多信息进入上下文")
        if not s:
            s.append("各项指标健康，无需调整。")
        return s


# ----------------------------------------------------------------------
# 混合级联压缩（习题 2.3 实现）
# ----------------------------------------------------------------------

class HybridCascadeContextBuilder(QualityAwareContextBuilder):
    """S0-S5 级联压缩器：把书中「超限硬截断」的 Compress 升级为多机制级联。

    设计原则（习题 2.3）：无损优先于有损、确定性优先于生成式、
    生成式只作升级手段且必有确定性兜底；每级只在仍超预算时进入下一级。

    级联各级：
      S0 无损去重        词集 Jaccard ≥ 阈值视为重复（含完全相同），保留分高者
      S1 价值驱逐        综合分 < EVICT_BAR 的包直接逐出（便宜的确定性垃圾清理）
      S2 分通道滑窗      对话历史保留最近 K 条 + 最前 1 条锚点（首因效应），中段入池；
                         超长包 head+tail 截中段；仍超预算则按分数升序继续逐出
      S3 摘要回收        用省出的预算把被逐内容压成一条摘要塞回（绝不超预算），
                         按内容哈希缓存，LLM 不可用则静默跳过
      S4 存活率校验      压缩后查询覆盖率 / 压缩前 ≥ SURVIVAL_MIN 才接受摘要，
                         不过关丢弃摘要回退确定性结果
      S5 硬截断兜底      任何未解决的溢出交给书中原版截断，绝对保证预算合规

    保护名单（任何一级不动）：系统指令（[Task] 里的查询由 _structure 恒定保留）；
    最新对话轮因 recency 加权天然最后被逐出。
    """

    # ---- 级联参数（可按任务调整的旋钮） ----
    DEDUP_JACCARD = 0.75       # S0 近重复阈值
    EVICT_BAR = 0.45           # S1 低于此综合分直接逐出
    HISTORY_KEEP_LAST = 3      # S2 对话历史保留最近 K 条
    HISTORY_ANCHOR = 1         # S2 额外保留最前 1 条（任务定义锚点）
    OVERSIZED_TOKENS = 80      # S2 超长包阈值 → head+tail 截中段
    HEAD_RATIO, TAIL_RATIO = 0.5, 0.25
    DIGEST_MIN_SLACK = 20      # S3 剩余预算低于此值不值得调 LLM
    SUMMARIZE_MIN_SCORE = 0.30 # S3 只回收价值达标的被逐包（低于它的视为纯噪声丢弃）
    SURVIVAL_MIN = 0.80        # S4 查询覆盖率存活率下限

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_query = ""
        self._summary_cache: Dict[str, str] = {}   # 内容哈希 → 摘要（摊薄 LLM 成本）

    def build(self, user_query: str, **kwargs) -> str:
        self._last_query = user_query   # 级联各级重建/校验都需要查询
        return super().build(user_query, **kwargs)

    # ================= 级联主流程 =================

    def _compress(self, context: str, max_tokens: int) -> str:
        t = self.last_trace
        before = self._count_tokens(context)
        if t is None or before <= max_tokens:
            return context
        t.tokens_before_compress = before
        t.compression_stages = ["S0"]

        kept, pool = self._s0_dedup(list(t.selected)), []
        if not self._fits(kept, max_tokens):
            t.compression_stages.append("S1")
            kept, pool = self._s1_evict(kept)
        if not self._fits(kept, max_tokens):
            t.compression_stages.append("S2")
            kept, more = self._s2_window(kept, max_tokens)
            pool += more

        # S3+S4：把被逐内容以摘要形式回收（利用省出的预算，不破坏预算合规）
        if pool and self.config.llm_compression:
            sp = self._s3_digest(pool, max_tokens - self._count_tokens(self._rebuild(kept)))
            if sp is not None:
                t.compression_stages.append("S3")
                cand = kept + [sp]
                t.compression_stages.append("S4")
                cov0 = self._query_coverage(context)
                cov1 = self._query_coverage(self._rebuild(cand))
                survival = cov1 / max(cov0, 1e-9)
                if survival >= self.SURVIVAL_MIN:
                    kept, t.summarized_packets = cand, sp.metadata["provenance"]
                else:
                    print(f"[S4] 覆盖率存活率 {survival:.2f} < {self.SURVIVAL_MIN:.2f}，"
                          f"丢弃摘要回退确定性结果")
        return self._finish(kept, context, before, max_tokens)

    def _fits(self, kept: List[ContextPacket], max_tokens: int) -> bool:
        return self._count_tokens(self._rebuild(kept)) <= max_tokens

    def _rebuild(self, kept: List[ContextPacket]) -> str:
        """从存活包重建结构化上下文（复用书中 Structure，保证格式一致）"""
        return self._structure(kept, self._last_query)

    def _query_coverage(self, text: str) -> float:
        """查询词项在 text 中的覆盖率（S4 存活率的分子/分母）"""
        terms = _query_terms(self._last_query)
        if not terms:
            return 1.0
        low = text.lower()
        return len({x for x in terms if x in low}) / len(terms)

    # ---- S0 无损去重 ----
    def _s0_dedup(self, packets: List[ContextPacket]) -> List[ContextPacket]:
        """词集 Jaccard ≥ DEDUP_JACCARD 视为重复（完全相同 Jaccard=1 自然覆盖），
        保留综合分高者；移除后恢复 Select 的原始顺序"""
        t = self.last_trace
        assert t is not None, "_s0_dedup 只应由 _compress 在 Trace 存在时调用"
        ordered = sorted(packets, key=self._combined_score, reverse=True)
        kept, seen = [], []
        for p in ordered:
            tk = _tokenize(p.content)
            dup = any((tk | s) and len(tk & s) / len(tk | s) >= self.DEDUP_JACCARD
                      for s in seen)
            if dup:
                t.dedup_removed += 1
                continue
            seen.append(tk)
            kept.append(p)
        order = {id(p): i for i, p in enumerate(packets)}
        kept.sort(key=lambda p: order[id(p)])
        return kept

    # ---- S1 价值驱逐 ----
    def _s1_evict(self, packets: List[ContextPacket]) -> Tuple[List[ContextPacket], List[ContextPacket]]:
        """低于 EVICT_BAR 的包直接逐出（确定性垃圾清理，零 LLM 成本）"""
        protected = [p for p in packets if p.metadata.get("type") == "system_instruction"]
        others = [p for p in packets if p.metadata.get("type") != "system_instruction"]
        pool = [p for p in others if self._combined_score(p) < self.EVICT_BAR]
        keep = [p for p in others if self._combined_score(p) >= self.EVICT_BAR]
        return protected + keep, pool

    # ---- S2 分通道滑窗 ----
    def _s2_window(self, packets: List[ContextPacket],
                   max_tokens: int) -> Tuple[List[ContextPacket], List[ContextPacket]]:
        """对话历史留尾 K 条 + 首条锚点，中段入池；超长包 head+tail 截中段；
        仍超预算则按分数升序继续逐出——确定性保证装得下的部分一定装得下"""
        protected = [p for p in packets if p.metadata.get("type") == "system_instruction"]
        others = [p for p in packets if p.metadata.get("type") != "system_instruction"]
        pool = []
        # 1) 历史窗口：Gather 按对话顺序追加，Select 稳定排序保持该相对顺序
        hist_idx = [i for i, p in enumerate(others)
                    if p.metadata.get("type") == "conversation_history"]
        keep_n = self.HISTORY_KEEP_LAST + self.HISTORY_ANCHOR
        if len(hist_idx) > keep_n:
            drop = set(hist_idx[self.HISTORY_ANCHOR:len(hist_idx) - self.HISTORY_KEEP_LAST])
            pool = [others[i] for i in sorted(drop)]
            others = [p for i, p in enumerate(others) if i not in drop]
        # 2) 超长包 head+tail（保两端逐字保真，丢中段）
        for p in others:
            if p.token_count > self.OVERSIZED_TOKENS:
                p.content = self._head_tail(p.content)
                p.token_count = self._count_tokens(p.content)
        # 3) 仍装不下：价值升序继续逐出，直到装下
        if not self._fits(protected + others, max_tokens):
            others = sorted(others, key=self._combined_score)
            while others and not self._fits(protected + others, max_tokens):
                pool.append(others.pop(0))
        return protected + others, pool

    def _head_tail(self, text: str) -> str:
        h, tl = int(len(text) * self.HEAD_RATIO), int(len(text) * self.TAIL_RATIO)
        return text[:h] + "\n[... 中段省略（head+tail 保两端）...]\n" + (text[-tl:] if tl else "")

    # ---- S3 摘要回收 ----
    def _s3_digest(self, pool: List[ContextPacket], slack: int) -> Optional[ContextPacket]:
        """把被逐的旧材料压成一条摘要，塞回剩余预算（slack）内。
        只回收价值达标的包；摘要按 (内容+查询) 哈希缓存，命中零成本；
        LLM 不可用 / slack 不足 / 调用失败 一律返回 None（级联自动退化为 S0-S2+S5）"""
        worth = [p for p in pool if self._combined_score(p) >= self.SUMMARIZE_MIN_SCORE]
        if not worth or slack < self.DIGEST_MIN_SLACK:
            return None
        key = hashlib.md5(("|".join(p.content for p in worth) + "||" + self._last_query)
                          .encode("utf-8")).hexdigest()
        if key in self._summary_cache:
            summary = self._summary_cache[key]
        else:
            llm = self._get_llm()
            if llm is None:
                return None
            items = "\n".join(f"[{i}] {p.content[:150]}" for i, p in enumerate(worth, 1))
            prompt = (f"把以下 {len(worth)} 条较旧的对话/记忆压缩成一段不超过 {max(30, slack)} tokens 的要点，"
                      "用于放回 LLM 上下文。要求：保留与查询相关的关键事实；"
                      "数字/ID/文件名逐字保留；只输出要点正文，不要解释。\n\n"
                      f"用户查询：{self._last_query}\n\n" + items)
            try:
                # 不设 max_tokens 上限：flash 模型内部推理计入补全额度，上限过小会返回空串
                summary = llm.invoke([{"role": "user", "content": prompt}]).strip()
                self._summary_cache[key] = summary
            except Exception as e:
                print(f"[WARNING] S3 摘要失败，退化为丢弃+截断: {e}")
                return None
        n = self._count_tokens(summary)
        if n > slack:   # 摘要超了剩余预算：截到装得下（保真降级，S4 仍会校验）
            summary = self._truncate_text(summary, slack)
            n = self._count_tokens(summary)
        return ContextPacket(f"[摘要回收 x{len(worth)}] {summary}", datetime.now(),
                             max(n, 1), 0.75,
                             {"type": "summary", "provenance": len(worth)})

    # ---- S5 兜底 ----
    def _finish(self, kept: List[ContextPacket], context: str,
                before: int, max_tokens: int) -> str:
        """任何一级没解决的溢出，交给书中原版截断按 section 边界处理；
        并以最初上下文为基准记录压缩统计与覆盖率存活率"""
        t = self.last_trace
        assert t is not None, "_finish 只应由 _compress 在 Trace 存在时调用"
        rebuilt = self._rebuild(kept)
        out = super()._compress(rebuilt, max_tokens)   # S5：硬截断兜底
        after = self._count_tokens(out)
        if after < self._count_tokens(rebuilt):
            t.compression_stages.append("S5")
        t.tokens_before_compress, t.tokens_after_compress = before, after
        t.compressed = after < before
        t.compressed_loss_tokens = max(0, before - after)
        cov0, cov1 = self._query_coverage(context), self._query_coverage(out)
        t.query_cov_survival = cov1 / max(cov0, 1e-9)
        return out


# ----------------------------------------------------------------------
# 演示
# ----------------------------------------------------------------------

def _demo():
    from datetime import timedelta
    use_llm = "--no-llm" not in sys.argv
    llm_cfg = dict(llm_relevance=use_llm, llm_suggestions=use_llm, llm_compression=use_llm)
    print(f"LLM 增强层：{'开启（批量相关性打分 + 建议复核）' if use_llm else '关闭（--no-llm，纯规则零成本）'}")
    now = datetime.now()
    sys_prompt = "You are a senior Python data engineering consultant."

    history = [
        {"role": "user", "content": "I am building a data analysis tool with Python and pandas"},
        {"role": "assistant", "content": "Pandas is a solid choice for data wrangling."},
        {"role": "user", "content": "I finished the CSV reading module last week"},
    ]

    def _mk_packets(with_dup=False):
        """每个场景独立新建信息包：Select 阶段会原地写 relevance_score，
        复用同一批对象会导致打分跨场景泄漏"""
        base = [
            ContextPacket("Use category dtype instead of object to cut memory up to 90%.",
                          now - timedelta(hours=2), 15, 0.9, {"type": "memory"}),
            ContextPacket("Downcast int64 to int32 or float64 to float32 for large DataFrames.",
                          now - timedelta(hours=3), 14, 0.8, {"type": "memory"}),
            ContextPacket("The user prefers simple solutions over clever ones in code review.",
                          now - timedelta(days=30), 13, 0.4, {"type": "memory"}),
            ContextPacket("Pandas memory optimization: dtype downcasting, chunking, and category encoding.",
                          now - timedelta(days=1), 12, 0.85, {"type": "rag_result"}),
            # 不带相关性分（默认 0.5 哨兵值）：模拟 memory_tool 检索未返回 score，
            # 这类包正是 LLM 批量打分 / Jaccard 现场计算的目标
            ContextPacket("Vectorization avoids intermediate object copies that blow up RAM.",
                          now - timedelta(hours=5), 12, metadata={"type": "knowledge"}),
        ]
        if with_dup:
            base += [
                ContextPacket("Use category dtype instead of object to cut memory up to 90%.",
                              now - timedelta(hours=1), 15, 0.92, {"type": "memory"}),
                ContextPacket("Downcast int64 to int32 for large DataFrames to save memory.",
                              now - timedelta(minutes=30), 12, 0.88, {"type": "memory"}),
                ContextPacket("Use category dtype instead of object to cut memory up to 90%.",
                              now - timedelta(minutes=10), 15, 0.91, {"type": "memory"}),
            ]
        return base

    print("\n########## 场景 1：健康配置 ##########")
    b1 = QualityAwareContextBuilder(
        config=ContextConfig(max_tokens=3000, min_relevance=0.1, **llm_cfg))
    b1.build("How to optimize pandas memory usage?",
             conversation_history=history, system_instructions=sys_prompt,
             custom_packets=_mk_packets())

    print("\n########## 场景 2：预算过小（触发截断 + 压缩） ##########")
    b2 = QualityAwareContextBuilder(
        config=ContextConfig(max_tokens=100, reserve_ratio=0.25, min_relevance=0.1, **llm_cfg))
    b2.build("How to optimize pandas memory usage?",
             conversation_history=history, system_instructions=sys_prompt,
             custom_packets=_mk_packets())

    print("\n########## 场景 3：阈值过高 + 高冗余 ##########")
    b3 = QualityAwareContextBuilder(
        config=ContextConfig(max_tokens=3000, min_relevance=0.85, **llm_cfg))
    b3.build("How to optimize pandas memory usage?",
             conversation_history=history, system_instructions=sys_prompt,
             custom_packets=_mk_packets(with_dup=True))

    print("\n########## 场景 4：混合级联压缩（习题 2.3） ##########")
    long_history = [
        {"role": "user", "content": "I am building a data analysis tool with Python and pandas"},
        {"role": "assistant", "content": "Pandas is a solid choice for data wrangling."},
        {"role": "user", "content": "The CSV reader works but RAM usage explodes on big files"},
        {"role": "assistant", "content": "Likely object dtype; consider category and downcasting."},
        {"role": "user", "content": "Also the merge step creates huge intermediate copies"},
        {"role": "assistant", "content": "Merge on category keys and avoid full-frame copies."},
        {"role": "user", "content": "I finished the CSV reading module last week"},
        {"role": "assistant", "content": "Great, next step is profiling with memory_usage()."},
    ]
    # 低价值包：S1 的价值驱逐目标（综合分 < EVICT_BAR 0.45）
    junk = [
        ContextPacket("The team had a pizza party last Friday and everyone enjoyed it.",
                      now - timedelta(days=2), 12, 0.2, {"type": "memory"}),
        ContextPacket("Office moved to the 3rd floor, update the address book.",
                      now - timedelta(days=3), 11, 0.3, {"type": "memory"}),
        ContextPacket("Yesterday I chatted with Sarah about weekend plans, then we walked "
                      "to the new coffee shop downtown and discussed vacation ideas for "
                      "the upcoming summer holiday trip. She recommended visiting Kyoto.",
                      now - timedelta(days=2), 44, 0.3, {"type": "memory"}),
    ]
    # reserve_ratio 调小让 Select 放进全部候选，把裁剪留给级联各级演示
    b4 = HybridCascadeContextBuilder(
        config=ContextConfig(max_tokens=175, reserve_ratio=0.05, min_relevance=0.1, **llm_cfg))
    b4.build("How to optimize pandas memory usage?",
             conversation_history=long_history, system_instructions=sys_prompt,
             custom_packets=_mk_packets() + junk)


if __name__ == "__main__":
    _demo()

