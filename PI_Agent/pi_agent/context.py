
# Day 4: 上下文压缩（Compaction）
# python3
# Create Date: 2026-07-24
# Author: Scc_hy
# Tip:
# 参考 earendil-works/pi/packages/coding-agent/src/core/compaction/compaction.ts
# 核心设计：
# - 绝对 token 预算驱动（reserve_tokens / keep_recent_tokens）
# - 从后往前找合法切割点，保护 Turn 边界与 tool-call/tool-result 配对
# - 双摘要：历史摘要 + 当前 Turn 前缀摘要（split turn 时）
# - 增量更新：在上一次摘要基础上追加更新
# - 文件操作追踪：保留 read / modified 文件列表
# ==========================================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass

from pi_agent.llm import LLMClient, Message
from pi_agent.compaction.cutpoint import (
    _is_compaction_summary,
    find_cut_point,
    find_turn_start_index,
)
from pi_agent.compaction.fileops import extract_file_operations
from pi_agent.compaction.summary import (
    create_compaction_summary_message,
    generate_summary,
    generate_turn_prefix_summary,
    merge_summaries,
)
from pi_agent.compaction.token_utils import estimate_context_tokens

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 压缩配置
# ---------------------------------------------------------------------------


@dataclass
class CompactionConfig:
    """Pi-style 上下文压缩配置。"""

    enabled: bool = True
    reserve_tokens: int = 16384  # 为模型生成预留的空间 + 安全余量
    keep_recent_tokens: int = 20000  # 保留最近原始消息的 token 预算

    # 摘要长度限制
    summary_max_tokens: int = 4096
    turn_prefix_summary_max_tokens: int = 2048


# 保留旧别名，避免已有导入失效
CompressionConfig = CompactionConfig


# ---------------------------------------------------------------------------
# 增量摘要：提取上一次压缩的摘要文本
# ---------------------------------------------------------------------------


def _extract_previous_summary(
    messages: list[Message], cut_index: int
) -> tuple[str | None, int]:
    """从待压缩的消息中提取上一次的 compaction 摘要。

    从 cut_index 向前扫描，找到最近的 ``[上下文摘要]`` 消息，
    提取其正文作为 previous_summary，用于增量更新（UPDATE_SUMMARIZATION_PROMPT）。

    Returns
    -------
    (summary_text, message_index)
        若未找到，返回 (None, 0)。message_index 用于跳过旧摘要，
        只把摘要之后的新消息传给 LLM。
    """
    for i in range(cut_index - 1, -1, -1):
        if _is_compaction_summary(messages[i]):
            content = messages[i].content or ""
            # 去掉 ``[上下文摘要]`` 前缀，保留纯文本
            summary_text = content.replace("[上下文摘要]", "", 1).strip()
            return summary_text, i
    return None, 0


# ---------------------------------------------------------------------------
# 上下文压缩器
# ---------------------------------------------------------------------------


class ContextCompactor:
    """上下文压缩器 —— Pi-style Compaction。"""

    def __init__(self, config: CompactionConfig | None = None):
        self.config = config or CompactionConfig()
        self._last_stats: dict = {}

    async def compress_if_needed(
        self,
        messages: list[Message],
        context_limit: int,
        client: LLMClient,
        *,
        api_usage_tokens: int | None = None
    ) -> list[Message]:
        """检查并在必要时压缩消息列表。

        Parameters
        ----------
        messages : 当前完整消息列表
        context_limit : 模型的上下文窗口大小
        client : LLM 客户端（用于摘要）
        api_usage_tokens : API 返回的真实 prompt_tokens（优先使用）
        """
        if not self.config.enabled:
            return messages

        # 估算当前上下文 token
        estimated = estimate_context_tokens(messages, api_usage_tokens)
        trigger_threshold = context_limit - self.config.reserve_tokens

        self._last_stats = {
            "messages": len(messages),
            "estimated_tokens": estimated,
            "context_limit": context_limit,
            "reserve_tokens": self.config.reserve_tokens,
            "trigger_threshold": trigger_threshold,
            "keep_recent_tokens": self.config.keep_recent_tokens,
        }

        # 未超过触发阈值，不压缩
        if estimated <= trigger_threshold:
            return messages

        # Pi 的 compaction 只处理非系统消息；系统提示永远保留
        system_msgs = [m for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]

        if not non_system:
            return list(system_msgs)

        # Stage 1/2: 找切割点
        cut_index, is_split_turn = find_cut_point(
            non_system,
            self.config.keep_recent_tokens,
        )

        if cut_index <= 0:
            # 即使超过阈值也找不到合适切割点（消息太少或都太短），不压缩
            logger.warning("上下文超过阈值但无法找到合法切割点，跳过压缩")
            return messages

        # 需要保留的原始消息
        kept_raw = non_system[cut_index:]

        # Stage 3/4: 生成摘要
        turn_start = find_turn_start_index(non_system, cut_index)

        # ── 增量摘要：检测上一次压缩摘要，避免全量重写 ──
        previous_summary, prev_summary_idx = _extract_previous_summary(
            non_system, cut_index
        )
        # 如果有旧摘要，只对摘要之后的新消息做增量更新（而不是重读全部历史）
        history_start = prev_summary_idx + 1 if previous_summary else 0

        if is_split_turn:
            # 双摘要：历史摘要 + 当前 Turn 前缀摘要
            history_messages = non_system[history_start:turn_start]
            turn_prefix_messages = non_system[turn_start:cut_index]

            history_summary = ""
            if history_messages or previous_summary:
                history_summary = await generate_summary(
                    client,
                    history_messages,
                    previous_summary=previous_summary,
                    max_tokens=self.config.summary_max_tokens,
                )

            turn_prefix_summary = await generate_turn_prefix_summary(
                client,
                turn_prefix_messages,
                max_tokens=self.config.turn_prefix_summary_max_tokens,
            )

            summary_text = merge_summaries(history_summary, turn_prefix_summary)
            # 文件追踪以全部被压缩消息为准（含旧摘要）
            compacted_messages = non_system[:cut_index]
        else:
            # 单摘要：切割点正好在 Turn 边界
            compacted_messages = non_system[:cut_index]
            new_messages = non_system[history_start:cut_index]
            summary_text = await generate_summary(
                client,
                new_messages,
                previous_summary=previous_summary,
                max_tokens=self.config.summary_max_tokens,
            )

        # Stage 5: 合并摘要 + 文件操作记录
        file_ops = extract_file_operations(compacted_messages)
        summary_msg = create_compaction_summary_message(summary_text, file_ops)

        result = system_msgs + [summary_msg] + kept_raw
        logger.info(
            "Compaction: compressed %d non-system messages into summary (incremental=%s), kept %d raw",
            len(non_system),
            previous_summary is not None,
            len(kept_raw),
        )
        return result

    def get_stats(self) -> dict:
        """返回最后一次压缩的统计信息。"""
        return dict(self._last_stats)


# 保留旧别名，避免已有导入失效
ContextCompressor = ContextCompactor
