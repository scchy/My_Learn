
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
from pi_agent.compaction.cutpoint import find_cut_point, find_turn_start_index
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

        if is_split_turn:
            # 双摘要：历史摘要 + 当前 Turn 前缀摘要
            history_messages = non_system[:turn_start]
            turn_prefix_messages = non_system[turn_start:cut_index]

            history_summary = ""
            if history_messages:
                history_summary = await generate_summary(
                    client,
                    history_messages,
                    previous_summary=None,
                    max_tokens=self.config.summary_max_tokens,
                )

            turn_prefix_summary = await generate_turn_prefix_summary(
                client,
                turn_prefix_messages,
                max_tokens=self.config.turn_prefix_summary_max_tokens,
            )

            summary_text = merge_summaries(history_summary, turn_prefix_summary)
            compacted_messages = non_system[:turn_start]
        else:
            # 单摘要：切割点正好在 Turn 边界
            compacted_messages = non_system[:cut_index]
            summary_text = await generate_summary(
                client,
                compacted_messages,
                previous_summary=None,
                max_tokens=self.config.summary_max_tokens,
            )

        # Stage 5: 合并摘要 + 文件操作记录
        file_ops = extract_file_operations(compacted_messages)
        summary_msg = create_compaction_summary_message(summary_text, file_ops)

        result = system_msgs + [summary_msg] + kept_raw
        logger.info(
            "Compaction: compressed %d non-system messages into summary, kept %d raw",
            len(non_system),
            len(kept_raw),
        )
        return result

    def get_stats(self) -> dict:
        """返回最后一次压缩的统计信息。"""
        return dict(self._last_stats)


# 保留旧别名，避免已有导入失效
ContextCompressor = ContextCompactor
