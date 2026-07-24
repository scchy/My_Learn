
# Day 4: 上下文压缩
# python3
# Create Datae: 2026-07-24
# Author: Scc_hy
# Tip:
# 核心机制：三层策略、块级截断（不切断工具调用对）、Token 驱动决策
# 关键修正（相比初版计划）：
# - 按"块"（block）截断，保证 tool-call ↔ tool-result 成对保留
# - 优先使用 API 返回的 usage，估算仅作 fallback
# ==========================================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass

from pi_agent.llm import LLMClient, Message, estimate_messages_tokens
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 压缩配置
# ---------------------------------------------------------------------------


@dataclass
class CompressionConfig:
    """三级压缩的阈值和保留策略。"""

    tier1_ratio: float = 0.50  # 超过 50% → 截断旧块
    tier2_ratio: float = 0.75  # 超过 75% → 摘要旧消息
    tier3_ratio: float = 0.90  # 超过 90% → 紧急压缩

    tier1_preserve_blocks: int = 4  # Tier 1: 保留最近 N 个块
    tier2_preserve_blocks: int = 3  # Tier 2: 保留最近 N 个块 + 摘要
    tier3_preserve_blocks: int = 1  # Tier 3: 保留最近 1 个块

    tier1_escalation_limit: int = 3  # Tier 1 连续触发 N 次后，强制升级到 Tier 2

    summary_max_tokens: int = 512
    emergency_summary_max_tokens: int = 200


# 保留旧别名，避免已有导入失效
CompressConfig = CompressionConfig


# ---------------------------------------------------------------------------
# 上下文压缩器
# ---------------------------------------------------------------------------


class ContextCompressor:
    """上下文压缩器 —— 按"块"截断，保证工具调用对的完整性。"""
    
    def __init__(self, config: CompressionConfig | None = None):
        self.config = config or CompressionConfig()
        self._last_stats: dict = {}
        self._tier1_count: int = 0  # Tier 1 连续触发计数

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
        # 使用 API 真实值或估算
        estimated = api_usage_tokens if api_usage_tokens is not None else  estimate_messages_tokens(messages)
        ratio = estimated / context_limit
        self._last_stats = {
            "messages": len(messages),
            "estimated_tokens": estimated,
            "context_limit": context_limit,
            "ratio": f"{ratio:.1%}",
        }
        if ratio < self.config.tier1_ratio:
            return messages

        # 分离系统提示
        system_msgs = [m for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]

        # 划分为块
        blocks = self._split_into_blocks(non_system)
        if ratio >= self.config.tier3_ratio:
            self._tier1_count = 0
            return await self._tier3_compress(system_msgs, blocks, client)
        elif ratio >= self.config.tier2_ratio:
            self._tier1_count = 0
            return await self._tier2_compress(system_msgs, blocks, client)
        else:
            # Tier 1 区域：累计触发次数，超限则强制升级到 Tier 2 摘要
            self._tier1_count += 1
            if self._tier1_count >= self.config.tier1_escalation_limit:
                logger.info(
                    "Tier 1 已连续触发 %d 次（≥%d），强制升级到 Tier 2 摘要压缩",
                    self._tier1_count, self.config.tier1_escalation_limit,
                )
                self._tier1_count = 0
                return await self._tier2_compress(system_msgs, blocks, client)

            compressed = self._tier1_compress(system_msgs, blocks)
            # 如果 Tier 1 没有真正减少消息（例如块数太少），直接升级到 Tier 2，
            # 避免“触发压缩但 token 不降”的循环。
            if len(compressed) >= len(messages):
                logger.info("Tier 1 未减少消息，升级到 Tier 2 摘要压缩")
                self._tier1_count = 0
                return await self._tier2_compress(system_msgs, blocks, client)
            return compressed


    def get_stats(self) -> dict:
        """返回最后一次压缩的统计信息。"""
        return dict(self._last_stats)

    # ------------------------------------------------------------------
    # 块划分 —— 保证工具调用对的完整性
    # ------------------------------------------------------------------

    def _split_into_blocks(self, messages: list[Message]) -> list[list[Message]]:
        """将消息列表划分为逻辑块。
        块类型
        - 单条 user 消息 -> 独立块
        - 单条 assitant 消息（无tool_calls) -> 独立块
        - assistant （含 tool_calls) + 后续所有tool消息 -> 合并为一块
        """
        blocks: list[list[Message]] = []
        current: list[Message] = []

        for msg in messages:
            if msg.role == 'assistant' and msg.tool_calls:
                # 开始一个新的工具调用块
                if current:
                    blocks.append(current)
                current = [msg]
            elif msg.role == 'tool' and current and current[-1].role == 'assistant' and current[-1].tool_calls:
                # 属于当前工具调用块的 tool 结果
                current.append(msg)
            else:
                # user 或 普通调用工具块
                if current:
                    blocks.append(current)
                current = [msg]
        
        if current:
            blocks.append(current)

        return blocks

    # ------------------------------------------------------------------
    # 三级压缩策略
    # ------------------------------------------------------------------


    def _tier1_compress(
        self,
        system_msgs: list[Message],
        blocks: list[list[Message]],
    ) -> list[Message]:
        """Tier 1 (50%): 丢弃最旧的块，保留系统提示 + 最近 N 个块。

        关键修正：当块数不超过 ``tier1_preserve_blocks`` 时，旧实现会
        直接返回所有块，导致“触发压缩但 token 不降”的死循环。现在至少
        丢弃 1 个旧块（只要存在 2 个以上块），确保压缩真正生效。
        """
        if len(blocks) <= 1:
            # 只剩一个非系统块，无法通过丢块降低 token；交给上层升级到摘要
            return system_msgs + [m for b in blocks for m in b]

        preserve = min(self.config.tier1_preserve_blocks, len(blocks) - 1)
        kept_blocks = blocks[-preserve:]
        result = system_msgs + [m for b in kept_blocks for m in b]
        dropped = len(blocks) - preserve
        logger.info("Tier 1 压缩: 丢弃 %d 个旧块，保留 %d 个块", dropped, preserve)
        return result

    async def _tier2_compress(
        self,
        system_msgs: list[Message],
        blocks: list[list[Message]],
        client: LLMClient,
    ) -> list[Message]:
        """Tier 2 (75%): 丢弃并摘要最旧的块，保留系统提示 + 摘要 + 最近 N 个块。

        关键修正：当块数不超过 ``tier2_preserve_blocks`` 时，旧实现会原样
        返回所有块。现在改为只保留最近 1 个完整块，其余全部摘要，确保
        70%+ 的高占用场景下压缩真正生效。
        """
        if len(blocks) <= 1:
            # 只剩一个块时，直接摘要整个非系统历史
            old_messages = [m for b in blocks for m in b]
            try:
                summary = await client.summarize(
                    old_messages,
                    max_summary_tokens=self.config.summary_max_tokens,
                )
            except Exception as exc:
                logger.warning("摘要生成失败: %s", exc)
                summary = "[对话历史摘要生成失败]"
            return system_msgs + [
                Message(role="user", content=f"[对话历史摘要]\n{summary}"),
            ]

        # 块数充足时按配置保留；块数较少时只保留最近 1 个，其余全部摘要
        if len(blocks) <= self.config.tier2_preserve_blocks:
            preserve = 1
        else:
            preserve = self.config.tier2_preserve_blocks

        kept_blocks = blocks[-preserve:]
        old_blocks = blocks[:-preserve]
        old_messages = [m for b in old_blocks for m in b]
        try:
            summary = await client.summarize(
                old_messages,
                max_summary_tokens=self.config.summary_max_tokens,
            )
        except Exception as exc:
            logger.warning("摘要生成失败: %s", exc)
            summary = f"[对话历史摘要生成失败，保留最近 {preserve} 个块]"

        result = system_msgs + [
            Message(role="user", content=f"[对话历史摘要]\n{summary}"),
        ] + [m for b in kept_blocks for m in b]

        logger.info("Tier 2 压缩: 摘要 %d 个旧块，保留 %d 个块", len(old_blocks), preserve)
        return result

    async def _tier3_compress(
        self,
        system_msgs: list[Message],
        blocks: list[list[Message]],
        client: LLMClient,
    ) -> list[Message]:
        """Tier 3 (90%): 紧急压缩，保留系统提示 + 超短摘要 + 最近 1 个块。"""
        if len(blocks) <= 1:
            # 只剩一个块时，直接摘要整个非系统历史
            old_messages = [m for b in blocks for m in b]
            try:
                summary = await client.summarize(
                    old_messages,
                    max_summary_tokens=self.config.emergency_summary_max_tokens,
                )
                summary = "[紧急摘要] " + summary
            except Exception:
                summary = "[紧急摘要生成失败]"
            return system_msgs + [Message(role="user", content=summary)]

        preserve = min(self.config.tier3_preserve_blocks, len(blocks) - 1)
        kept_blocks = blocks[-preserve:]
        old_blocks = blocks[:-preserve]
        old_messages = [m for b in old_blocks for m in b]

        try:
            summary = await client.summarize(
                old_messages,
                max_summary_tokens=self.config.emergency_summary_max_tokens,
            )
            summary = "[紧急摘要] " + summary
        except Exception:
            summary = "[紧急摘要生成失败]"

        result = system_msgs + [
            Message(role="user", content=summary),
        ] + [m for b in kept_blocks for m in b]

        logger.warning("Tier 3 紧急压缩: 摘要 %d 个旧块，保留 %d 个块", len(old_blocks), preserve)
        return result



