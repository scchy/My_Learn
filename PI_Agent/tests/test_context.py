"""
Day 4: 上下文压缩模块测试
======================
覆盖内容：
- 未达阈值时不压缩
- Tier 1 块级截断（含块数 ≤ preserve 时的兜底压缩）
- Tier 1 连续触发升级到 Tier 2
- Tier 2 / Tier 3 摘要压缩
- 工具调用对完整性
- 统计信息
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from pi_agent.context import CompressionConfig, ContextCompressor
from pi_agent.llm import Message, estimate_messages_tokens

pytestmark = pytest.mark.anyio


@pytest.fixture
def client():
    """返回一个模拟的 LLMClient，summarize 返回固定摘要。"""
    c = AsyncMock()
    c.summarize = AsyncMock(return_value="这是摘要。")
    return c


def _limit_for_ratio(messages: list[Message], ratio: float) -> int:
    """根据消息估算 token 与目标比例反推出 context_limit。"""
    return int(estimate_messages_tokens(messages) / ratio)


class TestNoCompression:
    async def test_ratio_below_threshold_returns_original(self, client):
        config = CompressionConfig(tier1_ratio=0.5)
        compressor = ContextCompressor(config)
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="hi"),
        ]
        result = await compressor.compress_if_needed(messages, 10000, client)
        assert result == messages
        assert client.summarize.call_count == 0


class TestTier1Compression:
    async def test_drop_old_blocks(self, client):
        """块数充足时，Tier 1 丢弃旧块保留最近 N 个。"""
        config = CompressionConfig(tier1_ratio=0.5, tier1_preserve_blocks=2)
        compressor = ContextCompressor(config)
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="a" * 500),
            Message(role="assistant", content="b" * 500),
            Message(role="user", content="c" * 500),
            Message(role="assistant", content="d" * 500),
        ]
        # 让比例落在 Tier 1 区间 [0.5, 0.7)
        context_limit = _limit_for_ratio(messages, 0.6)
        result = await compressor.compress_if_needed(messages, context_limit, client)

        # 保留 system + 最近 2 个块（user c, assistant d）
        assert len(result) == 3
        assert result[0].role == "system"
        assert [m.content for m in result[1:]] == ["c" * 500, "d" * 500]
        assert client.summarize.call_count == 0

    async def test_compress_when_blocks_less_than_preserve(self, client):
        """关键回归测试：块数 ≤ tier1_preserve_blocks 时仍应至少丢弃 1 个块。

        旧实现会直接返回所有块，导致“触发压缩但 token 不降”的死循环。
        """
        config = CompressionConfig(tier1_ratio=0.5, tier1_preserve_blocks=4)
        compressor = ContextCompressor(config)
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="x" * 500),
            Message(role="assistant", content="y" * 500),
            Message(role="user", content="z" * 500),
        ]
        context_limit = _limit_for_ratio(messages, 0.6)
        result = await compressor.compress_if_needed(messages, context_limit, client)

        # 应丢弃至少一个块
        assert len(result) < len(messages)
        # 不调用 LLM 摘要
        assert client.summarize.call_count == 0

    async def test_escalate_to_tier2_after_limit(self, client):
        """Tier 1 连续触发 3 次后升级到 Tier 2。"""
        config = CompressionConfig(
            tier1_ratio=0.5,
            tier1_preserve_blocks=4,
            tier1_escalation_limit=3,
        )
        compressor = ContextCompressor(config)
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="x" * 500),
            Message(role="assistant", content="y" * 500),
        ]
        context_limit = _limit_for_ratio(messages, 0.6)

        # 第 1、2 次仍应为 Tier 1
        await compressor.compress_if_needed(messages, context_limit, client)
        await compressor.compress_if_needed(messages, context_limit, client)
        assert client.summarize.call_count == 0

        # 第 3 次应升级到 Tier 2（摘要）
        r3 = await compressor.compress_if_needed(messages, context_limit, client)
        assert client.summarize.call_count == 1
        assert any("摘要" in (m.content or "") for m in r3)

    async def test_tier1_escalates_when_cannot_reduce(self, client):
        """只剩一个非系统块时，Tier 1 无法丢块，应直接升级到 Tier 2。"""
        config = CompressionConfig(tier1_ratio=0.5, tier1_preserve_blocks=4)
        compressor = ContextCompressor(config)
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="x" * 4000),
        ]
        context_limit = _limit_for_ratio(messages, 0.6)
        result = await compressor.compress_if_needed(messages, context_limit, client)

        assert client.summarize.call_count == 1
        assert any("摘要" in (m.content or "") for m in result)


class TestTier2Compression:
    async def test_summarize_old_blocks(self, client):
        config = CompressionConfig()
        compressor = ContextCompressor(config)
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="a" * 500),
            Message(role="assistant", content="b" * 500),
            Message(role="user", content="c" * 500),
            Message(role="assistant", content="d" * 500),
        ]
        # 比例落在 Tier 2 区间 [0.75, 0.9)
        context_limit = _limit_for_ratio(messages, 0.8)
        result = await compressor.compress_if_needed(messages, context_limit, client)

        assert client.summarize.call_count == 1
        assert any("对话历史摘要" in (m.content or "") for m in result)
        # 系统提示应保留
        assert result[0].role == "system"

    async def test_tier2_when_blocks_less_than_preserve(self, client):
        """Tier 2 在块数 ≤ preserve 时应只保留最近 1 个块，其余全摘要。"""
        config = CompressionConfig(tier2_ratio=0.5, tier2_preserve_blocks=4)
        compressor = ContextCompressor(config)
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="a" * 1000),
            Message(role="assistant", content="b" * 1000),
        ]
        context_limit = _limit_for_ratio(messages, 0.6)
        result = await compressor.compress_if_needed(messages, context_limit, client)

        assert client.summarize.call_count == 1
        # 原始 user 块应被摘要替换
        assert any("对话历史摘要" in (m.content or "") for m in result)
        # 系统提示保留
        assert result[0].role == "system"
        # 块数不足时只保留最近 1 个完整块（assistant b）
        assert any(m.content == "b" * 1000 for m in result)
        assert not any(m.content == "a" * 1000 for m in result)


class TestTier3Compression:
    async def test_emergency_summarize(self, client):
        config = CompressionConfig()
        compressor = ContextCompressor(config)
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="a" * 500),
            Message(role="assistant", content="b" * 500),
            Message(role="user", content="c" * 500),
            Message(role="assistant", content="d" * 500),
        ]
        # 比例落在 Tier 3 区间 [0.9, +∞)
        context_limit = _limit_for_ratio(messages, 0.95)
        result = await compressor.compress_if_needed(messages, context_limit, client)

        assert client.summarize.call_count == 1
        assert any("紧急摘要" in (m.content or "") for m in result)


class TestBlockIntegrity:
    async def test_tool_call_pair_kept_together(self, client):
        """assistant + tool 消息应被划分为同一块，截断时成对保留或丢弃。"""
        config = CompressionConfig(tier1_ratio=0.5, tier1_preserve_blocks=1)
        compressor = ContextCompressor(config)
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="q1"),
            Message(
                role="assistant",
                content="call tool",
                tool_calls=[{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
            ),
            Message(role="tool", content="result1", tool_call_id="c1"),
            Message(role="user", content="q2" * 500),
        ]
        context_limit = _limit_for_ratio(messages, 0.6)
        result = await compressor.compress_if_needed(messages, context_limit, client)

        # 保留 system + 最近 1 个块（q2）
        assert len(result) == 2
        assert result[0].role == "system"
        assert result[1].content == "q2" * 500


class TestStats:
    async def test_stats_recorded(self, client):
        compressor = ContextCompressor()
        messages = [Message(role="user", content="x" * 4000)]
        context_limit = _limit_for_ratio(messages, 0.6)
        await compressor.compress_if_needed(messages, context_limit, client)

        stats = compressor.get_stats()
        assert stats["messages"] == len(messages)
        assert stats["context_limit"] == context_limit
        assert "ratio" in stats
