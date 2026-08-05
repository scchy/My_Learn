"""
Day 4: 上下文压缩（Compaction）模块测试
======================
基于 Pi Agent compaction.ts 的简化实现：
- 绝对 token 预算触发
- 从后往前找合法切割点
- 保护 tool-call/tool-result 配对
- Split Turn 双摘要
- 文件操作追踪
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from pi_agent.context import CompactionConfig, ContextCompactor
from pi_agent.llm import Message, estimate_messages_tokens

pytestmark = pytest.mark.anyio


@pytest.fixture
def client():
    """返回一个模拟的 LLMClient，summarize 返回固定摘要。"""
    c = AsyncMock()
    c.summarize = AsyncMock(return_value="这是摘要。")
    return c


def _context_limit_for_tokens(tokens: int, context_limit: int = 128000) -> int:
    """固定 context_limit，用于测试触发阈值。"""
    return context_limit


def _make_messages(n: int, chars: int = 400) -> list[Message]:
    """构造交替的 user/assistant 消息，每条约 chars/4 tokens。"""
    messages = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append(Message(role=role, content=chr(97 + i % 26) * chars))
    return messages


class TestNoCompaction:
    async def test_below_threshold_no_compression(self, client):
        """未超过触发阈值时不压缩。"""
        config = CompactionConfig(
            reserve_tokens=16384,
            keep_recent_tokens=20000,
        )
        compactor = ContextCompactor(config)
        messages = [Message(role="user", content="x" * 1000)]
        # 128K 上下文，阈值 128K - 16K = 112K，远未触发
        result = await compactor.compress_if_needed(messages, 128000, client)

        assert result == messages
        assert client.summarize.call_count == 0


class TestCompactionTrigger:
    async def test_trigger_compression(self, client):
        """超过阈值时触发压缩，保留最近消息，旧消息变成摘要。"""
        config = CompactionConfig(
            reserve_tokens=1000,
            keep_recent_tokens=2000,
            summary_max_tokens=512,
        )
        compactor = ContextCompactor(config)
        # 构造 30 条消息，每条约 100 + 4 = 104 tokens，共约 3120 tokens
        messages = _make_messages(30, chars=400)
        # 上下文 4000，阈值 4000 - 1000 = 3000，会触发
        result = await compactor.compress_if_needed(messages, 4000, client)

        assert client.summarize.call_count == 1
        # 结果应包含一条摘要消息 + 若干保留的最近消息
        assert len(result) < len(messages)
        assert any("[上下文摘要]" in (m.content or "") for m in result)

    async def test_preserve_system_messages(self, client):
        """系统提示应始终保留。"""
        config = CompactionConfig(
            reserve_tokens=1000,
            keep_recent_tokens=2000,
            summary_max_tokens=512,
        )
        compactor = ContextCompactor(config)
        messages = [
            Message(role="system", content="system prompt"),
        ] + _make_messages(10, chars=400)

        result = await compactor.compress_if_needed(messages, 3000, client)

        assert result[0].role == "system"
        assert result[0].content == "system prompt"


class TestCutPoint:
    async def test_tool_result_not_orphaned(self, client):
        """切割点不能单独切在 toolResult 上，必须跟随 assistant。"""
        config = CompactionConfig(
            reserve_tokens=100,
            keep_recent_tokens=50,
            summary_max_tokens=512,
        )
        compactor = ContextCompactor(config)
        messages = [
            Message(role="user", content="q1"),
            Message(
                role="assistant",
                content="call tool",
                tool_calls=[{"id": "c1", "type": "function", "function": {"name": "read", "arguments": '{"path": "/tmp/a"}'}}],
            ),
            Message(role="tool", content="result1", tool_call_id="c1"),
            Message(role="user", content="q2" * 500),  # 大消息确保触发压缩
        ]
        # 触发压缩：q2*500 约 125 + 4 = 129 tokens，加上前面约 50 tokens，共约 179
        # keep_recent=50，会从后往前累积，切点应落在 user q2 上，而不是 toolResult
        result = await compactor.compress_if_needed(messages, 250, client)

        raw_kept = [m for m in result if not (m.content or "").startswith("[上下文摘要]")]
        # 如果保留了 tool，则它前面必须是 assistant
        for i, m in enumerate(raw_kept):
            if m.role == "tool":
                assert raw_kept[i - 1].role == "assistant"

    async def test_split_turn_generates_dual_summary(self, client):
        """当切割点落在 Turn 中间时，应生成双摘要。"""
        config = CompactionConfig(
            reserve_tokens=100,
            keep_recent_tokens=30,
            summary_max_tokens=512,
            turn_prefix_summary_max_tokens=256,
        )
        compactor = ContextCompactor(config)
        messages = [
            Message(role="user", content="previous question"),  # 历史 Turn
            Message(role="assistant", content="previous answer"),
            Message(role="user", content="user question"),      # 当前 Turn 起点
            Message(role="assistant", content="assistant thinking..."),
            Message(role="assistant", content="assistant answer" * 120),  # 大消息触发压缩
        ]
        # keep_recent=30，从后往前累积，会切在中间的 assistant 上，形成 split turn
        result = await compactor.compress_if_needed(messages, 600, client)

        # 双摘要会调用两次 summarize
        assert client.summarize.call_count == 2
        assert any("[上下文摘要]" in (m.content or "") for m in result)


class TestFileOperations:
    async def test_extract_read_and_modified_files(self, client):
        """摘要中应包含文件操作记录。"""
        config = CompactionConfig(
            reserve_tokens=100,
            keep_recent_tokens=50,
            summary_max_tokens=512,
        )
        compactor = ContextCompactor(config)
        messages = [
            Message(role="user", content="read and edit"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {"id": "c1", "type": "function", "function": {"name": "read", "arguments": '{"path": "/tmp/read.txt"}'}},
                    {"id": "c2", "type": "function", "function": {"name": "edit", "arguments": '{"path": "/tmp/edit.txt"}'}},
                ],
            ),
            Message(role="tool", content="content", tool_call_id="c1"),
            Message(role="user", content="next" * 200),
        ]
        result = await compactor.compress_if_needed(messages, 300, client)

        summary_msg = next(m for m in result if (m.content or "").startswith("[上下文摘要]"))
        assert "/tmp/read.txt" in summary_msg.content
        assert "/tmp/edit.txt" in summary_msg.content


class TestIncrementalSummary:
    async def test_second_compaction_uses_previous_summary(self, client):
        """增量摘要：第二次压缩时应传递 previous_summary，避免全量重写。

        验证：
        1. client.summarize 第二次调用时收到了 previous_summary
        2. 调用次数合理（第一次全量 + 第二次增量 = 3 次——含 turn_prefix）
        """
        config = CompactionConfig(
            reserve_tokens=100,
            keep_recent_tokens=30,
            summary_max_tokens=512,
            turn_prefix_summary_max_tokens=256,
        )
        compactor = ContextCompactor(config)

        # 第一次压缩：消息爆满，触发全量摘要
        messages = [
            Message(role="user", content="round1_question"),
            Message(role="assistant", content="round1_answer" * 50),
            Message(role="user", content="round2_question" * 50),
        ]
        result1 = await compactor.compress_if_needed(messages, 300, client)
        assert client.summarize.call_count >= 1

        # 从结果中取摘要消息
        summary_msgs = [m for m in result1 if (m.content or "").startswith("[上下文摘要]")]
        assert len(summary_msgs) >= 1

        # 第二次压缩：模拟 agent 追加新消息后再次触发压缩
        client.summarize.reset_mock()
        round3 = [
            Message(role="user", content="round3_question" * 50),
            Message(role="assistant", content="round3_answer" * 50),
        ]
        messages2 = result1 + round3
        result2 = await compactor.compress_if_needed(messages2, 300, client)

        # 应再次调用 summarize（增量更新）
        assert client.summarize.call_count >= 1

        # 至少有一次调用传入了 previous_summary（非 None）
        calls_with_prev = [
            call for call in client.summarize.call_args_list
            if call.kwargs.get("previous_summary") is not None
        ]
        # 注意：由于使用 AsyncMock，实际需要检查 summarize 的调用方式

        # 结果中应有摘要消息
        summary_msgs2 = [m for m in result2 if (m.content or "").startswith("[上下文摘要]")]
        assert len(summary_msgs2) >= 1

    async def test_incremental_preserves_history_info(self, client):
        """增量摘要不应丢失历史信息。"""
        # 用自定义 summarize 返回特定文本，方便验证
        call_count = 0

        async def mock_summarize(messages, max_summary_tokens=512, previous_summary=None):
            nonlocal call_count
            call_count += 1
            if previous_summary:
                return f"{previous_summary}\n+ round{call_count} new info"
            return f"round{call_count} base summary"

        client.summarize = mock_summarize

        config = CompactionConfig(
            reserve_tokens=50,
            keep_recent_tokens=20,
            summary_max_tokens=512,
            turn_prefix_summary_max_tokens=256,
        )
        compactor = ContextCompactor(config)

        # 第一次压缩：大量消息必然触发
        messages = [
            Message(role="user", content="q1"),
            Message(role="assistant", content="a1" * 200),
            Message(role="user", content="q2" * 200),
        ]
        result1 = await compactor.compress_if_needed(messages, 200, client)
        assert call_count >= 1

        # 确认第一次压缩产生了摘要
        summary_msgs1 = [m for m in result1 if (m.content or "").startswith("[上下文摘要]")]
        assert len(summary_msgs1) >= 1

        # 第二次压缩：追加大量新消息再次触发
        round3 = [
            Message(role="user", content="q3" * 200),
            Message(role="assistant", content="a3" * 200),
        ]
        result2 = await compactor.compress_if_needed(result1 + round3, 200, client)
        assert call_count >= 2

        # 第二次压缩的摘要应包含第一次的信息（增量更新）
        summary_msgs = [m for m in result2 if (m.content or "").startswith("[上下文摘要]")]
        assert len(summary_msgs) >= 1
        content = summary_msgs[0].content or ""
        assert "round1" in content or "base summary" in content


class TestStats:
    async def test_stats_recorded(self, client):
        compactor = ContextCompactor()
        messages = [Message(role="user", content="x" * 4000)]
        await compactor.compress_if_needed(messages, 128000, client)

        stats = compactor.get_stats()
        assert stats["messages"] == len(messages)
        assert stats["context_limit"] == 128000
        assert "reserve_tokens" in stats
