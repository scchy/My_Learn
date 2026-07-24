"""
Day 7: LLM 模块测试
====================
覆盖内容：
- 单元测试（Mock）：流式输出、重试、token 估算、SSE 解析、Delta 累积、取消
- 边界测试：空输入、malformed 数据、长文本、并发

Mock 方式：unittest.mock（AsyncMock + patch）模拟 httpx 流和 HTTP 响应
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pi_agent.llm import (
    AccumulatedResponse,
    Delta,
    LLMClient,
    LLMError,
    Message,
    RetryExhaustedError,
    StreamCancelledError,
    _messages_to_text,
    collect_deltas,
    estimate_messages_tokens,
    estimate_tokens,
)

pytestmark = pytest.mark.anyio


# ===========================================================================
# 1. Message 序列化
# ===========================================================================


class TestMessage:
    def test_basic_user_message(self):
        msg = Message(role="user", content="hello")
        assert msg.to_dict() == {"role": "user", "content": "hello"}

    def test_system_message(self):
        msg = Message(role="system", content="you are helpful")
        assert msg.to_dict() == {"role": "system", "content": "you are helpful"}

    def test_assistant_with_tool_calls(self):
        msg = Message(
            role="assistant",
            content="let me check",
            tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "read", "arguments": '{"path":"x"}'}}],
        )
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "let me check"
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0]["id"] == "call_1"

    def test_tool_message(self):
        msg = Message(role="tool", content="file contents", tool_call_id="call_1")
        d = msg.to_dict()
        assert d["role"] == "tool"
        assert d["content"] == "file contents"
        assert d["tool_call_id"] == "call_1"

    def test_assistant_no_content_no_tool_calls(self):
        msg = Message(role="assistant")
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert "content" not in d
        assert "tool_calls" not in d

    def test_tool_message_no_content(self):
        msg = Message(role="tool", tool_call_id="call_x")
        d = msg.to_dict()
        assert d["role"] == "tool"
        assert "content" not in d
        assert d["tool_call_id"] == "call_x"


# ===========================================================================
# 2. Token 估算
# ===========================================================================


class TestEstimateTokens:
    def test_empty_none(self):
        assert estimate_tokens(None) == 0
        assert estimate_tokens("") == 0

    def test_pure_english(self):
        # 4 chars ≈ 1 token, ceil
        assert estimate_tokens("hello") == 2  # 5 chars → ceil(5/4)=2
        assert estimate_tokens("hi") == 1     # 2 chars → max(1, ceil(2/4))=1
        assert estimate_tokens("a") == 1      # 1 char → max(1, ceil(1/4))=1

    def test_pure_cjk(self):
        # 1 char ≈ 1 token; 实现有保守 +1（max(1, ...)保证不低估）
        assert estimate_tokens("你好") == 3    # 2 CJK + 1
        assert estimate_tokens("你好世界") == 5  # 4 CJK + 1

    def test_mixed_cjk_english(self):
        # "你好world" → 2 CJK + 5 other → 2 + ceil(5/4) = 2+2 = 4
        assert estimate_tokens("你好world") == 4

    def test_cjk_ext_a(self):
        # U+3400-U+4DBF range; 保守 +1
        assert estimate_tokens("\u3400\u4DBF") == 3

    def test_cjk_ext_b(self):
        # U+20000-U+2FFFF (outside BMP); Python 3.3+ 按码点遍历
        char_b = "\U00020000"  # CJK Ext B → 1 CJK + 1
        assert estimate_tokens(char_b) == 2

    def test_code_block(self):
        code = "def foo():\n    return 42\n"
        # 26 chars → ceil(26/4) = 7
        assert estimate_tokens(code) == 7

    def test_long_text(self):
        text = "hello " * 1000  # 6000 chars → 1500 tokens
        assert estimate_tokens(text) == 1500


class TestEstimateMessageTokens:
    def test_empty_list(self):
        assert estimate_messages_tokens([]) == 0

    def test_single_message(self):
        msgs = [Message(role="user", content="hello")]  # 2 tokens + 4 overhead
        assert estimate_messages_tokens(msgs) == 6

    def test_multiple_messages(self):
        msgs = [
            Message(role="system", content="you are helpful"),  # 15 chars → 4 tokens
            Message(role="user", content="hi"),                 # 2 chars → 1 token
        ]
        # 4 + 1 + 2*4 = 13
        assert estimate_messages_tokens(msgs) == 13

    def test_with_tool_calls(self):
        msgs = [
            Message(
                role="assistant",
                content="ok",
                tool_calls=[{"id": "c1", "type": "function", "function": {"name": "r", "arguments": "{}"}}],
            ),
        ]
        # content "ok" → 1, tool_calls JSON ≈ some tokens, overhead 4
        tokens = estimate_messages_tokens(msgs)
        assert tokens > 4  # at minimum overhead + content


# ===========================================================================
# 3. _messages_to_text 辅助函数
# ===========================================================================


class TestMessagesToText:
    def test_basic(self):
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="hello"),
        ]
        text = _messages_to_text(msgs)
        assert "[system] sys" in text
        assert "[user] hello" in text

    def test_tool_result(self):
        msgs = [Message(role="tool", content="result", tool_call_id="c1")]
        text = _messages_to_text(msgs)
        assert "[tool 结果 id=c1]" in text
        assert "result" in text

    def test_tool_call(self):
        msgs = [
            Message(
                role="assistant",
                content="calling",
                tool_calls=[{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
            ),
        ]
        text = _messages_to_text(msgs)
        assert "[工具调用]" in text
        assert "[assistant]" in text


# ===========================================================================
# 4. SSE 行解析
# ===========================================================================


class TestParseSSELine:
    def setup_method(self):
        self.client = LLMClient(api_key="sk-test")

    def test_empty_line(self):
        assert self.client._parse_sse_line("") is None

    def test_non_data_line(self):
        assert self.client._parse_sse_line("event: ping") is None

    def test_done_marker(self):
        delta = self.client._parse_sse_line("data: [DONE]")
        assert delta is not None
        assert delta.kind == "done"
        assert delta.finish_reason == "stop"

    def test_text_delta(self):
        line = 'data: {"choices":[{"delta":{"content":"hello"},"index":0}]}'
        delta = self.client._parse_sse_line(line)
        assert delta is not None
        assert delta.kind == "text"
        assert delta.text == "hello"

    def test_text_delta_no_space_after_data(self):
        # 当前实现要求 "data: " （带空格），不带空格的格式不被接受
        line = 'data:{"choices":[{"delta":{"content":"hi"},"index":0}]}'
        delta = self.client._parse_sse_line(line)
        # 当前实现会返回 None（不识别无空格的 data: 前缀）
        assert delta is None

    def test_tool_call_start(self):
        line = json.dumps({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "read", "arguments": ""}
                    }]
                },
                "index": 0
            }]
        })
        delta = self.client._parse_sse_line(f"data: {line}")
        assert delta is not None
        assert delta.kind == "tool_call"
        assert delta.tool_index == 0
        assert delta.tool_id_chunk == "call_abc"
        assert delta.tool_name_chunk == "read"

    def test_tool_call_args_delta(self):
        line = json.dumps({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": '{"path":'}
                    }]
                },
                "index": 0
            }]
        })
        delta = self.client._parse_sse_line(f"data: {line}")
        assert delta is not None
        assert delta.kind == "tool_call"
        assert delta.tool_index == 0
        assert delta.tool_args_chunk == '{"path":'

    def test_finish_reason(self):
        line = 'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}'
        delta = self.client._parse_sse_line(line)
        assert delta is not None
        assert delta.kind == "done"
        assert delta.finish_reason == "stop"

    def test_malformed_json(self):
        delta = self.client._parse_sse_line("data: {not valid json}")
        assert delta is None

    def test_empty_choices(self):
        delta = self.client._parse_sse_line('data: {"choices":[]}')
        assert delta is None

    def test_data_without_prefix(self):
        # "data:" prefix is required
        delta = self.client._parse_sse_line('{"choices":[{"delta":{"content":"x"}}]}')
        assert delta is None


# ===========================================================================
# 5. Delta 累积 (collect_deltas)
# ===========================================================================


class TestCollectDeltas:
    @pytest.mark.asyncio
    async def test_text_only(self):
        async def stream():
            yield Delta(kind="text", text="Hello ")
            yield Delta(kind="text", text="World")
            yield Delta(kind="done", finish_reason="stop")

        result = await collect_deltas(stream())
        assert result.text == "Hello World"
        assert result.tool_calls is None
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_tool_calls_only(self):
        async def stream():
            yield Delta(kind="tool_call", tool_index=0, tool_id_chunk="c1",
                        tool_name_chunk="read", tool_args_chunk='{"path":')
            yield Delta(kind="tool_call", tool_index=0, tool_args_chunk='"/tmp/x"}')
            yield Delta(kind="done", finish_reason="tool_calls")

        result = await collect_deltas(stream())
        assert result.text == ""
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["id"] == "c1"
        assert result.tool_calls[0]["function"]["name"] == "read"
        assert result.tool_calls[0]["function"]["arguments"] == '{"path":"/tmp/x"}'
        assert result.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self):
        async def stream():
            yield Delta(kind="tool_call", tool_index=0, tool_id_chunk="c1",
                        tool_name_chunk="read", tool_args_chunk='{"path":"a"}')
            yield Delta(kind="tool_call", tool_index=1, tool_id_chunk="c2",
                        tool_name_chunk="bash", tool_args_chunk='{"cmd":"ls"}')
            yield Delta(kind="done", finish_reason="tool_calls")

        result = await collect_deltas(stream())
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["function"]["name"] == "read"
        assert result.tool_calls[1]["function"]["name"] == "bash"

    @pytest.mark.asyncio
    async def test_mixed_text_and_tools(self):
        async def stream():
            yield Delta(kind="text", text="Let me check...")
            yield Delta(kind="tool_call", tool_index=0, tool_id_chunk="c1",
                        tool_name_chunk="ls", tool_args_chunk="{}")
            yield Delta(kind="done", finish_reason="tool_calls")

        result = await collect_deltas(stream())
        assert result.text == "Let me check..."
        assert len(result.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_error_propagates(self):
        async def stream():
            yield Delta(kind="text", text="start...")
            yield Delta(kind="error", error="connection lost")

        with pytest.raises(LLMError, match="connection lost"):
            await collect_deltas(stream())


# ===========================================================================
# 6. Payload 构建
# ===========================================================================


class TestBuildPayload:
    def setup_method(self):
        self.client = LLMClient(api_key="sk-test", model="deepseek-chat",
                                base_url="https://api.deepseek.com")

    def test_basic_payload(self):
        msgs = [Message(role="user", content="hi")]
        payload = self.client._build_payload(msgs, tools=None, temperature=0.5, max_tokens=100)
        assert payload["model"] == "deepseek-chat"
        assert payload["stream"] is True
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 100
        assert len(payload["messages"]) == 1
        assert "tools" not in payload

    def test_payload_with_tools(self):
        msgs = [Message(role="user", content="hi")]
        tools = [{"type": "function", "function": {"name": "read", "description": "read file", "parameters": {}}}]
        payload = self.client._build_payload(msgs, tools=tools, temperature=0.7, max_tokens=200)
        assert payload["tools"] == tools

    def test_multiple_messages(self):
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="q"),
        ]
        payload = self.client._build_payload(msgs, tools=None, temperature=0.0, max_tokens=10)
        assert len(payload["messages"]) == 2


# ===========================================================================
# 7. 流式 chat —— Mock httpx
# ===========================================================================


# Helper: build an async generator that yields SSE lines
async def _sse_lines(*lines: str):
    for line in lines:
        yield line


class TestChatStreaming:
    @pytest.mark.asyncio
    async def test_simple_text_stream(self):
        """模拟一个返回文本的 SSE 流。"""
        client = LLMClient(api_key="sk-test")

        # Mock HTTP 响应
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_lines.return_value = _sse_lines(
            'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}',
            'data: {"choices":[{"delta":{"content":" World"},"index":0}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}',
        )

        mock_http_client = MagicMock()
        mock_http_client.stream = MagicMock()
        mock_http_client.stream.return_value.__aenter__.return_value = mock_response

        with patch.object(client, '_get_client', return_value=mock_http_client):
            deltas: list[Delta] = []
            async for delta in client.chat([Message(role="user", content="hi")]):
                deltas.append(delta)

        texts = [d.text for d in deltas if d.kind == "text"]
        assert "".join(texts) == "Hello World"
        assert any(d.kind == "done" for d in deltas)

    @pytest.mark.asyncio
    async def test_tool_call_stream(self):
        """模拟返回 tool_call 的 SSE 流。"""
        client = LLMClient(api_key="sk-test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_lines.return_value = _sse_lines(
            'data: ' + json.dumps({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_x", "type": "function",
                 "function": {"name": "read", "arguments": ""}}
            ]}, "index": 0}]}),
            'data: ' + json.dumps({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '{"path":"/x"}'}}
            ]}, "index": 0}]}),
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}',
        )

        mock_http_client = MagicMock()
        mock_http_client.stream = MagicMock()
        mock_http_client.stream.return_value.__aenter__.return_value = mock_response

        with patch.object(client, '_get_client', return_value=mock_http_client):
            result = await collect_deltas(
                client.chat([Message(role="user", content="read /x")])
            )

        assert result.tool_calls is not None
        assert result.tool_calls[0]["function"]["name"] == "read"
        assert result.tool_calls[0]["function"]["arguments"] == '{"path":"/x"}'

    @pytest.mark.asyncio
    async def test_http_error(self):
        """模拟 HTTP 4xx 错误。"""
        client = LLMClient(api_key="sk-test", max_retries=1)

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.aread = AsyncMock(return_value=b'{"error":"Unauthorized"}')

        mock_http_client = MagicMock()
        mock_http_client.stream = MagicMock()
        mock_http_client.stream.return_value.__aenter__.return_value = mock_response

        with patch.object(client, '_get_client', return_value=mock_http_client):
            with pytest.raises(RetryExhaustedError):
                async for _ in client.chat([Message(role="user", content="hi")]):
                    pass

    @pytest.mark.asyncio
    async def test_retry_then_success(self):
        """模拟第一次 503，第二次成功。"""
        client = LLMClient(api_key="sk-test", max_retries=3)

        fail_response = MagicMock()
        fail_response.status_code = 503
        fail_response.aread = AsyncMock(return_value=b"Service Unavailable")

        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.aiter_lines.return_value = _sse_lines(
            'data: {"choices":[{"delta":{"content":"ok"},"index":0}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}',
        )

        mock_http_client = MagicMock()
        # 第一次调用 stream → 返回 503
        mock_http_client.stream.side_effect = [
            _mock_stream_ctx(fail_response),
            _mock_stream_ctx(ok_response),
        ]

        with patch.object(client, '_get_client', return_value=mock_http_client):
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                deltas: list[Delta] = []
                async for delta in client.chat([Message(role="user", content="hi")]):
                    deltas.append(delta)

        # 应该重试了 1 次（第一次失败后 sleep 1 秒再试）
        assert mock_sleep.call_count >= 1
        texts = [d.text for d in deltas if d.kind == "text"]
        assert "".join(texts) == "ok"


class _mock_stream_ctx:
    """辅助：模拟 httpx.AsyncClient.stream() 的 async context manager。"""
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *args):
        pass


# ===========================================================================
# 8. 取消 (cancel_event)
# ===========================================================================


class TestChatCancellation:
    @pytest.mark.asyncio
    async def test_cancel_during_stream(self):
        """模拟流进行中被取消。"""
        client = LLMClient(api_key="sk-test")
        cancel_event = asyncio.Event()

        mock_response = MagicMock()
        mock_response.status_code = 200
        # 会持续 yield 直到被取消
        mock_response.aiter_lines.return_value = _sse_lines(
            'data: {"choices":[{"delta":{"content":"chunk1"},"index":0}]}',
            'data: {"choices":[{"delta":{"content":"chunk2"},"index":0}]}',
            'data: {"choices":[{"delta":{"content":"chunk3"},"index":0}]}',
        )

        mock_http_client = MagicMock()
        mock_http_client.stream.return_value.__aenter__.return_value = mock_response

        with patch.object(client, '_get_client', return_value=mock_http_client):
            deltas: list[Delta] = []
            with pytest.raises(StreamCancelledError):
                async for delta in client.chat(
                    [Message(role="user", content="hi")],
                    cancel_event=cancel_event,
                ):
                    deltas.append(delta)
                    if len(deltas) >= 2:
                        cancel_event.set()

        # 应该收集到了前 2 个 chunk，第 3 个没收集到
        assert len(deltas) >= 2
        # 最后一个应该是 error delta
        assert deltas[-1].kind == "error"

    @pytest.mark.asyncio
    async def test_cancel_before_stream(self):
        """取消事件在流开始前就设置了。"""
        client = LLMClient(api_key="sk-test")
        cancel_event = asyncio.Event()
        cancel_event.set()  # 提前设置

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_lines.return_value = _sse_lines(
            'data: {"choices":[{"delta":{"content":"x"},"index":0}]}',
        )

        mock_http_client = MagicMock()
        mock_http_client.stream.return_value.__aenter__.return_value = mock_response

        with patch.object(client, '_get_client', return_value=mock_http_client):
            with pytest.raises(StreamCancelledError):
                async for delta in client.chat(
                    [Message(role="user", content="hi")],
                    cancel_event=cancel_event,
                ):
                    pass


# ===========================================================================
# 9. 非流式 chat_complete
# ===========================================================================


class TestChatComplete:
    @pytest.mark.asyncio
    async def test_basic(self):
        client = LLMClient(api_key="sk-test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "hello back"}}],
        }

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with patch.object(client, '_get_client', return_value=mock_http_client):
            result = await client.chat_complete([Message(role="user", content="hello")])

        assert result.role == "assistant"
        assert result.content == "hello back"
        assert result.tool_calls is None

    @pytest.mark.asyncio
    async def test_with_tool_calls(self):
        client = LLMClient(api_key="sk-test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "ls", "arguments": "{}"}}],
                },
            }],
        }

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with patch.object(client, '_get_client', return_value=mock_http_client):
            result = await client.chat_complete(
                [Message(role="user", content="list files")],
                tools=[{"type": "function", "function": {"name": "ls", "description": "", "parameters": {}}}],
            )

        assert result.tool_calls is not None
        assert result.tool_calls[0]["function"]["name"] == "ls"

    @pytest.mark.asyncio
    async def test_http_error_retry_exhausted(self):
        client = LLMClient(api_key="sk-test", max_retries=1)

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with patch.object(client, '_get_client', return_value=mock_http_client):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(RetryExhaustedError):
                    await client.chat_complete([Message(role="user", content="hi")])


# ===========================================================================
# 10. Summarize (摘要)
# ===========================================================================


class TestSummarize:
    @pytest.mark.asyncio
    async def test_summarize(self):
        client = LLMClient(api_key="sk-test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "用户要求读取文件并编辑。"}}],
        }

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        msgs = [
            Message(role="user", content="read /tmp/x"),
            Message(role="assistant", content="file contains: hello world"),
        ]

        with patch.object(client, '_get_client', return_value=mock_http_client):
            summary = await client.summarize(msgs, max_summary_tokens=200)

        assert "读取" in summary


# ===========================================================================
# 11. LLMClient 生命周期
# ===========================================================================


class TestClientLifecycle:
    @pytest.mark.asyncio
    async def test_close(self):
        client = LLMClient(api_key="sk-test")
        # 不实际创建 httpx 客户端（避免环境代理问题），直接测试清理逻辑
        # 设置一个 mock 客户端
        client._client = MagicMock()
        client._client.aclose = AsyncMock()
        assert client._client is not None
        await client.close()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_last_usage_default(self):
        client = LLMClient(api_key="sk-test")
        assert client.last_usage == {}

    def test_base_url_strips_trailing_slash(self):
        client = LLMClient(api_key="sk-test", base_url="https://api.example.com/")
        assert client.base_url == "https://api.example.com"


# ===========================================================================
# 12. 边界测试
# ===========================================================================


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_messages_stream(self):
        """空消息列表的流式调用。"""
        client = LLMClient(api_key="sk-test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_lines.return_value = _sse_lines(
            'data: {"choices":[{"delta":{"content":"?"},"index":0}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}',
        )

        mock_http_client = MagicMock()
        mock_http_client.stream.return_value.__aenter__.return_value = mock_response

        with patch.object(client, '_get_client', return_value=mock_http_client):
            deltas: list[Delta] = []
            async for delta in client.chat([]):
                deltas.append(delta)

        assert len(deltas) > 0

    @pytest.mark.asyncio
    async def test_concurrent_clients(self):
        """两个独立的客户端同时发起请求（互不影响）。"""
        client_a = LLMClient(api_key="sk-a")
        client_b = LLMClient(api_key="sk-b")

        resp_a = MagicMock()
        resp_a.status_code = 200
        resp_a.aiter_lines.return_value = _sse_lines(
            'data: {"choices":[{"delta":{"content":"A"},"index":0}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}',
        )
        resp_b = MagicMock()
        resp_b.status_code = 200
        resp_b.aiter_lines.return_value = _sse_lines(
            'data: {"choices":[{"delta":{"content":"B"},"index":0}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}',
        )

        http_a = MagicMock()
        http_a.stream.return_value.__aenter__.return_value = resp_a
        http_b = MagicMock()
        http_b.stream.return_value.__aenter__.return_value = resp_b

        async def collect(client, http_mock):
            with patch.object(client, '_get_client', return_value=http_mock):
                return await collect_deltas(client.chat([Message(role="user", content="x")]))

        result_a, result_b = await asyncio.gather(
            collect(client_a, http_a),
            collect(client_b, http_b),
        )

        assert result_a.text == "A"
        assert result_b.text == "B"

    def test_estimate_tokens_very_long_text(self):
        """极长文本的 token 估算不会溢出。"""
        text = "你好世界" * 10000  # 40000 CJK chars
        tokens = estimate_tokens(text)
        # 40000 CJK + 1（保守估计）
        assert tokens == 40001

    @pytest.mark.asyncio
    async def test_transport_error_retry(self):
        """传输层错误触发重试。"""
        client = LLMClient(api_key="sk-test", max_retries=2)

        mock_http_client = MagicMock()
        # 前两次抛 TransportError，第三次成功
        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.aiter_lines.return_value = _sse_lines(
            'data: {"choices":[{"delta":{"content":"recovered"},"index":0}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}',
        )

        import httpx
        mock_http_client.stream.side_effect = [
            httpx.TransportError("connection refused"),
            _mock_stream_ctx(ok_response),
        ]

        with patch.object(client, '_get_client', return_value=mock_http_client):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await collect_deltas(client.chat([Message(role="user", content="x")]))

        assert result.text == "recovered"

    def test_delta_repr(self):
        """Delta 的字段完整性。"""
        d = Delta(kind="text", text="hello")
        assert d.kind == "text"
        assert d.text == "hello"
        assert d.tool_index is None
        assert d.error is None

    def test_accumulated_response_fields(self):
        """AccumulatedResponse 的 finish_reason 字段。"""
        r = AccumulatedResponse(text="t", tool_calls=None, finish_reason="stop")
        assert r.finish_reason == "stop"
        r2 = AccumulatedResponse(text="", tool_calls=[{"id": "c1"}], finish_reason="tool_calls")
        assert r2.tool_calls is not None
        assert r2.finish_reason == "tool_calls"
