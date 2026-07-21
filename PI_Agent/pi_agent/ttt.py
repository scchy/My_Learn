"""
Day 1: 统一 LLM 客户端
========================
核心机制：流式 SSE 解析、指数退避重试、Token 估算、LLM 自身摘要

设计原则：
- OpenAI 兼容接口，覆盖 DeepSeek / Kimi / vLLM 等所有兼容提供商
- 流式输出优先，非流式作为降级方案
- 保守的 Token 估算，避免低估导致上下文溢出
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. 统一消息格式
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """统一消息格式，兼容 OpenAI Chat Completions API。

    Attributes
    ----------
    role : str
        system / user / assistant / tool
    content : str | None
        消息正文。tool 角色可为 None（只传 tool_call_id）。
    tool_call_id : str | None
        仅 role="tool" 时有效，关联到 assistant 的 tool_calls。
    tool_calls : list[dict] | None
        仅 role="assistant" 且需要调用工具时有效。
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """转为 OpenAI API 兼容的字典格式。"""
        payload: dict[str, Any] = {"role": self.role}

        if self.content is not None:
            payload["content"] = self.content

        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id

        if self.tool_calls is not None:
            payload["tool_calls"] = self.tool_calls

        return payload


# ---------------------------------------------------------------------------
# 2. Token 估算
# ---------------------------------------------------------------------------

def estimate_tokens(text: str | None) -> int:
    """保守估算 token 数量。

    策略（基于经验规则，误差 ±20% 以内）：
    - CJK 字符：1 字 ≈ 1 token
    - 其他字符：4 字符 ≈ 1 token
    - 向上取整，保守估计避免溢出
    """
    if not text:
        return 0
    cjk_count = 0
    other_count = 0
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf':
            cjk_count += 1
        else:
            other_count += 1
    # CJK: 1 char ≈ 1 token; 其他: 4 chars ≈ 1 token (向上取整)
    return cjk_count + max(1, (other_count + 3) // 4)


def estimate_messages_tokens(messages: list[Message]) -> int:
    """估算消息列表的总 token 数。"""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.content)
        if msg.tool_calls:
            total += estimate_tokens(json.dumps(msg.tool_calls, ensure_ascii=False))
    # 每条消息附加 ~4 token 的元数据开销（role, 分隔符等）
    total += len(messages) * 4
    return total


# ---------------------------------------------------------------------------
# 3. LLM 客户端
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """LLM 调用错误基类。"""


class RetryExhaustedError(LLMError):
    """重试耗尽错误。"""


class LLMClient:
    """统一 LLM 客户端 —— OpenAI 兼容接口。

    Parameters
    ----------
    api_key : str
        API Key。
    base_url : str
        API 基础地址，默认 DeepSeek。
    model : str
        模型名称。
    max_retries : int
        最大重试次数（默认 3）。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries

        # httpx 客户端（惰性创建，确保在正确的 event loop 中）
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """获取或创建 httpx 客户端。"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        """关闭 HTTP 客户端。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # 核心 API：流式 chat
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """流式对话，每次 yield 一个增量文本 chunk。

        同时解析 tool_calls（流式累积），在流结束后附带到
        ``LLMClient._last_tool_calls`` 供调用方读取。

        Yields
        ------
        str
            增量文本内容（delta chunks）。
        """
        self._last_tool_calls: list[dict[str, Any]] | None = None
        payload = self._build_payload(messages, tools, temperature, max_tokens)

        for attempt in range(self.max_retries + 1):
            try:
                client = await self._get_client()
                async with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json=payload,
                ) as response:
                    # 非 2xx → 抛异常触发重试
                    if response.status_code >= 400:
                        body = await response.aread()
                        raise LLMError(
                            f"HTTP {response.status_code}: {body.decode(errors='replace')[:500]}"
                        )

                    tool_calls_acc: dict[int, dict[str, Any]] = {}
                    async for line in response.aiter_lines():
                        chunk_text = self._parse_sse_line(line, tool_calls_acc)
                        if chunk_text:
                            yield chunk_text

                    # 流结束，归档 tool_calls
                    if tool_calls_acc:
                        self._last_tool_calls = [
                            tool_calls_acc[i]
                            for i in sorted(tool_calls_acc.keys())
                        ]
                    else:
                        self._last_tool_calls = None

                return  # 成功，退出重试循环

            except (httpx.TransportError, httpx.TimeoutException, LLMError) as exc:
                # 网络错误 / 超时 / 服务端错误 → 重试
                if attempt >= self.max_retries:
                    raise RetryExhaustedError(
                        f"重试 {self.max_retries} 次后仍失败: {exc}"
                    ) from exc
                wait = 2 ** attempt
                logger.warning(
                    "LLM 调用失败 (attempt %d/%d)，%ds 后重试: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    wait,
                    exc,
                )
                await asyncio.sleep(wait)

    # ------------------------------------------------------------------
    # 非流式 chat（降级方案 / 摘要用）
    # ------------------------------------------------------------------

    async def chat_complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Message:
        """非流式对话，返回完整 assistant 消息。

        自动包含指数退避重试。
        """
        payload = self._build_payload(messages, tools, temperature, max_tokens)
        payload["stream"] = False

        for attempt in range(self.max_retries + 1):
            try:
                client = await self._get_client()
                response = await client.post(
                    "/v1/chat/completions",
                    json=payload,
                )
                if response.status_code >= 400:
                    body = response.text
                    raise LLMError(
                        f"HTTP {response.status_code}: {body[:500]}"
                    )
                data: dict[str, Any] = response.json()
                choice = data["choices"][0]
                msg_data = choice["message"]
                return Message(
                    role="assistant",
                    content=msg_data.get("content"),
                    tool_calls=msg_data.get("tool_calls"),
                )

            except (httpx.TransportError, httpx.TimeoutException, LLMError) as exc:
                if attempt >= self.max_retries:
                    raise RetryExhaustedError(
                        f"重试 {self.max_retries} 次后仍失败: {exc}"
                    ) from exc
                wait = 2 ** attempt
                logger.warning(
                    "LLM 调用失败 (attempt %d/%d)，%ds 后重试: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    wait,
                    exc,
                )
                await asyncio.sleep(wait)

        # 理论上不可达，但静态分析友好
        raise RetryExhaustedError("Unexpected retry exhaustion")

    # ------------------------------------------------------------------
    # 摘要功能
    # ------------------------------------------------------------------

    async def summarize(
        self,
        messages: list[Message],
        *,
        max_summary_tokens: int = 512,
    ) -> str:
        """调用 LLM 自身对历史消息做摘要（用于上下文压缩）。

        使用非流式调用，要求模型输出简洁摘要。
        """
        # 序列化消息列表
        conversation_text = _messages_to_text(messages)

        summary_prompt = Message(
            role="user",
            content=(
                "请将以下对话历史压缩为一份简洁的摘要（中文输出）。\n"
                "要求：\n"
                "1. 只保留关键决策、发现和结论\n"
                "2. 丢弃冗余的问答和工具调用细节\n"
                "3. 摘要总长度不超过 300 字\n\n"
                f"对话历史：\n{conversation_text}"
            ),
        )

        result = await self.chat_complete(
            messages=[summary_prompt],
            temperature=0.3,
            max_tokens=max_summary_tokens,
        )
        return result.content or ""

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """构建请求体。"""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            # 部分提供商要求指定 tool_choice
            # payload["tool_choice"] = "auto"
        return payload

    def _parse_sse_line(
        self,
        line: str,
        tool_calls_acc: dict[int, dict[str, Any]],
    ) -> str | None:
        """解析单行 SSE 数据。

        处理三种情况：
        - 空行 / 注释行 → 跳过
        - ``data: [DONE]`` → 流结束标记
        - ``data: {...}`` → 正常数据，提取 delta

        Parameters
        ----------
        line : str
            原始 SSE 行。
        tool_calls_acc : dict
            累积 tool_calls 的字典，按 index 聚合。

        Returns
        -------
        str | None
            增量文本；若为 None 表示无内容输出（如 tool_call delta 或无文本的 chunk）。
        """
        # 空行或非 data 行直接跳过
        if not line.startswith("data:"):
            return None

        data_str = line[5:].strip()

        # 流结束标记
        if data_str == "[DONE]":
            return None

        # 解析 JSON
        try:
            data: dict[str, Any] = json.loads(data_str)
        except json.JSONDecodeError:
            logger.debug("SSE JSON 解析失败: %s", data_str)
            return None

        choices: list[dict[str, Any]] = data.get("choices", [])
        if not choices:
            return None

        delta: dict[str, Any] = choices[0].get("delta", {})
        finish_reason: str | None = choices[0].get("finish_reason")

        # 处理 tool_calls delta（流式累积）
        tc_deltas: list[dict[str, Any]] = delta.get("tool_calls", [])
        for tc in tc_deltas:
            idx: int = tc.get("index", 0)
            if idx not in tool_calls_acc:
                tool_calls_acc[idx] = {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            acc = tool_calls_acc[idx]
            # id 只在第一个 chunk 中出现，但后续 chunk 可能是空字符串
            if tc.get("id"):
                acc["id"] = tc["id"]
            func = tc.get("function", {})
            if func.get("name"):
                acc["function"]["name"] += func["name"]
            if func.get("arguments"):
                acc["function"]["arguments"] += func["arguments"]

        # 优先返回文本 delta；若 finish_reason 出现（如 "stop" 或 "tool_calls"）且
        # 此时没有文本内容，也要继续等待可能的后继 chunk 不提前终止
        content: str | None = delta.get("content")
        if content:
            return content
        return None

    @property
    def last_tool_calls(self) -> list[dict[str, Any]] | None:
        """最后一次流式 chat 调用产生的 tool_calls。"""
        return getattr(self, "_last_tool_calls", None)


# ---------------------------------------------------------------------------
# 4. 辅助函数
# ---------------------------------------------------------------------------

def _messages_to_text(messages: list[Message]) -> str:
    """将消息列表转为可读文本（用于摘要 prompt）。"""
    lines: list[str] = []
    for msg in messages:
        role = msg.role
        content = msg.content or ""
        if msg.tool_calls:
            content += "\n[工具调用] " + json.dumps(msg.tool_calls, ensure_ascii=False)
        if msg.tool_call_id:
            lines.append(f"[tool 结果 id={msg.tool_call_id}] {content}")
        else:
            lines.append(f"[{role}] {content}")
    return "\n".join(lines)
