# Day 1: 统一 LLM 客户端 (~80行)
# python3
# Create Date: 2026-07-21
# Author: Scc_hy
# Tip:
# 设计原则：
# - OpenAI 兼容接口，覆盖 DeepSeek / Kimi / vLLM 等所有兼容提供商
# - 流式输出优先，非流式作为降级方案
# - 保守的 Token 估算，避免低估导致上下文溢出
# ==========================================================================================

from __future__ import annotations 

import asyncio
import json 
from dataclasses import dataclass
from typing import Any, AsyncIterator, Literal
import httpx
from loguru import logger 


# 1. 统一消息格式

@dataclass 
class Message:
    """
    统一消息格式，兼容 OpenAI Chat Completions API。
    Attributes
    -----------
    role: str
        system / user / assistant / tool
    content: str | None
        消息正文。 tool 角色可为None(只传 tool_call_id)
    tool_call_id: str | None
        仅 role="tool" 时有效，关联到 assistant 的 tool_calls。
    tool_calls: List[Dict] | None 
        仅 role="assistant"且需要调用工具时有效
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None 
    tool_call_id: str | None = None 
    tool_calls: list[dict[str, Any]] | None = None
    
    def to_dict(self): # -> dict[str, Any]:
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
# 2. Delta —— 流式数据块（无副作用设计）
# ---------------------------------------------------------------------------


@dataclass
class Delta:
    """统一的流式增量数据块。

    每种 kind 对应不同的有效字段：
    - text:           text 字段有值，增量文本
    - tool_call:      tool_index + tool_id_chunk/name_chunk/args_chunk 有值
    - done:           finish_reason 有值
    - error:          error 有值
    """

    kind: Literal["text", "tool_call", "done", "error"]
    text: str | None = None
    # tool_call 相关
    tool_index: int | None = None
    tool_id_chunk: str | None = None
    tool_name_chunk: str | None = None
    tool_args_chunk: str | None = None
    # done 相关
    finish_reason: str | None = None
    # error 相关
    error: str | None = None


# ---------------------------------------------------------------------------
# 3. Token 估算
# ---------------------------------------------------------------------------


def estimate_tokens(text: str | None) -> int:
    """保守估算 token 数量。

    策略（基于经验规则，误差 ±20% 以内）：
    - CJK 字符：1 字 ≈ 1 token
    - 其他字符：4 字符 ≈ 1 token
    - 向上取整，保守估计避免溢出

    注意：这是估算值（误差 ±20%），实际 token 数以 API 返回的 usage 为准。
    代码中的 CJK 覆盖了 Ext-A，码点更高的 Ext B-H 在现代 tokenizer 中
    也按 ~1 token/字 处理，误差可接受。
    """
    if not text:
        return 0
    cjk = 0
    other = 0
    for ch in text:
        cp = ord(ch)
        if (
            0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs
            or 0x3400 <= cp <= 0x4DBF  # CJK Unified Ideographs Extension A
            or 0xF900 <= cp <= 0xFAFF  # CJK Compatibility Ideographs
            or 0x20000 <= cp <= 0x2FFFF  # Ext B+ (surrogate-aware in Py 3.3+)
        ):
            cjk += 1
        else:
            other += 1
    # CJK: 1 char ≈ 1 token; 其他: 4 chars ≈ 1 token (向上取整)
    return cjk + max(1, (other + 3) // 4)


def estimate_message_tokens(messages: list[Message]): # -> int:
    total = 0 
    for msg in messages:
        total += estimate_tokens(msg.content)
        if msg.tool_calls:
            total += estimate_tokens((json.dumps(msg.tool_calls, ensure_ascii=False)))
    # 每条消息附加~4 token的元数据开销 (role, 分隔符等)
    total += len(messages) * 4
    return total


# ---------------------------------------------------------------------------
# 4. LLM 客户端
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """LLM 调用错误基类。"""


class RetryExhaustedError(LLMError):
    """重试耗尽错误。"""


class StreamCancelledError(LLMError):
    """流被取消（steering 触发）。"""


class LLMClient:
    """统一 LLM 客户端 —— OpenAI 兼容接口
    
    Parematers
    ----------
    api_key: str
        API Key
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
        model: str = "deepseek-v4-flash",
        max_retries: int = 3
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        # httpx 客户端(惰性创建, 确保在正确的evnt loop中)
        self._client: httpx.AsyncClient | None = None 
        # 最后一次请求的 usage（如果 API 返回）
        self._last_usage: dict[str, int] = {}
    async def _get_client(self): # -> httpx.AsyncClient:
        """获取或创建 httpx 客户端。"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        """关闭 HTTP 客户端。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # 核心 API：流式 chat —— 返回 AsyncIterator[Delta]
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[Delta]:
        """流式对话，每次 yield 一个增量文本 chunk。
        
        同时解析 tool_calss（流式累积），在流结束后附带到
        ``LLMClient._last_tool_calls`` 供调用方读取。
        
        Yields
        ------
        Delta
            text / tool_call / done / error 四种类型。
            调用方自行累积 text 和 tool_calls。
        """
        payload = self._build_payload(messages, tools, temperature, max_tokens)

        for attempt in range(self.max_retries + 1):
            try:
                client = await self._get_client()
                async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
                    # 非 2xx -> 抛出异常触发重试
                    if response.status_code >= 400:
                        body = await response.aread()
                        raise LLMError(
                            f"HTTP {response.status_code}: {body.decode(errors='replace')[:500]}"
                        )

                    async for line in response.aiter_lines():
                        # 检查取消
                        if cancel_event is not None and cancel_event.is_set():
                            yield Delta(kind="error", error="stream_cancelled")
                            raise StreamCancelledError("流被用户中断")

                        delta = self._parse_sse_line(line)
                        if delta is not None:
                            yield delta

                return  # 成功

            except StreamCancelledError:
                raise  # 不重试
            except (httpx.TransportError, httpx.TimeoutException, LLMError) as exc:
                # 网络错误 / 超时 / 服务端错误 → 重试
                if attempt >= self.max_retries:
                    yield Delta(kind="error", error=str(exc))
                    raise RetryExhaustedError(
                        f"重试 {self.max_retries} 次后仍失败: {exc}"
                    ) from exc
                wait = 2 ** attempt
                logger.warning(
                    "LLM 调用失败 (attempt %d/%d)，%ds 后重试: %s",
                    attempt + 1, self.max_retries + 1, wait, exc,
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
    ): # -> Message:
        """非流式对话。返回完整 assistant消息

        Args:
            messages (list[Message]): 消息体
            tools (list[dict[str, Any]] | None, optional): 工具列表. Defaults to None.
            temperature (float, optional): 推理的温度. Defaults to 0.7.
            max_tokens (int, optional): 最大输出token. Defaults to 4096.
        """
        payload = self._build_payload(
            messages, tools, temperature, max_tokens
        )
        payload['stream'] = False
        
        for attempt in range(self.max_retries + 1):
            try:
                client = await self._get_client()
                response = await client.post(
                    "v1/chat/completions",
                    json=payload,
                )
                if response.status_code >= 400:
                    body = response.text 
                    raise LLMError(
                        f"HTTP {response.status_code}: {body[:500]}"
                    )
                data: dict[str, Any] = response.json()
                choice = data['choices'][0]
                msg_data = choice["message"]
                return Message(
                    role='assistant',
                    content=msg_data.get("content"),
                    tool_calls=msg_data.get("tool_calls")
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

    # 摘要功能
    async def summarize(
            self,
            messages: list[Message],
            *,
            max_summary_tokens: int = 512
    ):
        """调用 LLM 自身对历史消息做摘要（用于上下文压缩）。
        使用非流式调用，要求模型输出简洁摘要。

        Args:
            messages (list[Message]): 消息体
            max_summary_tokens (int, optional): 摘要最大token. Defaults to 512.
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
            max_tokens=max_summary_tokens
        )
        return result.content or ""

    # 内部方法
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

    def _parse_sse_line(self, line: str) -> Delta | None:
        """解析单行 SSE 数据
        处理三种情况：
        - 空行 / 注释行 -> 跳过
        - ``data: [DONE]`` -> 流结束标记
        - ``data: {...}`` -> 正常数据, 提取delta

        Args:
            line (str): 原始 SSE 行。
            tool_calls_acc (dict[int, dict[str, Any]]): 累积 tool_calls 的字典，按 index 聚合。

        Returns:
            Delta | None: 增量文本；若为 None 表示无内容输出（如 tool_call delta 或无文本的 chunk）。
        """
        if not line.startswith("data: "):
            return None 
        
        data_str = line[5:].strip()
        # 流结束标记
        if data_str == "[DONE]":
            return Delta(kind="done", finish_reason="stop")
        # 解析 JSON
        try:
            data: dict[str, Any] = json.loads(data_str)
        except json.JSONDecodeError:
            logger.debug(f"SSE JSON 解析失败： {data_str}")
            return None

        choices: list[dict[str, Any]] = data.get("choices", [])
        if not choices:
            return None

        choice = choices[0]
        delta: dict[str, Any] = choice.get("delta", {})
        finish_reason: str | None = choice.get("finish_reason")

        # 1) 文本增量
        content: str | None = delta.get("content")
        if content:
            return Delta(kind="text", text=content)

        # 2) 工具调用增量
        tool_calls: list[dict[str, Any]] = delta.get("tool_calls", [])
        for tc in tool_calls:
            idx: int = tc.get("index", 0)
            func: dict[str, Any] = tc.get("function", {})
            # 如果没有实际内容（空 delta），跳过
            tid = tc.get("id") or None
            tname = func.get("name") or None
            targs = func.get("arguments") or None
            if tid is None and tname is None and targs is None:
                continue
            return Delta(
                kind="tool_call",
                tool_index=idx,
                tool_id_chunk=tid,
                tool_name_chunk=tname,
                tool_args_chunk=targs,
            )

        # 3) 结束标记
        if finish_reason:
            return Delta(kind="done", finish_reason=finish_reason)

        return None

    @property
    def last_usage(self) -> dict[str, int]:
        """最后一次请求的 usage（prompt_tokens / completion_tokens / total_tokens）。"""
        return dict(self._last_usage)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _messages_to_text(messages: list[Message]) -> str:
    """将消息列表转为可读文本（用于摘要 prompt）

    Args:
        messages (list[Message]): 消息列表

    Returns:
        str: 可读文本
    """
    lines: list[str] = []
    for msg in messages:
        role = msg.role
        content = msg.content or ""
        if msg.tool_calls:
            content += "\n[工具调用] " + json.dumps(msg.tool_calls, ensure_ascii=False, default=str)
        if msg.tool_call_id:
            lines.append(f"[tool 结果 id={msg.tool_call_id}] {content}")
        else:
            lines.append(f"[{role}] {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 工具函数：从 Delta 流中累积出完整文本和 tool_calls
# Agent 层的消费工具
# ---------------------------------------------------------------------------


@dataclass
class AccumulatedResponse:
    """从 Delta 流中完整的累积结果。"""
    text: str
    tool_calls: list[dict[str, Any]] | None
    finish_reason: str | None = None  # "stop" | "tool_calls" | "length" | etc.


async def collect_deltas(stream: AsyncIterator[Delta]) -> AccumulatedResponse:
    """消耗 Delta 流，累积出完整文本和 tool_calls。

    这是 agent.py 中使用的便利函数。
    """
    text_parts: list[str] = []
    tool_calls_acc: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None

    async for delta in stream:
        if delta.kind == 'text' and delta.text:
            text_parts.append(delta.text)
        elif delta.kind == "tool_call" and delta.tool_index is not None:
            idx = delta.tool_index
            if idx not in tool_calls_acc: # 初始化
                tool_calls_acc[idx] = {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            acc = tool_calls_acc[idx]
            if delta.tool_id_chunk:
                acc["id"] += delta.tool_id_chunk
            if delta.tool_name_chunk:
                acc["function"]["name"] += delta.tool_name_chunk
            if delta.tool_args_chunk:
                acc["function"]["arguments"] += delta.tool_args_chunk
        elif delta.kind == "done":
            finish_reason = delta.finish_reason
        elif delta.kind == "error":
            raise LLMError(delta.error or "stream error")

    text = "".join(text_parts)
    tool_calls = (
        [tool_calls_acc[i] for i in sorted(tool_calls_acc.keys())]
        if tool_calls_acc
        else None
    )
    return AccumulatedResponse(
        text=text,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
    )
