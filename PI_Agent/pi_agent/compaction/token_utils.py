"""Token 估算工具。

复用 ``pi_agent.llm`` 中更精确的 CJK-aware 估算器，同时提供 Pi-style
的上下文级估算（以 API usage 为锚点，后续消息用启发式补全）。
"""

from __future__ import annotations

import json
from typing import Any

from pi_agent.llm import Message, estimate_tokens


def estimate_message_tokens(message: Message) -> int:
    """单条 Message 的 token 估算。"""
    total = estimate_tokens(message.content)
    if message.tool_calls:
        total += estimate_tokens(json.dumps(message.tool_calls, ensure_ascii=False))
    # 每条消息附加元数据开销（role、tool_call_id 等）
    total += 4
    return total


def estimate_messages_tokens(messages: list[Message]) -> int:
    """消息列表总 token 估算。"""
    return sum(estimate_message_tokens(m) for m in messages)


def calculate_context_tokens(usage: dict[str, Any]) -> int:
    """从 API 返回的 usage 计算总 token 数。"""
    if not usage:
        return 0
    if "total_tokens" in usage:
        return int(usage["total_tokens"])
    return (
        int(usage.get("prompt_tokens", 0))
        + int(usage.get("completion_tokens", 0))
        + int(usage.get("cache_read_tokens", 0))
        + int(usage.get("cache_write_tokens", 0))
    )


def estimate_context_tokens(
    messages: list[Message],
    api_usage_tokens: int | None = None,
) -> int:
    """上下文级 token 估算。

    - 如果提供了 api_usage_tokens，视为最后一条 assistant 之前的总 token；
      后续消息用启发式补全。
    - 否则全部用启发式估算。
    """
    if api_usage_tokens is not None:
        # 简化处理：api_usage_tokens 表示已知前缀的 token 数
        # 这里假设它覆盖除最后一条 assistant 输出外的所有消息
        # 实际 Pi 的实现会精确找到最后一条非中断 assistant 的 usage 位置
        trailing = messages[-1:] if messages else []
        return api_usage_tokens + sum(estimate_message_tokens(m) for m in trailing)
    return estimate_messages_tokens(messages)
