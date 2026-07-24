"""摘要生成：历史摘要 + Turn 前缀摘要 + 增量更新。"""

from __future__ import annotations

from pi_agent.llm import LLMClient, Message


SUMMARIZATION_PROMPT = """You are summarizing a conversation between a user and a coding assistant.
Please produce a structured summary in the following format:

## Goal
[The user's goal or request]

## Constraints & Preferences
- [Any constraints or preferences mentioned]

## Progress
### Done
- [x] [Completed tasks]

### In Progress
- [ ] [Ongoing tasks]

### Blocked
- [Blocked items]

## Key Decisions
- **[Decision]**: [Reasoning]

## Next Steps
1. [Next steps]

## Critical Context
- [Key data, file paths, error messages]

Be concise but preserve all facts needed to continue the task."""


UPDATE_SUMMARIZATION_PROMPT = """You are updating an existing conversation summary with new messages.

Rules:
- PRESERVE all existing information
- ADD new progress, decisions, and context
- UPDATE Progress: move completed items from "In Progress" to "Done"
- UPDATE "Next Steps" based on current state
- PRESERVE exact file paths, function names, and error messages

Existing summary:
{previous_summary}

Please produce the updated structured summary in the same format."""


TURN_PREFIX_SUMMARIZATION_PROMPT = """You are summarizing the prefix of an in-progress turn that was cut off during context compaction.

## Original Request
[The user's request in this turn]

## Early Progress
- [Key decisions and work from the prefix]

## Context for Suffix
- [Information needed to understand the remaining suffix messages]

Be concise. The suffix messages will remain in context; this summary just helps the model understand what happened before them."""


async def generate_summary(
    client: LLMClient,
    messages: list[Message],
    previous_summary: str | None = None,
    max_tokens: int = 4096,
) -> str:
    """生成历史摘要（首次或增量更新）。"""
    if previous_summary:
        prompt = UPDATE_SUMMARIZATION_PROMPT.format(previous_summary=previous_summary)
    else:
        prompt = SUMMARIZATION_PROMPT

    # 把待摘要消息序列化为可读文本
    conversation = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            tool_names = [tc.get("function", {}).get("name", "tool") for tc in m.tool_calls]
            conversation.append(f"{m.role}: {m.content or ''} [tools: {', '.join(tool_names)}]")
        else:
            conversation.append(f"{m.role}: {m.content or ''}")

    content = f"{prompt}\n\n<conversation>\n" + "\n".join(conversation) + "\n</conversation>"
    return await client.summarize(
        [Message(role="user", content=content)],
        max_summary_tokens=max_tokens,
    )


async def generate_turn_prefix_summary(
    client: LLMClient,
    messages: list[Message],
    max_tokens: int = 2048,
) -> str:
    """生成当前 Turn 前缀摘要。"""
    conversation = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            tool_names = [tc.get("function", {}).get("name", "tool") for tc in m.tool_calls]
            conversation.append(f"{m.role}: {m.content or ''} [tools: {', '.join(tool_names)}]")
        else:
            conversation.append(f"{m.role}: {m.content or ''}")

    content = (
        f"{TURN_PREFIX_SUMMARIZATION_PROMPT}\n\n<turn_prefix>\n"
        + "\n".join(conversation)
        + "\n</turn_prefix>"
    )
    return await client.summarize(
        [Message(role="user", content=content)],
        max_summary_tokens=max_tokens,
    )


def merge_summaries(history_summary: str, turn_prefix_summary: str) -> str:
    """合并历史摘要与 Turn 前缀摘要。"""
    return f"""{history_summary}

---

**Turn Context (split turn):**

{turn_prefix_summary}"""


def create_compaction_summary_message(summary: str, file_ops: dict[str, list[str]]) -> Message:
    """把摘要和文件操作合并成一条 compaction summary 消息。

    使用 user 角色并加上前缀标记，便于后续识别这是压缩摘要。
    """
    from pi_agent.compaction.fileops import format_file_operations_section

    sections = ["[上下文摘要]", summary]
    file_section = format_file_operations_section(
        file_ops.get("read_files", []),
        file_ops.get("modified_files", []),
    )
    if file_section:
        sections.append(file_section)

    return Message(role="user", content="\n\n".join(sections))
