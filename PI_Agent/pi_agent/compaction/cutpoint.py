"""切割点查找：保护 Turn 边界与 tool-call/tool-result 配对。"""

from __future__ import annotations

from pi_agent.llm import Message
from pi_agent.compaction.token_utils import estimate_message_tokens


def _is_compaction_summary(message: Message) -> bool:
    """通过内容前缀判断是否为上一次的压缩摘要消息。"""
    return (message.content or "").startswith("[上下文摘要]")


def is_valid_cut_point(messages: list[Message], index: int) -> bool:
    """判断某条消息是否可以作为合法切割点。

    合法：
    - user（Turn 起点）
    - 带有 tool_calls 的 assistant（tool 调用单元的起点）
    - 普通文本 assistant，且前面不是 tool（即不是 tool 单元的 final response）

    非法：
    - toolResult（单独出现会形成孤儿）
    - 前面是 tool 的 assistant final response（会切断 tool 单元）
    - system 消息通常在最开头，不作为压缩切割点
    """
    msg = messages[index]
    if msg.role == "tool":
        return False
    if msg.role == "system":
        return False
    if msg.role == "user":
        return True

    # role == "assistant"
    if msg.tool_calls:
        # tool 调用单元的起点，可以作为切割点（后续 tool results 会跟随保留）
        return True

    # 普通 assistant 回复：如果它前面紧挨着 tool，说明是 tool 单元的 final response，
    # 不能单独切，否则会丢掉前面的 tool 结果。
    if index > 0 and messages[index - 1].role == "tool":
        return False

    return True


def find_turn_start_index(messages: list[Message], cut_index: int) -> int:
    """从切割点向前找当前 Turn 的起始位置。

    Turn 起点定义为最近的 user 消息（Pi 还包括 bashExecution / custom_message / branch_summary）。
    如果找不到，返回 0。
    """
    for i in range(cut_index, -1, -1):
        if messages[i].role == "user":
            return i
    return 0


def find_cut_point(messages: list[Message], keep_recent_tokens: int) -> tuple[int, bool]:
    """从最新消息倒推，找到合法切割点。

    Returns
    -------
    cut_index : int
        切割点索引（该索引及之后保留，之前需要压缩）。
        若无需压缩，返回 0。
    is_split_turn : bool
        切割点是否落在 Turn 中间（非 user 起始点）。
    """
    accumulated = 0
    cut_index = 0
    for i in range(len(messages) - 1, -1, -1):
        accumulated += estimate_message_tokens(messages[i])
        if accumulated >= keep_recent_tokens:
            # 从 i 开始向前找第一个合法切割点
            for j in range(i, len(messages)):
                if is_valid_cut_point(messages, j):
                    cut_index = j
                    break
            else:
                # 找不到更靠后的合法点，只能切在 i（如果 i 合法）
                if is_valid_cut_point(messages, i):
                    cut_index = i
                else:
                    # i 不合法，继续向后到 len(messages)，即不压缩
                    cut_index = len(messages)
            break

    # 如果切在 0 或无需切，直接返回
    if cut_index <= 0:
        return 0, False

    # 判断是否为 split turn：切割点不是当前 Turn 的起点
    turn_start = find_turn_start_index(messages, cut_index)
    is_split_turn = cut_index != turn_start

    return cut_index, is_split_turn
