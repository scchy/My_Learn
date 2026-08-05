# Pi Agent 上下文压缩（Compaction）辅助模块
# 参考：earendil-works/pi/packages/coding-agent/src/core/compaction/compaction.ts
# ==========================================================================================
#
# 模块结构：
# - token_utils : Token 估算（CJK-aware 启发式 + API usage 锚点）
# - cutpoint    : 切割点查找（Turn 边界保护 + tool-call/tool-result 配对）
# - summary     : 摘要生成（全量 SUMMARIZATION_PROMPT + 增量 UPDATE_SUMMARIZATION_PROMPT）
# - fileops     : 文件操作追踪（read_files / modified_files）
#
# 入口：
#   from pi_agent.context import ContextCompactor, CompactionConfig
#   # ContextCompactor.compress_if_needed() 串联全部子模块
# ==========================================================================================

from pi_agent.compaction.token_utils import (
    calculate_context_tokens,
    estimate_context_tokens,
    estimate_message_tokens,
    estimate_messages_tokens,
)
from pi_agent.compaction.cutpoint import (
    find_cut_point,
    find_turn_start_index,
    is_valid_cut_point,
)
from pi_agent.compaction.summary import (
    SUMMARIZATION_PROMPT,
    TURN_PREFIX_SUMMARIZATION_PROMPT,
    UPDATE_SUMMARIZATION_PROMPT,
    create_compaction_summary_message,
    generate_summary,
    generate_turn_prefix_summary,
    merge_summaries,
)
from pi_agent.compaction.fileops import (
    extract_file_operations,
    format_file_operations_section,
)

__all__ = [
    # token_utils
    "calculate_context_tokens",
    "estimate_context_tokens",
    "estimate_message_tokens",
    "estimate_messages_tokens",
    # cutpoint
    "find_cut_point",
    "find_turn_start_index",
    "is_valid_cut_point",
    # summary
    "SUMMARIZATION_PROMPT",
    "TURN_PREFIX_SUMMARIZATION_PROMPT",
    "UPDATE_SUMMARIZATION_PROMPT",
    "create_compaction_summary_message",
    "generate_summary",
    "generate_turn_prefix_summary",
    "merge_summaries",
    # fileops
    "extract_file_operations",
    "format_file_operations_section",
]
