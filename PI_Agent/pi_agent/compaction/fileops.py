"""从消息中提取文件操作记录（read / modified）。"""

from __future__ import annotations

import json
from typing import Any

from pi_agent.llm import Message


def _extract_paths_from_tool_call(tool_call: dict[str, Any]) -> tuple[set[str], set[str]]:
    """解析单个 tool_call，返回 (read_files, modified_files)。"""
    reads: set[str] = set()
    modified: set[str] = set()

    name = tool_call.get("function", {}).get("name", "")
    try:
        args = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
    except Exception:
        args = {}

    path = args.get("path", "") or args.get("file_path", "")
    if not path:
        return reads, modified

    if name in {"read", "grep", "find", "ls"}:
        reads.add(path)
    elif name in {"write", "edit", "bash"}:
        modified.add(path)

    return reads, modified


def extract_file_operations(messages: list[Message]) -> dict[str, list[str]]:
    """从 assistant tool_calls 中提取读取和修改过的文件列表。"""
    reads: set[str] = set()
    modified: set[str] = set()

    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                r, m = _extract_paths_from_tool_call(tc)
                reads.update(r)
                modified.update(m)

    return {
        "read_files": sorted(reads),
        "modified_files": sorted(modified),
    }


def format_file_operations_section(read_files: list[str], modified_files: list[str]) -> str:
    """把文件操作格式化为 markdown 区块。"""
    lines: list[str] = []
    if read_files:
        lines.append("## Files Read")
        for f in read_files:
            lines.append(f"- {f}")
    if modified_files:
        if lines:
            lines.append("")
        lines.append("## Files Modified")
        for f in modified_files:
            lines.append(f"- {f}")
    return "\n".join(lines)
