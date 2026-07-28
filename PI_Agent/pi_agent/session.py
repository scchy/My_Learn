# Day 5: 会话持久化
# python3
# Create Date: 2026-07-27
# Author: Scc_hy
# Tip:
# 设计原则： 
# 核心机制：内存会话树、JSONL 持久化、原子写入、书签标记

# 关键修正（相比初版计划）：
# - 启动时全量载入内存（避免 O(n) 扫描）
# - 退出时全量写回 + tempfile 原子写入（避免并发损坏）
# ====================================================================================

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pi_agent.llm import Message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 会话节点
# ---------------------------------------------------------------------------


@dataclass 
class SessionNode:
    """会话树种的一个节点"""
    id: str 
    parent_id: str | None = None 
    messages: list[dict[str, Any]] = field(default_factory=list)
    bookmark: str | None = None 
    created_at: str= ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            from datetime import datetime
            self.created_at = datetime.now().isoformat()


# ---------------------------------------------------------------------------
# 会话存储
# ---------------------------------------------------------------------------

class SessionStore:
    """JSONAL 持久化的会话存储
    
    启动时全量载入到内存中的 dict, 操作都是 O(1)
    退出时全量写回 JSONL(通过临时文件原子写入)
    """
    def __init__(self, filepath: str = ""):
        if not filepath:
            filepath = os.path.expanduser("~/.pi-agent/sessions.jsonl")

        self.filepath = Path(filepath)
        self._nodes: dict[str, SessionNode] = {}
        self._loaded = False
        self._dirty = False 

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def load(self) -> None:
        """从 JSONL 文件加载全部节点到内存。"""
        self._nodes.clear()
        if self.filepath.exists():
            with open(self.filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        node = SessionNode(
                            id=data['id'],
                            parent_id=data.get("parent_id"),
                            messages=data.get("messages", []),
                            bookmark=data.get("bookmark"),
                            created_at=data.get("created_at", ""),
                            metadata=data.get("metadata", {}),
                        )
                        self._nodes[node.id] = node
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning("跳过损坏的会话行: %s (行内容: %.80s)", e, line[:80])
        self._loaded = True
        self._dirty = False
    
    def save(self) -> None:
        """
        将内存中的全部节点原子写回 JSONL（仅在 dirty 时写入）。
        """
        if not self._dirty:
            return
        if not self._loaded:
            raise RuntimeError(
                "SessionStore 尚未加载数据，禁止 save() 以免空写覆盖现有文件。"
                "请先调用 load()。"
            )
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 写到临时文件，然后原子 rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.filepath.parent),
            prefix='.sessions_',
            suffix='.jsonl',
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for node in self._nodes.values():
                    line = json.dumps({
                        "id": node.id,
                        "parent_id": node.parent_id,
                        "messages": node.messages,
                        "bookmark": node.bookmark,
                        "created_at": node.created_at,
                        "metadata": node.metadata,
                    }, ensure_ascii=False)
                    f.write(line + "\n")
            os.replace(tmp_path, str(self.filepath))
            self._dirty = False
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_node(
        self,
        messages: list[Message],
        *,
        parent_id: str | None = None, 
        bookmark: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionNode:
        """创建新会话节点。"""
        self.ensure_loaded()
        node = SessionNode(
            id=_new_id(),
            parent_id=parent_id,
            messages=[_message_to_dict(m) for m in messages],
            bookmark=bookmark,
            metadata=metadata or {},
        )
        self._nodes[node.id] = node
        self._dirty = True
        return node

    def update_node(self, node_id: str, messages: list[Message]) -> SessionNode | None:
        """ 更新节点的消息列表 """
        self.ensure_loaded()
        node = self._nodes.get(node_id)
        if node is None:
            return None 
        node.messages = [_message_to_dict(m) for m in messages]
        self._dirty = True
        return node 
    
    def bookmark_node(self, node_id: str, name: str) -> bool: 
        """ 
        为节点打书签（允许多个节点使用同一书签名）。
        """
        self.ensure_loaded()
        node = self._nodes.get(node_id)
        if node is None:
            return False
        node.bookmark = name
        self._dirty = True
        return True

    def get_node(self, node_id: str) -> SessionNode | None:
        self.ensure_loaded()
        return self._nodes.get(node_id)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get_branch(self, node_id: str) -> list[SessionNode]:
        """ 获取从根到指定节点的完整路径 """
        self.ensure_loaded()
        path: list[SessionNode] = []
        current = self._nodes.get(node_id)
        while current is not None:
            path.append(current)
            current = self._nodes.get(current.parent_id) if current.parent_id else None
        path.reverse()
        return path

    def get_children(self, parent_id: str) -> list[SessionNode]:
        """ 获取某节点的直接子节点 """
        self.ensure_loaded()
        return [n for n in self._nodes.values() if n.parent_id == parent_id]
    
    def get_by_bookmark(self, name: str) -> SessionNode | None:
        """ 按书签名查找节点 """
        self.ensure_loaded()
        for node in self._nodes.values():
            if node.bookmark == name:
                return node
        return None 

    def list_bookmarks(self) -> list[tuple[str, str]]:
        """列出所有书签 (名称, 节点ID)。"""
        self.ensure_loaded()
        return [(n.bookmark, n.id) for n in self._nodes.values() if n.bookmark]

    def list_roots(self) -> list[SessionNode]:
        """ 列出所有根节点（parent_id 为 None） """
        self.ensure_loaded()
        return sorted(
            [n for n in self._nodes.values() if n.parent_id is None],
            key=lambda n: n.created_at
        )

    # ------------------------------------------------------------------
    # 消息序列化
    # ------------------------------------------------------------------

    def messages_from_node(self, node_id: str) -> list[Message] | None:
        """ 从节点恢复 Message 列表 """
        self.ensure_loaded()
        node = self._nodes.get(node_id)
        if node is None:
            return None 
        return [_dict_to_message(m) for m in node.messages]


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _message_to_dict(msg: Message) -> dict[str, Any]:
    d: dict[str, Any] = {'role': msg.role}
    if msg.content is not None:
        d["content"] = msg.content
    if msg.tool_call_id is not None:
        d["tool_call_id"] = msg.tool_call_id
    if msg.tool_calls is not None:
        d["tool_calls"] = msg.tool_calls
    return d


def _dict_to_message(d: dict[str, Any]) -> Message:
    return Message(
        role=d["role"],
        content=d.get("content"),
        tool_call_id=d.get("tool_call_id"),
        tool_calls=d.get("tool_calls"),
    )

