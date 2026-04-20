# hello_agents/core/conversation.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

@dataclass
class ConversationNode:
    """对话树节点，替代线性 Message 存储"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    message: 'Message' = None  # 复用现有 Message 类
    parent: Optional['ConversationNode'] = None
    children: List['ConversationNode'] = field(default_factory=list)
    branch_id: str = "main"      # 所属分支标识
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def path_to_root(self) -> List['ConversationNode']:
        """获取从根到当前节点的路径（即完整对话上下文）"""
        path = []
        node = self
        while node:
            path.append(node)
            node = node.parent
        return list(reversed(path))
    
    def get_siblings(self) -> List['ConversationNode']:
        """获取兄弟节点（同一父节点的其他分支）"""
        if not self.parent:
            return []
        return [c for c in self.parent.children if c.id != self.id]


class ConversationTree:
    """多分支对话树，管理所有对话历史"""
    
    def __init__(self, root_system_prompt: Optional[str] = None):
        self.root = ConversationNode(
            message=Message(content=root_system_prompt or "", role="system"),
            id="root"
        )
        self.active_node: ConversationNode = self.root  # 当前操作节点
        self.branches: Dict[str, ConversationNode] = {   # 分支头节点
            "main": self.root
        }
        self.current_branch: str = "main"
    
    # ─── 核心操作 ───
    
    def add_message(self, message: Message) -> ConversationNode:
        """在当前激活分支追加消息"""
        new_node = ConversationNode(
            message=message,
            parent=self.active_node,
            branch_id=self.current_branch
        )
        self.active_node.children.append(new_node)
        self.active_node = new_node
        return new_node
    
    def branch_at(self, node_id: str, branch_name: Optional[str] = None) -> str:
        """在指定节点创建新分支"""
        target = self.find_node(node_id)
        if not target:
            raise ValueError(f"Node {node_id} not found")
        
        branch_id = branch_name or f"branch_{len(self.branches)}"
        self.branches[branch_id] = target
        self.current_branch = branch_id
        self.active_node = target  # 新分支从该节点开始
        
        return branch_id
    
    def rewind_to(self, node_id: str) -> List[Message]:
        """回溯到指定节点，返回该路径的上下文"""
        target = self.find_node(node_id)
        if not target:
            raise ValueError(f"Node {node_id} not found")
        
        self.active_node = target
        # 清理该节点之后的子节点（可选：或保留作为废弃分支）
        target.children.clear()
        
        return [n.message for n in target.path_to_root()]
    
    def switch_branch(self, branch_id: str):
        """切换到已有分支"""
        if branch_id not in self.branches:
            raise ValueError(f"Branch {branch_id} not found")
        
        self.current_branch = branch_id
        # 找到该分支的最新节点
        self.active_node = self._get_branch_tip(self.branches[branch_id])
    
    # ─── 查询方法 ───
    
    def get_active_path(self) -> List[Message]:
        """获取当前激活路径的完整消息列表（用于 LLM 上下文）"""
        return [n.message for n in self.active_node.path_to_root()]
    
    def get_all_branches(self) -> Dict[str, List[Message]]:
        """获取所有分支的摘要"""
        result = {}
        for bid, head in self.branches.items():
            tip = self._get_branch_tip(head)
            result[bid] = [n.message for n in tip.path_to_root()]
        return result
    
    def find_node(self, node_id: str) -> Optional[ConversationNode]:
        """BFS 查找节点"""
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if node.id == node_id:
                return node
            queue.extend(node.children)
        return None
    
    def _get_branch_tip(self, head: ConversationNode) -> ConversationNode:
        """获取分支的最新节点"""
        node = head
        while node.children:
            # 优先走当前分支的子节点
            branch_children = [c for c in node.children if c.branch_id == self.current_branch]
            node = branch_children[-1] if branch_children else node.children[-1]
        return node
    
    def visualize(self) -> str:
        """文本可视化对话树"""
        lines = []
        def _walk(node: ConversationNode, prefix: str = "", is_last: bool = True):
            marker = "└── " if is_last else "├── "
            role = node.message.role
            content = node.message.content[:30] + "..." if len(node.message.content) > 30 else node.message.content
            branch_info = f"[{node.branch_id}]" if node.branch_id != "main" else ""
            lines.append(f"{prefix}{marker}{role}{branch_info}: {content}")
            
            children = node.children
            for i, child in enumerate(children):
                is_last_child = (i == len(children) - 1)
                extension = "    " if is_last else "│   "
                _walk(child, prefix + extension, is_last_child)
        
        _walk(self.root)
        return "\n".join(lines)


class ConversationManager:
    """对话管理器：封装树操作，提供业务级接口"""
    
    def __init__(self):
        self.conversations: Dict[str, ConversationTree] = {}  # 多会话支持
        self.active_conversation: Optional[str] = None
    
    def create_conversation(self, conv_id: Optional[str] = None, system_prompt: Optional[str] = None) -> str:
        """创建新对话"""
        cid = conv_id or str(uuid.uuid4())[:8]
        self.conversations[cid] = ConversationTree(system_prompt)
        self.active_conversation = cid
        return cid
    
    def fork_conversation(self, from_node_id: str, new_conv_id: Optional[str] = None) -> str:
        """从指定节点分叉新对话（类似 Git branch）"""
        tree = self._get_active_tree()
        new_id = new_conv_id or str(uuid.uuid4())[:8]
        
        # 创建新分支
        branch_id = tree.branch_at(from_node_id, new_id)
        return branch_id
    
    def get_context_for_llm(self, max_length: int = 4000) -> List[Dict[str, str]]:
        """获取用于 LLM 调用的上下文（自动截断）"""
        tree = self._get_active_tree()
        messages = tree.get_active_path()
        
        # 长度控制：保留系统消息，从后往前截断
        total = sum(len(m.content) for m in messages)
        while total > max_length and len(messages) > 1:
            removed = messages.pop(1)  # 保留 system，移除最早的 user/assistant
            total -= len(removed.content)
        
        return [m.to_dict() for m in messages]
    
    def _get_active_tree(self) -> ConversationTree:
        if not self.active_conversation or self.active_conversation not in self.conversations:
            raise ValueError("No active conversation")
        return self.conversations[self.active_conversation]


