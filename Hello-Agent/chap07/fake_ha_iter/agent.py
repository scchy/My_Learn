# hello_agents/core/agent.py

from abc import abstractmethod, ABC
from typing import Iterator, Optional


class ConversationTree:
    def to_message_list(self) -> List[Message]:
        """兼容现有接口：返回线性 Message 列表"""
        return self.get_active_path()
    
    def import_history(self, messages: List[Message]):
        """从现有历史导入：线性 → 树"""
        for msg in messages:
            self.add_message(msg)


# Agent 基类兼容
class Agent(ABC):
    """扩展流式接口"""
    def __init__(self, ...):
        # 原有：self._history: list[Message] = []
        # 新增：默认使用 ConversationTree，但保留兼容
        self._conversation = ConversationTree(system_prompt)
        self._legacy_history: List[Message] = []  # 兼容旧代码

    def get_history(self) -> List[Message]:
        """兼容现有接口"""
        if hasattr(self, '_conversation'):
            return self._conversation.get_active_path()
        return self._legacy_history
    
    def add_message(self, message: Message):
        """统一入口：自动路由到树或线性历史"""
        if hasattr(self, '_conversation'):
            self._conversation.add_message(message)
        else:
            self._legacy_history.append(message)

    # 原有同步接口
    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        pass
    
    # 新增：流式接口
    def run_stream(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        默认实现：基于 run 的模拟流式（子类可重写为真流式）
        为了兼容性，基类提供默认实现，避免破坏现有子类
        """
        result = self.run(input_text, **kwargs)
        # 模拟流式：按句子/词切分 yield
        for chunk in self._simulate_stream(result):
            yield chunk
    
    def _simulate_stream(self, text: str, chunk_size: int = 4) -> Iterator[str]:
        """模拟流式：将完整文本切分后逐段输出"""
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]
    
    # 可选：真流式抽象方法（强制子类实现）
    @abstractmethod
    def _execute_stream(self, input_text: str, **kwargs) -> Iterator[str]:
        """子类必须实现的真流式逻辑"""
        pass