# hello_agents/agents/simple_agent.py

from typing import Iterator


class SimpleAgent(Agent):
    """支持真流式的简单 Agent"""
    
    def run(self, input_text: str, **kwargs) -> str:
        """同步接口"""

        messages = self._build_messages(input_text)
        response = self.llm.invoke(messages, **kwargs)
        self._update_history(input_text, response)
        return response
    
    def run_stream(self, input_text: str, **kwargs) -> Iterator[str]:
        """真流式实现"""
        messages = self._build_messages(input_text)
        
        full_response = ""
        
        # 实时 yield 每个 token
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            yield chunk  # ← 实时返回给用户
        
        # 流结束后保存完整历史
        self._update_history(input_text, full_response)
    
    def _build_messages(self, input_text: str) -> list:
        """构建消息列表（复用逻辑）"""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": input_text})
        return messages
    


class SimpleAgent1(Agent):
    def run(self, input_text: str, branch: Optional[str] = None, **kwargs) -> str:
        # 支持指定分支运行
        if branch:
            self._conversation.switch_branch(branch)
        
        # 构建上下文（自动处理分支历史）
        messages = self._conversation.get_active_path()
        messages.append(Message(input_text, "user"))
        
        # LLM 调用
        response = self.llm.invoke([m.to_dict() for m in messages], **kwargs)
        
        # 保存到当前分支
        self._conversation.add_message(Message(input_text, "user"))
        self._conversation.add_message(Message(response, "assistant"))
        
        return response
    
    def fork_here(self, new_name: str) -> str:
        """从当前节点分叉"""
        current_id = self._conversation.active_node.id
        return self._conversation.branch_at(current_id, new_name)
    
    def rewind(self, turns: int = 1) -> str:
        """回溯 N 轮"""
        node = self._conversation.active_node
        for _ in range(turns):
            if node.parent and node.parent.id != "root":
                node = node.parent
        
        self._conversation.rewind_to(node.id)
        return f"已回溯到: {node.message.content[:50]}..."