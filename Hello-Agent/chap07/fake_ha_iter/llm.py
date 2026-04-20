# hello_agents/core/llm.py
# HelloAgentsLLM 类（核心层）

from typing import Iterator, Optional


class HelloAgentsLLM:
    """扩展流式调用能力"""
    
    def invoke(self, messages: list, **kwargs) -> str:
        """原有同步调用（保持兼容）"""
        ...
    
    def stream_invoke(self, messages: list, **kwargs) -> Iterator[str]:
        """
        新增：流式调用，实时返回 token 片段
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,  # ← 关键参数
            temperature=self.temperature,
            **kwargs
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def think_stream(self, messages: list, **kwargs) -> Iterator[str]:
        """带思考的流式调用（兼容扩展 thinking 模式）"""
        # 如果支持 extended thinking，先流式输出 thinking tokens
        if hasattr(self, 'thinking') and self.thinking:
            yield from self._stream_with_thinking(messages, **kwargs)
        else:
            yield from self.stream_invoke(messages, **kwargs)