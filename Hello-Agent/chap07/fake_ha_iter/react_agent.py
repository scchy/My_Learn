# hello_agents/agents/react_agent.py

from typing import Iterator

class ReActAgent(Agent):
    """ReAct 流式：思考过程实时可见"""
    
    def run_stream(self, input_text: str, **kwargs) -> Iterator[str]:
        """流式 ReAct：实时展示 Thought/Action/Observation"""
        self.current_history = []
        
        for step in range(self.max_steps):
            # 1. 流式生成 Thought
            yield f"\n🤔 [思考 {step+1}] "
            thought = ""
            for chunk in self._stream_thought(input_text, **kwargs):
                thought += chunk
                yield chunk
            
            # 2. 流式生成 Action
            yield f"\n⚡ [行动] "
            action = ""
            for chunk in self._stream_action(thought, **kwargs):
                action += chunk
                yield chunk
            
            # 3. 执行工具（非流式，但结果实时展示）
            yield f"\n🔧 [执行] "
            observation = self._execute_action(action)
            yield f"{observation}\n"
            
            if self._is_finish(action):
                yield f"\n✅ [完成] "
                answer = self._extract_answer(action)
                self._update_history(input_text, answer)
                yield answer
                return
    
    def _stream_thought(self, context: str, **kwargs) -> Iterator[str]:
        """流式生成思考过程"""
        prompt = self._build_thought_prompt(context)
        messages = [{"role": "user", "content": prompt}]
        return self.llm.stream_invoke(messages, **kwargs)