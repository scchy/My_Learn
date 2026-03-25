# python3
# Author: Scc_hy
# Create Date: 2026-03022
# ========================================================================
import re
import os 
import sys 
import json
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from os.path import dirname 
CUR_DIR = dirname(__file__)
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)
from HelloAgentLLM import HelloAgentsLLM, ToolExecutor, search, calculate
from error_recovery import ErrorRecoveryStrategy, ErrorType

# ReAct 提示词模板
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""


class ActionType(Enum):
    TOOL_CALL = "tool_call"
    FINISH = "finish"
    THINK_ONLY = "think_only"  # 仅思考，不行动


@dataclass
class ParsedAction:
    action_type: ActionType
    tool_name: Optional[str]
    tool_input: Optional[Dict] = None
    final_answer: Optional[str] = None
    reasoning: Optional[str] = None


class ReActAgent:
    def __init__(
            self,
            llm_client: HelloAgentsLLM,
            tool_executor: ToolExecutor,
            max_steps: int = 5,
            use_function_calling: bool = True,
            enable_error_recovery: bool = True
    ):
        self.use_function_calling = use_function_calling
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []
        self.enable_error_recovery = enable_error_recovery
        # 错误恢复策略
        self.error_recovery = ErrorRecoveryStrategy(max_retries=3) if enable_error_recovery else None
        # 构建工具 schema
        self.tools_schema = self.tool_executor.build_tools_schema()
        print(f'{self.tools_schema=}')

    def run(self, question: str) -> Optional[str]:
        if self.use_function_calling:
            return self.run_function_call(question)
        return self.run_legacy(question)
        
    def run_function_call(self, question: str) -> Optional[str]:
        """主循环（集成错误恢复）"""
        self.history = []
        if self.error_recovery:
            self.error_recovery.reset()
        
        current_step = 0
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": "你是一个能够调用工具解决问题的智能助手。请逐步思考，必要时调用工具。"
            },
            {
                "role": "user",
                "content": question
            }
        ]
        
        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")
            
            # 获取过滤后的工具 schema（排除已失效的工具）
            tools_schema = self._get_filtered_tools_schema()
            
            response = self.llm_client.think_with_fc(
                messages,
                tools_schema
            )
            if not response:
                print("错误:LLM未能返回有效响应。")
                break
            
            # 提取思考内容（content 字段就是 reasoning）
            reasoning = response.content or "（模型直接调用工具）"
            print(f"🤔 思考: {reasoning[:200]}...")
            
            # 检查是否有工具调用
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    action = self._parse_function_call(tool_call)
                    if action.action_type == ActionType.FINISH:
                        print(f"🎉 最终答案: {action.final_answer}")
                        return action.final_answer
                    elif action.action_type == ActionType.TOOL_CALL \
                        and  action.tool_name is not None \
                        and action.tool_input is not None:
                        print(f"🎬 调用工具: {action.tool_name}({action.tool_input})")
                        
                        # 执行工具（带错误恢复）
                        observation = self._execute_tool_with_recovery(
                            action.tool_name,
                            action.tool_input,
                            messages
                        )
                        
                        print(f"👀 观察: {str(observation)[:200]}...")
                        # 构建工具响应消息（OpenAI 格式）
                        messages.append({
                            "role": "assistant",
                            "content": reasoning,
                            "tool_calls": [{
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": action.tool_name,
                                    "arguments": json.dumps(action.tool_input, ensure_ascii=False)
                                }
                            }]
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(observation)
                        })
                        
                        # 如果观察结果包含错误恢复提示，也添加到消息中
                        if isinstance(observation, str) and observation.startswith("[RECOVERY]"):
                            messages.append({
                                "role": "system",
                                "content": f"提示: {observation}"
                            })
            else:
                # 没有工具调用，直接回答
                print(f"没有工具调用，直接回答")
                print(f"🎉 直接回答: {reasoning}")
                return reasoning
        
        print("⚠️ 达到最大步数限制")
        
        # 输出错误恢复统计
        if self.error_recovery:
            summary = self.error_recovery.get_error_summary()
            if summary["total_errors"] > 0:
                print(f"\n📊 错误恢复统计: {summary}")
        
        return None
    
    def _get_filtered_tools_schema(self) -> List[Dict]:
        """获取过滤后的工具 schema（排除已失效的工具）"""
        if not self.enable_error_recovery or not self.error_recovery:
            return self.tools_schema
        
        excluded_tools = set(self.error_recovery.get_excluded_tools())
        if not excluded_tools:
            return self.tools_schema
        
        filtered = []
        for tool in self.tools_schema:
            tool_name = tool.get("function", {}).get("name", "")
            if tool_name not in excluded_tools:
                filtered.append(tool)
        
        return filtered if filtered else self.tools_schema  # 至少保留原始 schema
    
    def _execute_tool_with_recovery(self, tool_name: str, tool_input: Dict, messages: List[Dict]) -> str:
        """带错误恢复的工具执行"""
        try:
            tool_func = self.tool_executor.getTool(tool_name)
            if not tool_func:
                # 工具不存在，触发错误恢复
                error = Exception(f"Tool not found: {tool_name}")
                if self.enable_error_recovery and self.error_recovery:
                    available_tools = list(self.tool_executor.tools.keys())
                    action = self.error_recovery.on_tool_error(error, tool_name, messages, available_tools)
                    
                    # 记录到历史
                    self.history.append({
                        "tool": tool_name,
                        "error": str(error),
                        "recovery_action": action.action
                    })
                    
                    result = f"[RECOVERY] {action.feedback_message}"
                    if action.suggested_alternative:
                        result += f" 建议尝试: {action.suggested_alternative}"
                    return result
                else:
                    return f"错误: 未找到工具 '{tool_name}'"
            
            # 执行工具
            observation = tool_func(**tool_input)
            
            # 成功后重置重试计数
            if self.error_recovery:
                self.error_recovery.retry_count = 0
            
            return observation
            
        except Exception as e:
            # 执行出错，触发错误恢复
            if self.enable_error_recovery and self.error_recovery:
                available_tools = list(self.tool_executor.tools.keys())
                action = self.error_recovery.on_tool_error(e, tool_name, messages, available_tools)
                
                # 记录到历史
                self.history.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "error": str(e),
                    "recovery_action": action.action
                })
                
                # 执行等待（如果需要）
                if action.wait_time > 0:
                    import time
                    time.sleep(action.wait_time)
                
                result = f"[RECOVERY] {action.feedback_message}"
                if action.suggested_alternative:
                    result += f" 建议尝试: {action.suggested_alternative}"
                return result
            else:
                return f"错误: 工具执行失败 - {str(e)}"

    def _parse_function_call(self, tool_call) -> ParsedAction:
        """解析 function call 结果"""
        name = tool_call.function.name
        import json
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            arguments = {"query": tool_call.function.arguments}
        if name == "Finish":
            return ParsedAction(
                action_type=ActionType.FINISH,
                tool_name=None,
                final_answer=arguments.get("answer"),
                reasoning=arguments.get("reasoning")
            )
        else:
            return ParsedAction(
                action_type=ActionType.TOOL_CALL,
                tool_name=name,
                tool_input=arguments
            )

    def run_legacy(self, question: str):
        """兼容旧版：正则解析（带改进)
        运行ReAct智能体来回答一个问题。
        格式化提示词 -> 调用LLM -> 执行动作 -> 整合结果
        """
        # 改进的模板：更严格的格式要求
        IMPROVED_PROMPT_TEMPLATE = """你是一个严格遵循格式的智能助手。

可用工具:
{tools}

【强制输出格式】
你必须且只能输出以下两个字段，不要有任何其他内容：

Thought: <你的思考过程，单行或多行均可，但不能包含"Action:"字样>
Action: <只能是以下三种格式之一>
- 工具调用: ToolName{{"query": "具体输入"}}
- 完成回答: Finish{{"answer": "最终答案", "reasoning": "得出结论的过程"}}

【示例】
Thought: 用户询问天气，我需要搜索
Action: Search{{"query": "北京今天天气"}}

现在解决问题:
Question: {question}
History: {history}
"""
        tools_desc = self.tool_executor.getAvailableTools()
        self.history = [] # 每次运行时重置历史记录
        current_step = 0
        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            # 1. 格式化提示词'
            prompt = IMPROVED_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=self._format_history() # 循环的全部上下文
            )
            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(
                messages=messages
            )

            if not response_text:
                print("错误:LLM未能返回有效响应。")
                break

            # LLM 返回的是纯文本，我们需要从中精确地提取出Thought和Action。
            # 改进的解析：多层容错
            parsed = self._robust_parse(response_text)
            if parsed.action_type == ActionType.FINISH:
                print(f"🎉 最终答案: {parsed.final_answer}")
                return parsed.final_answer

            elif parsed.action_type == ActionType.TOOL_CALL \
                and  parsed.tool_name is not None \
                and parsed.tool_input is not None:
                print(f"🎬 行动: {parsed.tool_name}({parsed.tool_input})")
                
                tool_func = self.tool_executor.getTool(parsed.tool_name)
                if tool_func:
                    query = parsed.tool_input.get("query", "")
                    observation = tool_func(query)
                else:
                    observation = f"错误: 未找到工具 '{parsed.tool_name}'"
                
                print(f"👀 观察: {str(observation)[:200]}...")
                
                self.history.append({
                    "thought": parsed.reasoning,
                    "action": f"{parsed.tool_name}({parsed.tool_input})",
                    "observation": observation
                })
            else:
                print(f"⚠️ 解析失败，重试...")
                continue
        
        return None
    
    def _format_history(self) -> str:
        """格式化历史记录"""
        if not self.history:
            return "（无历史记录）"
        
        lines = []
        for h in self.history:
            lines.append(f"Thought: {h.get('thought', '')}")
            lines.append(f"Action: {h.get('action', '')}")
            lines.append(f"Observation: {h.get('observation', '')}")
            lines.append("---")
        return "\n".join(lines)
    
    def _robust_parse(self, text: str) -> ParsedAction:
        """多层容错的正则解析"""
        # 清理文本
        text = text.strip()
        # 尝试 JSON 格式（新）
        try:
            import json
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if "answer" in data:
                    return ParsedAction(
                        action_type=ActionType.FINISH,
                        tool_name=None,
                        final_answer=data.get("answer"),
                        reasoning=data.get("reasoning", "")
                    )
                elif "query" in data:
                    # 推断工具名
                    tool_name = "Search"  # 默认，或从上下文推断
                    return ParsedAction(
                        action_type=ActionType.TOOL_CALL,
                        tool_name=tool_name,
                        tool_input=data
                    )
        except (json.JSONDecodeError, AttributeError):
            pass
        # 回退到严格正则（旧格式兼容）
        # 使用更严格的边界：Thought 必须以 Action: 结束
        thought_match = re.search(
            r"Thought:\s*(.*?)(?=\n\s*Action:\s*)",
            text,
            re.DOTALL | re.IGNORECASE
        )
        
        action_match = re.search(
            r"Action:\s*(\w+)\s*\{(.*)\}",
            text,
            re.DOTALL | re.IGNORECASE
        )
        
        reasoning = thought_match.group(1).strip() if thought_match else ""

        if action_match:
            tool_name = action_match.group(1)
            try:
                import json
                tool_input = json.loads("{" + action_match.group(2) + "}")
            except json.JSONDecodeError:
                tool_input = {"query": action_match.group(2)}
            
            if tool_name.lower() == "finish":
                return ParsedAction(
                    action_type=ActionType.FINISH,
                    tool_name=None,
                    final_answer=tool_input.get("answer", ""),
                    reasoning=tool_input.get("reasoning", reasoning)
                )
            else:
                return ParsedAction(
                    action_type=ActionType.TOOL_CALL,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    reasoning=reasoning
                )
        # 完全失败
        return ParsedAction(action_type=ActionType.THINK_ONLY, tool_name=None, reasoning=text)

    def _parse_output(self, text: str):
        """基于prompt的解析
        Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
        Action: 你决定采取的行动，必须是以下格式之一: ....
        """
        # Thought: 匹配Action 或文本末尾
        thought_match = re.search(
            r"Thought:\s*(.*?)(?=\nAction:|$)", 
            text, 
            re.DOTALL
        )
        # Action: 匹配到文本末尾
        action_match = re.search(
            r"Action:\s*(.*?)$",
            text, 
            re.DOTALL
        )
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。
        - `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
        """
        match = re.match(
            r"(\w+)\[(.*)\]",  # ` 阻断了所以直接\w+就能匹配全部tool名称
            action_text, 
            re.DOTALL
        )
        if match:
            return match.group(1), match.group(2)
        return None, None


if __name__ == '__main__':
    tool_executor = ToolExecutor()
    tool_executor.registerTool(
        "Search", 
        "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。", 
        search
    )
    # print("=" * 50)
    # print("方案1: Function Calling 模式")
    # print("=" * 50)
    # rec_agent = ReActAgent(
    #     HelloAgentsLLM(),
    #     tool_executor,
    #     use_function_calling=True
    # )
    # res_ = rec_agent.run('英伟达最新的GPU型号是什么')

    # print("\n" + "=" * 50)
    # print("方案2: 改进的正则解析模式")
    # print("=" * 50)
    # rec_agent = ReActAgent(
    #     HelloAgentsLLM(),
    #     tool_executor,
    #     use_function_calling=False
    # )
    # res_ = rec_agent.run('英伟达最新的GPU型号是什么')

    tool_executor.registerTool(
        "Calculate",
        "安全的数学计算器。当你需要进行数学计算时，应使用此工具。",
        calculate
    )
    rec_agent = ReActAgent(
        HelloAgentsLLM(),
        tool_executor,
        use_function_calling=True
    )
    res_ = rec_agent.run('帮忙计算(123 + 456) * 789 / 12') 


