# Day 3: Agent Loop 核心
# python3
# Create Date: 2026-07-22
# Author: Scc_hy
# Tip:
# 核心机制：ReAct 循环、Delta 流式处理、并行工具执行、steering 中断

# 关键修正（相比初版计划）：
# - 使用 collect_deltas() 消除 _last_tool_calls 副作用
# - asyncio.gather(return_exceptions=True) 保证全部工具执行完毕
# - 工具错误以 isError 语义反馈给 LLM（而非崩溃）
# - cancel_event 集成到 client.chat() 实现可控中断
# - rich.Live 单面板 + 工具结果延迟打印（避免嵌套冲突）
# ==========================================================================================

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from pi_agent.tools import ToolRegistry, ToolResult, create_default_registry
from pi_agent.context import CompactionConfig, ContextCompactor
from pi_agent.llm import (
    LLMClient,
    LLMError,
    Message,
    StreamCancelledError,
    collect_deltas,
    estimate_messages_tokens,
    AccumulatedResponse
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent 配置
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Agent 运行配置"""
    model: str = 'deepseek-v4-flash'
    max_turns: int = 50
    context_limit: int = 128000
    temperature: float = 0.7
    system_prompt: str = field(
        default_factory=lambda: (
        "你是一个智能编程助手，可以读取、编辑和搜索文件，以及执行 shell 命令。\n\n"
        "## 核心规则\n"
        "1. 使用工具前先思考：我真的需要这个信息吗？\n"
        "2. 每次只做一个逻辑操作，不要在一次响应中混合多个独立任务\n"
        "3. 读取文件时用 offset/limit 分页，不要一次读取整个大文件\n"
        "4. 编辑文件时 old_text 必须精确唯一匹配\n"
        "5. 执行 shell 命令前确认其影响范围\n"
        "6. 如果工具返回错误，分析原因并尝试修正\n\n"
        "## 响应格式\n"
        "- 用中文思考和回复\n"
        "- 代码块标注语言\n"
        "- 简洁明了，不要冗余"
        )
    )

    # 上下文压缩（Compaction）配置
    reserve_tokens: int = 16384      # 为模型生成预留的空间
    keep_recent_tokens: int = 20000  # 保留最近原始消息的 token 预算


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    """
    ReAct Agent: 边想边做-交错——推理 -> 行动 -> 观察 循环

    核心流程
    --------
    1. 接收用户输入 → 追加到 messages
    2. 调用 LLM（流式）→ 实时渲染文本 + 收集 tool_calls
    3. 无 tool_calls → 任务完成，返回最终文本
    4. 有 tool_calls → 并行执行（asyncio.gather），结果以 tool message 注入对话
    5. 重复步骤 2-4，直到完成或达到 max_turns

    关键机制
    --------
    - 流式渲染: rich.Live 单面板实时更新 Markdown，避免嵌套 Live 冲突
    - Steering: 用户中途输入通过 asyncio.Queue 投递，cancel_event 打断当前 LLM 流
    - 错误传播: 工具异常不崩溃，包装为 ToolResult(is_error=True) 反馈给 LLM 自行恢复
    - 上下文压缩: 每轮自动检查 token 用量，按三层阈值触发压缩
    - 危险确认: bash 等危险工具执行前标记，由 Agent 层决定是否确认（confirm_dangerous）
    """

    def __init__(
        self,
        client: LLMClient,
        tools: ToolRegistry,
        config: AgentConfig | None = None, 
        *,
        console: Console | None = None, 
        confirm_dangerous: bool = True
    ):
        # ---- 核心依赖 ----
        self.client: LLMClient = client          # LLM 客户端（流式 chat + 取消）
        self.tools: ToolRegistry = tools          # 工具注册中心（7 个内置工具 + 可扩展）
        self.config: AgentConfig = config or AgentConfig()  # 运行配置（max_turns / 压缩阈值等）

        # ---- 渲染 & 交互 ----
        self.console: Console = console or Console()       # Rich 终端渲染器
        self.confirm_dangerous: bool = confirm_dangerous   # 是否在执行危险工具前确认

        # ---- 上下文压缩器 ----
        # Pi-style compaction：保留最近 keep_recent_tokens 原始消息，
        # 更早的历史压缩成结构化摘要。
        self.compressor = ContextCompactor(
            CompactionConfig(
                reserve_tokens=self.config.reserve_tokens,
                keep_recent_tokens=self.config.keep_recent_tokens,
            )
        )
        
        # ---- 对话历史 ----
        # 完整保留 system + user + assistant + tool 消息，每轮追加
        self.messages: list[Message] = []
        
        # ---- Steering 控制（用户中途介入）----
        self._steer_queue: asyncio.Queue[str] = asyncio.Queue()  # 用户 steering 消息队列
        self._cancel_event: asyncio.Event = asyncio.Event()      # 取消令牌：set() 后 LLM 流中断
        
        # ---- 运行时状态 ----
        self._turn_count: int = 0    # 当前轮次计数
        self._aborted: bool = False  # 是否已被用户中止 

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    
    async def run(self, user_input: str) -> str: 
        """执行一次 Agent 运行"""
        self._aborted = False 
        self._cancel_event.clear()
        # 初始化系统提示
        if not self.messages or self.messages[0].role != 'system':
            self.messages.insert(0, Message(role='system', content=self.config.system_prompt))
        
        # 添加用户消息
        self.messages.append(Message(role='user', content=user_input))
        final_text = ''
        for turn in range(self.config.max_turns):
            self._turn_count = turn + 1
            # 检查 steering
            steering_msg = await self._drain_steering()
            if steering_msg:
                self._cancel_event.clear()
                self.messages.append(Message(role='user', content=f'[用户中途指示]\n{steering_msg}'))
        
            # 上下文压缩
            before_tokens = estimate_messages_tokens(self.messages)
            self.messages = await self.compressor.compress_if_needed(
                self.messages, self.config.context_limit, self.client,
                api_usage_tokens = before_tokens,
            )
            
            # 调用LLM
            try:
                accumulated = await self._stream_and_collect()
            except StreamCancelledError:
                self.console.print("[yellow]⚠ 流被中断[/yellow]")
                final_text = "[被用户中断]"
                break
            except LLMError as exc:
                self.console.print(f"[red]✗ LLM 错误: {exc}[/red]")
                self.messages.append(Message(
                    role="assistant",
                    content=f"发生错误: {exc}"
                ))
                final_text = f"错误: {exc}"
                break
            
            text = accumulated.text 
            tool_calls = accumulated.tool_calls 
            
            # 添加 assistant 消息
            self.messages.append(Message(
                role='assistant',
                content=text or None,  
                tool_calls=tool_calls 
            ))

            # 无工具调用 → 任务完成
            if not tool_calls:
                final_text = text
                break

            # 执行工具
            tool_msgs, tool_outputs = await self._execute_tools(tool_calls)
            self.messages.extend(tool_msgs)
            
            # 检查
            all_failed = all(t.is_error for t in tool_outputs)
            if all_failed and tool_outputs:
                self.console.print("[yellow]⚠ 所有工具执行失败，LLM 将尝试恢复[/yellow]")

            final_text = text

            if self._aborted:
                break

        else:
            final_text = f"达到最大轮次限制 ({self.config.max_turns})"
            self.console.print(f"[yellow]⚠ {final_text}[/yellow]")

        return final_text

    # ------------------------------------------------------------------
    # Steering
    # ------------------------------------------------------------------

    async def steer(self, message: str) -> None: 
        """发送中途指示。会中断当前 LLM 流"""
        await self._steer_queue.put(message)
        self._cancel_event.set()

    def abort(self) -> None:
        """中止 Agent 运行。"""
        self._aborted = True
        self._cancel_event.set()

    async def _drain_steering(self) -> str | None:
        """ 获取所有排队的 steering 消息，合并为一条 """
        msgs: list[str] = []
        while not self._steer_queue.empty():
            msgs.append(self._steer_queue.get_nowait())
        return "\n".join(msgs) if msgs else None 

    # ------------------------------------------------------------------
    # 流式响应 + Live 渲染
    # ------------------------------------------------------------------

    async def _stream_and_collect(self) -> AccumulatedResponse:
        """流式获取 LLM 响应，用 rich.live 实时渲染文本。"""
        tools_schema = self.tools.to_openai_schema() if self.tools._tools else None 
        stream = self.client.chat(
            self.messages,
            tools=tools_schema,
            temperature=self.config.temperature,
            cancel_event=self._cancel_event, 
        )

        text_parts: list[str] = []
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        tool_names: list[str] = []
        live_text = ''
        
        with Live("", console=self.console, refresh_per_second=10, transient=False) as live:
            async for delta in stream:
                if delta.kind == 'text' and delta.text:
                    text_parts.append(delta.text)
                    live_text += delta.text 
                    live.update(Markdown(live_text + "▌"))
                
                elif delta.kind == 'tool_call' and delta.tool_index is not None:
                    idx = delta.tool_index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    acc = tool_calls_acc[idx]
                    if delta.tool_id_chunk:
                        acc["id"] += delta.tool_id_chunk
                    if delta.tool_name_chunk:
                        acc["function"]["name"] += delta.tool_name_chunk
                    if delta.tool_args_chunk:
                        acc["function"]["arguments"] += delta.tool_args_chunk

                    # 更新工具调用状态显示
                    current_names: list[str] = []
                    for ti in sorted(tool_calls_acc.keys()):
                        name = tool_calls_acc[ti]["function"]["name"]
                        if name:
                            current_names.append(name)
                    
                    if current_names != tool_names:
                        tool_names = current_names
                        status = "🔧 " + ", ".join(tool_names) if tool_names else ""
                        live.update(Markdown(live_text + f"\n\n{status}"))

                elif delta.kind == 'done':
                    live.update(Markdown(live_text or "（调用工具中...）"))
                    finish_reason = delta.finish_reason

                elif delta.kind == "error":
                    live.update(Text(f"✗ 流错误: {delta.error}", style="red"))
        
        # 构建最终 tools_calls 列表
        tool_calls: list[dict[str, Any]] | None = None 
        if tool_calls_acc:
            tool_calls = [tool_calls_acc[i] for  i in sorted(tool_calls_acc.keys())]
        
        return AccumulatedResponse(text=''.join(text_parts), tool_calls=tool_calls, finish_reason=finish_reason)

    # ------------------------------------------------------------------
    # 工具执行
    # ------------------------------------------------------------------

    async def _execute_tools(
        self, tool_calls: list[dict[str, Any]]
    ) -> tuple[list[Message], list[ToolResult]]:
        """并行执行工具，返回(tool 消息列表，执行结果列表)

        关键：所有工具都执行完毕（return_exceptions=True），
        错误包装为 ToolResult(is_error=True)，以 tool message 反馈给 LLM。
        
        Args:
            tool_calls (list[dict[str, Any]]): 需要执行的工具列表

        Returns:
            tuple[list[Message], list[ToolResult]]: tool message 反馈给 LLM
        """
        tasks: list[asyncio.Task[ToolResult]] = []
        task_info: list[tuple[str, str, dict[str, Any]]] = []
        
        for tc in tool_calls:
            call_id = tc.get("id", 'unknown')
            func = tc.get('function', {})
            name = func.get("name", "")
            try:
                args: dict[str, Any] = json.loads(func.get('arguments', '{}'))
            except json.JSONDecodeError:
                args = {}
        
            # 危险工具确认
            if self.confirm_dangerous and name in self.tools.get_dangerous_tools():
                self.console.print(
                    f"[yellow]⚠ 执行危险操作: {name}({json.dumps(args, ensure_ascii=False)})[/yellow]"
                )
                # 非交互模式下自动确认
                self.console.print("[dim]自动确认危险操作[/dim]")

            tasks.append(asyncio.create_task(self.tools.execute(name, args)))
            task_info.append((call_id, name, args))
            
        # 并行执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 生成 tool 消息
        tool_msgs: list[Message] = []
        tool_results: list[ToolResult] = []
        
        for (call_id, name, args), result in zip(task_info, results):
            if isinstance(result, Exception):
                tr = ToolResult.error(f'{type(result).__name__}: {result}')
            elif isinstance(result, ToolResult):
                tr = result 
            else:
                tr = ToolResult.error(f'未知结果类型: {type(result).__name__}')
            
            tool_results.append(tr)
            
            # 显示工具结果
            label = "✓" if not tr.is_error else "✗"
            style = "green" if not tr.is_error else "red"
            preview = tr.output[:300].replace("\n", " ")
            self.console.print(
                Panel(
                    f"[bold]{label} {name}[/bold]\n{preview}{'...' if len(tr.output) > 300 else ''}",
                    style=style,
                    title=f"tool: {name}",
                )
            )

            # 构建 tool 消息
            tool_msgs.append(Message(
                role="tool",
                content=tr.output,
                tool_call_id=call_id,
            ))

        return tool_msgs, tool_results


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def build_agent(
    api_key: str,
    *,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    system_prompt: str | None = None,
    max_turns: int = 50,
    context_limit: int = 128000,
    confirm_dangerous: bool = False,
):
    """便利函数：构建一个配置好的 Agent。"""
    config = AgentConfig(
        model=model,
        max_turns=max_turns,
        context_limit=context_limit,
        system_prompt=system_prompt or AgentConfig.system_prompt,
    )
    client = LLMClient(api_key=api_key, base_url=base_url, model=model)
    tools = create_default_registry()
    return Agent(client, tools, config, confirm_dangerous=confirm_dangerous)

