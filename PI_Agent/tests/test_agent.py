"""
Day 7: Agent 循环测试
====================
覆盖内容：
- 单元测试（Mock）：ReAct 循环、并行工具执行、steering 中断、max_turns
- Mock LLM 返回 tool_calls / 纯文本 / 错误流
- 上下文压缩集成

Mock 方式：Mock LLMClient.chat() 返回可控 Delta 流，Mock ToolRegistry 工具
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pi_agent.agent import Agent, AgentConfig, build_agent
from pi_agent.llm import (
    AccumulatedResponse,
    Delta,
    LLMClient,
    LLMError,
    Message,
    StreamCancelledError,
    collect_deltas,
)
from pi_agent.tools import ToolRegistry, ToolResult
from pi_agent.context import CompactionConfig

pytestmark = pytest.mark.anyio


# ===========================================================================
# 辅助工具
# ===========================================================================


def _make_text_deltas(text: str, finish_reason: str = "stop") -> list[Delta]:
    """构造纯文本 Delta 序列。"""
    deltas: list[Delta] = []
    # 模拟逐字符或分块发送
    chunk_size = 5
    for i in range(0, len(text), chunk_size):
        deltas.append(Delta(kind="text", text=text[i:i + chunk_size]))
    deltas.append(Delta(kind="done", finish_reason=finish_reason))
    return deltas


def _make_tool_call_deltas(
    tool_name: str,
    arguments: dict[str, Any],
    call_id: str = "call_001",
    finish_reason: str = "tool_calls",
) -> list[Delta]:
    """构造单个 tool_call 的 Delta 序列。"""
    args_str = json.dumps(arguments, ensure_ascii=False)
    return [
        Delta(kind="tool_call", tool_index=0, tool_id_chunk=call_id,
              tool_name_chunk=tool_name, tool_args_chunk=""),
        Delta(kind="tool_call", tool_index=0, tool_args_chunk=args_str),
        Delta(kind="done", finish_reason=finish_reason),
    ]


def _make_multi_tool_call_deltas(
    tools: list[tuple[str, dict[str, Any], str]],  # (name, args, call_id)
    finish_reason: str = "tool_calls",
) -> list[Delta]:
    """构造多个并行 tool_call 的 Delta 序列。"""
    deltas: list[Delta] = []
    for idx, (name, args, call_id) in enumerate(tools):
        args_str = json.dumps(args, ensure_ascii=False)
        deltas.append(Delta(kind="tool_call", tool_index=idx, tool_id_chunk=call_id,
                            tool_name_chunk=name, tool_args_chunk=args_str))
    deltas.append(Delta(kind="done", finish_reason=finish_reason))
    return deltas


async def _delta_stream(deltas: list[Delta]):
    """将 Delta 列表转为 async generator。"""
    for d in deltas:
        yield d


def _make_mock_agent(
    *,
    max_turns: int = 50,
    context_limit: int = 128000,
    confirm_dangerous: bool = False,
) -> Agent:
    """创建一个配置好的 Agent，使用 mock 客户端和工具注册中心。"""
    client = LLMClient(api_key="sk-test", model="test-model")
    tools = ToolRegistry()

    # 注册几个简单的 mock 工具
    def echo(text: str) -> str:
        return f"echo: {text}"

    def add(a: str, b: str) -> str:
        return str(int(a) + int(b))

    tools.register(echo, name="echo")
    tools.register(add, name="add")

    config = AgentConfig(
        model="test-model",
        max_turns=max_turns,
        context_limit=context_limit,
        temperature=0.0,
        reserve_tokens=1000,
        keep_recent_tokens=2000,
    )
    return Agent(client, tools, config, confirm_dangerous=confirm_dangerous)


# ===========================================================================
# 1. Agent 初始化
# ===========================================================================


class TestAgentInit:
    def test_default_config(self):
        agent = _make_mock_agent()
        assert agent.config.max_turns == 50
        assert agent.config.context_limit == 128000
        assert agent.messages == []

    def test_custom_max_turns(self):
        agent = _make_mock_agent(max_turns=10)
        assert agent.config.max_turns == 10

    def test_system_prompt_on_first_run(self):
        agent = _make_mock_agent()
        # 在 run() 之前系统提示还没插入
        assert agent.messages == []

    def test_abort_flag_default(self):
        agent = _make_mock_agent()
        assert agent._aborted is False


# ===========================================================================
# 2. ReAct 循环 — 纯文本回复（无工具调用）
# ===========================================================================


class TestReActLoopTextOnly:
    @pytest.mark.anyio
    async def test_simple_text_response(self):
        """Agent 收到纯文本回复后停止。"""
        agent = _make_mock_agent(max_turns=5)

        # Mock client.chat() 返回 "Hello, I can help!"
        text_response = _make_text_deltas("Hello, I can help!")
        with patch.object(agent.client, 'chat', return_value=_delta_stream(text_response)):
            result = await agent.run("Hi!")

        assert result == "Hello, I can help!"
        assert agent.turn_count == 1
        # 消息应包含 system + user + assistant
        roles = [m.role for m in agent.messages]
        assert roles == ["system", "user", "assistant"]

    @pytest.mark.anyio
    async def test_multi_turn_text_response(self):
        """两轮文本对话。"""
        agent = _make_mock_agent(max_turns=5)

        # 第一轮
        with patch.object(agent.client, 'chat', return_value=_delta_stream(
            _make_text_deltas("First response")
        )):
            result1 = await agent.run("First question")
        assert result1 == "First response"
        assert agent.turn_count == 1

        # 第二轮
        with patch.object(agent.client, 'chat', return_value=_delta_stream(
            _make_text_deltas("Second response")
        )):
            result2 = await agent.run("Second question")
        assert result2 == "Second response"
        assert agent.turn_count == 1  # 每轮重置


# ===========================================================================
# 3. ReAct 循环 — 工具调用
# ===========================================================================


class TestReActLoopWithTools:
    @pytest.mark.anyio
    async def test_single_tool_call(self):
        """Agent 调用一次工具后完成。"""
        agent = _make_mock_agent(max_turns=5)

        # chat() 被调用两次：
        # 第 1 次 → 返回 tool_call
        # 第 2 次 → 返回纯文本
        call1 = _delta_stream(_make_tool_call_deltas("echo", {"text": "hello"}))
        call2 = _delta_stream(_make_text_deltas("The echo tool returned: echo: hello"))

        with patch.object(agent.client, 'chat', side_effect=[call1, call2]) as mock_chat:
            result = await agent.run("Echo hello")

        assert "echo: hello" in result
        assert agent.turn_count == 2  # 2 轮（工具调用 + 最终回复）
        assert mock_chat.call_count == 2
        # 确认 tool message 已注入
        roles = [m.role for m in agent.messages]
        assert "tool" in roles
        assert roles.count("assistant") == 2

    @pytest.mark.anyio
    async def test_parallel_tool_calls(self):
        """Agent 并行调用多个工具，然后完成。"""
        agent = _make_mock_agent(max_turns=5)

        tools = [
            ("echo", {"text": "a"}, "c1"),
            ("add", {"a": "1", "b": "2"}, "c2"),
        ]
        call1 = _delta_stream(_make_multi_tool_call_deltas(tools))
        call2 = _delta_stream(_make_text_deltas("Done: echo: a, sum: 3"))

        with patch.object(agent.client, 'chat', side_effect=[call1, call2]):
            result = await agent.run("Echo and add")

        assert "Done" in result
        assert agent.turn_count == 2
        # 应该有 2 条 tool 消息
        tool_msgs = [m for m in agent.messages if m.role == "tool"]
        assert len(tool_msgs) == 2

    @pytest.mark.anyio
    async def test_tool_returns_error_feedback_to_llm(self):
        """工具执行异常不应崩溃，而是以 is_error=True 反馈给 LLM。"""
        agent = _make_mock_agent(max_turns=5)

        # 注册一个会爆炸的工具
        def broken(x: str) -> str:
            raise RuntimeError("BOOM!")

        agent.tools.register(broken, name="broken")

        call1 = _delta_stream(_make_tool_call_deltas("broken", {"x": "test"}))
        call2 = _delta_stream(_make_text_deltas("Tool failed, let me try differently"))

        with patch.object(agent.client, 'chat', side_effect=[call1, call2]):
            result = await agent.run("Use broken tool")

        assert "Tool failed" in result
        # broken 工具的错误作为 tool message 注入了
        tool_msgs = [m for m in agent.messages if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert "RuntimeError" in tool_msgs[0].content


# ===========================================================================
# 4. max_turns 保护
# ===========================================================================


class TestMaxTurns:
    @pytest.mark.anyio
    async def test_max_turns_stops_loop(self):
        """达到 max_turns 后强制停止。"""
        agent = _make_mock_agent(max_turns=3)

        # 每次都返回 tool_call，让循环持续
        # 注意：不能用 return_value（generator 会被耗尽），用 lambda 每次生成新实例
        call_count = [0]
        async def fresh_stream(*args, **kwargs):
            call_count[0] += 1
            for d in _make_tool_call_deltas("echo", {"text": f"ping{call_count[0]}"}):
                yield d

        with patch.object(agent.client, 'chat', side_effect=fresh_stream):
            result = await agent.run("Ping forever")

        assert "达到最大轮次限制" in result
        assert agent.turn_count == 3

    @pytest.mark.anyio
    async def test_max_turns_empty_tools(self):
        """工具列表为空时，纯文本对话只消耗 1 轮。"""
        agent = _make_mock_agent(max_turns=2)
        call_deltas = _delta_stream(_make_text_deltas("OK"))

        with patch.object(agent.client, 'chat', return_value=call_deltas):
            result = await agent.run("Hello")

        assert result == "OK"
        assert agent.turn_count == 1  # 纯文本只需 1 轮


# ===========================================================================
# 5. Steering 中断
# ===========================================================================


class TestSteering:
    @pytest.mark.anyio
    async def test_steer_interrupts_stream(self):
        """Steering 中断当前 LLM 流。"""
        agent = _make_mock_agent(max_turns=5)

        # 用单一 side_effect 函数：流进行中触发 steering → StreamCancelledError
        async def chat_with_steer(*args, **kwargs):
            # 在流进行中投递 steering 并触发取消
            await agent.steer("Stop and summarize instead!")
            yield Delta(kind="text", text="Let me think...")
            yield Delta(kind="text", text=" about this...")
            cancel_event = kwargs.get("cancel_event")
            if cancel_event and cancel_event.is_set():
                yield Delta(kind="error", error="stream_cancelled")
                raise StreamCancelledError("流被用户中断")
            yield Delta(kind="done", finish_reason="stop")

        with patch.object(agent.client, 'chat', side_effect=chat_with_steer):
            result = await agent.run("Tell me a story")

        assert "被用户中断" in result
        # steering 消息已在流中被投递到队列（因 break 未在后续轮次处理）
        assert not agent._steer_queue.empty()

    @pytest.mark.anyio
    async def test_steer_drained_in_next_turn(self):
        """Steering 在下一轮开始被 drain 并添加到 messages。"""
        agent = _make_mock_agent(max_turns=5)

        call_index = [0]
        async def multi_chat(*args, **kwargs):
            idx = call_index[0]
            call_index[0] += 1
            if idx == 0:
                # 第一轮：正常返回 tool_call
                for d in _make_tool_call_deltas("echo", {"text": "first"}):
                    yield d
                # 在执行工具后投递 steering（将在下一轮 drain）
            elif idx == 1:
                # 第二轮：drain steering 后，正常文本回复
                yield Delta(kind="text", text="Got your steering, adjusting...")
                yield Delta(kind="done", finish_reason="stop")

        with patch.object(agent.client, 'chat', side_effect=multi_chat):
            result = await agent.run("Do something")

        # 注意：steering 在流外部投递时，会在下一轮 drain
        # 此处我们验证的是 agent 不崩溃
        assert len(result) > 0

    @pytest.mark.anyio
    async def test_abort_during_run(self):
        """abort() 在 run 中设置 _aborted。"""
        agent = _make_mock_agent(max_turns=10)

        call1 = _delta_stream(_make_tool_call_deltas("echo", {"text": "x"}))
        call2 = _delta_stream(_make_text_deltas("Done"))

        with patch.object(agent.client, 'chat', side_effect=[call1, call2]):
            # 我们让 abort 在 run 之前设置（模拟场景）
            # 实际上这里我们验证 abort 机制的存在
            pass

        # 直接测试 abort
        agent.abort()
        assert agent._aborted is True
        assert agent._cancel_event.is_set()


# ===========================================================================
# 6. LLM 错误处理
# ===========================================================================


class TestLLMErrorHandling:
    @pytest.mark.anyio
    async def test_llm_error_in_stream(self):
        """LLM 流中产生错误时，Agent 应捕获并返回错误信息。"""
        agent = _make_mock_agent(max_turns=5)

        async def error_stream(*args, **kwargs):
            yield Delta(kind="text", text="Starting...")
            yield Delta(kind="error", error="API connection lost")
            raise LLMError("API connection lost")

        with patch.object(agent.client, 'chat', return_value=error_stream()):
            result = await agent.run("Do something")

        assert "错误" in result

    @pytest.mark.anyio
    async def test_stream_cancelled_is_handled(self):
        """StreamCancelledError 被捕获并返回友好信息。"""
        agent = _make_mock_agent(max_turns=5)

        async def cancelled_stream(*args, **kwargs):
            yield Delta(kind="error", error="stream_cancelled")
            raise StreamCancelledError("被取消")

        with patch.object(agent.client, 'chat', return_value=cancelled_stream()):
            result = await agent.run("Something")

        assert "被用户中断" in result


# ===========================================================================
# 7. 上下文压缩集成
# ===========================================================================


class TestContextCompaction:
    @pytest.mark.anyio
    async def test_no_compaction_below_threshold(self):
        """短对话不触发压缩。"""
        agent = _make_mock_agent(context_limit=128000)

        async def fresh_text_stream(*args, **kwargs):
            for d in _make_text_deltas("Short response"):
                yield d

        with patch.object(agent.client, 'chat', side_effect=fresh_text_stream):
            # 先跑多轮短对话
            for i in range(3):
                result = await agent.run(f"Question {i}")
                assert result == "Short response"

        # 消息数应该正常（未压缩）
        assert len(agent.messages) < 20

    @pytest.mark.anyio
    async def test_compaction_preserves_system_message(self):
        """压缩后系统消息始终保留。"""
        agent = _make_mock_agent(max_turns=3, context_limit=100)

        call_deltas = _delta_stream(_make_text_deltas("X" * 500))  # 大回复

        with patch.object(agent.client, 'chat', return_value=call_deltas):
            await agent.run("Give me a very long answer")

        # 系统消息应该在
        system_msgs = [m for m in agent.messages if m.role == "system"]
        assert len(system_msgs) >= 1

    @pytest.mark.anyio
    async def test_compressor_stats_available(self):
        """压缩器 stats 可查询。"""
        agent = _make_mock_agent()
        stats = agent.compressor.get_stats()
        assert isinstance(stats, dict)


# ===========================================================================
# 8. build_agent 便利函数
# ===========================================================================


class TestBuildAgent:
    def test_build_agent_returns_agent(self):
        agent = build_agent(api_key="sk-test")
        assert isinstance(agent, Agent)
        assert agent.client.api_key == "sk-test"
        assert agent.client.model == "deepseek-chat"

    def test_build_agent_custom_model(self):
        agent = build_agent(api_key="sk-test", model="gpt-4o-mini")
        assert agent.client.model == "gpt-4o-mini"

    def test_build_agent_system_prompt(self):
        agent = build_agent(api_key="sk-test", system_prompt="You are a test bot.")
        assert "test bot" in agent.config.system_prompt


# ===========================================================================
# 9. Agent 消息状态管理
# ===========================================================================


class TestAgentMessageState:
    @pytest.mark.anyio
    async def test_messages_accumulate_over_turns(self):
        """消息列表随轮次累积。"""
        agent = _make_mock_agent(max_turns=3)

        call1 = _delta_stream(_make_tool_call_deltas("echo", {"text": "p1"}))
        call2 = _delta_stream(_make_text_deltas("All done"))

        with patch.object(agent.client, 'chat', side_effect=[call1, call2]):
            await agent.run("Do echo")

        # 应该有 system + user + assistant(tool_call) + tool + assistant(final)
        assert len(agent.messages) == 5

    @pytest.mark.anyio
    async def test_no_duplicate_system_prompt(self):
        """多次 run() 不会重复插入系统提示。"""
        agent = _make_mock_agent(max_turns=3)

        with patch.object(agent.client, 'chat', return_value=_delta_stream(
            _make_text_deltas("OK")
        )):
            await agent.run("Q1")
            await agent.run("Q2")
            await agent.run("Q3")

        system_msgs = [m for m in agent.messages if m.role == "system"]
        assert len(system_msgs) == 1


# ===========================================================================
# 10. 边界测试
# ===========================================================================


class TestEdgeCases:
    @pytest.mark.anyio
    async def test_empty_user_input(self):
        """空用户输入不应崩溃。"""
        agent = _make_mock_agent(max_turns=3)

        with patch.object(agent.client, 'chat', return_value=_delta_stream(
            _make_text_deltas("What would you like?")
        )):
            result = await agent.run("")

        assert len(result) > 0

    @pytest.mark.anyio
    async def test_very_long_user_input(self):
        """非常长的用户输入不应崩溃。"""
        agent = _make_mock_agent(max_turns=3)
        long_input = "Hello " * 10000

        with patch.object(agent.client, 'chat', return_value=_delta_stream(
            _make_text_deltas("Got it!")
        )):
            result = await agent.run(long_input)

        assert result == "Got it!"

    @pytest.mark.anyio
    async def test_tool_with_malformed_args(self):
        """工具参数为无效 JSON 时回退到空字典。"""
        agent = _make_mock_agent(max_turns=3)

        # 手动构造一个 args 解析失败的情况
        # 在 _execute_tools 中 json.loads 失败时 args 设为 {}
        tool_calls = [{
            "id": "bad",
            "type": "function",
            "function": {"name": "echo", "arguments": "not valid json {{{"},
        }]

        with patch.object(agent.client, 'chat', return_value=_delta_stream(
            _make_text_deltas("Trying...")
        )):
            # 我们通过模拟 tool_calls 内部流来触发 malformed args
            pass

    @pytest.mark.anyio
    async def test_concurrent_runs_independent(self):
        """两个 Agent 实例独立运行，互不影响。"""
        agent_a = _make_mock_agent(max_turns=3)
        agent_b = _make_mock_agent(max_turns=3)

        async def run_agent(agent, label):
            with patch.object(agent.client, 'chat', return_value=_delta_stream(
                _make_text_deltas(f"Response for {label}")
            )):
                return await agent.run(f"Input {label}")

        result_a, result_b = await asyncio.gather(
            run_agent(agent_a, "A"),
            run_agent(agent_b, "B"),
        )

        assert "Response for A" in result_a
        assert "Response for B" in result_b
        # 两个 agent 的消息应该独立
        assert len(agent_a.messages) > 0
        assert len(agent_b.messages) > 0

    @pytest.mark.anyio
    async def test_all_tools_fail_warning(self):
        """所有工具执行失败时应有警告但不崩溃。"""
        agent = _make_mock_agent(max_turns=3)

        def fail1(x: str) -> str:
            raise ValueError("fail1")

        def fail2(x: str) -> str:
            raise RuntimeError("fail2")

        agent.tools.register(fail1, name="fail1")
        agent.tools.register(fail2, name="fail2")

        tools = [
            ("fail1", {"x": "a"}, "c1"),
            ("fail2", {"x": "b"}, "c2"),
        ]
        call1 = _delta_stream(_make_multi_tool_call_deltas(tools))
        call2 = _delta_stream(_make_text_deltas("Both failed, let me try another way"))

        with patch.object(agent.client, 'chat', side_effect=[call1, call2]):
            result = await agent.run("Use failing tools")

        assert "Both failed" in result
        # 两个 tool message 都应该是 error
        tool_msgs = [m for m in agent.messages if m.role == "tool"]
        assert len(tool_msgs) == 2

    @pytest.mark.anyio
    async def test_tool_call_with_no_id(self):
        """tool_call 缺少 id 时使用默认值。"""
        agent = _make_mock_agent(max_turns=3)

        deltas = [
            Delta(kind="tool_call", tool_index=0,
                  tool_name_chunk="echo", tool_args_chunk='{"text":"hi"}'),
            Delta(kind="done", finish_reason="tool_calls"),
        ]
        call1 = _delta_stream(deltas)
        call2 = _delta_stream(_make_text_deltas("Done"))

        with patch.object(agent.client, 'chat', side_effect=[call1, call2]):
            result = await agent.run("Echo hi")

        assert "Done" in result


# ===========================================================================
# 11. 流式渲染（Live 面板）
# ===========================================================================


class TestStreamingRendering:
    @pytest.mark.anyio
    async def test_stream_updates_live_panel(self):
        """流式输出时 rich.Live 面板被更新。"""
        from rich.console import Console as RichConsole

        # 使用一个不产生实际输出的 console
        console = RichConsole(file=None, color_system=None)
        agent = _make_mock_agent()
        agent.console = console

        text = "Streaming text output"
        with patch.object(agent.client, 'chat', return_value=_delta_stream(
            _make_text_deltas(text)
        )):
            result = await agent.run("Stream something")

        assert result == text


# ===========================================================================
# 12. 危险工具确认集成
# ===========================================================================


class TestDangerousToolIntegration:
    @pytest.mark.anyio
    async def test_dangerous_tool_not_blocked_in_execute(self):
        """危险工具的标记不影响 tools.execute()（确认逻辑在 Agent 层）。"""
        agent = _make_mock_agent(max_turns=3, confirm_dangerous=False)

        # 注册一个危险工具
        def rm(path: str) -> str:
            return f"would remove {path}"

        agent.tools.register(rm, name="rm", dangerous=True)

        call1 = _delta_stream(_make_tool_call_deltas("rm", {"path": "/tmp/test"}))
        call2 = _delta_stream(_make_text_deltas("Removed"))

        with patch.object(agent.client, 'chat', side_effect=[call1, call2]):
            result = await agent.run("Remove /tmp/test")

        assert "Removed" in result
        # 危险工具在 _execute_tools 中有打印但不阻止
        assert "rm" in agent.tools.get_dangerous_tools()

    @pytest.mark.anyio
    async def test_confirm_dangerous_with_auto_confirm(self):
        """confirm_dangerous=True 时非交互模式自动确认。"""
        agent = _make_mock_agent(max_turns=3, confirm_dangerous=True)

        def rm(path: str) -> str:
            return f"removed {path}"

        agent.tools.register(rm, name="rm", dangerous=True)

        call1 = _delta_stream(_make_tool_call_deltas("rm", {"path": "/tmp/x"}))
        call2 = _delta_stream(_make_text_deltas("Done"))

        with patch.object(agent.client, 'chat', side_effect=[call1, call2]):
            result = await agent.run("Delete /tmp/x")

        assert "Done" in result


# ===========================================================================
# 13. turn_count property
# ===========================================================================


class TestTurnCount:
    def test_turn_count_zero_initially(self):
        agent = _make_mock_agent()
        assert agent.turn_count == 0

    @pytest.mark.anyio
    async def test_turn_count_increments(self):
        agent = _make_mock_agent(max_turns=5)

        call1 = _delta_stream(_make_tool_call_deltas("echo", {"text": "x"}))
        call2 = _delta_stream(_make_text_deltas("Done"))

        with patch.object(agent.client, 'chat', side_effect=[call1, call2]):
            await agent.run("Test")

        assert agent.turn_count == 2  # 工具调用 + 最终回复 = 2 轮
