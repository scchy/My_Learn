# Day 6：CLI 外壳
# python3
# Create Date: 2026-07-28
# Author: Scc_hy
# Tip:
# 设计原则： 
# 核心机制：typer 命令解析、rich 交互、配置优先级、/save 命令、--base-url

# 关键修正（相比初版计划）：
# - 增加 --base-url / --api-base 参数
# - 增加 --no-confirm 跳过危险工具确认
# - 配置优先级: CLI > 配置文件 > 环境变量
# - 异步入口通过 asyncio.run() 包装
# ====================================================================================

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional
import typer
import yaml
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from pi_agent.agent import Agent, AgentConfig
from pi_agent.llm import LLMClient
from pi_agent.session import SessionStore 
from pi_agent.tools import create_default_registry


app = typer.Typer(
    name='pi-agent(simple)',
    help='Pi Agent 最小核心Python实现 —— 交互式编程助手'
)
console = Console()


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

CONFIG_PATHS = [
    Path("./config.yaml"),
    Path(os.path.expanduser("~/.pi-agent/config.yaml")),
]


def load_config() -> dict:
    config: dict = {}
    for path in CONFIG_PATHS:
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {} 
                config.update(data)
            except Exception:
                pass
    return config 


def resolve_kwargs(cli_kwargs: dict) -> dict: 
    """ 按优先级合并：CLI > 配置文件 > 环境变量 > 默认值。 """
    config = load_config()
    result: dict = {}
    
    # API Key: CLI > env > config 
    result["api_key"] = (
        cli_kwargs.get("api_key")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or config.get("api_key")
    )

    # Base URL: CLI > env > config > default
    result["base_url"] = (
        cli_kwargs.get("base_url")
        or os.environ.get("OPENAI_BASE_URL")
        or config.get("base_url")
        or "https://api.deepseek.com"
    )

    # Model: CLI > config > default
    result["model"] = (
        cli_kwargs.get("model")
        or config.get("model")
        or "deepseek-chat"
    )
    
    # 其他
    result["max_turns"] = cli_kwargs.get("max_turns") or config.get("max_turns") or 50
    result["context_limit"] = cli_kwargs.get("context_limit") or config.get("context_limit") or 128000
    result["confirm_dangerous"] = not cli_kwargs.get("no_confirm", False)

    return result 


# ---------------------------------------------------------------------------
# 共享初始化
# ---------------------------------------------------------------------------


def _build_agent_and_store(kwargs: dict) -> tuple[Agent, SessionStore]:
    """根据 kwargs 构建 Agent 和 SessionStore。"""
    client = LLMClient(
        api_key=kwargs["api_key"],
        base_url=kwargs["base_url"],
        model=kwargs["model"],
    )
    tools = create_default_registry()
    config = AgentConfig(
        model=kwargs["model"],
        max_turns=kwargs.get("max_turns", 50),
        context_limit=kwargs.get("context_limit", 128000),
    )
    agent = Agent(
        client=client,
        tools=tools,
        config=config,
        console=console,
        confirm_dangerous=kwargs.get("confirm_dangerous", True),
    )
    store = SessionStore()
    store.load()
    return agent, store


def _agent_metadata(agent: Agent) -> dict:
    """提取 Agent 运行配置元数据，随会话节点一起持久化。"""
    return {
        "model": agent.config.model,
        "max_turns": agent.config.max_turns,
        "context_limit": agent.config.context_limit,
    }


# ---------------------------------------------------------------------------
# chat 命令
# ---------------------------------------------------------------------------


@app.command()
def chat(
    prompt: Optional[str] = typer.Argument(None, help="用户消息（省略则进入交互模式）"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="模型名称"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API Key"),
    base_url: Optional[str] = typer.Option(None, "--base-url", "-b", help="API 基础地址"),
    max_turns: Optional[int] = typer.Option(None, "--max-turns", help="最大轮次"),
    no_confirm: bool = typer.Option(False, "--no-confirm", help="跳过危险工具确认"),
) -> None:
    """启动交互式或非交互式对话。"""
    kwargs = resolve_kwargs({
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "max_turns": max_turns,
        "no_confirm": no_confirm,
    })

    if not kwargs["api_key"]:
        console.print(
            "[red]错误: 未提供 API Key。[/red]\n"
            "请通过以下方式之一提供：\n"
            "  --api-key sk-xxx\n"
            "  环境变量 OPENAI_API_KEY 或 DEEPSEEK_API_KEY\n"
            "  ~/.pi-agent/config.yaml 中设置 api_key"
        )
        raise typer.Exit(1)

    agent, store = _build_agent_and_store(kwargs)

    # 启动提示：显示最近书签
    bookmarks = store.list_bookmarks()
    last_hint = ""
    if bookmarks:
        name, nid = bookmarks[-1]
        last_hint = f"\n最近书签: [cyan]{name}[/cyan] → {nid}\n恢复: pi-agent resume {name}"

    console.print(
        Panel.fit(
            f"[bold]Pi Agent Mini[/bold]\n"
            f"模型: {kwargs['model']}\n"
            f"API: {kwargs['base_url']}\n"
            f"最大轮次: {kwargs['max_turns']}\n"
            f"危险确认: {'开启' if kwargs['confirm_dangerous'] else '关闭'}"
            f"{last_hint}\n"
            f"命令: /save <name> 打书签 | /exit 退出 | /stats 统计",
            title="启动信息",
        )
    )

    if prompt:
        # 非交互模式
        asyncio.run(_run_non_interactive(agent, store, prompt))
    else:
        # 交互模式
        asyncio.run(_run_interactive(agent, store))


# ---------------------------------------------------------------------------
# resume 命令
# ---------------------------------------------------------------------------


@app.command()
def resume(
    bookmark: str = typer.Argument(..., help="书签名称"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="模型名称"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API Key"),
    base_url: Optional[str] = typer.Option(None, "--base-url", "-b", help="API 基础地址"),
    max_turns: Optional[int] = typer.Option(None, "--max-turns", help="最大轮次"),
    no_confirm: bool = typer.Option(False, "--no-confirm", help="跳过危险工具确认"),
) -> None:
    """按书签恢复之前的会话。"""
    kwargs = resolve_kwargs({
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "max_turns": max_turns,
        "no_confirm": no_confirm,
    })
    
    if not kwargs["api_key"]:
        console.print("[red]错误: 未提供 API Key[/red]")
        raise typer.Exit(1)

    store = SessionStore()
    store.load()

    # 先按书签查找，再按节点 ID 查找（书签是用户主动命名的，优先匹配）
    node = store.get_by_bookmark(bookmark)
    if node is None:
        node = store.get_node(bookmark)

    if node is None:
        console.print(f"[red]错误: 未找到节点或书签 '{bookmark}'[/red]")
        # 列出所有书签
        bookmarks = store.list_bookmarks()
        if bookmarks:
            console.print("\n可用书签:")
            for name, nid in bookmarks:
                console.print(f"  [cyan]{name}[/cyan] → {nid}")
        # 也列出最近的根节点供参考
        roots = store.list_roots()
        if roots:
            console.print("\n最近的根节点:")
            for n in roots[-5:]:
                marker = f" [dim]({n.bookmark})[/dim]" if n.bookmark else ""
                console.print(f"  {n.id}{marker}  {n.created_at[:16]}")
        raise typer.Exit(1)

    # 从存储的 metadata 恢复运行参数（CLI 显式传入优先，否则用存储值）
    stored_meta = node.metadata or {}
    for key in ("max_turns", "context_limit", "model"):
        if kwargs.get(key) is None and stored_meta.get(key):
            kwargs[key] = stored_meta[key]

    # 恢复消息
    messages = store.messages_from_node(node.id)
    if messages is None:
        console.print(f"[red]错误: 无法从节点 {node.id} 恢复消息[/red]")
        raise typer.Exit(1)

    agent, _ = _build_agent_and_store(kwargs)
    agent.messages = messages  # 恢复历史
    agent.current_node_id = node.id  # 绑定到已有节点

    console.print(
        Panel.fit(
            f"[bold]恢复会话[/bold]\n"
            f"书签: [cyan]{bookmark}[/cyan]\n"
            f"节点: {node.id}\n"
            f"消息数: {len(messages)}\n"
            f"最大轮次: {agent.config.max_turns}",
            title="会话恢复",
        )
    )

    asyncio.run(_run_interactive(agent, store))


# ---------------------------------------------------------------------------
# 运行模式
# ---------------------------------------------------------------------------


async def _run_non_interactive(
    agent: Agent,
    store: SessionStore,
    prompt: str
) -> None:
    """ 非交互模式: 一次 prompt -> 输出结果 -> 退出 """
    console.print(f"[dim]用户: [/dim] {prompt}\n")
    result = await agent.run(prompt)
    console.print(f"\n[bold green]完成[/bold green]")
    console.print(Markdown(result  or "(无输出)"))
    
    # 自动保存（附带运行配置元数据）
    node = store.create_node(agent.messages, metadata=_agent_metadata(agent))
    agent.current_node_id = node.id
    store.save()
    console.print(f"[dim]会话已自动保存：{node.id}[/dim]")
    

async def _run_interactive(
    agent: Agent,
    store: SessionStore,
):
    """交互模式：持续对话，支持 /save、/exit、/stats 命令。
    
    /save 打书签（不创建新节点），/exit 自动保存。
    """

    def _save_current_state():
        """将 agent.messages 同步到当前节点（存在则更新，否则创建）。"""
        if agent.current_node_id is None:
            node = store.create_node(
                agent.messages,
                metadata=_agent_metadata(agent),
            )
            agent.current_node_id = node.id
            return node
        else:
            node = store.update_node(agent.current_node_id, agent.messages)
            if node is None:
                console.print("[yellow]警告: 当前会话节点已丢失，创建新节点[/yellow]")
                node = store.create_node(
                    agent.messages,
                    metadata=_agent_metadata(agent),
                )
                agent.current_node_id = node.id
            return node

    while True:
        try:
            user_input = Prompt.ask("\n[bold blue]你[/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]再见！[/dim]")
            if agent.messages:
                node = _save_current_state()
                console.print(f"[yellow]自动保存会话: {node.id}[/yellow]")
            store.save()
            console.print(f"[dim]存储位置: {store.filepath}[/dim]")
            break

        # 处理命令
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            
            if cmd == "/exit" or cmd == '/quit':
                if agent.messages:
                    node = _save_current_state()
                    console.print(f"[dim]会话已保存: {node.id}[/dim]")
                    console.print(f"[dim]恢复: pi-agent resume {node.id}[/dim]")
                store.save()
                console.print(f"[dim]存储位置: {store.filepath}[/dim]")
                break
            elif cmd == '/save':
                name = arg or "auto"
                if agent.current_node_id is None:
                    # 首次保存：创建节点 + 打书签
                    node = store.create_node(
                        agent.messages,
                        bookmark=name,
                        metadata=_agent_metadata(agent),
                    )
                    agent.current_node_id = node.id
                    console.print(f"[green]✓ 已保存: {name} ({node.id})[/green]")
                    console.print(f"[dim]恢复: pi-agent resume {name} 或 pi-agent resume {node.id}[/dim]")
                else:
                    # 已有节点：更新消息 + 打书签
                    node = store.update_node(agent.current_node_id, agent.messages)
                    if node is None:
                        console.print("[yellow]警告: 当前会话节点已丢失，创建新节点[/yellow]")
                        node = store.create_node(
                            agent.messages,
                            bookmark=name,
                            metadata=_agent_metadata(agent),
                        )
                        agent.current_node_id = node.id
                    else:
                        store.bookmark_node(agent.current_node_id, name)
                    console.print(f"[green]✓ 书签已更新: {name} ({agent.current_node_id})[/green]")
                    console.print(f"[dim]恢复: pi-agent resume {name} 或 pi-agent resume {agent.current_node_id}[/dim]")
                store.save()
            elif cmd == '/bookmarks':
                bookmarks = store.list_bookmarks()
                if bookmarks:
                    for name, nid in bookmarks:
                        marker = " ★" if nid == agent.current_node_id else ""
                        console.print(f'  [cyan]{name}[/cyan] → {nid}{marker}')
                else:
                    console.print("[dim]暂无书签[/dim]")
            elif cmd == "/stats":
                stats = agent.compressor.get_stats()
                est = stats.get('estimated_tokens', 0)
                limit = stats.get('context_limit', 1)
                ratio = f"{est / limit * 100:.0f}%" if limit > 0 else "?"
                console.print(
                    f"轮次: {agent.turn_count}/{agent.config.max_turns} | "
                    f"消息数: {stats.get('messages', '?')} | "
                    f"Token 估算: {est} | "
                    f"使用率: {ratio}"
                )
            elif cmd == '/help':
                console.print("""
[bold]可用命令:[/bold]
  /save <name>  - 打书签（不创建新节点）
  /bookmarks    - 列出所有书签
  /stats        - 查看上下文统计
  /exit         - 保存并退出
  /help         - 显示此帮助
""")
            else:
                console.print(f"[yellow]未知命令: {cmd}，输入 /help 查看帮助[/yellow]")
            continue
        
        # 正常对话
        console.print()
        try:
            result = await agent.run(user_input)
        except Exception as exc:
            console.print(f"[red]✗ 运行错误: {exc}[/red]")
            continue
        
        # 如果是简单对话(无工具调用)，显示结果
        if result and not any(m.tool_calls for m in agent.messages if m.role == 'assistant'):
            console.print(f'\n[bold green]Agent[/bold green]')
            console.print(Markdown(result))
        
        console.print(f'[dim]--- 轮次 {agent.turn_count} --[/dim]')


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main():
    app()


if __name__ == "__main__":
    main()
