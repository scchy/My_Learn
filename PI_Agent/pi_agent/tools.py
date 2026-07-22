# Day 2: 工具系统
# python3
# Create Date: 2026-07-22
# Author: Scc_hy
# Tip:
# 核心机制：装饰器注册、Schema 自动生成、危险操作确认、超时截断、错误包装

# 关键修正（相比初版计划）：
# - ToolResult 统一返回格式（含 is_error 标记）
# - bash 工具妥善杀死超时子进程
# - execute() 返回 ToolResult 而不抛异常（错误通过 is_error=True 传递）
# ==========================================================================================


from __future__ import annotations
import asyncio
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from openai import timeout

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ToolResult —— 工具执行结果
# ---------------------------------------------------------------------------


@dataclass 
class ToolResult:
    """工具执行结果。is_error=True 时 output 是错误信息"""
    output: str 
    is_error: bool = False 
    
    @staticmethod
    def ok(output: str) -> "ToolResult":
        return ToolResult(output=output, is_error=False) 

    @staticmethod
    def error(output: str) -> "ToolResult":
        return ToolResult(output=output, is_error=True)


# ---------------------------------------------------------------------------
# ToolSpec —— 工具元数据
# ---------------------------------------------------------------------------


@dataclass 
class ToolSpec:
    name: str
    description: str 
    parameters: dict[str, Any] 
    dangerous: bool = False 
    timeout: float = 30.0
    _func: Callable | None = field(default=None, repr=False)
    
    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """
    全局工具注册中心
    """
    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}
    
    def register(
        self,
        func: Callable,
        *,
        name: str | None = None,
        description: str | None = None,
        dangerous: bool = False,
        timeout: float = 30.0,
    ) -> ToolSpec:
        """注册一个工具函数。自动从函数签名和 docstring 生成 schema。"""
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "").strip().split("\n")[0]
        schema = self._build_schema(func)
        
        spec = ToolSpec(
            name=tool_name,
            description=tool_desc,
            parameters=schema,
            dangerous=dangerous,
            timeout=timeout,
            _func=func
        )
        self._tools[tool_name] = spec
        return spec 

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def to_openai_schema(self) -> list[dict[str, Any]]:
        return [
            spec.to_openai_schema() for spec in self._tools.values()
        ]
    
    def get_dangerous_tools(self) -> list[str]:
        return [name for name, spec in self._tools.items() if spec.dangerous]
    
    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """执行工具。始终返回 ToolResult，不抛异常。"""
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult.error(f"未知工具: {name}")

        if spec._func is None:
            return ToolResult.error(f"工具没有实现: {name}")

        try:
            result = await asyncio.waut_for(
                asyncio.to_thread(spec._func, **arguments),
                timeout=spec.timeout
            )
            # 截断过长的输出
            output = str(result) if not isinstance(result, str) else result
            if len(output) > 4000:
                output = output[:3900] + f"\n\n... [截断，原输出 {len(output)} 字符]"
            return ToolResult.ok(output)
        except asyncio.TimeoutError:
            return ToolResult.error(f"工具执行超时 ({spec.timeout}s): {name}")
        except Exception as exc:
            logger.exception(f"工具执行异常: {name}")
            return ToolResult.error(f"工具执行异常: {type(exc).__name__}: {exc}")

    def _build_schema(self, func: Callable) -> dict[str, Any]:
        """
        从函数签名构建JSON Schema。当前只能用 str 类型参数简化处理
        """
        import inspect 
        sig = inspect.signature(func)
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            # 跳过
            if param_name == 'self':
                continue
            # 所有参数统一视为string
            prop: dict[str, Any] = {"type": "string", "description": param_name}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
            properties[param_name] = prop 
            
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }


# ---------------------------------------------------------------------------
# 7 个内置工具
# ---------------------------------------------------------------------------


def _read_file(path: str, offset: str = "1", limit: str = "100") -> str:
    """
    读取文件内容。offset为起始行号 (1-indexed), limit 为最大行数。
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"错误: 文件不存在: {p}"
    if not p.is_file():
        return f"错误: 路径不是文件: {p}"
    
    try:
        # todo  大文件的情况改成流式情况
        off = int(offset)
        lim = int(limit)
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)
        if off < 1:
            off = 1
        start = off - 1
        end = min(total, start + lim)
        selected = lines[start:end]
        header = f"[{p}] 行 {start + 1}-{end} / 共 {total} 行\n"
        return header + "\n".join(selected)
    except Exception as exc:
        return f"读取文件失败: {exc}"


def _bash_command(command: str, timeout_sec: str = "30") -> str:
    """执行 shell 命令。需用户确认后执行。设置 timeout。"""
    try:
        timeout_val = float(timeout_sec)
    except ValueError:
        timeout_val = 30.0

    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_val,
            cwd=os.getcwd(),
        )
        output = proc.stdout 
        if proc.stderr:
            output += "\n[stderr]\n" + proc.stderr
        if proc.returncode != 0:
            output += f"\n[退出码: {proc.returncode}]"
        return output or "(无输出)"
    except subprocess.TimeoutExpired:
        return f"错误: 命令超时 ({timeout_val}s): {command}"
    except Exception as exc:
        return f"执行命令失败: {exc}"


def _edit_file(path: str, old_text: str, new_text: str) -> str:
    """在文件中替换字符串。old_text 必须精确匹配。"""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"错误: 文件不存在: {p}"
    if not p.is_file():
        return f"错误: 路径不是文件: {p}"   

    try:
        content = p.read_text(encoding="utf-8")
        count = content.count(old_text)
        if count == 0:
            return f"错误: 未找到匹配文本。请确认 old_text 精确匹配文件内容。"
        if count > 1:
            return f"错误: 找到 {count} 处匹配。old_text 不够精确，请包含更多上下文使匹配唯一。"
        new_content = content.replace(old_text, new_text, 1)
        p.write_text(new_content, encoding="utf-8")
        return f"已编辑 {p}：1 处替换。"
    except Exception as exc:
        return f"编辑文件失败: {exc}"


def _write_file(path: str, content: str) -> str:
    """写入文件（自动创建父目录）。"""
    p = Path(path).expanduser().resolve()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"已写入 {p} ({len(content)} 字符)"
    except Exception as exc:
        return f"写入文件失败: {exc}"


def _grep_search(pattern: str, path: str = ".", include: str = "*") -> str:
    """在文件中搜索正则模式。path 为目录或文件，include 为 glob 过滤。"""
    import fnmatch
    
    p = Path(path).expanduser().resolve()
    results: list[str] = []
    
    try:
        files: list[Path] = []
        if p.is_file():
            files = [p]
        elif p.is_dir():
            for root, _, filenames in os.walk(p):
                for fname in filenames:
                    fpath = Path(root) / fname
                    if fnmatch.fnmatch(fname, include):
                        files.append(fpath)
        else:
            return f"错误: 路径不存在: {p}"

        # 限制搜索文件数
        if len(files) > 500:
            return f"错误: 匹配文件过多 ({len(files)})。请缩小范围。"

        pat = re.compile(pattern)
        for fpath in sorted(files)[:200]:
            try:
                for lineno, line in enumerate(fpath.read_text(errors="replace").splitlines(), 1):
                    if pat.search(line):
                        results.append(f"{fpath}:{lineno}: {line.strip()[:200]}")
                        if len(results) >= 100:
                            break
            except Exception:
                pass
            if len(results) >= 100:
                break

        if not results:
            return f"未找到匹配模式 '{pattern}' 的内容。"
        return "\n".join(results)
    except Exception as exc:
        return f"搜索失败: {exc}"


def _find_files(pattern: str, path: str = ".") -> str:
    """查找匹配通配符的文件。"""
    import fnmatch

    p = Path(path).expanduser().resolve()
    results: list[str] = []

    try:
        if p.is_file():
            return str(p) if fnmatch.fnmatch(p.name, pattern) else "未找到匹配文件。"
        if not p.is_dir():
            return f"错误: 路径不存在: {p}"

        for root, _, filenames in os.walk(p):
            for fname in sorted(filenames):
                if fnmatch.fnmatch(fname, pattern):
                    fpath = Path(root) / fname
                    results.append(str(fpath))
                    if len(results) >= 200:
                        break
            if len(results) >= 200:
                break

        if not results:
            return f"未找到匹配 '{pattern}' 的文件。"
        return "\n".join(results)
    except Exception as exc:
        return f"查找失败: {exc}"


def _list_directory(path: str = ".") -> str:
    """列出目录内容。"""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"错误: 路径不存在: {p}"
    if p.is_file():
        return f"{p} ({p.stat().st_size} bytes)"

    items: list[str] = []
    try:
        for entry in sorted(p.iterdir()):
            suffix = "/" if entry.is_dir() else ""
            try:
                size = entry.stat().st_size
                size_str = f" ({size} bytes)" if not entry.is_dir() else ""
            except Exception:
                size_str = ""
            items.append(f"  {entry.name}{suffix}{size_str}")
    except PermissionError:
        return f"错误: 无权限访问: {p}"

    return f"[{p}] 共 {len(items)} 项:\n" + "\n".join(items)


# ---------------------------------------------------------------------------
# 注册所有内置工具
# ---------------------------------------------------------------------------


def register_builtin_tools(registry: ToolRegistry) -> ToolRegistry:
    """
    注册 7 个内置工具到 registry
    """
    registry.register(
        _read_file, 
        name="read", 
        description="读取文件内容（支持 offset/limit 分页）"
    )
    registry.register(
        _bash_command,
        name="bash",
        description="执行 shell 命令（危险操作，需用户确认）",
        dangerous=True,
        timeout=60.0,
    )
    registry.register(
        _edit_file,
        name="edit",
        description="在文件中精确替换字符串。oldText 必须唯一匹配。",
    )
    registry.register(
        _write_file,
        name="write",
        description="写入文件（自动创建父目录）",
    )
    registry.register(
        _grep_search,
        name="grep",
        description="在文件中搜索正则模式。path 为目录或文件路径。",
    )
    registry.register(
        _find_files,
        name="find",
        description="按通配符模式查找文件。",
    )
    registry.register(
        _list_directory,
        name="ls",
        description="列出目录内容。",
    )
    return registry


def create_default_registry() -> ToolRegistry:
    """创建并返回一个包含所有内置工具的 registry。"""
    registry = ToolRegistry()
    return register_builtin_tools(registry)

