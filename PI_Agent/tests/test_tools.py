"""
Day 7: 工具模块测试
====================
覆盖内容：
- 单元测试：ToolResult、ToolSpec、ToolRegistry（注册/Schema/执行）
- 7 个内置工具：read、bash、edit、write、grep、find、ls
- 边界测试：危险确认、超时截断、错误包装、输出截断

Mock 方式：直接调用工具函数（文件 IO 用 tmp_path），mock 危险确认
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pi_agent.tools import (
    ToolRegistry,
    ToolResult,
    ToolSpec,
    _bash_command,
    _edit_file,
    _find_files,
    _grep_search,
    _list_directory,
    _read_file,
    _write_file,
    create_default_registry,
    register_builtin_tools,
)

# tools.py 依赖 asyncio.to_thread，trio 后端由 conftest.py 过滤
pytestmark = pytest.mark.anyio


# ===========================================================================
# 1. ToolResult
# ===========================================================================


class TestToolResult:
    def test_ok(self):
        r = ToolResult.ok("success")
        assert r.output == "success"
        assert r.is_error is False

    def test_error(self):
        r = ToolResult.error("something went wrong")
        assert r.output == "something went wrong"
        assert r.is_error is True

    def test_ok_empty_string(self):
        r = ToolResult.ok("")
        assert r.output == ""
        assert r.is_error is False

    def test_error_with_exception_message(self):
        r = ToolResult.error("FileNotFoundError: /tmp/no_such_file")
        assert "FileNotFoundError" in r.output
        assert r.is_error is True


# ===========================================================================
# 2. ToolSpec
# ===========================================================================


class TestToolSpec:
    def test_basic_spec(self):
        spec = ToolSpec(
            name="read",
            description="read a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            dangerous=False,
            timeout=30.0,
        )
        assert spec.name == "read"
        assert spec.dangerous is False
        assert spec.timeout == 30.0

    def test_dangerous_spec(self):
        spec = ToolSpec(
            name="bash",
            description="run command",
            parameters={"type": "object", "properties": {}},
            dangerous=True,
        )
        assert spec.dangerous is True

    def test_to_openai_schema(self):
        spec = ToolSpec(
            name="read",
            description="read file",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "path"}},
                "required": ["path"],
            },
        )
        schema = spec.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "read"
        assert schema["function"]["description"] == "read file"
        assert schema["function"]["parameters"]["required"] == ["path"]

    def test_to_openai_schema_no_required(self):
        spec = ToolSpec(
            name="ls",
            description="list dir",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        )
        schema = spec.to_openai_schema()
        assert "required" not in schema["function"]["parameters"]


# ===========================================================================
# 3. ToolRegistry — 注册 & Schema
# ===========================================================================


class TestToolRegistryRegister:
    def test_register_single_tool(self):
        registry = ToolRegistry()

        def dummy_echo(text: str) -> str:
            """回显输入的文本"""
            return text

        spec = registry.register(dummy_echo)
        assert spec.name == "dummy_echo"
        assert spec.description == "回显输入的文本"
        assert "text" in spec.parameters["properties"]
        assert spec.dangerous is False
        assert spec.timeout == 30.0

    def test_register_with_custom_name(self):
        registry = ToolRegistry()

        def my_func(x: str) -> str:
            return x

        spec = registry.register(my_func, name="custom_name", description="custom desc")
        assert spec.name == "custom_name"
        assert spec.description == "custom desc"

    def test_register_with_dangerous_flag(self):
        registry = ToolRegistry()

        def risky_op(cmd: str) -> str:
            """Run a command"""
            return cmd

        spec = registry.register(risky_op, dangerous=True, timeout=60.0)
        assert spec.dangerous is True
        assert spec.timeout == 60.0

    def test_get_existing_tool(self):
        registry = ToolRegistry()

        def foo(x: str) -> str:
            return x

        registry.register(foo)
        spec = registry.get("foo")
        assert spec is not None
        assert spec.name == "foo"

    def test_get_nonexistent_tool(self):
        registry = ToolRegistry()
        assert registry.get("no_such_tool") is None

    def test_to_openai_schema_empty(self):
        registry = ToolRegistry()
        assert registry.to_openai_schema() == []

    def test_to_openai_schema_multiple(self):
        registry = ToolRegistry()

        def read(path: str) -> str:
            return path

        def write(path: str, content: str) -> str:
            return content

        registry.register(read)
        registry.register(write)
        schemas = registry.to_openai_schema()
        assert len(schemas) == 2
        names = [s["function"]["name"] for s in schemas]
        assert "read" in names
        assert "write" in names

    def test_get_dangerous_tools(self):
        registry = ToolRegistry()

        def safe_op(x: str) -> str:
            return x

        def dangerous_op(cmd: str) -> str:
            return cmd

        registry.register(safe_op, dangerous=False)
        registry.register(dangerous_op, dangerous=True)
        dangerous = registry.get_dangerous_tools()
        assert dangerous == ["dangerous_op"]

    def test_get_dangerous_tools_empty(self):
        registry = ToolRegistry()

        def safe_op(x: str) -> str:
            return x

        registry.register(safe_op)
        assert registry.get_dangerous_tools() == []

    def test_overwrite_tool(self):
        """重复注册同名工具会覆盖"""
        registry = ToolRegistry()

        def foo_v1(x: str) -> str:
            return "v1"

        def foo_v2(x: str) -> str:
            return "v2"

        registry.register(foo_v1, name="foo", description="v1")
        registry.register(foo_v2, name="foo", description="v2")
        spec = registry.get("foo")
        assert spec is not None
        assert spec.description == "v2"


# ===========================================================================
# 4. ToolRegistry — _build_schema
# ===========================================================================


class TestBuildSchema:
    def test_no_params(self):
        registry = ToolRegistry()

        def no_args() -> str:
            return "ok"

        schema = registry._build_schema(no_args)
        assert schema["type"] == "object"
        assert schema["properties"] == {}
        assert schema["required"] == []

    def test_skip_self_param(self):
        registry = ToolRegistry()

        class Dummy:
            def method(self, path: str) -> str:
                return path

        schema = registry._build_schema(Dummy().method)
        assert "self" not in schema["properties"]
        assert "path" in schema["properties"]

    def test_required_params(self):
        registry = ToolRegistry()

        def func(a: str, b: str, c: str = "default") -> str:
            return f"{a}-{b}-{c}"

        schema = registry._build_schema(func)
        assert schema["required"] == ["a", "b"]
        assert "c" not in schema["required"]

    def test_all_string_type(self):
        """所有参数统一为 string 类型（简化处理）"""
        registry = ToolRegistry()

        def func(path: str, count: int, flag: bool = False) -> str:
            return path

        schema = registry._build_schema(func)
        for prop in schema["properties"].values():
            assert prop["type"] == "string"

    def test_mixed_params(self):
        registry = ToolRegistry()

        def func(required_str: str, optional_str: str = "default") -> str:
            return required_str

        schema = registry._build_schema(func)
        assert schema["required"] == ["required_str"]
        assert set(schema["properties"].keys()) == {"required_str", "optional_str"}


# ===========================================================================
# 5. ToolRegistry — execute
# ===========================================================================


class TestToolRegistryExecute:
    @pytest.mark.anyio
    async def test_execute_success(self):
        registry = ToolRegistry()

        def echo(text: str) -> str:
            return f"echo: {text}"

        registry.register(echo)
        result = await registry.execute("echo", {"text": "hello"})
        assert result.is_error is False
        assert result.output == "echo: hello"

    @pytest.mark.anyio
    async def test_execute_unknown_tool(self):
        registry = ToolRegistry()
        result = await registry.execute("no_such_tool", {})
        assert result.is_error is True
        assert "未知工具" in result.output

    @pytest.mark.anyio
    async def test_execute_tool_without_func(self):
        """注册了但没有 _func 的工具"""
        registry = ToolRegistry()
        spec = ToolSpec(
            name="orphan",
            description="orphan",
            parameters={"type": "object", "properties": {}},
            _func=None,
        )
        registry._tools["orphan"] = spec
        result = await registry.execute("orphan", {})
        assert result.is_error is True
        assert "没有实现" in result.output

    @pytest.mark.anyio
    async def test_execute_raises_exception(self):
        registry = ToolRegistry()

        def blow_up(x: str) -> str:
            raise ValueError("Boom!")

        registry.register(blow_up)
        result = await registry.execute("blow_up", {"x": "test"})
        assert result.is_error is True
        assert "ValueError" in result.output
        assert "Boom!" in result.output

    @pytest.mark.anyio
    async def test_execute_timeout(self):
        """工具超时应返回 is_error=True"""
        registry = ToolRegistry()

        def slow_op(x: str) -> str:
            import time
            time.sleep(5)
            return x

        registry.register(slow_op, timeout=0.1)
        result = await registry.execute("slow_op", {"x": "test"})
        assert result.is_error is True
        assert "超时" in result.output

    @pytest.mark.anyio
    async def test_execute_output_truncation(self):
        """输出超过 4000 字符应被截断"""
        registry = ToolRegistry()

        def big_output(x: str) -> str:
            return "A" * 5000

        registry.register(big_output)
        result = await registry.execute("big_output", {"x": "x"})
        assert result.is_error is False
        assert len(result.output) < 5000
        assert "截断" in result.output
        assert "原输出 5000 字符" in result.output

    @pytest.mark.anyio
    async def test_execute_output_exact_4000_no_truncation(self):
        """恰好 4000 字符不应截断"""
        registry = ToolRegistry()

        def exact_output(x: str) -> str:
            return "B" * 4000

        registry.register(exact_output)
        result = await registry.execute("exact_output", {"x": "x"})
        assert result.is_error is False
        assert "截断" not in result.output
        assert len(result.output) == 4000


# ===========================================================================
# 6. 内置工具 — read
# ===========================================================================


class TestReadTool:
    def test_read_existing_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("line1\nline2\nline3\nline4\nline5\n")
        output = _read_file(str(f))
        assert "line1" in output
        assert "line2" in output
        assert "共 5 行" in output

    def test_read_with_offset_limit(self, tmp_path):
        f = tmp_path / "data.txt"
        lines = [f"line{i}" for i in range(1, 21)]
        f.write_text("\n".join(lines))
        output = _read_file(str(f), offset="5", limit="3")
        assert "line5" in output
        assert "line6" in output
        assert "line7" in output
        assert "line8" not in output  # 第 8 行不应出现

    def test_read_nonexistent_file(self):
        output = _read_file("/tmp/no_such_file_xyz_12345.txt")
        assert "文件不存在" in output

    def test_read_directory_not_file(self, tmp_path):
        output = _read_file(str(tmp_path))
        assert "不是文件" in output

    def test_read_offset_out_of_range(self, tmp_path):
        f = tmp_path / "small.txt"
        f.write_text("line1\nline2\n")
        output = _read_file(str(f), offset="10", limit="5")
        # 第 10 行开始，文件只有 2 行，结果应有 header 但内容为空
        assert "共 2 行" in output

    def test_read_invalid_offset_returns_error(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\n")
        output = _read_file(str(f), offset="invalid", limit="1")
        # 非数字 offset 被 int() 抛出 ValueError，外层捕获返回错误
        assert "读取文件失败" in output

    def test_read_expands_tilde(self, tmp_path, monkeypatch):
        """测试 ~ 展开"""
        f = tmp_path / "home_file.txt"
        f.write_text("content")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr(Path, "expanduser", lambda self: tmp_path / "home_file.txt")
        output = _read_file("~/home_file.txt")
        assert "content" in output

    def test_read_binary_file(self, tmp_path):
        """读取二进制文件不会崩溃，使用 errors='replace'"""
        f = tmp_path / "binary.bin"
        f.write_bytes(b"\x00\x01\x02\xff\xfe")
        output = _read_file(str(f))
        assert "共" in output  # 有 header 就行


# ===========================================================================
# 7. 内置工具 — bash
# ===========================================================================


class TestBashTool:
    def test_simple_command(self):
        output = _bash_command("echo hello world")
        assert "hello world" in output

    def test_command_with_stderr(self):
        output = _bash_command("echo ok && echo err >&2")
        assert "ok" in output
        # stderr 应包含在输出中
        assert "err" in output or "[stderr]" in output

    def test_command_nonzero_exit(self):
        output = _bash_command("exit 1")
        assert "退出码: 1" in output

    def test_command_no_output(self):
        output = _bash_command("true")
        assert output == "(无输出)" or "(无输出)" in output

    def test_command_timeout(self):
        output = _bash_command("sleep 10", timeout_sec="0.1")
        # 超时可能被杀死，预期有超时信息
        assert "超时" in output

    def test_invalid_timeout_falls_back(self):
        """无效 timeout_sec 应回退到默认值 30"""
        output = _bash_command("echo ok", timeout_sec="not_a_number")
        assert "ok" in output

    def test_command_not_found(self):
        output = _bash_command("nonexistent_command_xyz_12345")
        # 应该有错误输出
        assert len(output) > 0

    def test_multiline_output(self):
        output = _bash_command("printf 'a\nb\nc\n'")
        assert "a" in output
        assert "b" in output
        assert "c" in output


# ===========================================================================
# 8. 内置工具 — edit
# ===========================================================================


class TestEditTool:
    def test_single_replacement(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("x = 1\n")
        output = _edit_file(str(f), "x = 1", "x = 2")
        assert "已编辑" in output
        assert "1 处替换" in output
        assert f.read_text() == "x = 2\n"

    def test_multiple_matches_error(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("x = 1\ny = 1\n")
        output = _edit_file(str(f), "1", "2")
        assert "找到 2 处匹配" in output
        # 文件内容不应改变
        assert f.read_text() == "x = 1\ny = 1\n"

    def test_no_match_error(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("hello world\n")
        output = _edit_file(str(f), "no_such_text", "replacement")
        assert "未找到匹配文本" in output
        assert f.read_text() == "hello world\n"

    def test_file_not_found(self):
        output = _edit_file("/tmp/no_such_edit_file.txt", "old", "new")
        assert "文件不存在" in output

    def test_path_is_directory(self, tmp_path):
        output = _edit_file(str(tmp_path), "old", "new")
        assert "不是文件" in output

    def test_replace_unique_by_context(self, tmp_path):
        """通过更多上下文使匹配唯一"""
        f = tmp_path / "code.py"
        f.write_text("x = 1  # first\nx = 1  # second\n")
        output = _edit_file(str(f), "x = 1  # first", "y = 2")
        assert "已编辑" in output
        assert "y = 2" in f.read_text()
        assert "x = 1  # second" in f.read_text()

    def test_multiline_replacement(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("line1\nline2\nline3\n")
        output = _edit_file(str(f), "line1\nline2", "new1\nnew2")
        assert "已编辑" in output
        assert f.read_text() == "new1\nnew2\nline3\n"


# ===========================================================================
# 9. 内置工具 — write
# ===========================================================================


class TestWriteTool:
    def test_write_new_file(self, tmp_path):
        f = tmp_path / "new_file.txt"
        output = _write_file(str(f), "hello world")
        assert "已写入" in output
        assert f.read_text() == "hello world"

    def test_write_creates_parent_dirs(self, tmp_path):
        f = tmp_path / "deeply" / "nested" / "dir" / "file.txt"
        output = _write_file(str(f), "nested content")
        assert "已写入" in output
        assert f.read_text() == "nested content"

    def test_write_overwrites_existing(self, tmp_path):
        f = tmp_path / "existing.txt"
        f.write_text("old content")
        output = _write_file(str(f), "new content")
        assert "已写入" in output
        assert f.read_text() == "new content"

    def test_write_empty_content(self, tmp_path):
        f = tmp_path / "empty.txt"
        output = _write_file(str(f), "")
        assert "已写入" in output
        assert "(0 字符)" in output
        assert f.read_text() == ""

    def test_write_special_characters(self, tmp_path):
        f = tmp_path / "unicode.txt"
        content = "你好世界 🌍\néàç\n"
        output = _write_file(str(f), content)
        assert "已写入" in output
        assert f.read_text(encoding="utf-8") == content


# ===========================================================================
# 10. 内置工具 — grep
# ===========================================================================


class TestGrepTool:
    def test_grep_single_file(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("def foo():\n    return 42\n")
        output = _grep_search("def foo", path=str(f))
        assert "def foo" in output
        assert "code.py" in output or str(f) in output

    def test_grep_directory(self, tmp_path):
        (tmp_path / "a.py").write_text("import os\nprint('hello')\n")
        (tmp_path / "b.py").write_text("import sys\nprint('world')\n")
        output = _grep_search("import", path=str(tmp_path))
        assert "import os" in output or "import sys" in output

    def test_grep_no_match(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("hello world\n")
        output = _grep_search("XYZ_NO_MATCH_123", path=str(f))
        assert "未找到匹配模式" in output

    def test_grep_with_include_filter(self, tmp_path):
        (tmp_path / "a.py").write_text("TODO: fix this\n")
        (tmp_path / "b.txt").write_text("TODO: fix that\n")
        output = _grep_search("TODO", path=str(tmp_path), include="*.py")
        assert "a.py" in output
        assert "b.txt" not in output

    def test_grep_nonexistent_path(self):
        output = _grep_search("pattern", path="/tmp/no_such_grep_dir")
        assert "路径不存在" in output

    def test_grep_regex_pattern_case_sensitive(self, tmp_path):
        """默认大小写敏感：error 匹配，ERROR 不匹配（rror vs RROR）"""
        f = tmp_path / "log.txt"
        f.write_text("error: disk full\ninfo: started\nERROR: timeout\n")
        output = _grep_search(r"error", path=str(f))
        assert "disk full" in output
        # error（小写）匹配，ERROR（大写）不匹配
        assert "timeout" not in output

    def test_grep_regex_pattern_ignore_case(self, tmp_path):
        """用 (?i) 前缀实现大小写不敏感"""
        f = tmp_path / "log.txt"
        f.write_text("error: disk full\ninfo: started\nERROR: timeout\n")
        output = _grep_search(r"(?i)error", path=str(f))
        assert "disk full" in output
        assert "timeout" in output

    def test_grep_result_limit(self, tmp_path):
        """结果超过 100 条应截断"""
        f = tmp_path / "many.txt"
        lines = "\n".join([f"match_{i}" for i in range(150)])
        f.write_text(lines)
        output = _grep_search("match_", path=str(f))
        # 限制 100 条结果
        result_lines = output.strip().split("\n")
        assert len(result_lines) <= 100


# ===========================================================================
# 11. 内置工具 — find
# ===========================================================================


class TestFindTool:
    def test_find_py_files(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        output = _find_files("*.py", path=str(tmp_path))
        assert "a.py" in output
        assert "b.py" in output
        assert "c.txt" not in output

    def test_find_no_match(self, tmp_path):
        output = _find_files("*.xyz", path=str(tmp_path))
        assert "未找到匹配" in output

    def test_find_single_file(self, tmp_path):
        f = tmp_path / "hello.py"
        f.write_text("")
        output = _find_files("hello.py", path=str(f))
        assert str(f) in output

    def test_find_nonexistent_path(self):
        output = _find_files("*", path="/tmp/no_such_find_dir_xyz")
        assert "路径不存在" in output

    def test_find_recursive(self, tmp_path):
        nested = tmp_path / "sub" / "deep"
        nested.mkdir(parents=True)
        (nested / "deep_file.py").write_text("")
        output = _find_files("*.py", path=str(tmp_path))
        assert "deep_file.py" in output

    def test_find_result_limit(self, tmp_path):
        """结果超过 200 条应截断"""
        for i in range(250):
            (tmp_path / f"file_{i}.txt").write_text("")
        output = _find_files("*.txt", path=str(tmp_path))
        result_lines = output.strip().split("\n")
        assert len(result_lines) <= 200


# ===========================================================================
# 12. 内置工具 — ls
# ===========================================================================


class TestLsTool:
    def test_ls_directory(self, tmp_path):
        (tmp_path / "file1.txt").write_text("hello")
        (tmp_path / "file2.txt").write_text("world")
        (tmp_path / "subdir").mkdir()
        output = _list_directory(str(tmp_path))
        assert "file1.txt" in output
        assert "file2.txt" in output
        assert "subdir/" in output
        assert "共" in output

    def test_ls_empty_directory(self, tmp_path):
        output = _list_directory(str(tmp_path))
        assert "共 0 项" in output

    def test_ls_file(self, tmp_path):
        f = tmp_path / "single.txt"
        f.write_text("data")
        output = _list_directory(str(f))
        assert "single.txt" in output
        assert "bytes" in output

    def test_ls_nonexistent_path(self):
        output = _list_directory("/tmp/no_such_ls_dir_xyz")
        assert "路径不存在" in output

    def test_ls_default_cwd(self):
        """默认列出当前目录"""
        output = _list_directory()
        assert len(output) > 0


# ===========================================================================
# 13. 内置工具注册
# ===========================================================================


class TestBuiltinRegistry:
    def test_register_builtin_tools(self):
        registry = ToolRegistry()
        registry = register_builtin_tools(registry)
        specs = registry.to_openai_schema()
        assert len(specs) == 7

    def test_create_default_registry(self):
        registry = create_default_registry()
        # 7 个工具都已注册
        for name in ["read", "bash", "edit", "write", "grep", "find", "ls"]:
            assert registry.get(name) is not None

    def test_default_registry_dangerous(self):
        registry = create_default_registry()
        dangerous = registry.get_dangerous_tools()
        assert "bash" in dangerous
        # 其他工具不应标记为危险
        assert "read" not in dangerous

    def test_default_registry_to_openai_schema(self):
        registry = create_default_registry()
        schemas = registry.to_openai_schema()
        names = {s["function"]["name"] for s in schemas}
        assert names == {"read", "bash", "edit", "write", "grep", "find", "ls"}


# ===========================================================================
# 14. 集成测试 —— 工具链组合
# ===========================================================================


class TestToolIntegration:
    def test_write_then_read(self, tmp_path):
        """write → read 完整工作流"""
        f = tmp_path / "data.txt"
        write_output = _write_file(str(f), "line1\nline2\nline3\n")
        assert "已写入" in write_output
        read_output = _read_file(str(f))
        assert "line1" in read_output
        assert "共 3 行" in read_output

    def test_write_then_edit_then_read(self, tmp_path):
        """write → edit → read 完整工作流"""
        f = tmp_path / "code.py"
        _write_file(str(f), "x = 1\ny = 2\n")
        _edit_file(str(f), "x = 1", "x = 99")
        read_output = _read_file(str(f))
        assert "x = 99" in read_output
        assert "y = 2" in read_output

    def test_write_then_grep(self, tmp_path):
        """write → grep 完整工作流"""
        f = tmp_path / "log.txt"
        _write_file(str(f), "INFO: started\nERROR: failed\nINFO: done\n")
        grep_output = _grep_search("ERROR", path=str(f))
        assert "ERROR: failed" in grep_output

    def test_write_then_find(self, tmp_path):
        """write → find 完整工作流"""
        _write_file(str(tmp_path / "a.py"), "")
        _write_file(str(tmp_path / "b.py"), "")
        _write_file(str(tmp_path / "c.txt"), "")
        find_output = _find_files("*.py", path=str(tmp_path))
        assert "a.py" in find_output
        assert "b.py" in find_output
        assert "c.txt" not in find_output

    def test_write_then_ls(self, tmp_path):
        """write → ls 完整工作流"""
        _write_file(str(tmp_path / "test.txt"), "content")
        ls_output = _list_directory(str(tmp_path))
        assert "test.txt" in ls_output


# ===========================================================================
# 15. 边界测试
# ===========================================================================


class TestEdgeCases:
    def test_read_large_file(self, tmp_path):
        """大文件读取（10MB）—— 确认返回结果不崩溃"""
        f = tmp_path / "large.txt"
        # 生成 ~100K 行文本
        lines = "\n".join([f"line_{i:06d}" for i in range(100000)])
        f.write_text(lines)
        output = _read_file(str(f), offset="1", limit="100")
        assert "共 100000 行" in output
        # 只返回 limit 行
        assert len(output.split("\n")) <= 102  # 1 header + 100 lines + possible newline

    def test_grep_binary_file(self, tmp_path):
        """二进制文件 grep 不崩溃"""
        f = tmp_path / "binary.bin"
        f.write_bytes(b"\x00\x01\x02hello\xff\xfe")
        output = _grep_search("hello", path=str(f))
        # 应该能找到或用 errors='replace' 不崩溃
        assert isinstance(output, str)

    def test_ls_permission_error(self, tmp_path, monkeypatch):
        """模拟权限错误"""

        def mock_iterdir(_self):
            raise PermissionError("denied")

        monkeypatch.setattr(Path, "iterdir", mock_iterdir)
        output = _list_directory("/root")
        assert "无权限" in output

    def test_write_permission_error(self, tmp_path, monkeypatch):
        """写入权限错误返回错误信息"""

        def mock_mkdir(_self, **kwargs):
            raise PermissionError("denied")

        monkeypatch.setattr(Path, "mkdir", mock_mkdir)
        # 需要确保父目录不存在触发 mkdir
        output = _write_file("/root/denied_sub/file.txt", "data")
        assert "写入文件失败" in output

    def test_edit_large_file(self, tmp_path):
        """大文件的编辑"""
        f = tmp_path / "large.py"
        f.write_text("A" * 100000 + "UNIQUE_MARKER" + "B" * 100000)
        output = _edit_file(str(f), "UNIQUE_MARKER", "REPLACED")
        assert "已编辑" in output
        assert "UNIQUE_MARKER" not in f.read_text()
        assert "REPLACED" in f.read_text()

    def test_bash_empty_command(self):
        """空命令不应崩溃"""
        output = _bash_command("")
        # 返回输出（可能是空或错误）
        assert isinstance(output, str)

    def test_read_with_negative_offset(self, tmp_path):
        """负数 offset 自动调整为 1"""
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\n")
        output = _read_file(str(f), offset="-5", limit="1")
        assert "line1" in output

    def test_tool_result_non_string_return(self, tmp_path):
        """测试工具返回非字符串时的处理（通过 execute）"""
        registry = ToolRegistry()

        def return_int(x: str) -> int:
            return 42

        registry.register(return_int)
        # 同步调用
        import asyncio
        result = asyncio.run(registry.execute("return_int", {"x": "test"}))
        assert result.is_error is False
        assert "42" in result.output


# ===========================================================================
# 16. 危险确认流程测试
# ===========================================================================


class TestDangerousToolConfirmation:
    """测试危险工具确认逻辑。
    注意：确认逻辑在 agent.py 中实现，这里只验证 tools.py 的正确标记。"""

    def test_bash_is_marked_dangerous(self):
        registry = create_default_registry()
        assert "bash" in registry.get_dangerous_tools()

    def test_read_is_not_dangerous(self):
        registry = create_default_registry()
        assert "read" not in registry.get_dangerous_tools()

    def test_edit_is_not_dangerous_by_default(self):
        registry = create_default_registry()
        # edit 默认不标记为 dangerous
        assert "edit" not in registry.get_dangerous_tools()

    def test_write_is_not_dangerous_by_default(self):
        registry = create_default_registry()
        assert "write" not in registry.get_dangerous_tools()

    def test_custom_dangerous_tool(self):
        registry = ToolRegistry()

        def risky(cmd: str) -> str:
            return cmd

        registry.register(risky, dangerous=True)
        assert "risky" in registry.get_dangerous_tools()
        spec = registry.get("risky")
        assert spec is not None
        assert spec.dangerous is True


# ===========================================================================
# 17. execute 带危险标记的集成
# ===========================================================================


class TestExecuteDangerous:
    @pytest.mark.anyio
    async def test_execute_dangerous_tool(self):
        """危险工具执行本身不阻止，只是在 registry 中标记。确认在 agent 层。"""
        registry = ToolRegistry()

        def rm_cmd(path: str) -> str:
            return f"would remove {path}"

        registry.register(rm_cmd, dangerous=True, name="rm")
        result = await registry.execute("rm", {"path": "/tmp/test"})
        # execute 不检查 dangerous 标记，只执行
        assert result.is_error is False
        assert "would remove" in result.output


# ===========================================================================
# 18. 并发安全 —— 注册不冲突
# ===========================================================================


class TestConcurrency:
    @pytest.mark.anyio
    async def test_concurrent_registration(self):
        """并发注册不会互相影响"""
        registry = ToolRegistry()

        async def register_tool(name: str):
            def tool(x: str) -> str:
                return f"{name}: {x}"

            registry.register(tool, name=name)

        await asyncio.gather(
            register_tool("tool_a"),
            register_tool("tool_b"),
            register_tool("tool_c"),
        )
        assert len(registry._tools) == 3
        for name in ["tool_a", "tool_b", "tool_c"]:
            assert registry.get(name) is not None

    @pytest.mark.anyio
    async def test_concurrent_execution(self):
        """并发执行不同工具"""
        registry = ToolRegistry()

        def slow_echo(x: str) -> str:
            import time
            time.sleep(0.05)
            return f"echo: {x}"

        registry.register(slow_echo, name="echo")
        results = await asyncio.gather(
            registry.execute("echo", {"x": "a"}),
            registry.execute("echo", {"x": "b"}),
            registry.execute("echo", {"x": "c"}),
        )
        assert all(not r.is_error for r in results)
        outputs = [r.output for r in results]
        assert "echo: a" in outputs
        assert "echo: b" in outputs
        assert "echo: c" in outputs
