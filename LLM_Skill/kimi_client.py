"""KimiClient - 类似 anthropic.Anthropic() 的 Python SDK 封装"""

import subprocess
import os
import re
from typing import List, Dict, Optional, Generator, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


class BlockType(Enum):
    """内容块类型"""
    TURN_BEGIN = "TurnBegin"
    TURN_END = "TurnEnd"
    STEP_BEGIN = "StepBegin"
    STEP_END = "StepEnd"
    THINK_PART = "ThinkPart"
    TEXT_PART = "TextPart"
    TOOL_BEGIN = "ToolBegin"
    TOOL_END = "ToolEnd"
    TOOL_RESULT = "ToolResult"
    FILE_OP = "FileOperation"
    CODE = "CodeBlock"
    TEXT = "Text"
    STATUS_UPDATE = "StatusUpdate"
    UNKNOWN = "Unknown"


@dataclass
class ContentBlock:
    """内容块 - 类似 anthropic 的 ContentBlock"""
    type: str  # "text", "thinking", "tool_use", etc.
    text: str = ""
    thinking: str = ""
    tool_name: str = ""
    tool_input: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    data: Dict[str, Any] = field(default_factory=dict)  # 额外的元数据
    
    def __str__(self) -> str:
        return self.text or self.thinking or self.raw
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "text": self.text,
            "thinking": self.thinking,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "data": self.data,
        }


@dataclass
class Content:
    """响应内容 - 类似 anthropic.Content"""
    raw: str
    blocks: List[ContentBlock] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.blocks:
            self.blocks = ProtocolParser.parse(self.raw)
    
    @property
    def text(self) -> str:
        """获取主要文本内容"""
        texts = [b.text for b in self.blocks if b.text]
        return "\n\n".join(texts) if texts else self.raw
    
    @property
    def thinking_content(self) -> str:
        """获取思考内容"""
        thinks = [b.thinking for b in self.blocks if b.thinking]
        return "\n\n".join(thinks)
    
    def __str__(self) -> str:
        return self.text
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "blocks": [b.to_dict() for b in self.blocks],
        }


@dataclass 
class Message:
    """消息对象 - 类似 anthropic.Message"""
    content: Content
    model: str = "kimi-k2.5"
    role: str = "assistant"
    stop_reason: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return str(self.content)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content.to_dict(),
            "model": self.model,
            "role": self.role,
            "stop_reason": self.stop_reason,
            "usage": self.usage,
        }


class AnsiStripper:
    """去除 ANSI 颜色代码"""
    ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
    
    @classmethod
    def strip(cls, text: str) -> str:
        return cls.ANSI_RE.sub('', text)


class ProtocolParser:
    """kimi-cli 协议解析器"""
    
    @classmethod
    def parse(cls, text: str) -> List[ContentBlock]:
        """解析文本为 ContentBlock 列表"""
        clean_text = AnsiStripper.strip(text)
        blocks = []
        pos = 0
        
        while pos < len(clean_text):
            while pos < len(clean_text) and clean_text[pos] in ' \n\t':
                pos += 1
            
            if pos >= len(clean_text):
                break
            
            # 尝试解析协议消息
            block, new_pos = cls._parse_message(clean_text, pos)
            if block:
                blocks.append(block)
                pos = new_pos
                continue
            
            # 作为普通文本
            text_end = cls._find_next_protocol(clean_text, pos)
            if text_end > pos:
                content = clean_text[pos:text_end].strip()
                if content:
                    blocks.append(ContentBlock(type="text", text=content, raw=content))
                pos = text_end
            else:
                pos += 1
        
        return blocks
    
    @classmethod
    def _find_next_protocol(cls, text: str, start: int) -> int:
        match = re.search(r'\n\s*[A-Z][a-zA-Z]*\s*\(', text[start:])
        return start + match.start() if match else len(text)
    
    @classmethod
    def _parse_message(cls, text: str, start: int) -> Tuple[Optional[ContentBlock], int]:
        match = re.match(r'([A-Z][a-zA-Z0-9]*)\s*\(', text[start:])
        if not match:
            return None, start
        
        name = match.group(1)
        pos = start + match.end() - 1
        
        content, end_pos = cls._parse_parens(text, pos)
        if content is None:
            return None, start
        
        args_str = content[1:-1]
        args = cls._parse_args(args_str)
        raw = text[start:end_pos]
        
        # 转换为 ContentBlock
        return cls._to_content_block(name, args, raw), end_pos
    
    @classmethod
    def _parse_parens(cls, text: str, start: int) -> Tuple[Optional[str], int]:
        if text[start] != '(':
            return None, start
        
        depth = 0
        i = start
        
        while i < len(text):
            c = text[i]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    return text[start:i+1], i + 1
            elif c in "'\"":
                i += 1
                while i < len(text):
                    if text[i] == '\\':
                        i += 2
                    elif text[i] == c:
                        break
                    else:
                        i += 1
            i += 1
        
        return None, start
    
    @classmethod
    def _parse_args(cls, text: str) -> Dict[str, Any]:
        args = {}
        i = 0
        
        while i < len(text):
            while i < len(text) and text[i] in ' \n\t':
                i += 1
            
            if i >= len(text):
                break
            
            key_match = re.match(r'([a-zA-Z_]\w*)\s*=', text[i:])
            if not key_match:
                break
            
            key = key_match.group(1)
            i += key_match.end()
            
            while i < len(text) and text[i] in ' \n\t':
                i += 1
            
            value, consumed = cls._parse_value(text[i:])
            i += consumed
            args[key] = value
            
            while i < len(text) and text[i] in ', \n\t':
                i += 1
        
        return args
    
    @classmethod
    def _parse_value(cls, text: str) -> Tuple[Any, int]:
        text = text.lstrip()
        if not text:
            return None, 0
        
        if text[0] in "'\"":
            quote = text[0]
            result = []
            i = 1
            while i < len(text):
                if text[i] == '\\':
                    result.append(text[i+1] if i+1 < len(text) else '\\')
                    i += 2
                elif text[i] == quote:
                    return ''.join(result), i + 1
                else:
                    result.append(text[i])
                    i += 1
            return ''.join(result), len(text)
        
        obj_match = re.match(r'([A-Z][a-zA-Z0-9]*)\s*\(', text)
        if obj_match:
            obj_str, consumed = cls._parse_parens(text, obj_match.end() - 1)
            if obj_str:
                inner = cls._parse_args(obj_str[1:-1])
                return {"_type": obj_match.group(1), **inner}, consumed
        
        for word, val in [('None', None), ('True', True), ('False', False)]:
            if text.startswith(word):
                return val, len(word)
        
        num_match = re.match(r'-?\d+\.?\d*', text)
        if num_match:
            num_str = num_match.group()
            return (float if '.' in num_str else int)(num_str), len(num_str)
        
        end = 0
        while end < len(text) and text[end] not in ',)':
            end += 1
        return text[:end].strip(), end
    
    @classmethod
    def _to_content_block(cls, name: str, args: Dict[str, Any], raw: str) -> ContentBlock:
        """转换为 ContentBlock"""
        if name == 'ThinkPart':
            return ContentBlock(
                type="thinking",
                thinking=args.get('think', ''),
                raw=raw,
                data=args
            )
        elif name == 'TextPart':
            return ContentBlock(
                type="text",
                text=args.get('text', ''),
                raw=raw,
                data=args
            )
        elif name in ('ToolBegin', 'ToolUse'):
            return ContentBlock(
                type="tool_use",
                tool_name=args.get('tool', args.get('name', '')),
                tool_input=args,
                raw=raw,
                data=args
            )
        elif name == 'ToolResult':
            return ContentBlock(
                type="tool_result",
                text=args.get('result', ''),
                raw=raw,
                data=args
            )
        else:
            # 其他类型作为元数据块
            return ContentBlock(
                type=name.lower(),
                raw=raw,
                data=args
            )


class Messages:
    """Messages API - 类似 anthropic.messages"""
    
    def __init__(self, client: "KimiClient"):
        self.client = client
    
    def create(
        self,
        model: str = "kimi-k2.5",
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
        stream: bool = False,
        timeout: int = 120,
    ) -> Union[Message, Generator[str, None, None]]:
        """
        发送消息并获取回复 - 类似 anthropic.messages.create()
        
        Args:
            model: 模型名称
            messages: 消息列表 [{"role": "user", "content": "..."}]
            max_tokens: 最大 token 数（暂不支持）
            system: 系统提示
            stream: 是否流式输出
            timeout: 超时时间
        
        Returns:
            Message 对象 或 流式生成器
        """
        messages = messages or []
        
        # 构建提示词
        prompt_parts = []
        if system:
            prompt_parts.append(f"[系统指令] {system}")
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"[系统] {content}")
            elif role == "user":
                prompt_parts.append(content)
            elif role == "assistant":
                prompt_parts.append(f"[助手] {content}")
        
        prompt = "\n\n".join(prompt_parts)
        
        cmd = [
            "kimi", "--print", "--yes",
            "--work-dir", self.client.work_dir,
            "--command", prompt
        ]
        
        if self.client.skills_dir and os.path.isdir(self.client.skills_dir):
            cmd.extend(["--skills-dir", self.client.skills_dir])
        
        if stream:
            return self._stream(cmd, timeout)
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"Kimi 调用失败: {result.stderr}")
        
        return Message(
            content=Content(raw=result.stdout),
            model=model
        )
    
    def _stream(self, cmd: List[str], timeout: int) -> Generator[str, None, None]:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in proc.stdout:
            yield line
        proc.wait(timeout=timeout)


class KimiClient:
    """
    Kimi CLI 客户端 - 类似 anthropic.Anthropic()
    
    使用示例:
        client = KimiClient()
        
        # 方式1: 简单对话
        response = client.chat("你好")
        print(response)
        
        # 方式2: messages API
        response = client.messages.create(
            messages=[{"role": "user", "content": "你好"}]
        )
        print(response.content)
    """
    
    def __init__(self, work_dir: Optional[str] = None, skills_dir: Optional[str] = None):
        self.work_dir = work_dir or os.getcwd()
        self.skills_dir = skills_dir
        self.messages = Messages(self)
        self._check_kimi()
    
    def _check_kimi(self):
        if subprocess.run(["which", "kimi"], capture_output=True).returncode != 0:
            raise RuntimeError("kimi 命令未找到，请先安装 kimi-cli")
    
    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False,
        timeout: int = 120
    ) -> Union[str, Generator[str, None, None]]:
        """
        简单对话接口
        
        Args:
            prompt: 用户提示
            system: 系统提示
            stream: 是否流式输出
            timeout: 超时时间
        
        Returns:
            回复文本 或 流式生成器
        """
        if stream:
            return self.messages.create(
                messages=[{"role": "user", "content": prompt}],
                system=system,
                stream=True,
                timeout=timeout
            )
        
        response = self.messages.create(
            messages=[{"role": "user", "content": prompt}],
            system=system,
            stream=False,
            timeout=timeout
        )
        return response.content.text


def create_client(work_dir: Optional[str] = None, skills_dir: Optional[str] = None) -> KimiClient:
    """创建客户端"""
    return KimiClient(work_dir, skills_dir)


# 测试
if __name__ == "__main__":
    sample = """
什么是深度学习?\n\x1b[1;35mTurnBegin\x1b[0m(\x1b[33muser_input\x1b[0m=\x1b[32m'什么是深度学习?'\x1b[0m)
\x1b[1;35mStepBegin\x1b[0m(\x1b[33mn\x1b[0m=\x1b[1;36m1\x1b[0m)
\x1b[1;35mThinkPart\x1b[0m(
    \x1b[33mtype\x1b[0m=\x1b[32m'think'\x1b[0m,
    \x1b[33mthink\x1b[0m=\x1b[32m'用户问"什么是深度学习?"，这是一个关于深度学习概念的简单问题。'\x1b[0m
)
\x1b[1;35mTextPart\x1b[0m(
    \x1b[33mtype\x1b[0m=\x1b[32m'text'\x1b[0m,
    \x1b[33mtext\x1b[0m=\x1b[32m'**深度学习（Deep Learning）** 是机器学习的一个分支。'\x1b[0m
)
\x1b[1;35mTurnEnd\x1b[0m()
"""
    
    print("=== 测试 ===\n")
    
    # 测试 Content 解析
    content = Content(raw=sample)
    print(f"总块数: {len(content.blocks)}")
    print(f"文本内容: {content.text[:80]}...")
    print(f"思考内容: {content.thinking_content[:80]}...")
    print()
    
    # 测试 blocks
    for i, block in enumerate(content.blocks):
        print(f"{i+1}. {block.type}")
        if block.text:
            print(f"   text: {block.text[:60]}...")
        if block.thinking:
            print(f"   thinking: {block.thinking[:60]}...")
