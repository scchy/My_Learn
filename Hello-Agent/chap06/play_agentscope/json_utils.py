# -*- coding: utf-8 -*-
"""JSON 解析工具函数"""
import json
import re
from typing import Type, Optional, Any
from pydantic import BaseModel, ValidationError


def extract_json_from_text(text: str) -> Optional[dict]:
    """从文本中提取 JSON 对象"""
    if not text:
        return None
    
    # 尝试直接解析整个文本
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # 尝试提取代码块中的 JSON
    code_block_pattern = r'```(?:json)?\s*(.*?)\s*```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # 尝试提取花括号中的 JSON
    brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(brace_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    return None


def parse_structured_output(text: Any, model_class: Type[BaseModel]) -> Optional[BaseModel]:
    """解析结构化输出
    
    Args:
        text: 模型输出的文本（可以是 str 或其他类型）
        model_class: Pydantic 模型类
        
    Returns:
        解析后的模型实例，解析失败返回 None
    """
    try:
        # 处理 content 可能是 str 或其他类型的情况
        text_str = str(text) if not isinstance(text, str) else text
        data = extract_json_from_text(text_str)
        if data is None:
            return None
        
        return model_class(**data)
    except Exception:
        # 静默处理解析失败，避免输出干扰信息
        return None


def get_json_prompt_instructions(model_class: Type[BaseModel]) -> str:
    """获取 JSON 格式的提示说明
    
    Args:
        model_class: Pydantic 模型类
        
    Returns:
        提示字符串
    """
    schema = model_class.model_json_schema()
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    fields_desc = []
    for field_name, field_info in properties.items():
        desc = field_info.get("description", "")
        field_type = field_info.get("type", "any")
        is_required = "(必填)" if field_name in required else "(可选)"
        fields_desc.append(f'  "{field_name}": <{field_type}> {desc} {is_required}')
    
    fields_str = "\n".join(fields_desc)
    
    return f"""
【重要】请严格按照以下 JSON 格式回复，不要添加任何其他文字：
{{
{fields_str}
}}
"""
