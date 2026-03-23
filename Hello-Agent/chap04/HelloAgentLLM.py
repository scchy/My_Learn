# python
# Create Date: 2026-03-22
# Author: Scc_hy
# Func: Hello Agent 
# ============================================================

import os 
import inspect
from openai import OpenAI
from typing import List, Dict, Optional, Any, Callable, Annotated
from typing import get_type_hints, get_origin, get_args 
from serpapi import SerpApiClient
import numexpr as ne


class HelloAgentsLLM:
    def __init__(
            self, 
            model: Optional[str] = None, 
            api_key: Optional[str] = None,
            base_url: Optional[str] = None, 
            timeout: Optional[int] = None
        ):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        """
        self.model = model if model else 'qwen2.5-7B'
        if api_key is None:
            base_url = "http://localhost:8088/v1"
            api_key = "sk-test"
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
        )
    
    def think_with_fc(self, messages, tools_schema, temperature: float = 0.0):
        print(f"🧠 正在调用 {self.model} 模型(with fucntion call)...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools_schema,
                tool_choice="auto",  # 让模型自己决定
                temperature=temperature 
            )
            message = response.choices[0].message
            return message
        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None

            
    def think(self,  messages, temperature: float = 0.0):
        """
        调用大语言模型进行思考，并返回其响应。
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True
            )
            
            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None


def test_LLM():
    try:
        llmClient = HelloAgentsLLM()
        
        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]
        
        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)


class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    """
    
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        # 硬编码分类（符合你的业务：代码 + 数据 + 计算）
        self.hierarchy = {
            "code": {
                "desc": "代码开发与审查",
                "keywords": ["代码", "函数", "bug", "review", "python", "sql"],
                "tools": ["code_review", "generate_sql", "explain_code", "git_diff"]
            },
            "data": {
                "desc": "数据分析与查询", 
                "keywords": ["数据", "查询", "统计", "table", "db", "指标"],
                "tools": ["sql_executor", "pandas_analysis", "data_visualization", "search_data"]
            },
            "calc": {
                "desc": "数学计算与算法",
                "keywords": ["计算", "公式", "数学", "统计检验", "预测"],
                "tools": ["calculate", "statistical_test", "ml_predict", "optimize"]
            }
        }
    
    def registerTool(self, name: str, description: str, func: Callable):
        """向工具箱中注册一个新工具。

        Args:
            name (str): 工具名称
            description (str): 工具描述
            func (Callable): 工具方法
        """
        if name in self.tools:
            print(f"警告:工具 '{name}' 已存在，将被覆盖。")
        self.tools[name] = {
            "description": description, 
            "func": func
        }
        print(f"工具 '{name}' 已注册。")
    
    def getTool(self, name: str):
        return self.tools.get(name, {}).get("func")
    
    def getAllTools(self, user_query: Optional[str]=None):
        if user_query is not None:
            return self._get_dynamic_tools(user_query)
        return self._get_all_tools()

    def _get_all_tools(self):
        return [
            (name, info['description'], info['func']) for name, info in self.tools.items()
        ]

    def route(self, user_query: str) -> list:
        """简单关键词路由（无需模型，零延迟）"""
        query = user_query.lower()
        
        # 匹配分类（可多选）
        matched_categories = []
        for cat, meta in self.hierarchy.items():
            if any(kw in query for kw in meta["keywords"]):
                matched_categories.append(cat)
        
        # 默认 fallback
        if not matched_categories:
            matched_categories = ["code"]  # 默认代码类（你的场景）
        
        # 合并工具（去重）
        tools = []
        for cat in matched_categories:
            tools.extend(self.hierarchy[cat]["tools"])
        
        # 去重后限制数量（防止跨类过多）
        return list(dict.fromkeys(tools))[:8]  # 最多 8 个工具给模型选
    
    def _get_dynamic_tools(self, user_query: str):
        return [
            (name, self.tools[name]['description'], self.tools[name]['func']) for name in self.route(user_query)
        ]
    
    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        """
        return '\n'.join(
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        )

    def build_tools_schema(self, user_query: Optional[str]=None) -> List[Dict[str, Any]]:
        """
        函数定义示例
            def search(query: Annotated[str, "搜索关键词"]) -> Annotated[str, "搜索结果"]:
        """
        schemas = []
        for tool_name, tool_desc, tool_func in self.getAllTools(user_query):
            signature = inspect.signature(tool_func)
            # 处理 Annotated 类型需要特殊逻辑
            raw_hints = get_type_hints(tool_func, include_extras=True)  # 关键：include_extras=True
            # 2. 构建 parameters.properties
            properties = {}
            required = []
            for param_name, param in signature.parameters.items():
                if param_name in ('self', 'cls'):
                    continue
                
                # 获取原始类型和 Annotated 的 metadata
                hint = raw_hints.get(param_name, str)
                param_type = hint
                param_desc = f"Parameter: {param_name}"
                
                # 解析 Annotated[T, "description"]
                origin = get_origin(hint)
                if origin is not None and origin is Annotated:
                    args = get_args(hint)
                    param_type = args[0]  # 实际类型
                    if len(args) > 1:
                        param_desc = args[1]  # 描述字符串
                
                # 类型映射
                type_mapping = self._get_json_schema_type(param_type)
                
                properties[param_name] = {
                    "type": type_mapping["type"],
                    "description": param_desc
                }
                if type_mapping.get("items"):
                    properties[param_name]["items"] = type_mapping["items"]
                
                # 必填判断
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)
            
            schema = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_desc,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required if required else list(properties.keys())
                    }
                }
            }
            schemas.append(schema)
        
        return schemas

    def _get_json_schema_type(self, python_type) -> Dict[str, Any]:
        """
        Python 类型转 JSON Schema 类型
        """
        type_map = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            list: {"type": "array", "items": {"type": "string"}},
            dict: {"type": "object"},
            Any: {"type": "string"},
        }
        
        # 处理 List[str], List[int] 等泛型
        origin = getattr(python_type, '__origin__', None)
        if origin is list:
            args = getattr(python_type, '__args__', (str,))
            item_type = args[0] if args else str
            return {
                "type": "array",
                "items": type_map.get(item_type, {"type": "string"})
            }
        
        return type_map.get(python_type, {"type": "string"})


def calculate(expression: Annotated[str, "数学表达式字符串"]) -> str:
    """
    安全的数学计算器，支持复杂表达式。
    
    Args:
        expression: 数学表达式字符串，如 "(123 + 456) * 789 / 12"
                   支持: + - * / // % ** ( ) 
    """
    try:
        # 清理表达式：替换中文符号为英文，移除空格
        clean_expr = expression.replace('×', '*').replace('÷', '/').replace('x', '*').replace('X', '*')
        clean_expr = clean_expr.replace(' ', '').replace('=', '')
        
        # 使用 numexpr 安全计算（无代码注入风险）
        result = ne.evaluate(clean_expr).item()
        
        # 格式化输出（整数显示整数，小数保留4位）
        if isinstance(result, float) and result.is_integer():
            return f"计算结果: {int(result)}"
        return f"计算结果: {result:.4f}"
        
    except ZeroDivisionError:
        return "错误: 除数不能为零"
    except SyntaxError:
        return "错误: 表达式语法错误，请检查括号匹配"
    except Exception as e:
        return f"计算错误: {str(e)}"


def search(query: Annotated[str, "搜索关键词"]) -> Annotated[str, "搜索结果"]:
    """
    一个基于SerpApi的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "错误:SERPAPI_API_KEY 未在 .env 文件中配置。"

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",  # 国家代码
            "hl": "zh-cn", # 语言代码
        }
        
        client = SerpApiClient(params)
        results = client.get_dict()
        
        # 智能解析:优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        
        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"


def test_tool():
    tool_executor = ToolExecutor()
    # 2. 注册我们的实战搜索工具
    tool_executor.registerTool(
        "Search", 
        "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。", 
        search
    )
    
    # 3. 打印可用的工具
    print("\n--- 可用的工具 ---")
    print(tool_executor.getAvailableTools())

    # 4. 智能体的Action调用，这次我们问一个实时性的问题
    print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
    tool_name = "Search"
    tool_input = "英伟达最新的GPU型号是什么"
    tool_function = tool_executor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误:未找到名为 '{tool_name}' 的工具。")
        

def test_kimi_code_client():
    client = OpenAI(
        api_key=os.environ['KIM_CODE_API_KEY'],  # 注意：KIMI 不是 KIM
        base_url="https://api.moonshot.cn/v1"    
    )

    res = client.chat.completions.create(
        model="kimi-k2-5",  # ← 你的 OpenClaw 配置里用的是 k2p5，不是 kimi-k2p5
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你好！"}
        ]
    )
    print(res)


# --- 客户端使用示例 ---
if __name__ == '__main__':
    test_kimi_code_client()
    # test_LLM()
    # test_tool()

