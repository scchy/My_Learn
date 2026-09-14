# python3
# Author: Scc_hy
# Func: 30秒上手记忆功能 & 30秒上手RAG功能
# Reference: https://datawhalechina.github.io/hello-agents/#/./chapter8/%E7%AC%AC%E5%85%AB%E7%AB%A0%20%E8%AE%B0%E5%BF%86%E4%B8%8E%E6%A3%80%E7%B4%A2?id=_833-%e5%bf%ab%e9%80%9f%e4%bd%93%e9%aa%8c%ef%bc%9a30%e7%a7%92%e4%b8%8a%e6%89%8brag%e5%8a%9f%e8%83%bd
# 运行: /home/scc/anaconda3/envs/LLM/bin/python3 p_mem_and_rag.py
# 说明: 配置见同目录 .env（已固化嵌入式 Qdrant + DashScope embedding）
#       本 fork 的工具只支持 tool.run({"action": ...}) 字典传参，没有 execute()
# ===========================================================================================


import os
from pathlib import Path

from dotenv import load_dotenv

# 统一切到脚本所在目录：.env / memory_data / knowledge_base 都相对这里落盘，
# 这样从任何目录执行都不会把数据写到别处（必须在 import hello_agents 之前做，
# 因为它的 core/database_config.py 在导入时就会 load_dotenv）。
CHAP_DIR = Path(__file__).resolve().parent
os.chdir(CHAP_DIR)
load_dotenv(CHAP_DIR / ".env")

from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry  # noqa: E402
from hello_agents.tools import MemoryTool, RAGTool  # noqa: E402


# LLM
llm = HelloAgentsLLM(model_name=os.getenv("LLM_MODEL_ID", "deepseek-v4-flash"), temperature=1.0)


# ===========================================================================================
# 一、记忆功能（memory_tool）
# ===========================================================================================
memory_tool = MemoryTool(user_id="scc")
tool_reg = ToolRegistry()
tool_reg.register_tool(memory_tool)

mem_agent = SimpleAgent(name="memCheck", llm=llm, tool_registry=tool_reg)

print("=== 添加多个记忆 ===")

# 添加第一个记忆
result1 = memory_tool.run({"action": "add", "content": "用户张三是一名Python开发者，专注于机器学习和数据分析", "memory_type": "semantic", "importance": 0.8})
print(f"记忆1: {result1}")

# 添加第二个记忆
result2 = memory_tool.run({"action": "add", "content": "李四是前端工程师，擅长React和Vue.js开发", "memory_type": "semantic", "importance": 0.7})
print(f"记忆2: {result2}")

# 按语义检索
print("\n=== 检索记忆 ===")
result3 = memory_tool.run({"action": "search", "query": "张三 技术方向", "memory_type": "semantic", "limit": 3})
print(f"检索结果:\n{result3}")

# 定向检索（应精准命中第二条）
print("\n=== 搜索特定记忆 ===")
result4 = memory_tool.run({"action": "search", "query": "前端工程师", "limit": 3})
print(result4)

# 统计与摘要
print("\n=== 记忆统计 ===")
result5 = memory_tool.run({"action": "stats"})
print(result5)

print("\n=== 记忆摘要 ===")
result6 = memory_tool.run({"action": "summary"})
print(result6)


# ===========================================================================================
# 二、RAG 功能（rag_tool）
# ===========================================================================================
rag_tool = RAGTool(
    knowledge_base_path="./knowledge_base",
    collection_name="test_collection",
    rag_namespace="test",
)
tool_registry = ToolRegistry()
tool_registry.register_tool(rag_tool)

rag_agent = SimpleAgent(name="知识助手", llm=llm, tool_registry=tool_registry)

# 添加知识
print("\n=== 添加知识 ===")
KNOWLEDGE = [
    ("python_intro", "Python是一种高级编程语言，由Guido van Rossum于1991年首次发布。Python的设计哲学强调代码的可读性和简洁的语法。"),
    ("ml_basics", "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式。主要包括监督学习、无监督学习和强化学习三种类型。"),
    ("rag_concept", "RAG（检索增强生成）是一种结合信息检索和文本生成的AI技术。它通过检索相关知识来增强大语言模型的生成能力。"),
]
for doc_id, text in KNOWLEDGE:
    result = rag_tool.run({"action": "add_text", "text": text, "document_id": doc_id})
    print(f"知识[{doc_id}]: {result}")

# 检索知识（不调 LLM）
print("\n=== 检索知识 ===")
print(rag_tool.run({"action": "search", "query": "什么是RAG？", "limit": 3}))

# 知识库问答（检索 + LLM 生成）
print("\n=== 知识库问答 ===")
print(rag_tool.run({"action": "ask", "question": "RAG是什么？它解决什么问题？", "limit": 3}))

# 知识库统计
print("\n=== 知识库统计 ===")
print(rag_tool.run({"action": "stats"}))
