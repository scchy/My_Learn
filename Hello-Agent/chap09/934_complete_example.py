# python3
# Authore: Scc_hy
# Create Date: 2026-09-26
# Func: 9.3.4 完整使用示例
# Run : /home/scc/anaconda3/envs/LLM/bin/python3 934_complete_example.py
# 说明: 配置见同目录 .env（自 chap08 复制，嵌入式 Qdrant + DashScope embedding）；
#       hello-agents 0.2.8 需先 import _qdrant_embedded 打开嵌入式本地模式
# ==========================================================================================================

import os
from pathlib import Path

# 统一切到脚本所在目录：.env / memory_data / knowledge_base 都相对这里落盘
# （必须在 import hello_agents 之前做：它的 core/database_config.py 导入时就会 load_dotenv）
CHAP_DIR = Path(__file__).resolve().parent
os.chdir(CHAP_DIR)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(CHAP_DIR / ".env")

import _qdrant_embedded  # noqa: E402,F401  # 0.2.8 补丁：QDRANT_PATH 设置时走嵌入式本地模式

from hello_agents.context import ContextBuilder, ContextConfig  # noqa: E402
from hello_agents.tools import MemoryTool, RAGTool  # noqa: E402
from hello_agents.core.message import Message  # noqa: E402
from datetime import datetime  # noqa: E402

# 1. 初始化工具
memory_tool = MemoryTool(user_id='u123')
rag_tool = RAGTool(knowledge_base_path='./knowledge_base')

# 2. 创建 ContextBuilder
config = ContextConfig(
    max_tokens=3000,
    reserve_ratio=0.2,
    min_relevance=0.2,
    enable_compression=True
)

builder = ContextBuilder(
    memory_tool=memory_tool,
    rag_tool=rag_tool,
    config=config
)

# 3. 准备对话历史
conversation_history = [
    Message(content="我正在开发一个数据分析工具", role='user', timestamp=datetime.now()),
    Message(content="很好!数据分析工具通常需要处理大量数据。您计划使用什么技术栈?", role="assistant", timestamp=datetime.now()),
    Message(content="我打算使用Python和Pandas,已经完成了CSV读取模块", role="user", timestamp=datetime.now()),
    Message(content="不错的选择!Pandas在数据处理方面非常强大。接下来您可能需要考虑数据清洗和转换。", role="assistant", timestamp=datetime.now()),
]

# 4. 添加一些记忆
memory_tool.run({
    "action": 'add',
    "content": "用户正在开发数据分析工具，使用Python和Pandas",
    "memory_type": "semantic",
    "importance": 0.8
})

memory_tool.run({
    "action": "add",
    "content": "已完成CSV读取模块的开发",
    "memory_type": "episodic",
    "importance": 0.7
})

# 5. 构建上下文
context = builder.build(
    user_query="如何优化Pandas的内存占用?",
    conversation_history=conversation_history,
    system_instructions="你是一位资深的Python数据工程顾问。你的回答需要:1) 提供具体可行的建议 2) 解释技术原理 3) 给出代码示例"
)

print("=" * 80)
print("构建的上下文:")
print("=" * 80)
print(context)
print("=" * 80)




