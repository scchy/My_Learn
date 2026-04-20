# test_stream.py
from hello_agents import SimpleAgent, HelloAgentsLLM

llm = HelloAgentsLLM()
agent = SimpleAgent(name="流式助手", llm=llm)

# 同步调用（原有）
response = agent.run("讲个故事")
print(response)

# 流式调用（新增）
print("🌊 流式输出：")
for chunk in agent.run_stream("讲个故事"):
    print(chunk, end="", flush=True)  # 实时打字效果
print()

# ReAct 流式：看到思考过程
react_agent = ReActAgent(name="推理助手", llm=llm, tools=[search_tool])
for chunk in react_agent.run_stream("2024年诺贝尔物理学奖得主"):
    print(chunk, end="", flush=True)
# 输出：实时看到 🤔[思考] → ⚡[行动] → 🔧[执行] → ✅[完成]