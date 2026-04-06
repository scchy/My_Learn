# reference: https://github.com/datawhalechina/hello-agents/blob/main/code/chapter6/Langgraph/Dialogue_System.py
# chap06:  智能搜索助手 - 基于 LangGraph + Tavily API 的真实搜索系统
# Func:  
# =================================================================================


"""
智能搜索助手 - 基于 LangGraph + Tavily API 的真实搜索系统
1. 理解用户需求
2. 使用Tavily API真实搜索信息  
3. 生成基于搜索结果的回答
"""

import asyncio
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
import os
from dotenv import load_dotenv
from tavily import TavilyClient


load_dotenv()
# 定义状态结构
class SearchState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    search_query: str
    search_results: str
    final_answer: str
    step: str
    # 新增反思相关字段
    reflection_count: int      # 反思次数（防止无限循环）
    quality_score: float         # 质量评分 (0-1)
    reflection_feedback: str   # 反思反馈/改进建议
    retry_strategy: Literal["re_search", "re_generate", "continue"]  # 重试策略


# 初始化模型和Tavily客户端
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    temperature=0.7
)

# 初始化Tavily客户端
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def understand_query_node(state: SearchState) -> SearchState:
    """ 
    步骤1：理解用户查询并生成搜索关键词
    """
    user_message = ""
    for msg in reversed(state['messages']):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    # 如果有反思反馈，融入理解过程
    feedback_context = ""
    if state.get("reflection_feedback"):
        feedback_context = f"\n【上一轮反馈】{state['reflection_feedback']}"
    
    understand_prompt = f"""分析用户的查询："{user_message}"{feedback_context}

请完成两个任务：
1. 简洁总结用户想要了解什么
2. 生成最适合搜索的关键词（中英文均可，要精准）

格式：
理解：[用户需求总结]
搜索词：[最佳搜索关键词]"""

    response = llm.invoke([SystemMessage(content=understand_prompt)])
    response_text = response.content
    search_query = user_message


    if "搜索词：" in response_text:
        search_query = response_text.split("搜索词：")[1].strip()
    elif "搜索关键词：" in response_text:
        search_query = response_text.split("搜索关键词：")[1].strip()
    
    return {
        "user_query": response_text,
        "search_query": search_query,
        "step": "understood",
        "messages": [AIMessage(content=f"我理解您的需求：{response_text}")]
    }


def tavily_search_node(state: SearchState) -> SearchState:
    """步骤2：使用Tavily API进行真实搜索"""
    
    search_query = state["search_query"]
    
    # 如果有反思反馈建议重新搜索，调整搜索策略
    if state.get("retry_strategy") == "re_search" and state.get("reflection_feedback"):
        search_query = f"{search_query} {state['reflection_feedback'][:50]}"
        print(f"🔄 基于反馈优化搜索词: {search_query}")

    
    try:
        print(f"🔍 正在搜索: {search_query}")
        response = tavily_client.search(
            query=search_query,
            search_depth="basic",
            include_answer=True,
            include_raw_content=True,  # 增强：获取更多原始内容
            max_results=8 if state.get("reflection_count", 0) > 0 else 5  # 反思时扩大搜索
        )
        
        search_results = ""
        if response.get("answer"):
            search_results = f"综合答案：\n{response['answer']}\n\n"
        
        # 添加具体的搜索结果
        if response.get("results"):
            search_results += "详细信息：\n"
            for i, result in enumerate(response["results"][:5], 1):
                title = result.get("title", "")
                content = result.get("content", "")
                raw_content = result.get("raw_content", "")[:500]  # 截取更多内容
                url = result.get("url", "")
                search_results += f"{i}. {title}\n{content}\n{raw_content}\n来源：{url}\n\n"
        
        return {
            "search_results": search_results,
            "step": "searched",
            "messages": [AIMessage(content="✅ 搜索完成！")]
        }
        
    except Exception as e:
        return {
            "search_results": f"搜索失败: {str(e)}",
            "step": "search_failed",
            "messages": [AIMessage(content=f"⚠️ 搜索异常: {str(e)}")]
        }



def generate_answer_node(state: SearchState) -> SearchState:
    """步骤3：生成答案"""
    
    # 如果是重新生成，使用更严格的提示
    regenerate_hint = ""
    if state.get("retry_strategy") == "re_generate":
        regenerate_hint = """
【重要：这是改进版本】
上一轮答案被评估为质量不足，请确保：
1. 答案长度至少300字，结构完整
2. 必须包含具体的例子、数据或步骤
3. 明确引用搜索结果中的信息
4. 避免泛泛而谈
"""
    
    if state["step"] == "search_failed":
        fallback_prompt = f"""基于知识回答：{state['user_query']}"""
        response = llm.invoke([SystemMessage(content=fallback_prompt)])
    else:
        answer_prompt = f"""{regenerate_hint}
基于以下搜索结果回答：

{state['search_results']}

要求：
1. 直接回答用户问题，不绕弯子
2. 包含具体细节、数据或案例
3. 引用来源，结构清晰
4. 字数充足（>200字）"""
        
        response = llm.invoke([SystemMessage(content=answer_prompt)])
    
    return {
        "final_answer": response.content,
        "step": "answer_generated",
        "messages": [AIMessage(content=response.content)]
    }


# ========== 新增：反思节点 ==========

def reflect_node(state: SearchState) -> SearchState:
    """
    步骤4：质量反思与评估
    输出: quality_score + retry_strategy
    """
    
    answer = state.get("final_answer", "")
    search_results = state.get("search_results", "")
    reflection_count = state.get("reflection_count", 0)
    
    # 硬约束：最多反思2次，防止无限循环
    if reflection_count >= 2:
        print(f"⏹️ 已达最大反思次数({reflection_count})，强制结束")
        return {
            "quality_score": 0.6,  # 中等评分
            "retry_strategy": "continue",
            "step": "reflected",
            "reflection_count": reflection_count,
            "messages": [AIMessage(content="[系统：已达到质量检查上限，返回当前最佳答案]")]
        }
    
    # 构建反思提示
    reflect_prompt = f"""评估以下答案质量：

【用户问题】{state['user_query']}
【搜索结果长度】{len(search_results)} 字符
【生成答案】{answer[:800]}...

请从以下维度评分（0-1）：
1. 完整性：是否充分回答问题？
2. 具体性：是否包含细节、数据、案例？
3. 引用度：是否基于搜索结果？
4. 长度：是否过于简短（<150字）？

输出格式：
质量评分：[0-1之间的数字]
主要问题：[简短描述最大缺陷]
改进建议：[如何改进]
是否重新搜索：[是/否]
是否重新生成：[是/否]"""

    reflection = llm.invoke([SystemMessage(content=reflect_prompt)])
    reflection_text = reflection.content

    # 解析反思结果
    quality_score = 0.5
    if "质量评分：" in reflection_text:
        try:
            score_str = reflection_text.split("质量评分：")[1].split()[0]
            quality_score = float(score_str)
        except:
            pass

    # 确定重试策略
    retry_strategy: Literal["re_search", "re_generate", "continue"] = "continue"
    
    # 条件逻辑
    if quality_score < 0.6:
        if "重新搜索：是" in reflection_text or len(search_results) < 200:
            retry_strategy = "re_search"
        elif "重新生成：是" in reflection_text:
            retry_strategy = "re_generate"
    
    # 提取反馈用于下一轮
    feedback = ""
    if "改进建议：" in reflection_text:
        feedback = reflection_text.split("改进建议：")[1].split("\n")[0][:100]

    print(f"🤔 反思结果: 评分={quality_score:.2f}, 策略={retry_strategy}, 次数={reflection_count+1}")
    
    return {
        "quality_score": quality_score,
        "retry_strategy": retry_strategy,
        "reflection_feedback": feedback,
        "reflection_count": reflection_count + 1,
        "step": "reflected",
        "messages": [AIMessage(content=f"[质量检查: {quality_score:.0%}]")]
    }


# ========== 条件边路由函数 ==========

def route_by_quality(state: SearchState) -> Literal["re_search", "re_generate", "end"]:
    """
    条件边：根据反思结果决定下一步
    """
    strategy = state.get("retry_strategy", "continue")
    score = state.get("quality_score", 0.5)
    
    if strategy == "re_search":
        return "re_search"
    elif strategy == "re_generate":
        return "re_generate"
    else:
        return "end"


# 构建搜索工作流
def create_search_assistant():
    # 有点像airflow
    workflow = StateGraph(SearchState)

    # 添加三个节点
    workflow.add_node("understand", understand_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)
    workflow.add_node("reflect", reflect_node)  # 新增
    
    # 设置线性流程
    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", "reflect")
    # ========== 条件边：反思后的路由 ==========
    # 质量差 → 重新搜索（回到search节点，但带上反馈）
    workflow.add_conditional_edges(
        "reflect",
        route_by_quality,
        {
            "re_search": "search",      # 循环：重新搜索
            "re_generate": "answer",      # 循环：重新生成（不搜索）
            "end": END                    # 结束
        }
    )
    
    # 编译图
    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


async def main():
    """主函数：运行智能搜索助手"""
    
    # 检查API密钥
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ 错误：请在.env文件中配置TAVILY_API_KEY")
        return
    
    app = create_search_assistant()
    
    print("🔍 智能搜索助手（带质量反思）启动！")
    print("系统会自动检查答案质量，必要时重新搜索或生成\n")
    
    print("我会使用Tavily API为您搜索最新、最准确的信息")
    print("支持各种问题：新闻、技术、知识问答等")
    print("(输入 'quit' 退出)\n")
    
    session_count = 0
    
    while True:
        user_input = input("🤔 您想了解什么: ").strip()
        
        if user_input.lower() in ['quit', 'q', '退出', 'exit']:
            print("感谢使用！再见！👋")
            break
        
        if not user_input:
            continue
        
        session_count += 1
        config = {"configurable": {"thread_id": f"search-session-{session_count}"}}
        
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_query": "",
            "search_query": "",
            "search_results": "",
            "final_answer": "",
            "step": "start",
            "reflection_count": 0,
            "quality_score": 0.0,
            "reflection_feedback": "",
            "retry_strategy": "continue"
        }
        
        print("\n" + "="*60)
        try:
            async for output in app.astream(initial_state, config=config):
                for node_name, node_output in output.items():
                    if "messages" in node_output and node_output["messages"]:
                        latest = node_output["messages"][-1]
                        if isinstance(latest, AIMessage):
                            content = latest.content
                            
                            if node_name == "understand":
                                print(f"🧠 理解: {content[:80]}...")
                            elif node_name == "search":
                                print(f"🔍 搜索: {content}")
                            elif node_name == "answer":
                                if len(content) > 100:
                                    print(f"✍️ 生成答案 ({len(content)}字)")
                                else:
                                    print(f"✍️ 答案: {content}")
                            elif node_name == "reflect":
                                print(f"🔍 {content}")
                            
                            # 显示循环信息
                            if node_output.get("reflection_count", 0) > 0:
                                print(f"   [反思次数: {node_output['reflection_count']}, "
                                      f"质量: {node_output.get('quality_score', 0):.0%}, "
                                      f"策略: {node_output.get('retry_strategy', 'none')}]")
            
            # 获取最终状态显示结果
            final_state = await app.aget_state(config)
            if final_state and final_state.values.get("final_answer"):
                print(f"\n💡 最终回答:\n{final_state.values['final_answer'][:500]}...")
                print(f"\n📊 统计: 反思{final_state.values.get('reflection_count', 0)}次, "
                      f"最终质量: {final_state.values.get('quality_score', 0):.0%}")
            
            print("\n" + "="*60 + "\n")
        
        except Exception as e:
            print(f"❌ 错误: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())

