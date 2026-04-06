

plan_ = """
                    ┌─────────────────┐
         ┌─────────│   START (入口)   │─────────┐
         │         └─────────────────┘         │
         │                   │                 │
         ▼                   ▼                 ▼
┌─────────────────┐  ┌─────────────┐  ┌─────────────┐
│  outline_node   │  │ draft_node  │  │ review_node │
│  (大纲生成)      │  │  (章节写作)  │  │  (智能审阅)  │
│                 │  │             │  │             │
│ 输出: 论文大纲    │  │ 输出: 章节内容 │  │ 输出: 审阅意见 │
└────────┬────────┘  └──────┬──────┘  └──────┬──────┘
         │                  │                │
         │                  ▼                │
         │           ┌─────────────┐         │
         └──────────▶│  quality_check│◀────────┘
                     │  (质量检查节点) │
                     │              │
                     │  判断: 通过?   │
                     │  /需修改?/完成? │
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌─────────┐   ┌─────────┐   ┌─────────┐
        │  revise │   │ next_   │   │  final  │
        │  _node  │   │ section │   │ _output │
        │ (修改)  │   │ (下一章) │   │ (结束)  │
        │         │   │         │   │         │
        │ 回到review │   │ 回到draft │   │   END   │
        └────┬────┘   └────┬────┘   └─────────┘
             └─────────────┘
"""


from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, Annotated, Literal
import operator
import os
from dotenv import load_dotenv 


load_dotenv()
# 初始化模型和Tavily客户端
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    temperature=0.7
)


class PaperState(TypedDict):
    topic: str
    requirements: str
    outline: str
    sections: list
    current_section_idx: int
    current_draft: str
    full_draft: str
    review_result: dict
    revision_count: int
    final_paper: str


def outline_node(state: PaperState):
    """根据主题生成论文结构大纲"""
    prompt = f"""
    主题: {state['topic']}
    要求: {state['requirements']}
    请生成详细的论文大纲，包含章节标题和每章要点。
    """
    outline = llm.generate(prompt)
    return {
        "outline": outline,
        "current_section_idx": 0,
        "sections": parse_sections(outline),
        "revision_count": 0
    }


def draft_node(state: PaperState):
    """根据大纲撰写当前章节"""
    section = state['sections'][state['current_section_idx']]
    previous_content = state.get('full_draft', '')
    
    prompt = f"""
    论文主题: {state['topic']}
    已写内容: {previous_content[-1000:] if previous_content else '无'}
    
    当前章节: {section['title']}
    要点: {section['points']}
    字数要求: {section['word_count']}
    
    请撰写本章内容，保持与上下文的连贯性。
    """
    content = llm.generate(prompt)
    
    return {
        "current_draft": content,
        "full_draft": previous_content + "\n\n" + content
    }


def review_node(state: PaperState):
    """多维度审阅当前章节"""
    draft = state['current_draft']
    section_title = state['sections'][state['current_section_idx']]['title']
    
    # 并行多维度检查
    checks = {
        "logic": check_logic(draft),      # 逻辑连贯性
        "citation": check_citations(draft),  # 引用规范
        "style": check_academic_style(draft), # 学术风格
        "completeness": check_completeness(draft, section_title)  # 内容完整性
    }
    
    review_result = {
        "passed": all(c['passed'] for c in checks.values()),
        "issues": [c['feedback'] for c in checks.values() if not c['passed']],
        "scores": {k: v['score'] for k, v in checks.items()}
    }
    
    return {"review_result": review_result}


def quality_check(state: PaperState) -> Literal["revise", "next_section", "final_output", "human_node"]:
    """决定循环流向"""
    review = state['review_result']
    revision_count = state.get('revision_count', 0)
    current_idx = state['current_section_idx']
    total_sections = len(state['sections'])
    
    # 循环退出条件
    if revision_count >= 3 and not review['passed']:
        return "human_node"  # 超过3次修改仍不通过，转人工
    
    if not review['passed']:
        return "revise"  # 需要修改，进入内循环
    
    # 当前章节通过，检查是否还有下一章
    if current_idx < total_sections - 1:
        return "next_section"  # 进入下一章的外循环
    
    return "final_output"  # 全部完成


def revise_node(state: PaperState):
    """根据审阅意见修改"""
    draft = state['current_draft']
    issues = state['review_result']['issues']
    
    prompt = f"""
    当前草稿:
    {draft}
    
    审阅意见:
    {format_issues(issues)}
    
    请逐条解决上述问题，输出修改后的内容。
    修改说明: [列出具体修改点]
    """
    revised = llm.generate(prompt)
    
    return {
        "current_draft": revised,
        "revision_count": state.get('revision_count', 0) + 1  # 计数+1
    }


def next_section(state: PaperState):
    """推进到下一章节，重置计数器"""
    return {
        "current_section_idx": state['current_section_idx'] + 1,
        "revision_count": 0,  # 重置修改计数
        "current_draft": ""   # 清空当前草稿
    }




# 构建图
workflow = StateGraph(PaperState)

# 添加节点
workflow.add_node("outline_node", outline_node)
workflow.add_node("draft_node", draft_node)
workflow.add_node("review_node", review_node)
workflow.add_node("revise_node", revise_node)
workflow.add_node("next_section", next_section)
workflow.add_node("human_node", human_node)
workflow.add_node("final_output", final_output_node)

# 设置入口
workflow.set_entry_point("outline_node")

# 添加边
workflow.add_edge("outline_node", "draft_node")
workflow.add_edge("draft_node", "review_node")

# 条件边：核心循环逻辑
workflow.add_conditional_edges(
    "review_node",
    quality_check,
    {
        "revise": "revise_node",           # 内循环：修改→重新审阅
        "next_section": "next_section",    # 外循环：下一章→重新写作
        "final_output": "final_output",     # 结束循环
        "human_node": "human_node"          # 异常退出
    }
)

# 修改后回到审阅（内循环闭环）
workflow.add_edge("revise_node", "review_node")

# 下一章回到写作（外循环闭环）
workflow.add_edge("next_section", "draft_node")

# 人工介入后回到写作
workflow.add_edge("human_node", "draft_node")

# 结束
workflow.add_edge("final_output", END)

# 编译
app = workflow.compile()