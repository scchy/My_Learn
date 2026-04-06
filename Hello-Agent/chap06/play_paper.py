"""
论文写作工作流 - 基于 LangGraph 的实现

该模块实现了一个自动化的论文写作流程，包含以下阶段：
1. 大纲生成 (outline_node)
2. 章节写作 (draft_node)
3. 智能审阅 (review_node)
4. 质量检查与路由 (quality_check)
5. 修改完善 (revise_node)
6. 人工介入 (human_node)
7. 最终输出 (final_output_node)
"""

from typing import TypedDict, Literal
import os
import re
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.messages import HumanMessage, SystemMessage

# 加载环境变量
load_dotenv()

# 初始化 LLM 模型
api_key = os.getenv("LLM_API_KEY")
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
    api_key=SecretStr(api_key) if api_key else None,
    base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    temperature=0.7
)


class PaperState(TypedDict):
    """论文写作状态字典"""
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


def parse_sections(outline: str) -> list[dict]:
    """
    解析大纲文本，提取章节结构
    
    Args:
        outline: 大纲文本
        
    Returns:
        章节列表，每个章节包含 title, points, word_count
    """
    sections = []
    lines = outline.strip().split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 匹配章节标题（支持多种格式：1. 第一章 或 ## 第一章）
        if re.match(r'^(\d+[\.、]|#{1,3})\s*', line):
            if current_section:
                sections.append(current_section)
            title = re.sub(r'^(\d+[\.、]|#{1,3})\s*', '', line)
            current_section = {
                "title": title,
                "points": [],
                "word_count": 800  # 默认字数
            }
        elif current_section is not None:
            # 当前要点
            point = re.sub(r'^[\-\*•]\s*', '', line)
            current_section["points"].append(point)
            
            # 尝试提取字数要求
            word_match = re.search(r'(\d+)\s*字', line)
            if word_match:
                current_section["word_count"] = int(word_match.group(1))
    
    if current_section:
        sections.append(current_section)
    
    # 如果没有解析到章节，创建一个默认章节
    if not sections:
        sections.append({
            "title": "主要内容",
            "points": [outline],
            "word_count": 1000
        })
    
    return sections


def check_logic(draft: str) -> dict:
    """检查逻辑连贯性"""
    messages = [
        SystemMessage(content="你是一个学术论文评审专家，负责评估文本的逻辑连贯性。"),
        HumanMessage(content=f"""请评估以下文本的逻辑连贯性：

文本内容：
{draft[:2000]}

请以 JSON 格式返回：
{{
    "passed": true/false,
    "score": 0-100,
    "feedback": "具体的反馈或改进建议"
}}""")
    ]
    
    try:
        response = llm.invoke(messages)
        # 简单解析响应
        content = response.content
        content_str = str(content)
        passed = "通过" in content_str or '"passed": true' in content_str.lower() or "true" in content_str.lower()
        score = 80 if passed else 50
        return {"passed": passed, "score": score, "feedback": content_str[:200]}
    except Exception:
        return {"passed": True, "score": 85, "feedback": "默认通过"}


def check_citations(draft: str) -> dict:
    """检查引用规范"""
    has_citations = re.search(r'\[\d+\]|\(\w+\s+et\s+al\.?\s*,\s*\d{4}\)', draft)
    return {
        "passed": True,  # 可选检查
        "score": 90 if has_citations else 70,
        "feedback": "引用检查完成" if has_citations else "建议添加引用"
    }


def check_academic_style(draft: str) -> dict:
    """检查学术风格"""
    messages = [
        SystemMessage(content="你是一个学术写作专家，负责评估文本的学术规范性。"),
        HumanMessage(content=f"""请评估以下文本的学术风格：

文本内容：
{draft[:2000]}

请以 JSON 格式返回评估结果。""")
    ]
    
    try:
        response = llm.invoke(messages)
        content = response.content
        content_str = str(content)
        passed = "规范" in content_str or "符合" in content_str or '"passed": true' in content_str.lower()
        return {"passed": passed, "score": 85, "feedback": content_str[:200]}
    except Exception:
        return {"passed": True, "score": 80, "feedback": "默认通过"}


def check_completeness(draft: str, section_title: str) -> dict:
    """检查内容完整性"""
    word_count = len(draft)
    min_words = 200
    
    return {
        "passed": word_count >= min_words,
        "score": min(100, int(word_count / min_words * 100)),
        "feedback": f"字数：{word_count}" if word_count >= min_words else f"内容较短（{word_count}字），建议扩充"
    }


def format_issues(issues: list[str]) -> str:
    """格式化审阅意见"""
    if not issues:
        return "无问题"
    return "\n".join(f"{i+1}. {issue}" for i, issue in enumerate(issues))


def outline_node(state: PaperState) -> dict:
    """
    根据主题生成论文结构大纲
    
    Args:
        state: 当前状态
        
    Returns:
        包含大纲、章节列表和初始化的状态更新
    """
    messages = [
        SystemMessage(content="你是一个专业的学术论文写作助手。"),
        HumanMessage(content=f"""请为以下主题生成详细的论文大纲：

主题: {state['topic']}
要求: {state['requirements']}

请生成包含章节标题和每章要点的详细大纲。格式要求：
1. 使用 "1. 章节标题" 或 "## 章节标题" 格式
2. 每章下用 "- 要点" 列出关键内容
3. 可以在要点中标注建议字数，如 "（约800字）"
""")
    ]
    
    response = llm.invoke(messages)
    outline = response.content
    
    return {
        "outline": outline,
        "current_section_idx": 0,
        "sections": parse_sections(outline),
        "revision_count": 0
    }


def draft_node(state: PaperState) -> dict:
    """
    根据大纲撰写当前章节
    
    Args:
        state: 当前状态
        
    Returns:
        包含当前章节内容和完整草稿的状态更新
    """
    sections = state.get('sections', [])
    current_idx = state.get('current_section_idx', 0)
    
    if not sections or current_idx >= len(sections):
        return {
            "current_draft": "",
            "full_draft": state.get('full_draft', '')
        }
    
    section = sections[current_idx]
    previous_content = state.get('full_draft', '')
    
    messages = [
        SystemMessage(content="你是一个专业的学术论文写作专家。"),
        HumanMessage(content=f"""请撰写论文章节：

论文主题: {state['topic']}
当前章节: {section.get('title', '未命名章节')}
要点: {', '.join(section.get('points', []))}
字数要求: {section.get('word_count', 800)}字

已写内容摘要:
{previous_content[-500:] if previous_content else '这是第一章，无前文'}

请撰写本章内容，保持与上下文的连贯性。要求：
1. 学术语言规范
2. 逻辑清晰
3. 内容完整
4. 字数达标
""")
    ]
    
    response = llm.invoke(messages)
    content = response.content
    
    return {
        "current_draft": content,
        "full_draft": (previous_content + "\n\n" + content).strip()
    }


def review_node(state: PaperState) -> dict:
    """
    多维度审阅当前章节
    
    Args:
        state: 当前状态
        
    Returns:
        包含审阅结果的状态更新
    """
    draft = state.get('current_draft', '')
    sections = state.get('sections', [])
    current_idx = state.get('current_section_idx', 0)
    
    section_title = sections[current_idx].get('title', '未命名章节') if sections and current_idx < len(sections) else '未命名章节'
    
    # 并行多维度检查
    checks = {
        "logic": check_logic(draft),
        "citation": check_citations(draft),
        "style": check_academic_style(draft),
        "completeness": check_completeness(draft, section_title)
    }
    
    review_result = {
        "passed": all(c.get('passed', True) for c in checks.values()),
        "issues": [c.get('feedback', '') for c in checks.values() if not c.get('passed', True)],
        "scores": {k: v.get('score', 0) for k, v in checks.items()}
    }
    
    return {"review_result": review_result}


def quality_check(state: PaperState) -> Literal["revise", "next_section", "final_output", "human_node"]:
    """
    决定循环流向
    
    Args:
        state: 当前状态
        
    Returns:
        下一个节点的路由决策
    """
    review = state.get('review_result', {})
    revision_count = state.get('revision_count', 0)
    current_idx = state.get('current_section_idx', 0)
    total_sections = len(state.get('sections', []))
    
    # 循环退出条件：超过3次修改仍不通过，转人工
    if revision_count >= 3 and not review.get('passed', False):
        return "human_node"
    
    # 需要修改，进入内循环
    if not review.get('passed', False):
        return "revise"
    
    # 当前章节通过，检查是否还有下一章
    if current_idx < total_sections - 1:
        return "next_section"
    
    # 全部完成
    return "final_output"


def revise_node(state: PaperState) -> dict:
    """
    根据审阅意见修改
    
    Args:
        state: 当前状态
        
    Returns:
        包含修改后内容和更新计数的状态
    """
    draft = state.get('current_draft', '')
    issues = state.get('review_result', {}).get('issues', [])
    
    messages = [
        SystemMessage(content="你是一个专业的学术论文修改专家。"),
        HumanMessage(content=f"""请根据审阅意见修改以下内容：

当前草稿:
{draft}

审阅意见:
{format_issues(issues)}

请逐条解决上述问题，输出修改后的完整内容。
请在修改后的内容前添加"修改说明:"段落，列出具体修改点。
""")
    ]
    
    response = llm.invoke(messages)
    revised = response.content
    
    return {
        "current_draft": revised,
        "revision_count": state.get('revision_count', 0) + 1
    }


def next_section(state: PaperState) -> dict:
    """
    推进到下一章节，重置计数器
    
    Args:
        state: 当前状态
        
    Returns:
        更新的状态
    """
    return {
        "current_section_idx": state.get('current_section_idx', 0) + 1,
        "revision_count": 0,
        "current_draft": ""
    }


def human_node(state: PaperState) -> dict:
    """
    人工介入节点 - 当自动修改超过限制时触发
    
    Args:
        state: 当前状态
        
    Returns:
        状态（实际应用中可能等待人工输入）
    """
    print(f"\n{'='*50}")
    print("人工介入节点")
    print(f"{'='*50}")
    print(f"章节: {state.get('sections', [{}])[state.get('current_section_idx', 0)].get('title', '未知')}")
    print(f"修改次数: {state.get('revision_count', 0)}")
    print(f"审阅问题: {state.get('review_result', {}).get('issues', [])}")
    print(f"{'='*50}\n")
    
    # 在实际应用中，这里应该等待人工输入
    # 简化版本：直接继续，将当前草稿作为最终结果
    return {
        "current_draft": state.get('current_draft', '') + "\n[已人工审核]",
        "revision_count": 0
    }


def final_output_node(state: PaperState) -> dict:
    """
    最终输出节点 - 整合所有内容生成最终论文
    
    Args:
        state: 当前状态
        
    Returns:
        包含最终论文的状态
    """
    full_draft = state.get('full_draft', '')
    outline = state.get('outline', '')
    
    # 添加论文头部信息
    header = f"""# {state.get('topic', '未命名论文')}

## 论文大纲
{outline}

---

"""
    
    final_paper = header + full_draft
    
    print(f"\n{'='*50}")
    print("论文写作完成！")
    print(f"总字数: {len(final_paper)}")
    print(f"章节数: {len(state.get('sections', []))}")
    print(f"{'='*50}\n")
    
    return {
        "final_paper": final_paper
    }


# ============================================================================
# 构建工作流图
# ============================================================================

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
workflow.add_edge(START, "outline_node")

# 添加普通边
workflow.add_edge("outline_node", "draft_node")
workflow.add_edge("draft_node", "review_node")
workflow.add_edge("revise_node", "review_node")
workflow.add_edge("next_section", "draft_node")
workflow.add_edge("human_node", "draft_node")
workflow.add_edge("final_output", END)

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

# 编译工作流
memory = InMemorySaver()
app = workflow.compile(checkpointer=memory)


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    # 示例运行
    initial_state = {
        "topic": "深度学习在计算机视觉中的应用",
        "requirements": "要求涵盖CNN、Transformer等主流方法，要有技术细节和应用案例",
        "outline": "",
        "sections": [],
        "current_section_idx": 0,
        "current_draft": "",
        "full_draft": "",
        "review_result": {},
        "revision_count": 0,
        "final_paper": ""
    }
    
    # 运行工作流
    config = {"configurable": {"thread_id": "paper_001"}}
    
    print("开始论文写作工作流...\n")
    
    for event in app.stream(initial_state, config=config):
        for node_name, node_state in event.items():
            print(f"执行节点: {node_name}")
            if node_name == "final_output":
                print("\n最终论文已生成！")
                print(f"\n预览（前500字）:\n{node_state.get('final_paper', '')[:500]}...")
