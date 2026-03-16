# python3
# Create Date: 2026-03-06
# Func: Basic Agent
# Learning-Url: https://github.com/datawhalechina/hello-agents/blob/main/docs/chapter1/Chapter1-Introduction-to-Agents.md
# ----------------------------------------------------------------------------------------------------------------------------
import re
import requests
import os
import json 
import sys 
from typing import Dict, List, Optional, Callable, Any, Tuple
from tavily import TavilyClient
from openai import OpenAI

PROJ_DIR = os.path.dirname(__file__)
if PROJ_DIR not in sys.path:
    sys.path.append(PROJ_DIR)
from agent_utils import UserProfile, StrategyReflector, ToolRegistry


# （1）指令模板 
AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str, preference: str = "")`: 根据城市、天气和用户偏好搜索景点。
- `check_ticket(attraction: str, date: str)`: 查询景点门票 availability，售罄时自动触发备选推荐。
- `record_preference(key: str, value: str)`: 记录用户偏好到档案。
- `ask_user(question: str)`: 向用户提问以明确需求（策略反思模式下使用）。
- `finish(answer: str)`: 结束任务并给出最终答案。


# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对Thought和Action：

Thought: [你的思考过程和下一步计划，考虑用户档案中的偏好和策略提示]
Action: [你要执行的具体行动]

Action格式：
1. 调用工具：tool_name(arg="value")
2. 结束任务：finish(answer="最终答案")

# 特殊规则:
1. 【记忆功能】如果用户表达了偏好（如"我喜欢历史文化"、"预算500以内"），必须使用 record_preference 记录
2. 【备选方案】如果 check_ticket 返回售罄或不可用，系统会自动提供备选，你只需基于 Observation 继续
3. 【策略反思】如果收到[策略调整提示]，必须严格遵循：
   - interactive模式：先使用 ask_user 询问，不要直接推荐
   - budget_flexible模式：强调性价比，推荐免费/低价选项
   - crowd_avoid模式：强调小众、人少、独特体验
   - diverse模式：明确避开历史拒绝列表中的景点
4. 【反馈跟踪】每次推荐后，等待用户反馈（接受/拒绝），并在Thought中记录

请开始吧！
"""
API_KEY = os.environ['DEEPSEEK_API_KEY']
BASE_URL = 'https://api.deepseek.com'
MODEL_ID = "deepseek-chat"
TAVILY_API_KEY = os.environ['TAVILY_API_KEY']


#（2）工具 1：查询真实天气
def get_weather(city: str) -> str:
    """
    通过调用 wttr.in API 查询真实的天气信息。
    """
    # API端点，我们请求JSON格式的数据
    url = f"https://wttr.in/{city}?format=j1"
    
    try:
        # 发起网络请求
        response = requests.get(url)
        # 检查响应状态码是否为200 (成功)
        response.raise_for_status() 
        # 解析返回的JSON数据
        data = response.json()
        
        # 提取当前天气状况
        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']
        
        # 格式化成自然语言返回
        return f"{city}当前天气:{weather_desc}，气温{temp_c}摄氏度"
        
    except requests.exceptions.RequestException as e:
        # 处理网络错误
        return f"错误:查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        # 处理数据解析错误
        return f"错误:解析天气数据失败，可能是城市名称无效 - {e}"


# （3）工具 2：搜索并推荐旅游景点
def get_attraction(city: str, weather: str, preference: str = "") -> str:
    """
    根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐。
    """
    # 1. 从环境变量中读取API密钥
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误:未配置TAVILY_API_KEY环境变量。"

    # 2. 初始化Tavily客户端
    tavily = TavilyClient(api_key=api_key)
    
    # 3. 构造一个精确的查询
    # 构建查询，融入用户偏好
    pref_str = f"，偏好{preference}" if preference else ""
    query = f"{city}{pref_str} 在{weather}天气下最值得去的旅游景点推荐"
    
    try:
        # 4. 调用API，include_answer=True会返回一个综合性的回答
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        
        # 5. Tavily返回的结果已经非常干净，可以直接使用
        # response['answer'] 是一个基于所有搜索结果的总结性回答
        if response.get("answer"):
            return response["answer"]
        
        # 如果没有综合性回答，则格式化原始结果
        results = []
        for r in response.get("results", [])[:3]:
            results.append(f"• {r['title']}: {r['content'][:100]}...")
      
        return "推荐景点:\n" + "\n".join(results) if results else "未找到推荐"

    except Exception as e:
        return f"错误:搜索失败 - {e}"



def check_ticket(attraction: str, date: str) -> str:
    """
    模拟查票功能
    实际应用中应调用真实API（如携程、美团等）
    """
    # 模拟数据：部分景点售罄
    sold_out_attractions = ["故宫", "国家博物馆", "颐和园"]
    
    if any(sold in attraction for sold in sold_out_attractions):
        return f"{attraction} {date}: 门票已售罄"
    
    return f"{attraction} {date}: 有票，余量充足"


def get_similar_attraction(city: str, original: str, weather: str) -> str:
    """获取相似备选景点（当原景点售罄时使用）"""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误:未配置TAVILY_API_KEY"
    
    tavily = TavilyClient(api_key=api_key)
    
    query = f"{city} 类似{original}的替代景点推荐，{weather}天气适合"
    
    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        
        if response.get("answer"):
            return f"[备选方案] {original}不可用，为您推荐类似景点：\n{response['answer']}"
        
        return f"[备选方案] 建议游览{city}其他景点替代{original}"
    except Exception as e:
        return f"错误:备选搜索失败 - {e}"


def record_preference(key: str, value: str) -> str:
    """记录用户偏好（实际由主循环处理，这里返回确认）"""
    return f"偏好已记录: {key}={value}"


def ask_user(question: str) -> str:
    """向用户提问（策略反思模式使用）"""
    return f"[系统提问] {question}"


def finish(answer: str) -> str:
    """结束任务"""
    return answer



# ============ 主程序 ============

def parse_thought_action(llm_output: str) -> Optional[Tuple[str, str]]:
    """解析 Thought 和 Action"""
    thought_match = re.search(r'Thought:\s*(.*?)(?=Action:|$)', llm_output, re.DOTALL)
    action_match = re.search(r'Action:\s*(.*?)(?=\n|$)', llm_output, re.DOTALL)
    
    if not action_match:
        return None
    
    thought = thought_match.group(1).strip() if thought_match else ""
    action = action_match.group(1).strip()
    
    return thought, action


def parse_tool_call(action_str: str) -> Optional[Dict]:
    """解析工具调用"""
    # 匹配 finish(answer="...") 或 tool_name(arg="value")
    if action_str.startswith("finish"):
        match = re.search(r'finish\((.*?)\)', action_str, re.DOTALL)
        if match:
            return {"type": "finish", "raw": action_str}
        return {"type": "finish", "raw": action_str}
    
    match = re.search(r'(\w+)\((.*?)\)', action_str, re.DOTALL)
    if not match:
        return None
    
    tool_name = match.group(1)
    args_str = match.group(2)
    
    # 解析参数
    kwargs = {}
    for key, value in re.findall(r'(\w+)\s*=\s*["\']([^"\']*)["\']', args_str):
        kwargs[key] = value
    
    return {"type": "tool", "name": tool_name, "kwargs": kwargs}



def extract_preferences_from_thought(thought: str) -> Dict[str, str]:
    """从思考过程中提取偏好"""
    prefs = {}
    
    # 简单关键词匹配
    if "历史文化" in thought or "古迹" in thought:
        prefs["景点类型"] = "历史文化"
    elif "自然" in thought or "风景" in thought:
        prefs["景点类型"] = "自然风光"
    elif "现代" in thought or "都市" in thought:
        prefs["景点类型"] = "现代都市"
    
    # 预算提取
    if re.search(r'预算[高低]|便宜|贵|省钱', thought):
        if "高" in thought or "贵" in thought:
            prefs["预算"] = "高"
        elif "低" in thought or "便宜" in thought or "省钱" in thought:
            prefs["预算"] = "低"
        else:
            prefs["预算"] = "中"
    
    return prefs



def simulate_user_feedback(profile: UserProfile, recommendation: str) -> Optional[str]:
    """
    模拟用户反馈（用于演示）
    实际应用中应通过真实用户输入获取
    """
    # 演示：前3次模拟拒绝以触发策略反思
    if profile.rejection_count < 3:
        reasons = [
            "这个有点贵，超出预算了",
            "人太多了，不想排队", 
            "不太感兴趣这种类型"
        ]
        return reasons[profile.rejection_count]
    
    # 第4次及以后模拟接受
    return "accept"


class OpenAICompatibleClient:
    """
    一个用于调用任何兼容OpenAI接口的LLM服务的客户端。
    """
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用LLM API来生成回应。"""
        print("正在调用大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误:调用语言模型服务时出错。"


def test_enhanced():
    """
    Thought-Action-Observation
    增强版旅行助手演示
    集成：记忆功能、备选方案推荐、策略反思
    """
    # 1. 初始化
    llm = OpenAICompatibleClient(
        model=MODEL_ID,
        api_key=API_KEY,
        base_url=BASE_URL
    )
    # 用户档案和策略反思器
    profile = UserProfile()
    reflector = StrategyReflector(profile)
    tools = ToolRegistry()
    tools.register("get_weather", get_weather)
    tools.register("get_attraction", get_attraction)
    tools.register("check_ticket", check_ticket, 
                   fallbacks=["get_similar_attraction"])  # 售罄时自动找备选
    tools.register("get_similar_attraction", get_similar_attraction)
    tools.register("record_preference", record_preference)
    tools.register("ask_user", ask_user)
    tools.register("finish", finish)
    
    
    # 对话初始化
    user_input = "你好，我想去北京旅游，喜欢历史文化景点，预算中等。请帮我查天气并推荐景点。"
    prompt_history = [
        f"[系统] 用户档案: {json.dumps(profile.to_dict(), ensure_ascii=False)}",
        f"用户: {user_input}"
    ]
    
    print("=" * 60)
    print("🤖 增强版智能旅行助手")
    print("功能: 记忆 | 备选方案 | 策略反思")
    print("=" * 60)
    print(f"\n👤 用户: {user_input}\n")
    

    # 主循环
    max_iterations = 8
    for i in range(max_iterations): # 设置最大循环次数
        print(f"\n{'='*60}")
        print(f"🔄 循环 {i+1}/{max_iterations}")
        print(f"📊 当前策略: {profile.last_strategy} | 连续拒绝: {profile.rejection_count}")
        print(f"{'='*60}\n")
        # 3.1 构建prompt 
        strategy_context = reflector.get_strategy_context()
        current_prompt = '\n'.join(prompt_history) + strategy_context

        # 3.2 调用LLM进行思考
        llm_output = llm.generate(
            current_prompt, system_prompt=AGENT_SYSTEM_PROMPT
        )
        # 3.3 解析
        parsed = parse_thought_action(llm_output)
        if not parsed:
            print("❌ 解析失败，重新尝试")
            prompt_history.append("Observation: 错误：请严格使用 Thought:... Action:... 格式")
            continue
        
        thought, action_str = parsed
        print(f"💭 Thought: {thought[:200]}...")
        print(f"🎯 Action: {action_str}")
        
        # 3.4. 提取并记录偏好（记忆功能）
        prefs = extract_preferences_from_thought(thought)
        if prefs:
            profile.preferences.update(prefs)
            print(f"📝 [记忆更新] 记录偏好: {prefs}")
        
        # 3.5. 检查是否结束
        if "finish" in action_str.lower():
            # 提取最终答案
            match = re.search(r'finish\(["\']?(.*?)["\']?\)', action_str, re.DOTALL)
            final_answer = match.group(1) if match else "任务完成"
            print(f"\n{'='*60}")
            print(f"✅ 任务完成: {final_answer[:200]}...")
            print(f"{'='*60}")
            print(f"\n📋 最终用户档案:")
            print(json.dumps(profile.to_dict(), indent=2, ensure_ascii=False))
            break

        # 3.6. 解析工具调用
        tool_call = parse_tool_call(action_str)
        if not tool_call:
            print("❌ 工具调用解析失败")
            prompt_history.append(f"Observation: 错误：无法解析Action")
            continue

        # 3.7. 执行工具（支持备选方案自动降级）
        if tool_call["type"] == "tool":
            tool_name = tool_call["name"]
            kwargs = tool_call["kwargs"]
            
            # 注入用户偏好到相关工具
            if tool_name in ["get_attraction", "get_similar_attraction"]:
                if "preference" not in kwargs and profile.preferences.get("景点类型"):
                    kwargs["preference"] = profile.preferences["景点类型"]
            
            # 执行
            result = tools.execute(tool_name, **kwargs)
            
            # 构建 Observation
            if result.success:
                observation = result.data
                if result.is_fallback:
                    print(f"⚠️ [备选方案已激活] {result.fallback_reason}")
                    print(f"   使用工具: {result.used_tool}")
            else:
                observation = f"错误: {result.data}"
                print(f"❌ 工具执行失败: {observation}")
            
            prompt_history.append(f"Thought: {thought}")
            prompt_history.append(f"Action: {action_str}")
            prompt_history.append(f"Observation: {observation}")
            
            # 3.8. 模拟用户反馈（演示策略反思）
            # 实际应用中这里等待真实用户输入
            if tool_name in ["get_attraction", "get_similar_attraction", "check_ticket"]:
                # 提取推荐内容
                rec_match = re.search(r'推荐|景点|(\w+公园|\w+博物馆|\w+故宫)', observation)
                current_rec = rec_match.group(0) if rec_match else "当前推荐"
                profile.current_recommendation = current_rec
                
                feedback = simulate_user_feedback(profile, current_rec)
                
                if feedback == "accept":
                    print(f"\n👤 用户反馈: 接受推荐 ✓")
                    reflector.record_acceptance(current_rec)
                    prompt_history.append(f"用户反馈: 接受")
                else:
                    print(f"\n👤 用户反馈: 拒绝 - '{feedback}'")
                    analysis = reflector.record_rejection(current_rec, feedback)
                    
                    if analysis:
                        print(f"\n🔔 [策略反思触发!]")
                        print(f"   检测到模式: {analysis['pattern']}")
                        print(f"   调整策略: {analysis['suggested_strategy']}")
                        print(f"   行动指令: {analysis['action_prompt'][:100]}...")
                        print(f"   约束条件: {analysis['constraints']}")
                    
                    prompt_history.append(f"用户反馈: 拒绝，原因: {feedback}")
        
        # 更新档案状态
        prompt_history[0] = f"[系统] 用户档案: {json.dumps(profile.to_dict(), ensure_ascii=False)}"
        
    else:
        print(f"\n⚠️ 达到最大循环次数")


if __name__ == '__main__':
    test_enhanced()

