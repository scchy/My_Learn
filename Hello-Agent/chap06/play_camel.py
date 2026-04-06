# reference: https://github.com/datawhalechina/hello-agents/blob/main/code/chapter6/CAMEL
# chap06:  短篇电子书 camel-ai==0.2.75
# Func: 心理学家 + 作家
# =================================================================================

from colorama import Fore
from camel.societies import RolePlaying
from camel.utils import print_text_animated
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from dotenv import load_dotenv
import os
import re


load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL_ID")


model = ModelFactory.create(
    model_platform=ModelPlatformType.MOONSHOT,
    model_type=LLM_MODEL, # 'kimi-k2-0711-preview' 
    url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    model_config_dict={
        "max_tokens":  2*4096,
        "temperature": 1, 
    }
)


# 定义协作任务
task_prompt = """
创作一本关于"拖延症心理学"的短篇电子书，目标读者是对心理学感兴趣的普通大众。
要求：
1. 内容科学严谨，基于实证研究
2. 语言通俗易懂，避免过多专业术语
3. 包含实用的改善建议和案例分析
4. 篇幅控制在8000-10000字
5. 结构清晰，包含引言、核心章节和总结

【重要】当你认为任务已完成时，回复中必须包含 <CAMEL_TASK_DONE> 标志。
当你认为对方提出的完成提议不合理时，回复中必须包含 <CAMEL_TASK_REJECT> 标志并说明理由。
"""

print(Fore.YELLOW + f"协作任务:\n{task_prompt}\n")



# 初始化角色扮演会话
role_play_session = RolePlaying(
    assistant_role_name="心理学家", 
    user_role_name="作家", 
    task_prompt=task_prompt,
    model=model
)

print(Fore.CYAN + f"具体任务描述:\n{role_play_session.task_prompt}\n")


def check_termination_status(assistant_msg: str, user_msg: str) -> tuple:
    """
    检测双方终止意图
    返回: (assistant_done, user_done, conflict_detected, reject_reason)
    """
    assistant_done = "CAMEL_TASK_DONE" in assistant_msg
    user_done = "CAMEL_TASK_DONE" in user_msg
    assistant_reject = "CAMEL_TASK_REJECT" in assistant_msg
    user_reject = "CAMEL_TASK_REJECT" in user_msg

    # 冲突情况：一方想终止，另一方明确反对
    conflict = (assistant_done and user_reject) or (user_done and assistant_reject)
    # 提取拒绝理由（简单正则）
    reject_reason = ""
    if conflict:
        reject_match = re.search(r'<CAMEL_TASK_REJECT>(.*?)(?=<|$)', 
                                assistant_msg if user_done else user_msg, 
                                re.DOTALL)
        if reject_match:
            reject_reason = reject_match.group(1).strip()[:200]  # 限制长度
    
    return assistant_done, user_done, conflict, reject_reason



def resolve_conflict(assistant_msg: str, user_msg: str, turn: int, max_turns: int) -> str:
    """
    极简仲裁：基于回合数和信心度做决策
    """
    assistant_conf = _extract_confidence(assistant_msg)
    user_conf = _extract_confidence(user_msg)
    
    # 规则1：接近上限，强制终止
    if turn >= max_turns - 3:
        return "force_terminate"
    
    # 规则2：高信心方胜（信心差距>2）
    if abs(assistant_conf - user_conf) > 2:
        return "terminate" if max(assistant_conf, user_conf) == assistant_conf else "continue"
    
    # 规则3：保守策略，默认继续
    return "continue"


def _extract_confidence(msg: str) -> int:
    """从消息中提取信心值（1-10），默认5"""
    match = re.search(r'信心[:：]\s*(\d+)', msg)
    if match:
        return min(10, max(1, int(match.group(1))))
    # 通过关键词估算
    strong_words = ['完成', '达成', '满足', '符合']
    weak_words = ['需要', '改进', '不足', '缺少']
    score = 5 + sum(1 for w in strong_words if w in msg) - sum(1 for w in weak_words if w in msg)
    return min(10, max(1, score))


# 开始协作对话
chat_turn_limit, n = 30, 0
input_msg = role_play_session.init_chat()
termination_pending = False  # 标记是否有待确认的终止提议
pending_proposer = None      # "心理学家" 或 "作家"

while n < chat_turn_limit:
    n += 1
    assistant_response, user_response = role_play_session.step(input_msg)
        
    a_msg = assistant_response.msg.content
    u_msg = user_response.msg.content
    
    print_text_animated(Fore.BLUE + f"作家:\n\n{user_response.msg.content}\n")
    print_text_animated(Fore.GREEN + f"心理学家:\n\n{assistant_response.msg.content}\n")
    
    # 检查任务完成标志
    if "CAMEL_TASK_DONE" in user_response.msg.content:
        print(Fore.MAGENTA + "✅ 电子书创作完成！")
        break
    
    # 检测终止状态
    a_done, u_done, conflict, reason = check_termination_status(a_msg, u_msg)
    # 情况1：双方一致同意终止
    if a_done and u_done:
        print(Fore.MAGENTA + "✅ 双方确认：电子书创作完成！")
        break
    
    # 情况2：冲突检测（一方想终止，另一方反对）
    if conflict:
        print(Fore.RED + f"⚠️ 检测到分歧！{('心理学家' if u_done else '作家')}反对终止")
        if reason:
            print(Fore.RED + f"反对理由: {reason}")
        
        # 极简仲裁
        decision = resolve_conflict(a_msg, u_msg, n, chat_turn_limit)
        
        if decision == "force_terminate":
            print(Fore.MAGENTA + "⏰ 接近回合上限，强制终止")
            break
        elif decision == "terminate":
            print(Fore.MAGENTA + "✅ 仲裁结果：满足终止条件，任务完成")
            break
        else:
            print(Fore.YELLOW + "🔄 仲裁结果：继续协作，完善内容")
            # 在输入中注入仲裁提示，引导双方聚焦分歧点
            input_msg = assistant_response.msg
            input_msg.content += f"\n\n【系统提示】双方对完成度有分歧，请针对以下问题继续讨论：{reason or '完善内容细节'}"
            continue
    
    # 情况3：单方提议终止，等待对方确认
    if (a_done or u_done) and not termination_pending:
        termination_pending = True
        pending_proposer = "心理学家" if a_done else "作家"
        print(Fore.YELLOW + f"⏸️ {pending_proposer}提议终止，等待另一方确认...")
        # 不break，继续下一轮让对方表态
    
    # 情况4：单方提议但已pending（对方未反对也未同意，继续讨论）
    elif termination_pending and not (a_done or u_done):
        termination_pending = False  # 重置，继续正常流程
    
    input_msg = assistant_response.msg

print(Fore.YELLOW + f"总共进行了 {n} 轮协作对话")





