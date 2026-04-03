# reference: https://github.com/datawhalechina/hello-agents/blob/main/code/chapter6/AutoGenDemo
# chap06: 智能体团队角色
#   ProductManager (产品经理): 负责将用户的模糊需求转化为清晰、可执行的开发计划。
#   Engineer (工程师): 依据开发计划，负责编写具体的应用程序代码。
#   CodeReviewer (代码审查员): 负责审查工程师提交的代码，确保其质量、可读性和健壮性。
#   UserProxy (用户代理): 代表最终用户，发起初始任务，并负责执行和验证最终交付的代码。
# Func: 实时显示比特币当前价格
# 增加 状态机驱动的显式状态流转（更精细控制）
# ================================================================================

import os
import asyncio
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
# 先测试一个版本，使用 OpenAI 客户端
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console


# 加载环境变量
load_dotenv()

class DevState(Enum):
    """扩展的开发流程状态"""
    REQUIREMENT_ANALYSIS = "需求分析"          # PM
    IMPLEMENTATION = "开发实现"                # Engineer  
    CODE_REVIEW = "代码审查"                   # CodeReviewer
    QA_TESTING = "QA测试"                      # QualityAssurance ⭐新增
    BUG_FIX = "缺陷修复"                       # Engineer（来自QA回退）
    PRODUCT_VALIDATION = "产品验收"            # PM（最终确认）
    USER_ACCEPTANCE = "用户验收"               # UserProxy
    COMPLETED = "完成"

@dataclass
class StateTransition:
    """状态转移定义"""
    from_state: DevState
    to_state: DevState
    condition: str  # 触发条件关键词
    reason: str     # 转移原因说明
    

# 扩展状态转移图（新增QA相关流转）
STATE_TRANSITIONS = [
    # 正向流程
    StateTransition(DevState.REQUIREMENT_ANALYSIS, DevState.IMPLEMENTATION, 
                   "请工程师开始实现", "需求分析完成"),
    StateTransition(DevState.IMPLEMENTATION, DevState.CODE_REVIEW, 
                   "请代码审查员检查", "开发完成"),
    StateTransition(DevState.CODE_REVIEW, DevState.QA_TESTING, 
                   "请QA工程师测试", "代码审查通过"),  # ⭐新增
    StateTransition(DevState.QA_TESTING, DevState.USER_ACCEPTANCE, 
                   "QA测试通过，请用户代理验收", "测试通过"),  # ⭐新增
    StateTransition(DevState.USER_ACCEPTANCE, DevState.COMPLETED, 
                   "TERMINATE", "用户验收通过"),
    
    # 回退流程（原有 + 新增QA相关）
    StateTransition(DevState.IMPLEMENTATION, DevState.REQUIREMENT_ANALYSIS, 
                   "需求变更", "开发中发现需求问题"),
    StateTransition(DevState.CODE_REVIEW, DevState.REQUIREMENT_ANALYSIS, 
                   "需求不清晰", "审查发现需求缺陷"),
    StateTransition(DevState.CODE_REVIEW, DevState.IMPLEMENTATION, 
                   "代码不通过", "代码质量不达标"),
    
    # ⭐ QA测试相关回退
    StateTransition(DevState.QA_TESTING, DevState.IMPLEMENTATION, 
                   "【回退-工程师】", "发现严重缺陷需修复"),
    StateTransition(DevState.QA_TESTING, DevState.REQUIREMENT_ANALYSIS, 
                   "【回退-产品经理】", "实现与需求存在偏差"),
    StateTransition(DevState.BUG_FIX, DevState.QA_TESTING, 
                   "修复完成，请重新测试", "缺陷修复后回归测试"),
]



class StateMachineOrchestrator:
    """状态机驱动的协作编排器"""
    
    # 更新状态-智能体映射
    STATE_AGENT_MAP = {
        DevState.REQUIREMENT_ANALYSIS: "ProductManager",
        DevState.IMPLEMENTATION: "Engineer",
        DevState.CODE_REVIEW: "CodeReviewer",
        DevState.QA_TESTING: "QualityAssurance",  # ⭐新增
        DevState.BUG_FIX: "Engineer",  # 缺陷修复仍由工程师执行
        DevState.PRODUCT_VALIDATION: "ProductManager",
        DevState.USER_ACCEPTANCE: "UserProxy",
    }

    def __init__(self, agents: Dict[str, AssistantAgent], model_client):
        self.agents = agents
        self.model_client = model_client
        self.current_state = DevState.REQUIREMENT_ANALYSIS
        self.history: List[tuple] = []  # (state, agent, message)
        self.transition_count = 0
        self.max_transitions = 20
        
    def get_next_agent(self) -> str:
        """根据当前状态获取下一个执行的智能体"""
        return self.STATE_AGENT_MAP.get(self.current_state, "ProductManager")
    
    def detect_transition(self, message: str) -> Optional[StateTransition]:
        """检测消息中是否包含状态转移信号"""
        message_lower = message.lower()
        
        for transition in STATE_TRANSITIONS:
            # 判断状态流转
            if transition.from_state == self.current_state:
                if transition.condition.lower() in message_lower:
                    return transition
        return None
    
    async def run(self, initial_task: str):
        """执行状态机驱动的协作流程"""
        current_message = initial_task
        
        print(f"🚀 启动状态机协作流程")
        print(f"📍 初始状态: {self.current_state.value}")
        print("=" * 60)
        
        while self.current_state != DevState.COMPLETED and self.transition_count < self.max_transitions:
            # 1. 确定当前执行者
            agent_name = self.get_next_agent()
            agent = self.agents[agent_name]
            
            print(f"\n🔹 [{self.current_state.value}] → {agent_name}")
            
            # 2. 执行当前智能体
            response = await self._execute_agent(agent, current_message)
            print(f"   {agent_name}: {response[:100]}...")
            
            # 3. 记录历史
            self.history.append((self.current_state, agent_name, response))
            
            # 4. 检测状态转移
            transition = self.detect_transition(response)
            
            if transition:
                print(f"   ↺ 状态转移: {transition.from_state.value} → {transition.to_state.value}")
                print(f"      原因: {transition.reason}")
                self.current_state = transition.to_state
                self.transition_count += 1
                
                # 如果回退到上游，需要构造上下文消息
                if self._is_backward_transition(transition):
                    current_message = self._construct_backtrack_message(transition, response)
                else:
                    current_message = response
            else:
                # 无明确转移信号，尝试默认流转
                next_state = self._get_default_next_state()
                if next_state:
                    self.current_state = next_state
                    current_message = response
                else:
                    print("⚠️ 无法确定下一步，终止流程")
                    break
        
        print("=" * 60)
        print(f"✅ 流程结束 | 最终状态: {self.current_state.value}")
        return self.history
    
    def _is_backward_transition(self, transition: StateTransition) -> bool:
        """判断是否为回退转移"""
        # 根据枚举定义的顺序判断
        state_order = list(DevState)
        from_idx = state_order.index(transition.from_state)
        to_idx = state_order.index(transition.to_state)
        return to_idx < from_idx

    def _construct_backtrack_message(self, transition: StateTransition, response: str) -> str:
        """构造回退时的上下文消息，保留历史信息"""
        # 获取相关历史上下文
        context = f"""
【流程回退通知】
当前步骤: {transition.from_state.value}
目标步骤: {transition.to_state.value}
回退原因: {transition.reason}

原始消息:
{response}

历史上下文:
{self._format_history(3)}  # 最近3轮

请基于以上信息继续处理。
"""
        return context
    
    def _format_history(self, n: int) -> str:
        """格式化最近n轮历史"""
        recent = self.history[-n:] if len(self.history) >= n else self.history
        return "\n".join([
            f"  [{h[0].value}] {h[1]}: {h[2][:50]}..." 
            for h in recent
        ])
    
    def _get_default_next_state(self) -> Optional[DevState]:
        """获取默认的下一个状态（简单顺序流转）"""
        flow = [
            DevState.REQUIREMENT_ANALYSIS,
            DevState.IMPLEMENTATION,
            DevState.CODE_REVIEW,
            DevState.USER_ACCEPTANCE,
            DevState.COMPLETED
        ]
        try:
            idx = flow.index(self.current_state)
            return flow[idx + 1] if idx + 1 < len(flow) else None
        except ValueError:
            return None
    
    async def _execute_agent(self, agent: AssistantAgent, message: str) -> str:
        """执行单个智能体（简化版，实际使用AutoGen的run）"""
        # 这里使用AutoGen的实际调用方式
        from autogen_agentchat.messages import TextMessage
        response = await agent.on_messages(
            [TextMessage(content=message, source="user")],
            cancellation_token=None
        )
        return response.chat_message.content


def create_qa_engineer(model_client):
    """创建测试工程师（QA）智能体"""
    system_message = """你是一位资深的测试工程师（QA），专注于软件质量保障和自动化测试。

## 核心职责
1. **测试策略制定**：根据需求分析设计全面的测试方案
2. **自动化测试开发**：编写可执行的自动化测试代码
3. **测试执行与报告**：运行测试并生成详细的测试报告
4. **缺陷跟踪**：识别、分类和跟踪软件缺陷

## 技术专长
- **测试框架**：pytest、unittest、Selenium、Playwright
- **API测试**：requests、httpx、Postman脚本转换
- **性能测试**：locust、k6基础测试
- **测试类型**：单元测试、集成测试、E2E测试、边界测试

## 工作流程

### 阶段1：测试分析（收到代码后）
1. 理解产品需求和代码实现
2. 识别关键功能路径和边界场景
3. 设计测试用例（正常/异常/边界）
4. 确定测试优先级（P0必须通过，P1重要，P2一般）

### 阶段2：测试实现
1. 编写自动化测试脚本
2. 准备测试数据（mock数据、fixtures）
3. 配置测试环境

### 阶段3：测试执行与反馈
执行测试后，根据结果选择下一步：

✅ **测试全部通过**（P0/P1无失败）
→ 回复："QA测试通过，请用户代理验收" + 测试摘要

⚠️ **发现严重缺陷**（P0失败或核心功能异常）
→ 回复："【回退-工程师】发现严重缺陷：[具体描述]" + 测试报告

🔍 **发现需求偏差**（实现与需求不符）
→ 回复："【回退-产品经理】实现与需求存在偏差：[具体描述]"

📋 **需要澄清**（需求模糊无法测试）
→ 回复："【回退-产品经理】需求需要澄清：[具体问题]"

## 输出格式规范

### 测试报告模板
-------- markdown --------
## 测试执行报告

### 1. 测试概览
- 测试范围：[功能模块]
- 测试类型：[单元/集成/E2E]
- 执行时间：[timestamp]

### 2. 测试用例执行结果
| 用例ID | 描述 | 优先级 | 状态 | 备注 |
|--------|------|--------|------|------|
| TC-001 | [描述] | P0 | ✅通过 | - |
| TC-002 | [描述] | P0 | ❌失败 | [错误信息] |

### 3. 缺陷汇总
- **严重（Blocker）**：X个
- **高（Critical）**：X个  
- **中（Major）**：X个
- **低（Minor）**：X个

### 4. 测试代码
-------- markdown --------
# 可执行的测试代码
## 协作原则
- 测试是质量的守门员，但不是代码的敌人
- 提供可复现的测试步骤和明确的缺陷描述
- 优先保障核心功能（P0）的稳定性
- 与工程师协作修复缺陷，而非仅报告问题

收到代码审查通过的代码后，请立即开始测试分析和执行。"""

    return AssistantAgent(
        name="QualityAssurance",
        model_client=model_client,
        system_message=system_message,
    )



def create_openai_model_client():
    """创建 OpenAI 模型客户端用于测试"""
    return OpenAIChatCompletionClient(
        model=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        model_info={
            "function_calling": True,
            "max_tokens": 4096,
            "context_length": 32768,
            "vision": False,
            "json_output": True,
            "family": "deepseek",
            "structured_output": True,
        }
    )


def create_product_manager(model_client):
    """创建产品经理智能体"""
    system_message = """你是一位经验丰富的产品经理，专门负责软件产品的需求分析和项目规划。

你的核心职责包括：
1. **需求分析**：深入理解用户需求，识别核心功能和边界条件
2. **技术规划**：基于需求制定清晰的技术实现路径
3. **风险评估**：识别潜在的技术风险和用户体验问题
4. **协调沟通**：与工程师和其他团队成员进行有效沟通

当接到开发任务时，请按以下结构进行分析：
1. 需求理解与分析
2. 功能模块划分
3. 技术选型建议
4. 实现优先级排序
5. 验收标准定义

请简洁明了地回应，并在分析完成后说"请工程师开始实现"。"""

    return AssistantAgent(
        name="ProductManager",
        model_client=model_client,
        system_message=system_message,
    )


def create_engineer(model_client):
    """创建软件工程师智能体"""
    system_message = """你是一位资深的软件工程师，擅长 Python 开发和 Web 应用构建。

你的技术专长包括：
1. **Python 编程**：熟练掌握 Python 语法和最佳实践
2. **Web 开发**：精通 Streamlit、Flask、Django 等框架
3. **API 集成**：有丰富的第三方 API 集成经验
4. **错误处理**：注重代码的健壮性和异常处理

当收到开发任务时，请：
1. 仔细分析技术需求
2. 选择合适的技术方案
3. 编写完整的代码实现
4. 添加必要的注释和说明
5. 考虑边界情况和异常处理

请提供完整的可运行代码，并在完成后说"请代码审查员检查"。"""

    return AssistantAgent(
        name="Engineer",
        model_client=model_client,
        system_message=system_message,
    )


def create_code_reviewer(model_client):
    """创建代码审查员智能体"""
    system_message = """你是一位经验丰富的代码审查专家，专注于代码质量和最佳实践。

你的审查重点包括：
1. **代码质量**：检查代码的可读性、可维护性和性能
2. **安全性**：识别潜在的安全漏洞和风险点
3. **最佳实践**：确保代码遵循行业标准和最佳实践
4. **错误处理**：验证异常处理的完整性和合理性

审查流程：
1. 仔细阅读和理解代码逻辑
2. 检查代码规范和最佳实践
3. 识别潜在问题和改进点
4. 提供具体的修改建议
5. 评估代码的整体质量

请提供具体的审查意见，完成后说"代码审查完成，请用户代理测试"。"""

    return AssistantAgent(
        name="CodeReviewer",
        model_client=model_client,
        system_message=system_message,
    )


def create_user_proxy():
    """创建用户代理智能体"""
    return UserProxyAgent(
        name="UserProxy",
        description="""用户代理，负责以下职责：
1. 代表用户提出开发需求
2. 执行最终的代码实现
3. 验证功能是否符合预期
4. 提供用户反馈和建议

完成测试后请回复 TERMINATE。""",
    )
    

async def run_software_development_team():
    """运行软件开发团队协作"""
    
    print("🔧 正在初始化模型客户端...")
    model_client = create_openai_model_client()
    print("👥 正在创建智能体团队...")
    # 创建智能体团队
    product_manager = create_product_manager(model_client)
    engineer = create_engineer(model_client)
    code_reviewer = create_code_reviewer(model_client)
    user_proxy = create_user_proxy()
    
    # 添加终止条件
    termination = TextMentionTermination("TERMINATE")
    
    # 创建团队聊天
    team_chat = RoundRobinGroupChat(
        participants=[
            product_manager,
            engineer, 
            code_reviewer,
            user_proxy
        ],
        termination_condition=termination,
        max_turns=20,  # 增加最大轮次
    )

    # 定义开发任务
    task = """我们需要开发一个比特币价格显示应用，具体要求如下：

核心功能：
- 实时显示比特币当前价格（USD）
- 显示24小时价格变化趋势（涨跌幅和涨跌额）
- 提供价格刷新功能

技术要求：
- 使用 Streamlit 框架创建 Web 应用
- 界面简洁美观，用户友好
- 添加适当的错误处理和加载状态

请团队协作完成这个任务，从需求分析到最终实现。"""
    
    # 执行团队协作
    print("🚀 启动 AutoGen 软件开发团队协作...")
    print("=" * 60)
    
    # 使用 Console 来显示对话过程
    result = await Console(team_chat.run_stream(task=task))
    
    print("\n" + "=" * 60)
    print("✅ 团队协作完成！")
    
    return result


# 使用示例
async def run_with_state_machine():
    model_client = create_openai_model_client()
    
    agents = {
        "ProductManager": create_product_manager(model_client),
        "Engineer": create_engineer(model_client),
        "CodeReviewer": create_code_reviewer(model_client),
        "UserProxy": create_user_proxy(),
    }
    
    orchestrator = StateMachineOrchestrator(agents, model_client)
    
    task = """开发比特币价格显示应用...
    
【重要】请使用以下状态转移信号：
- 需求分析完成: "请工程师开始实现"
- 开发完成: "请代码审查员检查"  
- 代码需修改: "代码不通过，需要修改"
- 需求有问题: "需求变更/需求不清晰"
- 审查通过: "请QA工程师测试"                    # ⭐修改
- QA测试通过: "QA测试通过，请用户代理验收"      # ⭐新增
- QA回退工程师: "【回退-工程师】缺陷描述：..."    # ⭐新增
- QA回退产品: "【回退-产品经理】需求偏差：..."   # ⭐新增
- 验收通过: "TERMINATE"

核心功能：
- 实时显示比特币当前价格（USD）
- 显示24小时价格变化趋势（涨跌幅和涨跌额）
- 提供价格刷新功能

技术要求：
- 使用 Streamlit 框架创建 Web 应用
- 界面简洁美观，用户友好
- 添加适当的错误处理和加载状态

请团队协作完成这个任务，从需求分析到最终实现。"""
    
    return await orchestrator.run(task)
    

# 主程序入口
if __name__ == "__main__":
    try:
        # 运行异步协作流程
        # result = asyncio.run(run_software_development_team())
        result = asyncio.run(run_with_state_machine())
        
        print(f"\n📋 协作结果摘要：")
        print(f"- 参与智能体数量：4个")
        print(f"- 任务完成状态：{'成功' if result else '需要进一步处理'}")
        
    except ValueError as e:
        print(f"❌ 配置错误：{e}")
        print("请检查 .env 文件中的配置是否正确")
    except Exception as e:
        print(f"❌ 运行错误：{e}")
        import traceback
        traceback.print_exc()
