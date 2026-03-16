
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime

# ============ 数据类定义 ============

@dataclass
class UserProfile:
    """用户偏好记忆"""
    preferences: Dict[str, Any] = field(default_factory=dict)  # 如 {"景点类型": "历史文化", "预算": "中等"}
    rejected_items: List[Dict] = field(default_factory=list)   # 被拒绝的推荐历史
    accepted_items: List[Dict] = field(default_factory=list)   # 被接受的推荐历史
    rejection_count: int = 0                                   # 连续拒绝计数
    last_strategy: str = "default"                             # 当前策略标识
    current_recommendation: str = ""                           # 当前推荐项（用于跟踪反馈）
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ToolResult:
    """工具执行结果，支持备选方案"""
    success: bool
    data: str
    used_tool: str
    fallback_reason: Optional[str] = None
    is_fallback: bool = False



class StrategyReflector:
    """策略反思模块"""
    
    STRATEGIES = {
        "default": "基于用户原始请求和当前条件进行标准推荐",
        "diverse": "避开历史拒绝项，推荐不同类型/风格的景点",
        "budget_flexible": "调整预算范围，推荐性价比更高的选项",
        "crowd_avoid": "避开热门景点，推荐小众但优质的替代方案",
        "interactive": "主动询问用户具体偏好，而非直接推荐",
        "weather_adaptive": "根据天气变化调整推荐类型"
    }
    
    def __init__(self, profile: UserProfile):
        self.profile = profile
    
    def record_rejection(self, item: str, reason: str = None) -> Optional[Dict]:
        """记录拒绝，检查是否需要策略调整"""
        self.profile.rejected_items.append({
            "item": item,
            "reason": reason or "未说明原因",
            "time": datetime.now().isoformat(),
            "strategy": self.profile.last_strategy
        })
        self.profile.rejection_count += 1
        
        # 触发反思
        if self.profile.rejection_count >= 3:
            return self._reflect_and_adjust()
        return None
    
    def record_acceptance(self, item: str):
        """记录接受，重置拒绝计数"""
        self.profile.accepted_items.append({
            "item": item,
            "time": datetime.now().isoformat()
        })
        self.profile.rejection_count = 0
        self.profile.current_recommendation = ""
    
    def _reflect_and_adjust(self) -> Dict[str, Any]:
        """策略反思：分析拒绝模式，生成调整建议"""
        recent_rejections = self.profile.rejected_items[-3:]
        
        # 分析拒绝原因模式
        reasons = [r.get("reason", "") for r in recent_rejections]
        items = [r["item"] for r in recent_rejections]
        
        analysis = self._analyze_pattern(reasons, items)
        
        # 更新策略
        self.profile.last_strategy = analysis["suggested_strategy"]
        self.profile.rejection_count = 0  # 重置计数
        
        return analysis
    
    def _analyze_pattern(self, reasons: List[str], items: List[str]) -> Dict[str, Any]:
        """分析拒绝模式并给出建议"""
        
        # 模式1: 预算相关
        if any("贵" in r or "预算" in r or "钱" in r for r in reasons):
            return {
                "pattern": "budget_sensitive",
                "suggested_strategy": "budget_flexible",
                "reasoning": "连续3次因预算原因拒绝，建议调整价格区间或推荐免费景点",
                "action_prompt": "用户预算敏感，请推荐：1)免费景点 2)高性价比选项 3)明确询问预算范围",
                "constraints": ["避开高价景点", "强调免费或低价优势"]
            }
        
        # 模式2: 人多/拥挤
        if any("人多" in r or "拥挤" in r or "排队" in r for r in reasons):
            return {
                "pattern": "crowd_averse",
                "suggested_strategy": "crowd_avoid",
                "reasoning": "连续拒绝热门景点，偏好小众、人少体验",
                "action_prompt": "用户不喜欢拥挤，请推荐：1)小众景点 2)非热门时段 3)强调'人少安静'特点",
                "constraints": ["避开网红打卡地", "强调私密性和独特体验"]
            }
        
        # 模式3: 类型不匹配
        if len(set(items)) >= 3:
            return {
                "pattern": "type_mismatch",
                "suggested_strategy": "interactive",
                "reasoning": "推荐多样性已尝试但未命中，需明确具体需求",
                "action_prompt": "连续推荐未符合预期，请先停止推荐，询问：1)具体景点类型偏好 2)活动强度要求 3)特殊兴趣点",
                "constraints": ["不要直接推荐", "必须先问清楚需求"]
            }
        
        # 模式4: 默认策略调整
        return {
            "pattern": "general_rejection",
            "suggested_strategy": "diverse",
            "reasoning": "未明确模式，建议扩大推荐多样性并避开历史拒绝项",
            "action_prompt": f"请避开以下已拒绝选项：{', '.join(items)}。尝试完全不同类型，并说明推荐理由",
            "constraints": [f"禁止推荐: {', '.join(items)}"]
        }
    
    def get_strategy_context(self) -> str:
        """获取当前策略的上下文提示"""
        strategy_desc = self.STRATEGIES.get(self.profile.last_strategy, "")
        rejected_list = [r["item"] for r in self.profile.rejected_items[-5:]]
        
        context = f"\n[当前策略: {self.profile.last_strategy}] {strategy_desc}\n"
        context += f"[历史拒绝] {len(self.profile.rejected_items)}次，近期: {', '.join(rejected_list)}\n"
        
        if self.profile.rejection_count > 0:
            context += f"[注意] 已连续拒绝{self.profile.rejection_count}次，再拒绝{3-self.profile.rejection_count}次将触发策略反思\n"
        
        return context


class ToolRegistry:
    """工具注册表，支持自动备选降级"""
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.fallback_chains: Dict[str, List[str]] = {}  # 主工具 -> 备选工具列表
    
    def register(self, name: str, func: Callable, fallbacks: List[str] = None):
        self.tools[name] = func
        if fallbacks:
            self.fallback_chains[name] = fallbacks
    
    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """
        执行工具，支持自动降级到备选方案
        """
        chain = [tool_name] + self.fallback_chains.get(tool_name, [])
        
        for i, current_tool in enumerate(chain):
            if current_tool not in self.tools:
                continue
            
            try:
                result = self.tools[current_tool](**kwargs)
                
                # 检查结果是否有效
                if self._is_valid(result):
                    return ToolResult(
                        success=True,
                        data=result,
                        used_tool=current_tool,
                        fallback_reason=f"主工具{tool_name}不可用，已自动降级" if i > 0 else None,
                        is_fallback=(i > 0)
                    )
                else:
                    # 结果无效（如售罄），继续尝试下一个备选
                    continue
                    
            except Exception as e:
                # 执行失败，继续尝试备选
                continue
        
        # 所有工具都失败
        return ToolResult(
            success=False,
            data=f"错误: {tool_name}及其备选方案均不可用",
            used_tool=None,
            fallback_reason="所有备选 exhausted"
        )
    
    def _is_valid(self, result: str) -> bool:
        """检查结果是否有效（未售罄、无错误）"""
        if not result or "错误" in result:
            return False
        
        invalid_signals = [
            "售罄", "sold out", "无票", "不可用", 
            "约满", "关闭", "维护中", "null", "None"
        ]
        result_lower = result.lower()
        return not any(signal in result_lower for signal in invalid_signals)
